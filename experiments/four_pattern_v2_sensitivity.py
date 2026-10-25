"""Frozen synthetic audit of v2 MAP policies (not calibration or auto-tuning).

Run from the engine repository root with PYTHONPATH=expdoe-dk/src. Only
EngineError failures are contained; unexpected harness errors abort the audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import permutations, product
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from expdoe_dk import EngineError, ErrorCode, Knowledge, ObservationBatch, Parameter, Space, fit_surrogate
from expdoe_dk.bo.evaluation import _reference_classes, _subset
from expdoe_dk.bo.knowledge_runtime import compile_runtime
from expdoe_dk.bo.structured_fit import fit_structured_gp
from expdoe_dk.bo.surrogates import FittedSurrogate, _PhysicalFeatures, _model_inputs


ROOT = Path(__file__).resolve().parents[1]
SIGNS = {"increasing": 1., "decreasing": -1., "flat": 0.}
CAVEATS = [
    "GP intervals are latent-response intervals conditional on point-estimate MAP parameters; parameter uncertainty is not integrated.",
    "Coverage is against noiseless generating truth, not a calibrated observation-prediction interval.",
    "Mean and residual GP can be confounded; fitted signs do not establish empirical or causal validity, and the full GP need not preserve component shape.",
    "Prior sensitivity means a strict maximum pairwise grid-mean difference greater than one training response scale; it is not a significance test.",
    "Constant training responses use S=1. Confidence and prior constants are engineering policy, not calibrated probabilities.",
    "Ridge and random forest provide point predictions only; their uncertainty is unavailable. They are never production fallbacks.",
    "No fixture, fold, generator, start, policy or audit-grid point is selected after observing a fit result.",
]


def default_policy():
    return {"seeds": [17, 29, 43], "magnitude_scale_multipliers": [.5, 1., 2.],
            "width_prior_centers": [.025, .05, .10]}


def policy_grid(seeds=None, scale_multipliers=None, width_centers=None):
    defaults = default_policy()
    return [dict(seed=int(seed), magnitude_scale_multiplier=float(scale), width_prior_center=float(width))
            for seed, scale, width in product(
                defaults["seeds"] if seeds is None else seeds,
                defaults["magnitude_scale_multipliers"] if scale_multipliers is None else scale_multipliers,
                defaults["width_prior_centers"] if width_centers is None else width_centers)]


def is_prior_sensitive(means: np.ndarray, training_scale: float) -> bool:
    """Strict max pairwise difference; fewer than two successes are not evidence."""
    values = np.asarray(means, dtype=float)
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1
            or not np.isfinite(values).all() or not np.isfinite(training_scale) or training_scale <= 0):
        raise ValueError("Need at least two finite prediction grids and a positive training scale")
    return bool(np.max(np.ptp(values, axis=0)) > training_scale)


def summarize_sensitivity(means, training_scale):
    if len(means) < 2:
        return {"status": "not_evaluable", "successful_configurations": len(means),
                "max_pairwise_change": None, "training_scale": training_scale}
    return {"status": "prior_sensitive" if is_prior_sensitive(np.asarray(means), training_scale) else "not_flagged",
            "successful_configurations": len(means),
            "max_pairwise_change": float(np.max(np.ptp(np.asarray(means), axis=0))),
            "training_scale": training_scale}


@dataclass
class Fixture:
    name: str
    space: Space
    observations: ObservationBatch
    truth: np.ndarray
    grid: pd.DataFrame
    correct: Knowledge
    wrong: Knowledge
    generator: dict
    default_noiseless: bool
    folds: tuple[tuple[int, ...], ...]

    def describe(self):
        result = dict(name=self.name, space=self.space.to_dict(),
            observations={"X": self.observations.X.to_dict(orient="records"),
                          "Y": self.observations.Y.to_dict(orient="records"),
                          "Yvar": self.observations.Yvar.to_dict(orient="records"),
                          "ids": list(self.observations.ids)},
            latent_truth=self.truth.tolist(), grid=self.grid.to_dict(orient="records"),
            correct=self.correct.to_envelope(), wrong=self.wrong.to_envelope(),
            generator=self.generator, default_noiseless=self.default_noiseless,
            folds=[list(fold) for fold in self.folds])
        result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result


def _knowledge(pattern, *, direction="increasing", below="increasing", above="flat",
               response_effect="higher", origin=0., span=1., composed=False, two_shapes=False):
    k = Knowledge()
    if pattern == "saturation" or composed:
        k.with_saturation("x", direction=direction, half_response=origin + .5 * span,
                          origin=origin, objective="response", version="2.0", confidence=.8)
    if pattern == "threshold":
        k.with_threshold("x", threshold=.5, below_behavior=below, above_behavior=above,
                         objective="response", version="2.0", confidence=.8)
    if two_shapes:
        k.with_threshold("z", threshold=.5, below_behavior="increasing", above_behavior="flat",
                         objective="response", version="2.0", confidence=.8)
    if pattern in {"synergy", "antagonism"} or composed:
        k.with_interaction("x", "z", kind=pattern if not composed else "synergy",
                           reference="anchored_additive", response_effect=response_effect,
                           objective="response", version="2.0", confidence=.8)
    return k


def build_fixtures():
    """Analytic coefficients are fixed independently of fitted prior constants.

    Complementary even/odd folds leave eight distinct x values in 1D and all
    eight x levels with two z levels in 2D. No randomized fold search is used.
    """
    definitions = [(f"saturation_{d}", "saturation", {"direction": d}) for d in ("increasing", "decreasing")]
    definitions += [(f"threshold_{a}_{b}", "threshold", {"below": a, "above": b})
                    for a, b in permutations(SIGNS, 2)]
    definitions += [(f"{k}_{d}", k, {"response_effect": d})
                    for k, d in product(("synergy", "antagonism"), ("higher", "lower"))]
    definitions += [("additive_only", "synergy", {}), ("zero_effect", "threshold", {}),
                    ("weak_noisy", "synergy", {}), ("changed_units", "saturation", {}),
                    ("missing_transition", "threshold", {}), ("composed", "synergy", {"composed": True}),
                    ("composed_two_shapes", "synergy", {"composed": True, "two_shapes": True})]
    fixtures = []
    for name, pattern, options in definitions:
        two = pattern in {"synergy", "antagonism"}
        points = (np.array(list(product((0., .1, .25, .4, .6, .75, .9, 1.), (0., .3, .7, 1.))))
                  if two else np.linspace(0., 1., 16)[:, None])
        if options.get("two_shapes"):
            points = np.array(list(product((0., .1, .25, .4, .6, .75, .9, 1.), repeat=2)))
        if name == "missing_transition":
            points = np.linspace(0., .4, 16)[:, None]
        u = points[:, 0]
        v = points[:, 1] if two else np.zeros(len(u))
        amplitude = 1.5 if pattern == "saturation" else 1.2
        width, p, q = .08, 1.1, .7
        if pattern == "saturation":
            truth = 4. + SIGNS[options.get("direction", "increasing")] * amplitude * u / (.5 + u)
        elif pattern == "threshold":
            p *= SIGNS[options.get("below", "increasing")]
            q *= SIGNS[options.get("above", "flat")]
            H = width * (np.logaddexp(0., (u - .5) / width) - np.logaddexp(0., -.5 / width))
            truth = 4. + p * u + (q - p) * H
        else:
            sign = (1. if pattern == "synergy" else -1.) * (1. if options.get("response_effect", "higher") == "higher" else -1.)
            amplitude = 0. if name == "additive_only" else (.03 if name == "weak_noisy" else amplitude)
            truth = 4. + .4 * u - .2 * v + sign * amplitude * u * v
            if options.get("composed"):
                truth = 4. + 1.5 * u / (.5 + u) - .2 * v + amplitude * u * v
            if options.get("two_shapes"):
                p, q = .9, 0.
                Hz = width * (np.logaddexp(0., (v - .5) / width) - np.logaddexp(0., -.5 / width))
                truth = 4. + 1.5 * u / (.5 + u) + p * (v - Hz) + amplitude * u * v
        if name == "zero_effect":
            truth = np.full(len(u), 4.)
        origin, span, response_offset, response_multiplier = ((100., 1000., 273.15, 100.)
            if name == "changed_units" else (0., 1., 0., 1.))
        truth = response_offset + response_multiplier * truth
        noise_sd = .08 if name == "weak_noisy" else 0.
        noise = np.random.default_rng(20261024).normal(0., noise_sd, len(u)) * response_multiplier
        names = ["x", "z"] if two else ["x"]
        physical = points.copy()
        physical[:, 0] = origin + span * u
        space = Space([Parameter("x", bounds=(origin, origin + span))] +
                      ([Parameter("z", bounds=(0., 1.))] if two else []), objectives="response")
        X = pd.DataFrame(physical, columns=names)
        variance = max(noise_sd ** 2, .0004) * response_multiplier ** 2
        batch = ObservationBatch(X, pd.DataFrame({"response": truth + noise}),
            Yvar=pd.DataFrame({"response": np.full(len(u), variance)}),
            ids=tuple(f"{name}:{i}" for i in range(len(u))))
        kwargs = dict(options)
        if pattern == "saturation":
            kwargs.update(origin=origin, span=span)
        correct = _knowledge(pattern, **kwargs)
        wrong_kwargs = dict(kwargs)
        if pattern == "saturation":
            wrong_kwargs["direction"] = "decreasing" if kwargs.get("direction", "increasing") == "increasing" else "increasing"
        elif pattern == "threshold":
            wrong_kwargs.update(below=kwargs.get("above", "flat"), above=kwargs.get("below", "increasing"))
        else:
            wrong_kwargs["response_effect"] = "lower" if kwargs.get("response_effect", "higher") == "higher" else "higher"
        grid = (np.array(list(product(np.linspace(0., 1., 9), np.linspace(0., 1., 5))))
                if two else np.linspace(0., 1., 41)[:, None])
        grid[:, 0] = origin + span * grid[:, 0]
        fixtures.append(Fixture(name, space, batch, truth, pd.DataFrame(grid, columns=names), correct,
            _knowledge(pattern, **wrong_kwargs),
            {"formula": "anchored saturation / integrated softplus / additive plus signed product",
             "pattern": pattern, "options": options, "baseline": 4., "amplitude": amplitude,
             "below_slope": p, "above_slope": q, "width": width,
             "saturation_amplitude": 1.5 if options.get("composed") else (amplitude if pattern == "saturation" else None),
             "linear_x": .4 if two and not options.get("composed") else 0.,
             "linear_z": -.2 if two and not options.get("two_shapes") else 0., "noise_sd": noise_sd, "noise_seed": 20261024,
             "factor_origin": origin, "factor_multiplier": span, "response_offset": response_offset,
             "response_multiplier": response_multiplier, "known_variance": variance,
             "zero_response_override": name == "zero_effect"},
            name not in {"additive_only", "zero_effect", "weak_noisy", "missing_transition"},
            tuple(tuple(range(start, len(u), 2)) for start in (0, 1))))
    return fixtures


def _stats(training):
    y = training.Y.to_numpy(dtype=float)
    return float(y.mean()), 1. if np.all(y == y[0, 0]) else float(y.std())


def _fit_policy(fixture, training, knowledge, policy):
    """Use F4's internal override seam with the public route's physical transforms."""
    runtime = compile_runtime(fixture.space, knowledge, training)
    features = _PhysicalFeatures(fixture.space, training.X)
    location, scale = _stats(training)
    sign = -1. if fixture.space.maximize[0] else 1.
    model, diagnostics = fit_structured_gp(fixture.space, knowledge, training,
        runtime=runtime, X=_model_inputs(fixture.space, training.X),
        Y=torch.tensor(sign * (training.Y.to_numpy() - location) / scale, dtype=torch.double),
        Yvar=torch.tensor(training.Yvar.to_numpy() / scale ** 2, dtype=torch.double),
        feature_transform=features, feature_groups=features.groups,
        response_location=location, response_scale=scale, **policy)
    fitted = FittedSurrogate(fixture.space, model, location, scale, sign, features.describe(), ())
    fitted._diagnostics = {"structured_mean": diagnostics}
    return fitted


def _invariants(fitted, fixture, training, context):
    """Inspect actual component values/derivatives, not just fitted sign labels."""
    mean = fitted._model.mean_module
    checks = {}
    origin = torch.zeros((1, fixture.space.n_dims), dtype=torch.double)
    for index, component in enumerate(mean.components):
        checks[f"component_{index}_anchor"] = bool(torch.allclose(component(origin), torch.zeros(1, dtype=torch.double), atol=1e-12))
        probe = torch.full((1, fixture.space.n_dims), .37, dtype=torch.double, requires_grad=True)
        first = torch.autograd.grad(component(probe).sum(), probe, create_graph=True)[0]
        dims = component.payload["dimensions"]
        params = component.payload["parameters"]
        if component.pattern == "saturation":
            checks[f"component_{index}_direction"] = bool(first[0, dims[0]] * SIGNS[params["direction"]] >= 0)
        elif component.pattern == "threshold":
            slopes = [float(s.detach()) for s in component.slopes()]
            checks[f"component_{index}_branches"] = all(
                value == 0. if params[f"{branch}_behavior"] == "flat" else value * SIGNS[params[f"{branch}_behavior"]] >= 0.
                for branch, value in zip(("below", "above"), slopes))
            checks[f"component_{index}_width"] = bool(.01 <= component.width <= .25)
        else:
            sign = (1. if component.pattern == "synergy" else -1.) * (1. if params["response_effect"] == "higher" else -1.)
            mixed = torch.autograd.grad(first[0, dims[0]], probe)[0][0, dims[1]]
            checks[f"component_{index}_mixed_sign"] = bool(sign * mixed >= 0)
            axes = torch.ones((2, fixture.space.n_dims), dtype=torch.double)
            axes[0, dims[0]], axes[1, dims[1]] = 0., 0.
            checks[f"component_{index}_both_axes"] = bool(torch.allclose(component(axes), torch.zeros(2, dtype=torch.double), atol=1e-12))
    location, scale = _stats(training)
    checks["training_only_response_statistics"] = bool(float(mean.response_location) == location and float(mean.response_scale) == scale)
    checks["disjoint_training_validation_ids"] = set(context["training_ids"]).isdisjoint(context["validation_ids"])
    X = _model_inputs(fixture.space, fixture.grid)
    with torch.no_grad():
        physical = mean.physical(X)
        decoded = mean(X) * mean.internal_sign * scale + location
    checks["single_physical_response_normalization"] = bool(torch.allclose(physical, decoded, atol=1e-10, rtol=1e-12))
    return {"passed": all(checks.values()), "checks": checks}


def _metrics(truth, observed, means, variances):
    residual = np.asarray(observed) - np.asarray(means)
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))), "mae": float(np.mean(np.abs(residual))),
            "coverage_95": None if variances is None else float(np.mean(
                np.abs(np.asarray(truth) - means) <= 1.959963984540054 * np.sqrt(variances)))}


def _failure(error):
    return {"code": error.code.value, "message": error.message, "details": error.details}


def _finite_prediction(prediction):
    if not np.isfinite(prediction.mean).all() or not np.isfinite(prediction.variance).all() or (prediction.variance < 0).any():
        raise EngineError(ErrorCode.MODEL_FIT_FAILED, "Nonfinite or negative-variance audit prediction")


def _run(fixture, training, context, track, knowledge, policy):
    result = {**context, "track": track, "policy": policy,
              "width_policy_active": any(s.pattern == "threshold" for s in knowledge.specs),
              "uncertainty": "conditional_on_MAP"}
    try:
        fitted = _fit_policy(fixture, training, knowledge, policy)
        validation = fitted.predict(fixture.observations.X.iloc[context["validation_indices"]])
        grid = fitted.predict(fixture.grid)
        _finite_prediction(validation)
        _finite_prediction(grid)
        result.update(status="success", grid_mean=grid.mean[:, 0].tolist(),
            grid_variance=grid.variance[:, 0].tolist(), validation_mean=validation.mean[:, 0].tolist(),
            validation_variance=validation.variance[:, 0].tolist(),
            diagnostics=fitted.diagnostics, invariants=_invariants(fitted, fixture, training, context))
        indices = context["validation_indices"]
        result.update(_metrics(fixture.truth[indices], fixture.observations.Y.iloc[indices, 0].to_numpy(),
                               validation.mean[:, 0], validation.variance[:, 0]))
    except EngineError as error:
        result.update(status="failed", failure=_failure(error))
    return result


def _reference_runs(fixture, training, context, seeds):
    Ridge, RandomForest = _reference_classes()
    features = _PhysicalFeatures(fixture.space, training.X)
    train_X = features(_model_inputs(fixture.space, training.X)).numpy()
    test_X = fixture.observations.X.iloc[context["validation_indices"]]
    test_features = features(_model_inputs(fixture.space, test_X)).numpy()
    grid_features = features(_model_inputs(fixture.space, fixture.grid)).numpy()
    results = []
    for seed in seeds:
        for track in ("M0", "ridge", "random_forest"):
            result = {**context, "track": track, "policy": {"seed": seed},
                      "uncertainty": "conditional_on_fitted_hyperparameters" if track == "M0" else "unavailable",
                      "uncertainty_reason": "Point-estimate GP" if track == "M0" else "Reference point predictions only"}
            try:
                if track == "M0":
                    fitted = fit_surrogate(fixture.space, training, seed=seed)
                    pred, grid = fitted.predict(test_X), fitted.predict(fixture.grid)
                    _finite_prediction(pred)
                    _finite_prediction(grid)
                    means, variances, grid_mean = pred.mean[:, 0], pred.variance[:, 0], grid.mean[:, 0]
                else:
                    estimator = Ridge(alpha=1.) if track == "ridge" else RandomForest(n_estimators=200, random_state=seed, n_jobs=1)
                    estimator.fit(train_X, training.Y.iloc[:, 0].to_numpy())
                    means, variances, grid_mean = estimator.predict(test_features), None, estimator.predict(grid_features)
                if not np.isfinite(means).all() or not np.isfinite(grid_mean).all():
                    raise EngineError(ErrorCode.MODEL_FIT_FAILED, "Nonfinite reference output")
                indices = context["validation_indices"]
                result.update(status="success", validation_mean=means.tolist(),
                    validation_variance=None if variances is None else variances.tolist(), grid_mean=grid_mean.tolist())
                result.update(_metrics(fixture.truth[indices], fixture.observations.Y.iloc[indices, 0].to_numpy(), means, variances))
            except EngineError as error:
                result.update(status="failed", failure=_failure(error), coverage_95=None)
            results.append(result)
    return results


def _pooled(fixture, runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["track"], json.dumps(run["policy"], sort_keys=True)), []).append(run)
    pooled = []
    for (track, policy), group in groups.items():
        complete = len(group) == len(fixture.folds) and all(r["status"] == "success" for r in group)
        item = {"track": track, "policy": json.loads(policy), "complete": complete,
                "rmse": None, "mae": None, "coverage_95": None}
        if complete:
            means = np.empty(len(fixture.truth))
            variances = np.empty(len(fixture.truth)) if group[0]["validation_variance"] is not None else None
            for run in group:
                means[run["validation_indices"]] = run["validation_mean"]
                if variances is not None:
                    variances[run["validation_indices"]] = run["validation_variance"]
            item.update(_metrics(fixture.truth, fixture.observations.Y.iloc[:, 0].to_numpy(), means, variances))
        pooled.append(item)
    return pooled


def _metadata():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "engine_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "python": platform.python_version(), "executable": sys.executable,
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "torch", "botorch", "gpytorch", "scikit-learn")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run_audit(*, fixtures=None, seeds=None, scale_multipliers=None, width_centers=None, progress=False):
    fixtures = build_fixtures() if fixtures is None else fixtures
    defaults = default_policy()
    seeds = defaults["seeds"] if seeds is None else list(seeds)
    scale_multipliers = defaults["magnitude_scale_multipliers"] if scale_multipliers is None else list(scale_multipliers)
    width_centers = defaults["width_prior_centers"] if width_centers is None else list(width_centers)
    policies = policy_grid(seeds, scale_multipliers, width_centers)
    if not policies or any(p["seed"] < 0 or not np.isfinite(p["magnitude_scale_multiplier"]) or p["magnitude_scale_multiplier"] <= 0
                           or not np.isfinite(p["width_prior_center"]) or p["width_prior_center"] <= 0 for p in policies):
        raise ValueError("Policy grid must be nonempty with nonnegative seeds and positive finite constants")
    report = {"schema_version": "1.0", "metadata": _metadata(),
        "policy": {"seeds": seeds, "magnitude_scale_multipliers": scale_multipliers,
                   "width_prior_centers": width_centers, "configurations_per_fixture_fold_track": len(policies)},
        "caveats": CAVEATS, "fold_policy": "two fixed complementary even/odd validation-index folds; never policy-seeded",
        "fixtures": [], "default_noiseless_failures": 0, "default_noiseless_runs": 0}
    for fixture in fixtures:
        item = {"definition": fixture.describe(), "runs": [], "references": [], "reproducibility": []}
        for fold_index, validation in enumerate(fixture.folds):
            training_indices = [i for i in range(len(fixture.truth)) if i not in validation]
            training = _subset(fixture.observations, training_indices)
            location, scale = _stats(training)
            context = {"fold": fold_index, "training_indices": training_indices, "training_ids": list(training.ids),
                "validation_indices": list(validation), "validation_ids": [fixture.observations.ids[i] for i in validation],
                "response_location": location, "response_scale": scale}
            for track, knowledge in (("correct", fixture.correct), ("wrong", fixture.wrong)):
                group = [_run(fixture, training, context, track, knowledge, policy) for policy in policies]
                sensitivity = summarize_sensitivity([r["grid_mean"] for r in group if r["status"] == "success"], scale)
                for run in group:
                    run["sensitivity"] = sensitivity
                    default = run["policy"]["magnitude_scale_multiplier"] == 1. and run["policy"]["width_prior_center"] == .05
                    if track == "correct" and fixture.default_noiseless and default:
                        report["default_noiseless_runs"] += 1
                        report["default_noiseless_failures"] += int(run["status"] != "success")
                    # Refit each seed's default first fold; no reuse of the first estimator.
                    if track == "correct" and fold_index == 0 and default:
                        repeated = _run(fixture, training, context, track, knowledge, run["policy"])
                        same = ({k: v for k, v in run.items() if k != "sensitivity"} == repeated)
                        item["reproducibility"].append({"policy": run["policy"], "exact_match": same,
                                                        "status": repeated["status"]})
                item["runs"].extend(group)
            item["references"].extend(_reference_runs(fixture, training, context, seeds))
        item["pooled"] = _pooled(fixture, item["runs"] + item["references"])
        item["failure_counts"] = dict(Counter(r["failure"]["code"] for r in item["runs"] if r["status"] == "failed"))
        item["max_prediction_change"] = max((r["sensitivity"]["max_pairwise_change"] for r in item["runs"]
            if r["sensitivity"]["max_pairwise_change"] is not None), default=None)
        report["fixtures"].append(item)
        if progress:
            print(f"{fixture.name}: {len(item['runs'])} runs, failures={item['failure_counts']}, max change={item['max_prediction_change']}", flush=True)
    report["default_noiseless_gate"] = ("not_evaluated" if not report["default_noiseless_runs"] else
                                         "failed" if report["default_noiseless_failures"] else "passed")
    report["invariant_failures"] = sum(not run["invariants"]["passed"] for f in report["fixtures"]
                                       for run in f["runs"] if run["status"] == "success")
    report["reproducibility_failures"] = sum(not r["exact_match"] for f in report["fixtures"] for r in f["reproducibility"])
    report["complete_default_grid"] = (report["policy"]["seeds"] == defaults["seeds"]
        and scale_multipliers == defaults["magnitude_scale_multipliers"] and width_centers == defaults["width_prior_centers"]
        and [f.name for f in fixtures] == [f.name for f in build_fixtures()])
    report["changed_units_comparison"] = _unit_comparison(report["fixtures"])
    json.dumps(report, allow_nan=False)
    return report


def _unit_comparison(fixtures):
    by_name = {f["definition"]["name"]: f for f in fixtures}
    if not {"saturation_increasing", "changed_units"} <= by_name.keys():
        return {"status": "not_evaluated"}
    original, changed = by_name["saturation_increasing"], by_name["changed_units"]
    pairs = []
    for a, b in zip(original["runs"], changed["runs"], strict=True):
        if a["status"] == b["status"] == "success":
            delta = float(np.max(np.abs(np.asarray(a["grid_mean"]) - (np.asarray(b["grid_mean"]) - 273.15) / 100.)))
            pairs.append({"track": a["track"], "fold": a["fold"], "policy": a["policy"],
                          "max_original_units_difference": delta,
                          "within_tolerance": delta <= .001 * a["response_scale"]})
    return {"status": "evaluated" if pairs else "not_evaluable", "pairs": pairs,
            "tolerance": "0.001 times original training response scale, declared before the sweep",
            "failures": sum(not p["within_tolerance"] for p in pairs)}


def write_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n")
    lines = ["# Four-pattern v2 numerical sensitivity evidence", "",
        f"Command: `{report['metadata'].get('command', 'run_audit (library invocation)')}`", "",
        f"Engine source commit: `{report['metadata']['engine_commit']}` (see JSON for dirty state and dependency versions).", "",
        f"Full required grid: {report['complete_default_grid']}. Default noiseless gate: **{report['default_noiseless_gate']}** "
        f"({report['default_noiseless_failures']} failures / {report['default_noiseless_runs']} fits).", "",
        f"Signed/normalization/training-isolation invariant failures: {report['invariant_failures']}; exact replay failures: {report['reproducibility_failures']}.", "",
        f"Changed-units comparison: {report['changed_units_comparison']['status']}; "
        f"tolerance failures: {report['changed_units_comparison'].get('failures', 'not evaluated')}.", "",
        "| Fixture | Structured runs | Typed failures | Max grid-mean change (physical Y) |", "|---|---:|---|---:|"]
    for f in report["fixtures"]:
        change = "not_evaluable" if f["max_prediction_change"] is None else f"{f['max_prediction_change']:.9g}"
        lines.append(f"| {f['definition']['name']} | {len(f['runs'])} | {json.dumps(f['failure_counts'])} | {change} |")
    lines += ["", "## Default-policy pooled comparisons", "", "RMSE / MAE / latent truth 95% coverage; seed 17. Incomplete folds have no pooled score.", "",
              "| Fixture | Track | RMSE | MAE | Coverage |", "|---|---|---:|---:|---:|"]
    for f in report["fixtures"]:
        for row in f["pooled"]:
            policy = row["policy"]
            if policy["seed"] == 17 and policy.get("magnitude_scale_multiplier", 1.) == 1. and policy.get("width_prior_center", .05) == .05:
                values = ["unavailable" if row[k] is None else f"{row[k]:.7g}" for k in ("rmse", "mae", "coverage_95")]
                lines.append(f"| {f['definition']['name']} | {row['track']} | {' | '.join(values)} |")
    lines += ["", "## Interpretation and gates", ""] + [f"- {c}" for c in report["caveats"]]
    lines += ["- Missing-transition INSUFFICIENT_DATA is coverage failure, not default-policy optimizer failure.",
              "- Every full record, policy, fold, physical grid, diagnostic, failed start, and pooled metric is in audit.json.",
              "- E6 is not approved: independent result review, unresolved slow tests, version/compatibility and clean offline wheel gates remain separate.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = default_policy()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/validation/four-pattern-v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults["seeds"])
    parser.add_argument("--scale-multipliers", nargs="+", type=float, default=defaults["magnitude_scale_multipliers"])
    parser.add_argument("--width-centers", nargs="+", type=float, default=defaults["width_prior_centers"])
    parser.add_argument("--fixtures", nargs="+", help="Optional named subset for regression/development; never full acceptance")
    args = parser.parse_args(argv)
    fixtures = build_fixtures()
    if args.fixtures:
        if not set(args.fixtures) <= {f.name for f in fixtures}:
            parser.error("Unknown fixture name")
        fixtures = [f for f in fixtures if f.name in args.fixtures]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = run_audit(fixtures=fixtures, seeds=args.seeds, scale_multipliers=args.scale_multipliers,
                           width_centers=args.width_centers, progress=True)
    finally:
        torch.set_num_threads(previous)
    report["metadata"]["command"] = "PYTHONPATH=expdoe-dk/src " + shlex.join([sys.executable,
        "experiments/four_pattern_v2_sensitivity.py", *(sys.argv[1:] if argv is None else argv)])
    write_report(report, args.output_dir)
    return int(report["default_noiseless_failures"] > 0 or report["invariant_failures"] > 0
               or report["reproducibility_failures"] > 0 or report["changed_units_comparison"].get("failures", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
