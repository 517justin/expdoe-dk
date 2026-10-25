"""Deterministic joint MAP fitting of physical v2 means and residual GPs."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import warnings

import torch
from botorch import settings
from botorch.models import SingleTaskGP
from botorch.optim.core import OptimizationStatus
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..domain import ObservationBatch, Space
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge
from .gp import build_gp
from .structured_coverage import check_structured_coverage


class _StructuredMarginalLogLikelihood(ExactMarginalLogLikelihood):
    """Total exact MLL including registered GP priors, plus one mean prior.

    GPyTorch divides both the likelihood and its registered prior terms by n.
    Undo that division before adding the unregistered structured MAP penalty.
    Optimization and restart selection use this identical scalar objective.
    """

    def __init__(self, model, *, magnitude_scale_multiplier, width_prior_center):
        super().__init__(model.likelihood, model)
        self.prior_policy = dict(magnitude_scale_multiplier=magnitude_scale_multiplier,
                                 width_prior_center=width_prior_center)

    def forward(self, function_dist, target, *params, **kwargs):
        total = super().forward(function_dist, target, *params, **kwargs) * function_dist.event_shape.numel()
        return total + self.model.mean_module.log_prior(**self.prior_policy)


def fit_structured_gp(
    space: Space, knowledge: Knowledge, observations: ObservationBatch, *,
    runtime: dict[str, tuple], X: torch.Tensor, Y: torch.Tensor,
    Yvar: torch.Tensor | None, feature_transform: torch.nn.Module,
    feature_groups: tuple[tuple[int, ...], ...], response_location: float,
    response_scale: float, seed: int, magnitude_scale_multiplier: float = 1.0,
    width_prior_center: float = 0.05,
) -> tuple[SingleTaskGP, dict]:
    """Fit fresh models at the three coverage-eligible paired initializations.

    Only converged, finite candidates can be returned. No retries from sampled
    priors, model substitution, or unfit-model fallback are permitted here.
    Policy overrides are internal sensitivity/audit controls, not auto-tuning.
    """
    if type(seed) is not int or seed < 0:
        raise EngineError(ErrorCode.CONFIG_INVALID, "seed must be a non-negative integer")
    for name, value in (("magnitude_scale_multiplier", magnitude_scale_multiplier),
                        ("width_prior_center", width_prior_center)):
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise EngineError(ErrorCode.CONFIG_INVALID, f"{name} must be finite and positive")
    coverage = check_structured_coverage(space, observations, runtime["mean_components"])
    diagnostics = {**coverage, "seed": seed, "response_location": response_location,
        "response_scale": response_scale, "coverage_starts": coverage["starts"], "starts": [],
        "selected_start": None, "selected_objective": None,
        "prior_policy": {"magnitude_scale_multiplier": magnitude_scale_multiplier,
            "width_prior_center": width_prior_center,
            "objective": "total_exact_mll_with_gp_priors_plus_structured_log_prior"}}
    selected = None
    with torch.random.fork_rng(devices=[]), settings.validate_input_scaling(False):
        torch.manual_seed(seed)
        for index, (magnitude, width) in enumerate(((.05, .02), (.25, .05), (1., .15))):
            start = {"index": index, "initial_magnitude": magnitude, "initial_width": width,
                     "status": "ineligible", "objective": None, "warnings": []}
            diagnostics["starts"].append(start)
            if index not in coverage["eligible_starts"]:
                start["reason"] = "rank_deficient_initial_mean_basis"
                continue
            # Artifact/configuration errors remain typed; only optimization and
            # numerical candidate failures are contained within a restart.
            model, _ = build_gp(space, knowledge, X, Y, response_location=response_location,
                response_scale=response_scale, train_Yvar=Yvar,
                feature_transform=deepcopy(feature_transform), feature_groups=feature_groups,
                runtime=runtime, normalize_outcome=False)
            model.mean_module.initialize(magnitude=magnitude, width=width)
            mll = _StructuredMarginalLogLikelihood(model,
                magnitude_scale_multiplier=magnitude_scale_multiplier,
                width_prior_center=width_prior_center)
            model.train()
            mll.train()
            with warnings.catch_warnings(record=True) as captured:
                try:
                    result = fit_gpytorch_mll_scipy(mll)
                    start["optimizer_status"] = result.status.name
                    start["optimizer_message"] = result.message
                    if result.status != OptimizationStatus.SUCCESS:
                        raise RuntimeError(f"Optimizer did not converge: {result.status.name}: {result.message}")
                    if not math.isfinite(result.fval) or not all(torch.isfinite(p).all() for p in model.parameters()):
                        raise ValueError("Nonfinite optimizer objective or fitted parameter")
                    with torch.no_grad():
                        objective = float(mll(model(*model.train_inputs), model.train_targets).item())
                        if not math.isfinite(objective):
                            raise ValueError("Nonfinite fitted MAP objective")
                        description = model.mean_module.describe()
                        model.eval()
                        posterior = model.posterior(X)
                        if not torch.isfinite(posterior.mean).all() or not torch.isfinite(posterior.variance).all():
                            raise ValueError("Nonfinite fitted posterior")
                    start.update(status="converged", objective=objective)
                    if selected is None or objective > diagnostics["selected_objective"]:
                        selected = model
                        diagnostics.update(selected_start=index, selected_objective=objective)
                        selected_description = description
                except Exception as error:
                    start.update(status="failed", reason=f"{type(error).__name__}: {error}")
                finally:
                    start["warnings"] = [str(w.message) for w in captured]
    if selected is None:
        raise EngineError(ErrorCode.MODEL_FIT_FAILED,
            "All eligible structured MAP starts failed", details=diagnostics)
    diagnostics.update(selected_description)
    # F2 describes the default prior. Report the actual internal audit policy.
    for component in diagnostics["components"]:
        component["prior"]["magnitude_scale"] *= magnitude_scale_multiplier
        if component["source_pattern"] == "threshold":
            component["prior"]["log_width_center"] = math.log(width_prior_center)
    return selected, json.loads(json.dumps(diagnostics, allow_nan=False))


__all__ = ["fit_structured_gp"]
