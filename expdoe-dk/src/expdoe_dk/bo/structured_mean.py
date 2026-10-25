"""Standalone physical v2 means; deliberately not wired into fitting or runtime.

Constrained magnitudes and free nuisance coefficients are in response-scale
units. Only the baseline is a physical-response parameter. Priors are explicit
MAP penalties, not registered GPyTorch priors (which would count them twice).
"""
from __future__ import annotations

import json
import math

import torch
from gpytorch.constraints import Interval, Positive
from gpytorch.means import Mean
from gpytorch.module import Module
from torch.nn import ModuleList, Parameter, ParameterDict
from torch.nn.functional import softplus

from ..domain import Space
from ..errors import EngineError, ErrorCode
from ..knowledge import Knowledge
from ..knowledge._frame import flip_for_minimize
from ..knowledge.artifacts import OptimizationArtifact
from ..knowledge.specs import KnowledgeScope, make_pattern_spec


_KINDS = {"saturation": "saturation_mean_v2", "threshold": "threshold_mean_v2",
          "synergy": "signed_interaction_mean_v2", "antagonism": "signed_interaction_mean_v2"}
_BEHAVIOR_SIGN = {"increasing": 1., "decreasing": -1., "flat": 0.}


def _invalid(reason, artifact=None):
    return EngineError(ErrorCode.KNOWLEDGE_INVALID, reason, details={
        "pattern_id": artifact.source_pattern_id if artifact else None,
        "kind": artifact.kind if artifact else None,
    })


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or _finite_number(value)


def _validate(space, artifact):
    """Recompile through F1 to verify schema, provenance and physical metadata.

    This uses only built-in declarations, never provider loading or runtime
    consumers. Comparing the full canonical payload also rejects extra fields,
    anchor/range/transform tampering and noncanonical interaction ordering.
    """
    try:
        if not isinstance(artifact, OptimizationArtifact):
            raise ValueError("Expected OptimizationArtifact")
        if (artifact.source_version != "2.0"
                or _KINDS.get(artifact.source_pattern) != artifact.kind):
            raise ValueError("Unsupported structured mean artifact or provenance")
        payload = artifact.to_dict()["payload"]
        if not _finite_tree(payload):
            raise ValueError("Nonfinite artifact metadata")
        dimensions = payload["dimensions"]
        if not all(type(d) is int and 0 <= d < space.n_dims for d in dimensions):
            raise ValueError("Invalid factor dimensions")
        declaration = make_pattern_spec(pattern_id=artifact.source_pattern_id,
            pattern=artifact.source_pattern, version=artifact.source_version,
            scope=KnowledgeScope(factors=tuple(payload["factors"]), objectives=tuple(payload["objectives"])),
            parameters=payload["parameters"], confidence=payload["confidence"])
        expected, = Knowledge().add_spec(declaration).compile(space).mean_components
        if payload != expected.to_dict()["payload"]:
            raise ValueError("Artifact metadata differs from its Space declaration")
    except (KeyError, TypeError, ValueError, OverflowError, EngineError) as error:
        raise _invalid(f"Invalid structured mean artifact: {error}",
                       artifact if isinstance(artifact, OptimizationArtifact) else None) from error


class _Component(Module):
    """One zero-at-anchor contribution, with no intercept or nuisance slopes."""

    def __init__(self, artifact):
        super().__init__()
        self.artifact = artifact
        self.payload = artifact.to_dict()["payload"]
        self.pattern = artifact.source_pattern
        if self.pattern == "threshold":
            for branch in ("below", "above"):
                if self.payload["parameters"][f"{branch}_behavior"] != "flat":
                    self._magnitude(f"raw_{branch}")
            self.register_parameter("raw_width", Parameter(torch.zeros((), dtype=torch.double)))
            # Interval casts inputs to the global default dtype. Restore exact
            # double bounds without mutating that process-wide default.
            constraint = Interval(.01, .25).double()
            constraint.lower_bound.fill_(.01)
            constraint.upper_bound.fill_(.25)
            self.register_constraint("raw_width", constraint)
        else:
            self._magnitude("raw_amplitude")

    def _magnitude(self, name):
        self.register_parameter(name, Parameter(torch.zeros((), dtype=torch.double)))
        self.register_constraint(name, Positive().double())

    def magnitude(self, name):
        return getattr(self, f"{name}_constraint").transform(getattr(self, name))

    @property
    def width(self):
        return self.raw_width_constraint.transform(self.raw_width)

    def slopes(self):
        return tuple(_BEHAVIOR_SIGN[self.payload["parameters"][f"{branch}_behavior"]]
                     * self.magnitude(f"raw_{branch}")
                     if hasattr(self, f"raw_{branch}") else self.raw_width.new_zeros(())
                     for branch in ("below", "above"))

    def displacements(self, X):
        """Decode x-L directly, preserving small steps above large anchors."""
        result = []
        for dim, coordinate in zip(self.payload["dimensions"], self.payload["coordinates"], strict=True):
            model = X[..., dim]
            lower, span = coordinate["lower"], coordinate["range"]
            if coordinate["transform"] == "log":
                ratio = span / lower
                if math.isfinite(ratio):
                    displacement = lower * torch.expm1(model * math.log1p(ratio))
                else:
                    # Avoid both the overflowing ratio and expm1(log(upper/lower)).
                    log_lower = math.log(lower)
                    z = model * (math.log(coordinate["upper"]) - log_lower)
                    positive = z.clamp_min(0)
                    # Keep inactive branches finite for autograd; these clamps
                    # do not clip inputs or the selected displacement formula.
                    displacement = torch.where(z >= 0,
                        torch.exp(log_lower + positive) * (-torch.expm1(-positive)),
                        lower * torch.expm1(z.clamp_max(0)))
            else:
                displacement = span * model
            result.append(displacement)
        return result

    def forward(self, X):
        displacements = self.displacements(X)
        coordinates, parameters = self.payload["coordinates"], self.payload["parameters"]
        u = [value / coordinate["range"]
             for value, coordinate in zip(displacements, coordinates, strict=True)]
        if self.pattern == "saturation":
            t = displacements[0]  # F1 validation requires origin == lower.
            K = parameters["half_response"] - parameters["origin"]
            d = _BEHAVIOR_SIGN[parameters["direction"]]
            if not torch.isfinite(t).all():
                raise _invalid("Saturation requires finite physical displacement", self.artifact)
            # Divide the smaller positive operand by the larger before adding:
            # K+t can overflow even when the contribution and gradient are finite.
            # Select operands first so inactive branches stay safe for autograd
            # (including the t=0 anchor); no tensor is detached or scalarized.
            large = t >= K
            ratio = torch.where(large, K, t) / torch.where(large, t, K)
            fraction = torch.where(large, 1. / (1. + ratio), ratio / (1. + ratio))
            return d * self.magnitude("raw_amplitude") * fraction
        if self.pattern == "threshold":
            p, q = self.slopes()
            h = (parameters["threshold"] - coordinates[0]["lower"]) / coordinates[0]["range"]
            w = self.width
            H = w * softplus((u[0] - h) / w)
            H0 = w * softplus(-h / w)
            return p * u[0] + (q - p) * (H - H0)
        d = 1. if parameters["response_effect"] == "higher" else -1.
        s = 1. if self.pattern == "synergy" else -1.
        return d * s * self.magnitude("raw_amplitude") * u[0] * u[1]


class StructuredMean(Mean):
    """Physical signed structured mean with one baseline and explicit MAP prior.

    ``X`` uses Space's model coordinates, including logarithmic encoding.
    ``physical`` returns raw response Y; ``forward`` applies the existing
    engine's minimization sign and response normalization exactly once.
    """

    def __init__(self, space: Space, artifacts: tuple[OptimizationArtifact, ...], *,
                 response_location: float, response_scale: float) -> None:
        super().__init__()
        if (not isinstance(space, Space) or len(space.objectives) != 1
                or not _finite_number(response_location) or not _finite_number(response_scale)
                or response_scale <= 0):
            raise _invalid("StructuredMean requires a single response and finite location/positive scale")
        for artifact in artifacts:
            _validate(space, artifact)
        self.n_dims = space.n_dims
        self.register_buffer("response_location", torch.tensor(response_location, dtype=torch.double))
        self.register_buffer("response_scale", torch.tensor(response_scale, dtype=torch.double))
        sign = -1. if flip_for_minimize("increases_objective", space.maximize[0]) == "decreasing" else 1.
        self.register_buffer("internal_sign", torch.tensor(sign, dtype=torch.double))
        self.register_parameter("baseline", Parameter(self.response_location.clone()))
        self.components = ModuleList([_Component(artifact) for artifact in artifacts])
        univariate = {c.payload["dimensions"][0] for c in self.components
                      if c.pattern in {"saturation", "threshold"}}
        interaction = {dim for c in self.components if c.pattern in {"synergy", "antagonism"}
                       for dim in c.payload["dimensions"]}
        self.nuisance = ParameterDict({str(dim): Parameter(torch.zeros((), dtype=torch.double))
                                       for dim in sorted(interaction - univariate)})
        self.initialize(magnitude=.1, width=.05)

    def initialize(self, *, magnitude: float, width: float) -> None:
        """Initialize constrained normalized effects, without changing the baseline."""
        if (not _finite_number(magnitude) or magnitude <= 0 or not _finite_number(width)
                or not .01 <= width <= .25):
            raise _invalid("Initialization requires positive magnitude and width in [.01, .25]")
        with torch.no_grad():
            for component in self.components:
                for name, parameter in component.named_parameters():
                    value = parameter.new_tensor(width if name == "raw_width" else magnitude)
                    constraint = component.constraint_for_parameter_name(name)
                    parameter.copy_(constraint.inverse_transform(value))

    def physical(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate in physical response units, without Space prediction clipping."""
        if (X.ndim < 1 or X.shape[-1] != self.n_dims or not torch.isfinite(X).all()):
            raise _invalid("Structured mean requires finite model coordinates of the Space dimension")
        value = self.baseline.expand(X.shape[:-1])
        for component in self.components:
            value = value + self.response_scale * component(X)
        for dim, beta in self.nuisance.items():
            component = next(c for c in self.components if int(dim) in c.payload["dimensions"])
            index = component.payload["dimensions"].index(int(dim))
            coordinate = component.payload["coordinates"][index]
            displacement = component.displacements(X)[index]
            value = value + self.response_scale * beta * (displacement / coordinate["range"])
        if not torch.isfinite(value).all():
            raise _invalid("Structured mean produced nonfinite physical response")
        return value

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.internal_sign * (self.physical(X) - self.response_location) / self.response_scale

    def log_prior(self, *, magnitude_scale_multiplier: float = 1.0,
                  width_prior_center: float = 0.05) -> torch.Tensor:
        """Sum densities once in normalized-magnitude, beta/S and log-width space.

        No transformation Jacobians: these are the specified MAP penalties,
        not densities induced on unconstrained raw optimization parameters.
        """
        if (not _finite_number(magnitude_scale_multiplier) or magnitude_scale_multiplier <= 0
                or not _finite_number(width_prior_center) or width_prior_center <= 0):
            raise _invalid("Prior scale multiplier and width center must be finite and positive")
        result = self.baseline.new_zeros(())
        for component in self.components:
            tau = (.1 + .9 * component.payload["confidence"]) * magnitude_scale_multiplier
            prior = torch.distributions.HalfNormal(self.baseline.new_tensor(tau))
            for name, _ in component.named_parameters():
                if name != "raw_width":
                    result = result + prior.log_prob(component.magnitude(name))
            if component.pattern == "threshold":
                prior = torch.distributions.Normal(self.baseline.new_tensor(math.log(width_prior_center)),
                                                   self.baseline.new_tensor(.5))
                result = result + prior.log_prob(torch.log(component.width))
        normal = torch.distributions.Normal(self.baseline.new_zeros(()), self.baseline.new_ones(()))
        for beta in self.nuisance.values():
            result = result + normal.log_prob(beta)
        return result

    def describe(self) -> dict:
        """Return finite, declarative physical parameters and conditional diagnostics."""
        scale = self.response_scale.item()
        components = []
        for component in self.components:
            artifact, payload = component.artifact, component.payload
            item = {"kind": artifact.kind, "source_pattern_id": artifact.source_pattern_id,
                    "source_pattern": artifact.source_pattern, "source_version": artifact.source_version,
                    "coordinates": payload["coordinates"], "parameters": payload["parameters"],
                    "anchors": [c["lower"] for c in payload["coordinates"]],
                    "confidence": payload["confidence"], "flags": [],
                    "prior": {"magnitude_distribution": "HalfNormal", "magnitude_scale": .1 + .9 * payload["confidence"]}}
            if component.pattern == "threshold":
                p, q = [value.item() for value in component.slopes()]
                width = component.width.item()
                R = payload["coordinates"][0]["range"]
                item.update(normalized_below_slope=p, normalized_above_slope=q,
                            physical_below_slope=scale * p / R, physical_above_slope=scale * q / R,
                            width=width, physical_width=width * R,
                            width_at_lower_bound=width <= .01, width_at_upper_bound=width >= .25)
                item["prior"].update(log_width_distribution="Normal", log_width_center=math.log(.05), log_width_scale=.5)
                if max(abs(p), abs(q)) < 1e-3:
                    item["flags"].append("width_unidentified_near_zero_effect")
            else:
                magnitude = component.magnitude("raw_amplitude").item()
                sign = (_BEHAVIOR_SIGN[payload["parameters"]["direction"]] if component.pattern == "saturation"
                        else (1. if payload["parameters"]["response_effect"] == "higher" else -1.)
                        * (1. if component.pattern == "synergy" else -1.))
                item.update(normalized_magnitude=magnitude, physical_amplitude=scale * magnitude,
                            physical_coefficient=sign * scale * magnitude)
            components.append(item)
        nuisance = []
        for dim, beta in self.nuisance.items():
            sources = [c for c in self.components if int(dim) in c.payload["dimensions"]]
            coordinate = sources[0].payload["coordinates"][sources[0].payload["dimensions"].index(int(dim))]
            nuisance.append({"factor": coordinate["factor"], "anchor": coordinate["lower"],
                             "normalized_slope": beta.item(), "physical_slope": scale * beta.item() / coordinate["range"],
                             "source_pattern_ids": [c.artifact.source_pattern_id for c in sources],
                             "prior": {"distribution": "Normal", "location": 0., "scale": 1.}})
        result = {"baseline": self.baseline.item(), "response_location": self.response_location.item(),
                  "response_scale": scale, "internal_sign": self.internal_sign.item(),
                  "components": components, "nuisance": nuisance,
                  "parameter_identifiability": "not_certified", "uncertainty": "conditional_on_MAP"}
        try:
            # Detach the returned metadata, so callers cannot mutate future evaluations.
            return json.loads(json.dumps(result, allow_nan=False))
        except (ValueError, OverflowError) as error:
            raise _invalid("Structured mean description contains nonfinite parameters") from error


__all__ = ["StructuredMean"]
