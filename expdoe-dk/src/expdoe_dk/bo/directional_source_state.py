"""Audited, storage-independent state for the private fixed-GP moment source.

This module deliberately has no multiprecision arithmetic. Source eligibility
is decided once; a prepared generation never refreshes or changes route.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields, is_dataclass
import inspect
import math
import struct
import types

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize, InputStandardize, Log10, ChainedInputTransform
from botorch.models.transforms.outcome import Standardize, ChainedOutcomeTransform
from gpytorch.constraints import GreaterThan, Positive, Interval, LessThan
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, ProductKernel
from gpytorch.kernels.kernel import Distance
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise, FixedGaussianNoise
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.means.constant_mean import _ensure_updated_strategy_flag_set
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior
from gpytorch.utils.transforms import inv_softplus, inv_sigmoid

from .. import Parameter
from ..errors import EngineError, ErrorCode
from ..knowledge.artifacts import OptimizationArtifact
from ..knowledge.shape import ArrheniusMeanFrozen, QuadraticMeanFrozen, CombinedMean
from .gp import _FeatureKernel, _CategorySimilarityKernel, _PhysicalFormulaMean
from .structured_mean import StructuredMean, _Component
from .surrogates import _PhysicalFeatures
from . import structured_mean as _structured_mean


def failure(reason, **details):
    return EngineError(ErrorCode.MODEL_FIT_FAILED, "Frozen GP source failed validation",
                       details={"reason": reason, **details})


class _Unsupported(Exception):
    pass


_CLASSES = (
    SingleTaskGP, Normalize, InputStandardize, Log10, ChainedInputTransform,
    Standardize, ChainedOutcomeTransform, GreaterThan, Positive, Interval, LessThan,
    MaternKernel, RBFKernel, ScaleKernel, ProductKernel, Distance,
    GaussianLikelihood, FixedNoiseGaussianLikelihood, HomoskedasticNoise, FixedGaussianNoise,
    ConstantMean, ZeroMean, GammaPrior, LogNormalPrior, NormalPrior,
    _FeatureKernel, _CategorySimilarityKernel, _PhysicalFormulaMean, _PhysicalFeatures,
    ArrheniusMeanFrozen, QuadraticMeanFrozen, CombinedMean, StructuredMean, _Component,
    torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.Softplus,
)
_PAYLOADS = (Parameter, OptimizationArtifact)
_FUNCTIONS = (inv_softplus, inv_sigmoid, torch.exp, torch.log, torch.sigmoid,
              _ensure_updated_strategy_flag_set)


def _methods(cls):
    return tuple((name, inspect.getattr_static(cls, name)) for name in sorted(dir(cls))
                 if isinstance(inspect.getattr_static(cls, name),
                               (types.FunctionType, property, staticmethod, classmethod)))


# Imported alongside the directional adapter, before any preparation/evaluation.
_METHODS = {cls: _methods(cls) for cls in _CLASSES}
_METHOD_FUNCTIONS = {value for methods in _METHODS.values() for _, value in methods
                     if isinstance(value, types.FunctionType)}
_GLOBAL_HOOKS = tuple(name for name in vars(torch.nn.modules.module)
                      if name.startswith("_global_") and "hook" in name)
_CACHES = {"prediction_strategy", "_memoize_cache"}
_GLOBAL_FUNCTIONS = tuple((module, name, getattr(module, name)) for module, names in (
    (torch, ("exp", "expm1", "log", "log10", "sin", "where", "cat", "sigmoid")),
    (torch.nn.functional, ("softplus",)),
    (math, ("log", "log1p")),
    (_structured_mean, ("softplus",)),
) for name in names)


def _tensor_record(value):
    if value.layout != torch.strided or value.device.type != "cpu":
        raise _Unsupported("tensor layout/device")
    raw = value.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
    return (id(value), value.untyped_storage().data_ptr(), value.storage_offset(),
            tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
            value.requires_grad, value._version, raw)


def _derived_values(model):
    """Freshly evaluated constrained values have value, not object, identity."""
    values = {}
    with torch.no_grad():
        for path, module in model.named_modules():
            for name, constraint in getattr(module, "_constraints", {}).items():
                raw_name = name.removesuffix("_constraint")
                attribute = raw_name.removeprefix("raw_")
                value = (getattr(module, attribute) if hasattr(module, attribute) else
                         constraint.transform(getattr(module, raw_name)))
                values[f"{path}.{attribute}"] = value
        values["likelihood.noise"] = model.likelihood.noise
        if type(model.mean_module) is ConstantMean:
            values["mean_module.constant"] = model.mean_module.constant
        return tuple((name, (tuple(value.shape), value.dtype, value.device,
                            value.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()))
                     for name, value in sorted(values.items()))


def _global_hooks():
    if any(getattr(module, name) is not function for module, name, function in _GLOBAL_FUNCTIONS):
        raise _Unsupported("global function identity")
    result = []
    for name in _GLOBAL_HOOKS:
        registry = getattr(torch.nn.modules.module, name)
        if registry:
            raise _Unsupported("global hooks")
        result.append((name, type(registry)))
    return tuple(result)


def _manifest(root, *, owned=False):
    """Typed tree plus exact tensor content; unknown objects/callbacks fail closed.

    Ordinary attributes are traversed too: fixed noise, original training inputs,
    feature origins and artifact payloads are not all parameters or buffers.
    Only the two explicitly nonsemantic prediction caches are excluded.
    """
    seen = {}
    affine_views = ({id(value) for module in root.modules() if type(module) in (Normalize, InputStandardize)
                     for name, value in module.named_buffers(recurse=False)
                     if name in ("_coefficient", "_offset")} if owned else set())

    def visit(value):
        typ = type(value)
        if value is None or typ in (bool, int, str, bytes, torch.dtype, torch.device):
            return typ, value
        if typ is float:
            return typ, struct.pack("!d", value)
        if isinstance(value, torch.Tensor):
            record = _tensor_record(value)
            # Native owned affine evaluation reassigns detach() views. Their
            # storage, metadata and exact content remain fixed; original source
            # tensors never receive this exemption from object-identity checks.
            return typ, record[1:] if id(value) in affine_views else record
        if typ in (tuple, list, torch.Size, dict, OrderedDict, types.MappingProxyType, set, frozenset):
            if id(value) in seen:
                return "reference", seen[id(value)]
            seen[id(value)] = len(seen)
            if typ in (dict, OrderedDict, types.MappingProxyType):
                items = tuple((visit(k), visit(v)) for k, v in value.items())
            elif typ in (set, frozenset):
                items = tuple(sorted((visit(v) for v in value), key=repr))
            else:
                items = tuple(visit(v) for v in value)
            return typ, id(value), items
        if isinstance(value, types.MethodType):
            if value.__func__ not in _METHOD_FUNCTIONS:
                raise _Unsupported("method identity")
            return "method", value.__func__, id(value.__self__)
        if typ is torch.nn.modules.module._WrappedHook:
            hook = value.hook
            function = getattr(hook, "__func__", hook)
            permitted = (function is _ensure_updated_strategy_flag_set or
                         function is SingleTaskGP._load_state_hook_ignore_shapes)
            if not permitted or value.with_module:
                raise _Unsupported("load hook identity")
            return typ, id(value), visit(hook), value.with_module
        if isinstance(value, (types.FunctionType, types.BuiltinFunctionType)):
            if value not in _FUNCTIONS and value not in _METHOD_FUNCTIONS:
                raise _Unsupported("callback identity")
            if isinstance(value, types.FunctionType) and value.__closure__:
                raise _Unsupported("callback closure")
            return "function", value
        if typ in _CLASSES:
            if _methods(typ) != _METHODS[typ]:
                raise _Unsupported("class method changed")
            if id(value) in seen:
                return "reference", seen[id(value)]
            seen[id(value)] = len(seen)
            contents = []
            for name, child in vars(value).items():
                if name in _CACHES:
                    if name == "prediction_strategy" and typ is not SingleTaskGP:
                        raise _Unsupported("unknown cache owner")
                    continue
                if name in dict(_METHODS[typ]) and callable(child):
                    raise _Unsupported("instance method override")
                if "hook" in name and child and name != "_load_state_dict_pre_hooks":
                    raise _Unsupported("module hooks")
                contents.append((name, visit(child)))
            return typ, id(value), tuple(contents)
        if typ in _PAYLOADS and is_dataclass(value):
            return typ, id(value), tuple((f.name, visit(getattr(value, f.name))) for f in fields(value))
        # LogNormalPrior has ordinary Torch distribution/transform state. These
        # are fitting-only, but their parameters and flags must still be guarded.
        if typ in (torch.distributions.Normal, torch.distributions.transforms.ExpTransform):
            return typ, id(value), tuple((k, visit(v)) for k, v in vars(value).items())
        raise _Unsupported(f"uninspectable state: {typ.__name__}")

    return _global_hooks(), visit(_structured_mean._BEHAVIOR_SIGN), visit(root)


def _grammar(model):
    if type(model) is not SingleTaskGP or model.num_outputs != 1:
        raise _Unsupported("scalar exact GP required")
    if type(model.likelihood) not in (GaussianLikelihood, FixedNoiseGaussianLikelihood):
        raise _Unsupported("likelihood")
    if any(t.dtype != torch.double or t.device.type != "cpu"
           for t in (*model.train_inputs, model.train_targets)):
        raise _Unsupported("CPU double training required")
    stationary = []

    def kernel(node, featured=False):
        if type(node) in (MaternKernel, RBFKernel):
            if type(node) is MaternKernel and node.nu != 2.5:
                raise _Unsupported("Matern smoothness")
            stationary.append(node)
        elif type(node) is ScaleKernel:
            kernel(node.base_kernel, featured)
        elif type(node) is ProductKernel:
            for child in node.kernels:
                kernel(child, featured)
        elif type(node) is _FeatureKernel:
            if featured or type(node.transform) is not _PhysicalFeatures:
                raise _Unsupported("feature transform")
            kernel(node.kernel, True)
        elif type(node) is _CategorySimilarityKernel:
            if featured:
                raise _Unsupported("category feature frame")
        else:
            raise _Unsupported("kernel grammar")

    kernel(model.covar_module)
    if len(stationary) > 1:
        raise _Unsupported("multiple continuous factors")
    mean_types = (ConstantMean, ZeroMean, ArrheniusMeanFrozen, QuadraticMeanFrozen,
                  _PhysicalFormulaMean, StructuredMean)

    def mean(node):
        if type(node) is CombinedMean:
            for child in node.means:
                mean(child)
        elif type(node) not in mean_types:
            raise _Unsupported("mean grammar")

    mean(model.mean_module)
    for name, chain, leaves in (
        ("input_transform", ChainedInputTransform, (Normalize, InputStandardize, Log10)),
        ("outcome_transform", ChainedOutcomeTransform, (Standardize,)),
    ):
        def transform(node):
            if type(node) is chain:
                for child in node.values():
                    transform(child)
            elif type(node) not in leaves:
                raise _Unsupported("transform grammar")
            elif type(node) is Standardize and (node._m != 1 or node._outputs is not None
                                                or not node._is_trained):
                raise _Unsupported("scalar trained standardization required")
        if hasattr(model, name):
            transform(getattr(model, name))


def eligible(model):
    try:
        _grammar(model)
        _manifest(model)
        return True
    except _Unsupported:
        return False


class OperandSnapshot:
    def __init__(self, tensor):
        self.tensor = tensor
        self.record = _tensor_record(tensor)
        self.value = tensor.detach().clone()

    def check(self):
        if _tensor_record(self.tensor) != self.record:
            raise failure("autograd_operand_changed")


class OwnedState:
    def __init__(self, model):
        try:
            _grammar(model)
            _manifest(model)
        except _Unsupported as exc:
            raise failure("unsupported_source", error=str(exc)) from exc
        model.eval()
        self.source = model
        self.generation = object()
        self._record = _manifest(model)
        self._derived_record = _derived_values(model)
        # Declarative artifacts own recursively frozen JSON mappings. Python's
        # generic deepcopy cannot copy mappingproxy; preserve its audited type
        # while giving every copied artifact a separately owned mapping.
        memo, visited = {}, set()
        def payloads(value):
            if id(value) in visited:
                return
            visited.add(id(value))
            if isinstance(value, torch.Tensor):
                copied = value.detach().clone()
                memo[id(value)] = (torch.nn.Parameter(copied, requires_grad=False)
                                   if isinstance(value, torch.nn.Parameter) else copied)
            elif type(value) in (dict, OrderedDict, types.MappingProxyType):
                for child in value.values():
                    payloads(child)
                if type(value) is types.MappingProxyType:
                    memo[id(value)] = types.MappingProxyType(deepcopy(dict(value), memo))
            elif type(value) in (tuple, list):
                for child in value:
                    payloads(child)
            elif (type(value) in _CLASSES or type(value) in _PAYLOADS
                  or type(value) in (torch.distributions.Normal, torch.distributions.transforms.ExpTransform)):
                for name, child in vars(value).items():
                    if name in _CACHES:
                        if child is not None:
                            memo[id(child)] = None if name == "prediction_strategy" else {}
                    else:
                        payloads(child)
        payloads(model)
        self.model = deepcopy(model, memo)
        self.model.prediction_strategy = None
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # deepcopy detaches parameters, but ordinary tensors can require grad.
        # Clone every stored tensor, including nonregistered training/noise data.
        seen = {}

        def own(value):
            if id(value) in seen:
                return seen[id(value)]
            if isinstance(value, torch.Tensor):
                result = value.detach().clone()
                if isinstance(value, torch.nn.Parameter):
                    result = torch.nn.Parameter(result, requires_grad=False)
                seen[id(value)] = result
                return result
            if isinstance(value, (dict, OrderedDict)):
                for key in value:
                    value[key] = own(value[key])
            elif isinstance(value, list):
                value[:] = [own(v) for v in value]
            elif isinstance(value, tuple):
                result = type(value)(own(v) for v in value)
                seen[id(value)] = result
                return result
            elif type(value) in _CLASSES or type(value) in (
                    torch.distributions.Normal, torch.distributions.transforms.ExpTransform):
                seen[id(value)] = value
                for name, child in vars(value).items():
                    if name not in _CACHES:
                        vars(value)[name] = own(child)
            return value

        own(self.model)
        self._owned_record = _manifest(self.model, owned=True)
        self._owned_derived_record = _derived_values(self.model)
        self.check()

    def check(self):
        try:
            record = _manifest(self.source)
            owned_record = _manifest(self.model, owned=True)
        except _Unsupported as exc:
            raise failure("source_state_changed", error=str(exc)) from exc
        if (record != self._record or owned_record != self._owned_record
                or _derived_values(self.source) != self._derived_record
                or _derived_values(self.model) != self._owned_derived_record):
            raise failure("source_state_changed")
