"""Private frozen-GP orchestration and explicit reverse/double-reverse bridge."""
from __future__ import annotations

import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, ProductKernel

from ..errors import EngineError
from .directional_source_state import OwnedState, OperandSnapshot, failure, _Unsupported
from .gp import _FeatureKernel, _CategorySimilarityKernel


class _CallGuard:
    def __init__(self, source, X):
        self.source = source
        self.operands = [OperandSnapshot(X)]

    def check(self):
        self.source.check()
        for operand in self.operands:
            operand.check()


class _Relay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, guard):
        guard.check()
        ctx.guard = guard
        result = value.clone()
        guard.check()
        return result

    @staticmethod
    def backward(ctx, grad):
        ctx.guard.check()
        return _Relay.apply(grad, ctx.guard), None


class _Affine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale, offset, guard, errors, variance, actual, box):
        if guard is not None:
            guard.check()
        ctx.guard = guard
        ctx.operand = OperandSnapshot(value)
        ctx.scale = scale.detach().clone()
        _affine_preflight(value, scale, offset, errors, variance)
        product = value*scale
        computed = product if offset is None else product+offset
        output = computed if actual is None else actual.detach().clone()
        from .directional_source_arithmetic import audit_affine
        values = value.detach().flatten().tolist()
        propagated = audit_affine(values, scale.expand_as(value).flatten().tolist(),
            [0.] * len(values) if offset is None else offset.expand_as(value).flatten().tolist(),
            product.detach().flatten().tolist(), output.detach().flatten().tolist(),
            [0.] * len(values) if errors is None else errors,
            [False] * len(values) if variance is None else variance.expand_as(value).flatten().tolist())
        box.extend(propagated)
        ctx.operand.check()
        if guard is not None:
            guard.check()
        return output

    @staticmethod
    def backward(ctx, gradient):
        if ctx.guard is not None:
            ctx.guard.check()
        ctx.operand.check()
        return checked_affine(gradient, ctx.scale, guard=ctx.guard), None, None, None, None, None, None, None


def _affine_preflight(value, scale, offset, errors, variance):
    from .directional_source_arithmetic import preflight_affine
    flat = value.detach().flatten().tolist()
    return preflight_affine(flat, scale.expand_as(value).detach().flatten().tolist(),
        [0.]*len(flat) if offset is None else offset.expand_as(value).detach().flatten().tolist(),
        [0.]*len(flat) if errors is None else errors,
        [False]*len(flat) if variance is None else variance.expand_as(value).flatten().tolist())


def checked_affine(value, scale, offset=None, *, guard=None, errors=None,
                   variance=None, actual=None, _return_errors=False):
    scale = torch.as_tensor(scale, dtype=torch.double, device=value.device)
    offset = None if offset is None else torch.as_tensor(offset, dtype=torch.double, device=value.device)
    if variance is True:
        variance = torch.ones_like(value, dtype=torch.bool)
    box = []
    result = _Affine.apply(value, scale, offset, guard, errors, variance, actual, box)
    return (result, box) if _return_errors else result


def _outcome_stages(transform):
    from botorch.models.transforms.outcome import ChainedOutcomeTransform
    if type(transform) is ChainedOutcomeTransform:
        return [stage for child in reversed(list(transform.values())) for stage in _outcome_stages(child)]
    return [transform]


class PhysicalGuard:
    """Keep the existing physical adapter's scalar map fixed across passes."""
    def __init__(self, owner, model_attribute):
        self.owner, self.model_attribute = owner, model_attribute
        self.model = getattr(owner, model_attribute)
        self.record = self._record()

    def _record(self):
        return (id(getattr(self.owner, self.model_attribute)),
                tuple((type(value), value) for value in
                      (self.owner._sign, self.owner._scale, self.owner._location)))

    def check(self):
        if self._record() != self.record:
            raise failure("source_state_changed", component="physical affine state")
        if self.model._moment_source is not None:
            self.model._moment_source.check()


def checked_square(scale, guard):
    from .directional_source_arithmetic import preflight_affine, audit_affine
    guard.check()
    preflight_affine([scale], [scale], [0.], [0.], [True])
    squared = scale**2
    audit_affine([scale], [scale], [0.], [squared], [squared], [0.], [True])
    guard.check()
    return squared


class _Moments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, mean, guard, batch, categories, known, nvar, result_box):
        guard.check()
        ctx.guard, ctx.batch, ctx.categories, ctx.known, ctx.nvar = guard, batch, categories, known, nvar
        ctx.features = features
        ctx.mean = mean.detach().clone().tolist()
        result = guard.source.blocks[batch].moments(features.detach().tolist(), categories, ctx.mean, known)
        result_box.append(result)
        guard.check()
        return features.new_tensor(result.mean), features.new_tensor(result.covariance)

    @staticmethod
    def backward(ctx, gm, G):
        ctx.guard.check()
        gradient = _VJP.apply(ctx.features, gm, G, ctx.guard, ctx.batch,
                              ctx.categories, ctx.mean, ctx.known, ctx.nvar)
        ctx.guard.check()
        return gradient, _Relay.apply(gm, ctx.guard), None, None, None, None, None, None


class _VJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, gm, G, guard, batch, categories, mean, known, nvar):
        guard.check()
        ctx.guard, ctx.batch, ctx.categories, ctx.mean, ctx.known, ctx.nvar = (
            guard, batch, categories, mean, known, nvar)
        ctx.operands = [OperandSnapshot(v) for v in (features, gm, G)]
        x, h, weights = (v.value.tolist() for v in ctx.operands)
        result = guard.source.blocks[batch].vjp(x, categories, mean, h, weights, nvar=nvar, known=known)
        output = features.new_zeros(features.shape)
        output[:nvar] = features.new_tensor(result).reshape(nvar, features.shape[-1])
        for operand in ctx.operands:
            operand.check()
        guard.check()
        return output

    @staticmethod
    def backward(ctx, direction):
        ctx.guard.check()
        direction_snapshot = OperandSnapshot(direction)
        for operand in ctx.operands:
            operand.check()
        features, h, weights = (v.value.tolist() for v in ctx.operands)
        hv, dh, dG = ctx.guard.source.blocks[ctx.batch].hvp(
            features, ctx.categories, ctx.mean, h, weights, direction_snapshot.value.tolist(),
            nvar=ctx.nvar, known=ctx.known)
        output = direction.new_zeros(direction.shape)
        output[:ctx.nvar] = direction.new_tensor(hv).reshape(ctx.nvar, direction.shape[-1])
        direction_snapshot.check()
        for operand in ctx.operands:
            operand.check()
        ctx.guard.check()
        return output, direction.new_tensor(dh), direction.new_tensor(dG), None, None, None, None, None, None


def _coordinates(node, X, *, select=True):
    """Mirror actual kernel __call__/forward placement, including ScaleKernel."""
    if select and node.active_dims is not None:
        X = X.index_select(-1, node.active_dims)
    if type(node) in (MaternKernel, RBFKernel):
        return [(node, X)], [], []
    if type(node) is _CategorySimilarityKernel:
        indices = X[..., 0]
        if (not torch.isfinite(indices).all() or (indices != indices.round()).any()
                or (indices < 0).any() or (indices >= len(node.similarities)).any()):
            raise failure("invalid_source", component="categorical indices")
        return [], [(node, indices.long())], []
    if type(node) is _FeatureKernel:
        return _coordinates(node.kernel, node.transform(X))
    if type(node) is ScaleKernel:
        # ScaleKernel.forward calls base_kernel.forward, not __call__; applying
        # the child's active_dims again would silently change the fitted kernel.
        continuous, categories, scales = _coordinates(node.base_kernel, X, select=False)
        return continuous, categories, scales + [node.outputscale]
    if type(node) is ProductKernel:
        continuous, categories, scales = [], [], []
        for child in node.kernels:
            a, b, c = _coordinates(child, X)
            continuous.extend(a)
            categories.extend(b)
            scales.extend(c)
        return continuous, categories, scales
    raise failure("unsupported_source", component="kernel grammar")


class _FrozenExactGPSource:
    def __copy__(self):
        raise failure("source_state_changed", component="copy requires fresh directional preparation")

    def __deepcopy__(self, memo):
        raise failure("source_state_changed", component="copy requires fresh directional preparation")

    def __init__(self, model, endpoints=None):
        self.state = OwnedState(model)
        owned = self.state.model
        self.batch_shape = model.batch_shape
        self.endpoints = (owned.train_inputs[0].new_empty((0, owned.train_inputs[0].shape[-1]))
                          if endpoints is None else endpoints.detach().clone().reshape(-1, endpoints.shape[-1]))
        self._endpoint_guard = OperandSnapshot(self.endpoints)
        self._adapter = None
        self.blocks = []
        training = owned.train_inputs[0]
        continuous, categories, scales = _coordinates(owned.covar_module, training)
        if continuous:
            leaf, features = continuous[0]
            kind = "matern" if type(leaf) is MaternKernel else "rbf"
            length = leaf.lengthscale.expand(self.batch_shape + (1, features.shape[-1]))
        else:
            features, kind = training[..., :0], None
            length = training.new_empty(self.batch_shape + (1, 0))
        self._feature_dimension = features.shape[-1]
        n = training.shape[-2]
        features = features.expand(self.batch_shape + (n, features.shape[-1]))
        cat = (torch.stack([v for _, v in categories], -1) if categories else
               torch.empty(training.shape[:-1] + (0,), dtype=torch.long))
        cat = cat.expand(self.batch_shape + (n, len(categories)))
        means = owned.mean_module(training).expand(self.batch_shape + (n,))
        targets = owned.train_targets.expand(self.batch_shape + (n,))
        noise = owned.likelihood.noise.expand(self.batch_shape + (n,))
        if not all(torch.isfinite(v).all() for v in (features, means, targets, noise, length)):
            raise failure("invalid_source", component="nonfinite frozen operands")
        self.training = training.expand(self.batch_shape + training.shape[-2:]).detach().clone()
        self.noise = noise.detach().clone()
        from .directional_source_arithmetic import Conditional
        count = self.batch_shape.numel()
        lengths = length.reshape(count, self._feature_dimension).tolist()
        scale_values = [v.expand(self.batch_shape).reshape(count).tolist() for v in scales]
        for i in range(count):
            specification = {"kind": kind, "lengthscale": lengths[i],
                             "scales": [v[i] for v in scale_values],
                             "tables": [node.similarities.detach().tolist() for node, _ in categories]}
            self.blocks.append(Conditional(features.reshape(count, n, self._feature_dimension)[i].tolist(),
                targets.reshape(count, n)[i].tolist(), means.reshape(count, n)[i].detach().tolist(),
                noise.reshape(count, n)[i].tolist(), specification, cat.reshape(count, n, len(categories))[i].tolist()))
        if len(self.endpoints):
            transformed = owned.transform_inputs(self.endpoints.expand(self.batch_shape + self.endpoints.shape))
            continuous, endpoint_categories, _ = _coordinates(owned.covar_module, transformed)
            ef = continuous[0][1] if continuous else transformed[..., :0]
            ec = (torch.stack([v for _, v in endpoint_categories], -1) if endpoint_categories else
                  torch.empty(transformed.shape[:-1] + (0,), dtype=torch.long))
            for i, block in enumerate(self.blocks):
                block.set_endpoints(ef.reshape(count, len(self.endpoints), self._feature_dimension)[i].tolist(),
                                    ec.reshape(count, len(self.endpoints), len(categories))[i].tolist())
        self._numeric_record = self._numeric_state()
        self.check()

    def _numeric_state(self):
        from .directional_source_state import _manifest, _tensor_record
        fields = ("train", "targets", "train_mean", "noise", "kernel", "categories",
                  "_endpoints", "_endpoint_categories")
        return (id(self.state), self.batch_shape, self._feature_dimension,
                _tensor_record(self.training), _tensor_record(self.noise), id(self.blocks),
                tuple((id(block), tuple((name, _manifest(getattr(block, name))) for name in fields))
                      for block in self.blocks))

    def bind(self, adapter):
        if self._adapter is not None:
            raise failure("source_state_changed", component="directional generation already bound")
        self._adapter = adapter
        self._fit_record = self._fit_state()

    def _fit_state(self):
        from .directional_source_state import _tensor_record
        adapter = self._adapter
        return (id(adapter._source), id(adapter._fits), _tensor_record(adapter._endpoints),
                tuple((id(fit), tuple((name, _tensor_record(value)) for name, value in fit.named_buffers()))
                      for fit in adapter._fits))

    def check(self):
        self.state.check()
        try:
            self._endpoint_guard.check()
            if self._numeric_state() != self._numeric_record:
                raise failure("source_state_changed", component="extracted operands")
            if self._adapter is not None and self._fit_state() != self._fit_record:
                raise failure("source_state_changed")
        except (EngineError, _Unsupported) as exc:
            raise failure("source_state_changed") from exc

    def joint_moments(self, X, *, adapter=None):
        self.check()
        if adapter is not None and adapter is not self._adapter:
            raise failure("source_state_changed", component="directional generation")
        if X.dtype != torch.double or X.device.type != "cpu" or not torch.isfinite(X).all():
            raise failure("invalid_source", component="query dtype/device/values")
        guard = _CallGuard(self, X)
        checked = _Relay.apply(X, guard)
        batch_shape = torch.broadcast_shapes(X.shape[:-2], self.batch_shape)
        checked = checked.expand(batch_shape + X.shape[-2:])
        endpoints = self.endpoints.expand(batch_shape + self.endpoints.shape)
        Z = torch.cat((checked, endpoints), dim=-2)
        transformed = self.state.model.transform_inputs(Z)
        continuous, categories, _ = _coordinates(self.state.model.covar_module, transformed)
        features = continuous[0][1] if continuous else transformed[..., :0]
        mean = self.state.model.mean_module(transformed).expand(batch_shape + (Z.shape[-2],))
        if not all(torch.isfinite(value).all() for value in (transformed, features, mean)):
            raise failure("invalid_source", component="nonfinite Torch query operands")
        guard.operands += [OperandSnapshot(features), OperandSnapshot(mean)]
        cat = (torch.stack([v for _, v in categories], -1) if categories else
               torch.empty(features.shape[:-1] + (0,), dtype=torch.long))
        indices = torch.arange(len(self.blocks)).reshape(self.batch_shape).expand(batch_shape).flatten().tolist()
        full_q = Z.shape[-2]
        features = features.reshape(len(indices), full_q, self._feature_dimension)
        mean = mean.reshape(-1, full_q)
        cat = cat.reshape(len(indices), full_q, len(categories))
        transformed = transformed.reshape(-1, full_q, transformed.shape[-1])
        training = self.training.reshape(len(self.blocks), self.training.shape[-2], self.training.shape[-1])
        noise = self.noise.reshape(len(self.blocks), -1)
        means, covariances, errors = [], [], []
        for row, batch in enumerate(indices):
            matches = (transformed[row, :, None, :] == training[batch, None, :, :]).all(-1)
            matches = matches & (noise[batch] == 0)[None, :]
            known = [int(indices[0]) if len(indices) else -1 for indices in
                     (torch.nonzero(v, as_tuple=False).flatten().tolist() for v in matches)]
            box = []
            m, C = _Moments.apply(features[row], mean[row], guard, batch, cat[row].tolist(), known, X.shape[-2], box)
            means.append(m)
            covariances.append(C)
            errors.extend(box)
        result = (torch.stack(means).reshape(batch_shape + (full_q,)),
                  torch.stack(covariances).reshape(batch_shape + (full_q, full_q)))
        if hasattr(self.state.model, "outcome_transform"):
            from botorch.posteriors.gpytorch import GPyTorchPosterior
            from gpytorch.distributions import MultivariateNormal
            from linear_operator.operators import DenseLinearOperator
            mean, covariance = result
            me = [v for block in errors for v in block.mean_error]
            ce = [v for block in errors for row in block.covariance_error for v in row]
            diagonal = torch.eye(full_q, dtype=torch.bool).expand_as(covariance)
            for stage in _outcome_stages(self.state.model.outcome_transform):
                posterior = GPyTorchPosterior(MultivariateNormal(mean, DenseLinearOperator(covariance)))
                scale = stage.stdvs[..., 0].expand_as(mean)
                _affine_preflight(mean, scale, stage.means[..., 0], me, None)
                nominal, ne = _affine_preflight(covariance, scale.unsqueeze(-1), None, ce, diagonal)
                _affine_preflight(covariance.new_tensor(nominal).reshape_as(covariance),
                                  scale.unsqueeze(-2), None, ne, diagonal)
                actual = stage.untransform_posterior(posterior, X=transformed)
                mean, me = checked_affine(mean, scale, stage.means[..., 0], guard=guard, errors=me,
                    actual=actual.mean[..., 0], _return_errors=True)
                intermediate, ce = checked_affine(covariance, scale.unsqueeze(-1), guard=guard, errors=ce,
                    variance=diagonal, _return_errors=True)
                covariance, ce = checked_affine(intermediate, scale.unsqueeze(-2), guard=guard, errors=ce,
                    variance=diagonal, actual=actual.distribution.covariance_matrix, _return_errors=True)
            result = mean, covariance
        guard.check()
        return result
