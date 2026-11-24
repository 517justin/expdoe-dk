"""Bounded, non-acceptance latency measurement of the actual directional adapter.

Run with python -m tests.monotone_v2.source_latency --case n32-e128.
No optimizer, campaign settings, seeds, or search pools are changed by this tool.
"""
import argparse
import json
import resource
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
from botorch import settings
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel

from expdoe_dk.bo.directional_pairs import DirectionalPairs
from expdoe_dk.bo.directional_posterior import condition_directional_posterior
from expdoe_dk.bo.surrogates import _PhysicalAcquisitionModel
from .test_source import frozen_fit, tensor


class TimeLimit(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["frozen-current", "frozen-floor", "n32-e6", "n32-e128"], required=True)
    parser.add_argument("--limit", type=float, default=120.)
    args = parser.parse_args()
    result = {"case": args.case, "torch": torch.__version__, "stage_limit_seconds": args.limit,
              "samples": 4, "scope": "posterior and physical affine, not acquisition optimization"}
    def timeout(signum, frame):
        raise TimeLimit()
    signal.signal(signal.SIGALRM, timeout)
    def measured(name, function):
        started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, args.limit)
        try:
            value = function()
        except TimeLimit:
            result.update(status="time_limit", limited_stage=name)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result[name+"_seconds"] = time.perf_counter()-started
            result["max_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(json.dumps(result, sort_keys=True), flush=True)
        return value
    if args.case.startswith("frozen"):
        data, base = frozen_fit.__wrapped__(SimpleNamespace(param=args.case.split("-")[1]))
        n, d, batches = len(data["training"]), 1, 1
        endpoints = tensor(data["query"][n:]).reshape(3, 2, 1)
        widths = endpoints[:, 1, 0]-endpoints[:, 0, 0]
        y = np.asarray(data["physical_gain"])
        location, scale, sign = float(y.mean()), float(y.std()), -1.
        query = tensor([[.41], [.57]]).requires_grad_()
    else:
        n, d, batches = 32, 6, 32
        count = 3 if args.case.endswith("e6") else 64
        factors = tensor([1., 3., 5., 7., 11., 13.])
        training = ((torch.arange(n, dtype=torch.double)[:, None]*factors+1) % 37)/37
        targets = torch.sin(3*training[:, :1])+.2*training[:, 1:2]
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)).double()
        kernel.base_kernel.lengthscale = tensor([.4, .5, .6, .7, .8, .9])
        kernel.outputscale = 1.1
        with settings.validate_input_scaling(False):
            base = SingleTaskGP(training, targets, train_Yvar=torch.full_like(targets, .03),
                                covar_module=kernel, outcome_transform=None)
        centers = .1+.8*((torch.arange(count, dtype=torch.double)[:, None]*factors+3) % 131)/131
        endpoints = centers[:, None, :].expand(count, 2, d).clone()
        for i in range(count):
            endpoints[i, 0, i % d] -= .03
            endpoints[i, 1, i % d] += .03
        widths = tensor([endpoints[i, 1, i % d]-endpoints[i, 0, i % d] for i in range(count)])
        location, scale, sign = 0., 1., 1.
        query = torch.stack((torch.linspace(.2, .6, d, dtype=torch.double),
                             torch.linspace(.3, .7, d, dtype=torch.double)))
        query = (query[None]+torch.arange(batches, dtype=torch.double)[:, None, None]*.001).requires_grad_()
    count = len(endpoints)
    pairs = DirectionalPairs(endpoints, tensor([-1.] * count), widths,
                             tensor([1.] * count), tensor([.8/count] * count), ("latency",)*count, [])
    result.update(n=n, d=d, q=2, endpoints=2*count, t_batch=batches)
    try:
        model, _ = measured("setup", lambda: condition_directional_posterior(base, pairs))
        assert model._moment_source is not None
        measured("cached_setup", lambda: model._moment_source.joint_moments(endpoints.new_empty((0, d))))
        physical = _PhysicalAcquisitionModel(model, location, scale, sign)
        base_samples = torch.linspace(-.7, .7, 4*query.shape[:-1].numel(), dtype=torch.double).reshape(
            (4,)+query.shape[:-1])
        def forward():
            posterior = physical.posterior(query)
            samples = posterior.rsample_from_base_samples(torch.Size([4]), base_samples)
            return posterior.mean.square().sum()+posterior.variance.sum()+samples.square().mean()
        value = measured("forward", forward)
        gradient, = measured("backward", lambda: torch.autograd.grad(value, query))
        assert torch.isfinite(value) and torch.isfinite(gradient).all()
        result.update(status="completed", precision_states=len(model._moment_source.blocks[0]._states),
                      solver_states=len(model._moment_source.blocks[0]._solvers))
    except TimeLimit:
        pass
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
