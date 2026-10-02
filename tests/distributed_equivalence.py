"""Run with torchrun --nproc-per-node=2 tests/distributed_equivalence.py."""

import argparse
import json

import torch
import torch.distributed as dist

from mgflow import Branch, GaussianMixture, MomentState
from mgflow.distributed import gather, rank, setup, sum_gradients, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    device = setup(parser.parse_args().device)
    torch.manual_seed(0)
    result = []
    n, d = 64, 8
    for objective, families in [("kl", [1, 4, 16]), ("w2", [1, 4])]:
        for k in families:
            model = torch.nn.Linear(d, d, bias=False).to(device)
            full_input = torch.randn(n, d, device=device)
            local_input = full_input.chunk(world_size())[rank()]
            p = GaussianMixture(
                torch.ones(k, device=device),
                torch.randn(k, d, device=device, dtype=torch.float64),
                torch.eye(d, device=device, dtype=torch.float64)[None].repeat(k, 1, 1),
            )
            q = MomentState.from_reference(p)
            local = model(local_input)
            full = gather(local.detach())
            route = Branch(p, q, objective, ridge=0.03)
            indices = torch.arange(rank() * len(local), (rank() + 1) * len(local), device=device)
            gradient, _ = route.feature_gradient(full, 0.995, indices)
            local.backward(gradient)
            sum_gradients(model.parameters())
            baseline = torch.nn.Linear(d, d, bias=False).to(device)
            baseline.load_state_dict(model.state_dict())
            all_output = baseline(full_input)
            reference_route = Branch(p, MomentState.from_reference(p), objective, ridge=0.03)
            all_gradient, _ = reference_route.feature_gradient(all_output, 0.995)
            all_output.backward(all_gradient)
            error = float((model.weight.grad - baseline.weight.grad).abs().max())
            torch.testing.assert_close(
                model.weight.grad, baseline.weight.grad, atol=2e-6, rtol=2e-6
            )
            result.append(
                {"objective": objective, "K": k, "world": world_size(), "max_abs_error": error}
            )
            if rank() == 0:
                print("PASS DISTRIBUTED", json.dumps(result[-1]), flush=True)
    if rank() == 0:
        print("ALL_DISTRIBUTED_EQUIVALENCE_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
