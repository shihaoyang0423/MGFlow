from functools import lru_cache

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from ..distributed import rank, world_size


@lru_cache(maxsize=8)
def _constraints(n, k):
    matrix = lil_matrix((n + k - 1, n * k), dtype=np.float64)
    for i in range(n):
        matrix[i, i * k : (i + 1) * k] = 1.0
    for j in range(k - 1):
        matrix[n + j, j::k] = 1.0
    return matrix.tocsr()


@torch.no_grad()
def log_density(features, means, cholesky, weights):
    columns = []
    for k in range(len(weights)):
        delta = (features.double() - means[k]).T
        whitened = torch.linalg.solve_triangular(cholesky[k], delta, upper=False)
        logdet = 2 * cholesky[k].diagonal().log().sum()
        columns.append(
            weights[k].clamp_min(1e-300).log() - 0.5 * (whitened.square().sum(0) + logdet)
        )
    return torch.stack(columns, dim=1)


@torch.no_grad()
def balanced_assignment(log_scores, weights):
    """Rows sum to one; component columns sum to N times the reference weights."""
    n, k = log_scores.shape
    if k == 1:
        return torch.ones((n, 1), dtype=torch.float64, device=log_scores.device)
    plan = torch.empty_like(log_scores, dtype=torch.float64)
    status = torch.ones((), dtype=torch.int32, device=log_scores.device)
    detail = "assignment failed on rank zero"
    if rank() == 0:
        try:
            costs = -log_scores.cpu().double().numpy()
            if not np.isfinite(costs).all():
                raise ValueError("nonfinite assignment costs")
            costs -= costs.min(axis=1, keepdims=True)
            positive = costs[costs > 0]
            costs /= max(float(np.median(positive)) if positive.size else 1.0, 1e-12)
            result = linprog(
                costs.ravel(),
                A_eq=_constraints(n, k),
                b_eq=np.concatenate([np.ones(n), n * weights[:-1].cpu().numpy()]),
                bounds=(0.0, None),
                method="highs",
                options={
                    "primal_feasibility_tolerance": 1e-10,
                    "dual_feasibility_tolerance": 1e-10,
                },
            )
            if not result.success:
                raise RuntimeError(result.message)
            plan.copy_(torch.as_tensor(result.x.reshape(n, k), device=plan.device))
        except Exception as error:
            status.zero_()
            detail = str(error)
    if world_size() > 1:
        dist.broadcast(status, 0)
    if not status.item():
        raise RuntimeError(detail)
    if world_size() > 1:
        dist.broadcast(plan, 0)
    if not bool(torch.isfinite(plan).all()) or float(plan.min()) < -1e-10:
        raise RuntimeError("invalid assignment")
    plan.clamp_min_(0.0)
    error = max(
        float((plan.sum(1) - 1).abs().max()), float((plan.sum(0) - n * weights).abs().max())
    )
    if error > 1e-8:
        raise RuntimeError(f"assignment marginal error: {error:.3e}")
    return plan


def assign(features, reference):
    n, k = len(features), reference.k
    if k == 1 or rank() != 0:
        scores = torch.empty((n, k), dtype=torch.float64, device=features.device)
    else:
        scores = log_density(features, reference.mean, reference.gate_cholesky, reference.weight)
    return balanced_assignment(scores, reference.weight)
