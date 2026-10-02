from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..distributed import rank, world_size
from .assignment import assign
from .gaussian import psd_sqrt, symmetric
from .moments import moments


@torch.no_grad()
def kl_velocity(
    features, assignment, reference, state, ridge, recipe="imagenet", chol_p=None, inverse_p=None
):
    x = features.double()
    p_covariance = reference.covariance
    if recipe == "imagenet":
        outer = reference.mean[:, :, None] * reference.mean[:, None, :]
        p_covariance = symmetric((p_covariance + outer) - outer)
    if chol_p is None:
        chol_p = torch.linalg.cholesky(p_covariance + ridge * reference.eye)
    chol_q = torch.linalg.cholesky(state.covariance + ridge * reference.eye)
    if recipe == "t2i":
        if inverse_p is None:
            inverse_p = torch.cholesky_inverse(chol_p)
        velocity = torch.zeros_like(x)
        for k in range(reference.k):
            sp = (reference.mean[k] - x) @ inverse_p[k]
            sq = torch.cholesky_solve((state.mean[k] - x).T.contiguous(), chol_q[k]).T
            velocity.add_((sp - sq) * assignment[:, k, None])
        return velocity

    def score(mean, chol):
        delta = mean[None] - x[:, None, :]
        rhs = delta.permute(1, 2, 0).contiguous()
        return torch.cholesky_solve(rhs, chol).permute(2, 0, 1).contiguous()

    return (
        assignment[:, :, None] * (score(reference.mean, chol_p) - score(state.mean, chol_q))
    ).sum(1)


def gaussian_cost(mean, covariance, reference, ridge=0.0, sqrt_reference=None):
    covariance = symmetric(covariance) + ridge * reference.eye
    p_covariance = reference.covariance + ridge * reference.eye
    root = psd_sqrt(p_covariance) if sqrt_reference is None else sqrt_reference
    product = torch.bmm(torch.bmm(root, covariance), root)
    cross = torch.linalg.eigvalsh(symmetric(product)).clamp_min(0).sqrt().sum(-1)
    return (
        (mean - reference.mean).square().sum(-1)
        + covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        + p_covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        - 2 * cross
    )


def regression_loss(features, velocity):
    x = features.double()
    target = (x.detach() + velocity.double()).detach()
    return (x - target).square().sum(1).mean().float()


@dataclass
class Branch:
    reference: object
    state: object
    objective: str = "kl"
    ridge: float = 0.03
    weight: float = 1.0
    domain: str = "imagenet"
    owner: int = 0

    def __post_init__(self):
        if self.objective not in ("kl", "w2") or self.domain not in ("imagenet", "t2i"):
            raise ValueError("unsupported objective or domain")
        if self.domain == "t2i" and self.objective != "kl":
            raise ValueError("text-to-image training uses KL")
        if self.ridge < 0 or self.weight <= 0:
            raise ValueError("ridge must be nonnegative and weight positive")
        self._sqrt_reference = (
            psd_sqrt(self.reference.covariance + self.ridge * self.reference.eye)
            if self.objective == "w2"
            else None
        )
        self._pending = None
        self._warmup_tail = None
        self._warmup_mass = torch.zeros_like(self.reference.weight)
        self._chol_reference = None
        self._inverse_reference = None
        if self.objective == "kl":
            covariance = self.reference.covariance
            if self.domain == "imagenet":
                outer = self.reference.mean[:, :, None] * self.reference.mean[:, None, :]
                covariance = symmetric((covariance + outer) - outer)
            self._chol_reference = torch.linalg.cholesky(
                covariance + self.ridge * self.reference.eye
            )
            if self.domain == "t2i":
                self._inverse_reference = torch.cholesky_inverse(self._chol_reference)

    def feature_gradient(self, features, beta, indices=None):
        """Return a global-mean feature gradient without retaining an encoder graph."""
        if self.objective == "w2" and world_size() > 1:
            return self._distributed_w2_gradient(features, beta, indices)
        x = features.detach().double().requires_grad_(True)
        if self.objective == "kl":
            local_x = x if indices is None else x[indices]
            velocity = self.velocity(x, beta, indices)
            loss = regression_loss(local_x, velocity) * (len(local_x) / len(x))
        else:
            assignment = assign(x, self.reference)
            recipe = "w2" if self.reference.k > 1 else "kl"
            mean, second = moments(x, assignment, self.reference.weight, recipe)
            self._pending = mean.detach(), second.detach(), beta
            q_mean = beta * self.state.mean.detach() + (1 - beta) * mean
            q_second = beta * self.state.second.detach() + (1 - beta) * second
            covariance = q_second - q_mean[:, :, None] * q_mean[:, None, :]
            loss = (
                self.reference.weight
                * gaussian_cost(
                    q_mean, covariance, self.reference, self.ridge, self._sqrt_reference
                )
            ).sum()
        (gradient,) = torch.autograd.grad(self.weight * loss, x)
        if indices is not None:
            gradient = gradient[indices]
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("nonfinite feature gradient")
        return gradient.to(features.dtype), float(loss.detach())

    def _distributed_w2_gradient(self, features, beta, indices):
        owner = self.owner % world_size()
        x = features.detach().double().requires_grad_(rank() == owner)
        assignment = assign(x, self.reference)
        mean, second = moments(
            x, assignment, self.reference.weight, "w2" if self.reference.k > 1 else "kl"
        )
        self._pending = mean.detach(), second.detach(), beta
        gradient = torch.empty_like(x)
        value = torch.empty((), device=x.device, dtype=torch.float64)
        status = torch.ones((), device=x.device, dtype=torch.int32)
        detail = "W2 objective failed on its owner rank"
        if rank() == owner:
            try:
                q_mean = beta * self.state.mean + (1 - beta) * mean
                q_second = beta * self.state.second + (1 - beta) * second
                covariance = q_second - q_mean[:, :, None] * q_mean[:, None, :]
                loss = (
                    self.reference.weight
                    * gaussian_cost(
                        q_mean, covariance, self.reference, self.ridge, self._sqrt_reference
                    )
                ).sum()
                (computed,) = torch.autograd.grad(self.weight * loss, x)
                if not bool(torch.isfinite(computed).all()):
                    raise FloatingPointError("nonfinite W2 gradient")
                gradient.copy_(computed)
                value.copy_(loss.detach())
            except Exception as error:
                status.zero_()
                detail = str(error)
        dist.broadcast(status, owner)
        if not status.item():
            raise RuntimeError(detail)
        dist.broadcast(gradient, owner)
        dist.broadcast(value, owner)
        if indices is not None:
            gradient = gradient[indices]
        return gradient.to(features.dtype), float(value)

    @torch.no_grad()
    def velocity(self, features, beta, indices=None):
        if self.objective != "kl":
            raise ValueError("velocity regression is used for KL, not the W2 moment objective")
        x = features.detach().double()
        assignment = assign(x, self.reference)
        mean, second = moments(x, assignment, self.reference.weight)
        self._pending = mean, second, beta
        if self.domain == "t2i":
            self.commit()
        local_x = x if indices is None else x[indices]
        local_assignment = assignment if indices is None else assignment[indices]
        return kl_velocity(
            local_x,
            local_assignment,
            self.reference,
            self.state,
            self.ridge,
            self.domain,
            self._chol_reference,
            self._inverse_reference,
        )

    @torch.no_grad()
    def warmup(self, features):
        if self.domain == "imagenet":
            if self.reference.k == 1:
                self._accumulate_warmup(features)
                return
            pending = features.detach().double()
            if self._warmup_tail is not None:
                pending = torch.cat((self._warmup_tail, pending))
            used = 0
            while used + 1024 <= len(pending):
                self._accumulate_warmup(pending[used : used + 1024])
                used += 1024
            self._warmup_tail = pending[used:].clone() if used < len(pending) else None
            return
        assignment = assign(features, self.reference)
        recipe = "w2" if self.objective == "w2" and self.reference.k > 1 else "kl"
        mean, second = moments(features, assignment, self.reference.weight, recipe)
        self.state.accumulate(mean, second, len(features))

    @torch.no_grad()
    def _accumulate_warmup(self, features):
        x = features.detach().double()
        assignment = assign(x, self.reference)
        mass = assignment.sum(0) if self.objective == "w2" else len(x) * self.reference.weight
        self._warmup_mass.add_(mass)
        self.state.mean.add_(assignment.T @ x)
        if self.objective == "w2" and self.reference.k > 1:
            self.state.second.add_(torch.einsum("bk,bi,bj->kij", assignment, x, x))
        else:
            for k in range(self.reference.k):
                self.state.second[k].add_(x.T @ (assignment[:, k, None] * x))
        self.state.count += len(x)

    @torch.no_grad()
    def finalize_warmup(self):
        if self.domain == "imagenet":
            if self._warmup_tail is not None:
                self._accumulate_warmup(self._warmup_tail)
                self._warmup_tail = None
            if self.state.count == 0:
                raise ValueError("no generated features accumulated during warmup")
            self.state.mean.div_(self._warmup_mass[:, None])
            self.state.second.div_(self._warmup_mass[:, None, None])
            self._warmup_mass = None

    @torch.no_grad()
    def commit(self):
        if self._pending is not None:
            mean, second, beta = self._pending
            self.state.update(mean, second, beta, self.domain)
            self._pending = None
