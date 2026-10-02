from dataclasses import dataclass

import torch

from .gaussian import symmetric


def moments(features, assignment, weight, recipe="kl"):
    x = features.double()
    if len(weight) == 1:
        return x.mean(0)[None], (x.T @ x / len(x))[None]
    if recipe == "w2":
        mass = assignment.sum(0).clamp_min(1e-12)
        mean = assignment.T @ x / mass[:, None]
        second = torch.einsum("bk,bi,bj->kij", assignment, x, x) / mass[:, None, None]
    else:
        mass = len(x) * weight
        mean = assignment.T @ x / mass[:, None]
        second = (
            torch.stack([x.T @ (assignment[:, k, None] * x) for k in range(len(weight))])
            / mass[:, None, None]
        )
    return mean, second


@dataclass
class MomentState:
    mean: torch.Tensor
    second: torch.Tensor
    count: int = 0

    @classmethod
    def empty(cls, reference):
        return cls(torch.zeros_like(reference.mean), torch.zeros_like(reference.covariance))

    @classmethod
    def from_reference(cls, reference):
        return cls(
            reference.mean.clone(),
            reference.covariance + reference.mean[:, :, None] * reference.mean[:, None, :],
        )

    @property
    def covariance(self):
        return symmetric(self.second - self.mean[:, :, None] * self.mean[:, None, :])

    @torch.no_grad()
    def accumulate(self, mean, second, n):
        fraction = n / (self.count + n)
        self.mean.mul_(1 - fraction).add_(mean, alpha=fraction)
        self.second.mul_(1 - fraction).add_(second, alpha=fraction)
        self.count += n

    @torch.no_grad()
    def update(self, mean, second, beta, recipe="imagenet"):
        if recipe == "t2i":
            self.mean.lerp_(mean, 1 - beta)
            self.second.lerp_(second, 1 - beta)
        else:
            self.mean.mul_(beta).add_(mean, alpha=1 - beta)
            self.second.mul_(beta).add_(second, alpha=1 - beta)
