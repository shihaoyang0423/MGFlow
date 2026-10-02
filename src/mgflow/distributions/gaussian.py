import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def symmetric(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def psd_sqrt(matrix):
    values, vectors = torch.linalg.eigh(symmetric(matrix))
    return (vectors * values.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)


@dataclass
class GaussianMixture:
    weight: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor
    gate_jitter: float = 1e-5

    def __post_init__(self):
        self.weight = self.weight.double().reshape(-1)
        self.mean = self.mean.double()
        self.covariance = self.covariance.double()
        if self.mean.ndim != 2 or min(self.mean.shape) == 0:
            raise ValueError("reference mean must be a nonempty [K,D] array")
        k, d = self.mean.shape
        if self.weight.shape != (k,) or self.covariance.shape != (k, d, d):
            raise ValueError("reference shapes must be [K], [K,D], [K,D,D]")
        if not all(
            bool(torch.isfinite(t).all()) for t in (self.weight, self.mean, self.covariance)
        ) or bool((self.weight <= 0).any()):
            raise ValueError("reference must be finite with positive weights")
        if not torch.allclose(
            self.covariance, self.covariance.transpose(-1, -2), atol=1e-8, rtol=1e-8
        ):
            raise ValueError("reference covariance must be symmetric")
        self.covariance = symmetric(self.covariance)
        self.weight = self.weight / self.weight.sum()
        self.gate_cholesky = (
            torch.linalg.cholesky(self.covariance + self.gate_jitter * self.eye) if k > 1 else None
        )

    @property
    def k(self):
        return len(self.weight)

    @property
    def dimension(self):
        return self.mean.shape[1]

    @property
    def eye(self):
        return torch.eye(self.dimension, dtype=self.mean.dtype, device=self.mean.device)

    @classmethod
    def load(cls, path, device="cuda", gate_jitter=1e-5):
        with np.load(Path(path), allow_pickle=False) as arrays:
            weight = torch.as_tensor(arrays["pi"], device=device)
            mean = torch.as_tensor(arrays["mu"], device=device)
            covariance = torch.as_tensor(arrays["sigma"], device=device)
        if mean.ndim == 1:
            mean, covariance = mean[None], covariance[None]
        return cls(weight, mean, covariance, gate_jitter)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial.npz")
        np.savez(
            temporary,
            pi=self.weight.cpu().numpy(),
            mu=self.mean.cpu().numpy(),
            sigma=self.covariance.cpu().numpy(),
        )
        os.replace(temporary, path)

    def validate_covariance(self):
        eigenvalues = torch.linalg.eigvalsh(self.covariance)
        tolerance = 1e-8 * eigenvalues.abs().amax(-1).clamp_min(1)
        if bool((eigenvalues.amin(-1) < -tolerance).any()):
            raise ValueError("reference covariance must be positive semidefinite")
