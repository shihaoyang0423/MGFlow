from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

from ..distributed import rank, world_size
from .assignment import log_density
from .gaussian import GaussianMixture, symmetric


@dataclass
class FitOptions:
    seed: int = 3407
    chunk: int = 2048
    initialization_samples: int = 65536
    kmeans_iterations: int = 20
    kmeans_min_iterations: int = 8
    center_tolerance: float = 1e-4
    weight_tolerance: float = 0.002
    mean_tolerance: float = 0.006
    covariance_tolerance: float = 0.020


def feature_bank(path):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or len(values) == 0 or values.dtype != np.float32:
        raise ValueError("features must be a nonempty float32 [N,D] NPY array")
    return values


def blocks(features, chunk, device):
    start = len(features) * rank() // world_size()
    stop = len(features) * (rank() + 1) // world_size()
    for first in range(start, stop, chunk):
        yield torch.from_numpy(np.array(features[first : min(first + chunk, stop)], copy=True)).to(
            device
        )


def _reduce(*tensors):
    if world_size() > 1:
        for tensor in tensors:
            dist.all_reduce(tensor)


def _nearest(values, centers):
    distance = values.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
    distance -= 2 * values @ centers.T
    return distance.argmin(1)


@torch.no_grad()
def _centers(features, k, options, device):
    centers = torch.empty(k, features.shape[1], device=device, dtype=torch.float32)
    if rank() == 0:
        rng = np.random.default_rng(options.seed)
        indices = np.sort(
            rng.choice(
                len(features), min(len(features), options.initialization_samples), replace=False
            )
        )
        sample = torch.from_numpy(np.array(features[indices], copy=True)).to(device)
        generator = torch.Generator(device=device).manual_seed(options.seed)
        selected = torch.randint(len(sample), (1,), generator=generator, device=device).item()
        centers[0] = sample[selected]
        distance = (sample - centers[0]).square().sum(1)
        for index in range(1, k):
            total = distance.double().sum()
            if not torch.isfinite(total) or total <= 0:
                raise ValueError("too few distinct features for the requested components")
            selected = torch.multinomial(distance.double() / total, 1, generator=generator).item()
            centers[index] = sample[selected]
            distance = torch.minimum(distance, (sample - centers[index]).square().sum(1))
    if world_size() > 1:
        dist.broadcast(centers, 0)
    for iteration in range(options.kmeans_iterations):
        mass = torch.zeros(k, device=device, dtype=torch.float64)
        first = torch.zeros_like(centers, dtype=torch.float64)
        for values in blocks(features, options.chunk, device):
            labels = _nearest(values, centers)
            mass += torch.bincount(labels, minlength=k)
            first.index_add_(0, labels, values.double())
        _reduce(mass, first)
        if (mass == 0).any():
            raise ValueError("empty component during k-means initialization")
        updated = (first / mass[:, None]).float()
        change = ((updated - centers).norm(dim=1) / centers.norm(dim=1).clamp_min(1e-12)).max()
        centers = updated
        if iteration + 1 >= options.kmeans_min_iterations and change <= options.center_tolerance:
            break
    return centers


@torch.no_grad()
def _statistics(features, k, options, device, centers=None, reference=None):
    d = features.shape[1]
    mass = torch.zeros(k, device=device, dtype=torch.float64)
    first = torch.zeros(k, d, device=device, dtype=torch.float64)
    second = torch.zeros(k, d, d, device=device, dtype=torch.float64)
    for values in blocks(features, options.chunk, device):
        x = values.double()
        if k == 1:
            mass += len(x)
            first[0] += x.sum(0)
            second[0].addmm_(x.T, x)
        elif reference is None:
            labels = _nearest(values, centers)
            mass += torch.bincount(labels, minlength=k)
            first.index_add_(0, labels, x)
            for index in range(k):
                selected = x[labels == index]
                second[index].addmm_(selected.T, selected)
        else:
            scores = log_density(x, reference.mean, reference.gate_cholesky, reference.weight)
            probabilities = scores.softmax(1)
            mass += probabilities.sum(0)
            first += probabilities.T @ x
            for index in range(k):
                second[index].addmm_(x.T, probabilities[:, index, None] * x)
    _reduce(mass, first, second)
    if (mass <= 0).any():
        raise ValueError("an EM component has zero mass")
    mean = first / mass[:, None]
    covariance = symmetric(second / mass[:, None, None] - mean[:, :, None] * mean[:, None, :])
    return GaussianMixture(mass / mass.sum(), mean, covariance)


@torch.no_grad()
def em_step(features, reference, options):
    return _statistics(features, reference.k, options, reference.mean.device, reference=reference)


def parameter_change(old, new):
    def relative(a, b):
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    return (
        float((old.weight - new.weight).abs().max()),
        relative(old.mean, new.mean),
        relative(old.covariance, new.covariance),
    )


def converged(change, options):
    limits = (options.weight_tolerance, options.mean_tolerance, options.covariance_tolerance)
    return all(value <= limit for value, limit in zip(change, limits))


@torch.no_grad()
def fit(features, k, domain="imagenet", device="cuda", options=None):
    if domain not in ("imagenet", "t2i") or k not in (1, 4, 16):
        raise ValueError("fit supports K=1,4,16 for ImageNet and text-to-image features")
    if len(features) < k:
        raise ValueError("the feature bank must contain at least K samples")
    options = options or FitOptions()
    if k == 1:
        return _statistics(features, 1, options, device)
    centers = _centers(features, k, options, device)
    reference = _statistics(features, k, options, device, centers=centers)
    previous, consecutive = None, 0
    for iteration in range(96 if domain == "imagenet" else 384):
        updated = em_step(features, reference, options)
        change = parameter_change(reference, updated)
        if converged(change, options):
            nonincreasing = previous is not None and all(a <= b for a, b in zip(change, previous))
            consecutive = consecutive + 1 if nonincreasing else 1
            previous = change
        else:
            consecutive, previous = 0, None
        reference = updated
        if rank() == 0:
            print(
                f"EM {iteration + 1}: weight={change[0]:.6g} mean={change[1]:.6g} "
                f"covariance={change[2]:.6g}",
                flush=True,
            )
        if consecutive >= 2:
            if domain == "imagenet" and not converged(
                parameter_change(reference, em_step(features, reference, options)), options
            ):
                consecutive = 0
                continue
            return reference
    raise RuntimeError("reference fit did not meet the convergence thresholds")


@torch.no_grad()
def check(features, reference, options=None):
    options = options or FitOptions()
    reference.validate_covariance()
    if reference.k == 1:
        return {"components": 1, "dimension": reference.dimension}
    if features.shape[1] != reference.dimension:
        raise ValueError("feature and reference dimensions differ")
    change = parameter_change(reference, em_step(features, reference, options))
    return {
        "components": reference.k,
        "dimension": reference.dimension,
        "weight_change": change[0],
        "mean_change": change[1],
        "covariance_change": change[2],
        "converged": converged(change, options),
        "weight_ratio": float(reference.weight.max() / reference.weight.min()),
    }
