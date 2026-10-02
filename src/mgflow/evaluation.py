import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import ImageNetImages
from .distributed import rank, world_size
from .encoders import EvaluationEncoder

IMAGENET_STATS = {
    "Inception": ("guided_diffusion_stats.npz", 1.68),
    "MAE": ("vit_large_patch16_224_mae_in256_t224_stats.npz", 0.04),
    "DINOv2": ("vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz", 14.19),
    "CLIP": ("vit_large_patch14_clip_224_openai_in256_t256_stats.npz", 5.60),
    "SigLIP": ("vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz", 0.60),
    "ConvNeXt": ("convnext_in256_t224_stats.npz", 56.87),
}
FDR3_ENCODERS = ("ConvNeXt", "DINOv2", "CLIP")


class FeatureMoments:
    def __init__(self, dimension, device="cpu"):
        self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.total = torch.zeros(dimension, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dimension, dimension, dtype=torch.float64, device=device)

    def update(self, features):
        values = features.double()
        if (
            values.ndim != 2
            or values.shape[1] != len(self.total)
            or not torch.isfinite(values).all()
        ):
            raise ValueError("features must be finite [N,D] arrays")
        self.count += len(values)
        self.total += values.sum(0)
        self.outer.addmm_(values.T, values)

    def finalize(self, distributed=False):
        if distributed:
            for value in (self.count, self.total, self.outer):
                dist.all_reduce(value)
        n = int(self.count)
        if n < 2:
            raise ValueError("Frechet evaluation requires at least two samples")
        mean = self.total / n
        covariance = (self.outer - self.total[:, None] * mean[None]) / (n - 1)
        return mean.cpu().numpy(), covariance.cpu().numpy()


def gaussian_distance(mean, covariance, ref_mean, ref_covariance):
    from scipy.linalg import sqrtm

    mean, ref_mean = np.asarray(mean, dtype=np.float64), np.asarray(ref_mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    ref_covariance = np.asarray(ref_covariance, dtype=np.float64)
    d = len(mean)
    if mean.shape != ref_mean.shape or covariance.shape != (d, d) or ref_covariance.shape != (d, d):
        raise ValueError("generated and reference statistic shapes differ")
    if not all(np.isfinite(value).all() for value in (mean, ref_mean, covariance, ref_covariance)):
        raise ValueError("Frechet statistics must be finite")
    root = sqrtm(covariance @ ref_covariance)
    if not np.isfinite(root).all():
        offset = np.eye(d) * 1e-6
        root = sqrtm((covariance + offset) @ (ref_covariance + offset))
    if np.iscomplexobj(root):
        if not np.allclose(root.diagonal().imag, 0, atol=1e-3):
            raise ValueError("non-real covariance square root")
        root = root.real
    difference = mean - ref_mean
    value = difference @ difference + np.trace(covariance + ref_covariance) - 2 * np.trace(root)
    if not np.isfinite(value):
        raise ValueError("nonfinite Frechet distance")
    return float(max(0, value))


def frechet(features, reference_stats, device="cuda"):
    values = np.load(features, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2:
        raise ValueError("features must be [N,D]")
    moments = FeatureMoments(values.shape[1], device)
    for first in range(0, len(values), 1024):
        moments.update(torch.tensor(np.array(values[first : first + 1024]), device=device))
    mean, covariance = moments.finalize()
    with np.load(reference_stats, allow_pickle=False) as arrays:
        return gaussian_distance(
            mean, covariance, arrays["mu"].reshape(-1), arrays["sigma"].reshape(covariance.shape)
        )


def fdr(distances, reference_validation_distances):
    numerator = np.asarray(distances, dtype=np.float64)
    denominator = np.asarray(reference_validation_distances, dtype=np.float64)
    if numerator.ndim != 1 or numerator.size == 0 or numerator.shape != denominator.shape:
        raise ValueError("one normalization denominator is required per encoder")
    if (
        not np.isfinite(numerator).all()
        or not np.isfinite(denominator).all()
        or (denominator <= 0).any()
    ):
        raise ValueError("FDr requires finite distances and positive denominators")
    return float(np.mean(numerator / denominator))


def inception_score(logits, splits=10):
    logits = torch.as_tensor(logits).double()
    if logits.ndim != 2 or len(logits) < splits or not torch.isfinite(logits).all():
        raise ValueError("Inception Score requires finite [N,C] logits and N >= splits")
    permutation = np.random.RandomState(2020).permutation(len(logits))
    logits = logits[permutation]
    scores = []
    for i in range(splits):
        part = logits[i * len(logits) // splits : (i + 1) * len(logits) // splits]
        probabilities = part.softmax(1)
        marginal = probabilities.mean(0).clamp_min(torch.finfo(torch.float64).tiny)
        score = (probabilities * (part.log_softmax(1) - marginal.log())).sum(1).mean().exp()
        scores.append(float(score))
    return float(np.mean(scores)), float(np.std(scores))


def imagenet_summary(distances, score, count):
    if set(distances) != set(IMAGENET_STATS):
        raise ValueError("ImageNet evaluation requires all six encoders")
    normalized = {name: distances[name] / IMAGENET_STATS[name][1] for name in distances}
    return {
        "count": int(count),
        "FID": distances["Inception"],
        "FDr6": fdr(list(distances.values()), [IMAGENET_STATS[name][1] for name in distances]),
        "FDr3": float(np.mean([normalized[name] for name in FDR3_ENCODERS])),
        "IS": score[0],
        "IS_std": score[1],
        "FD": distances,
        "FDr": normalized,
    }


@torch.inference_mode()
def evaluate_imagenet(
    images, stats, weights, output, count=50000, batch=64, device="cuda", workers=4
):
    dataset = ImageNetImages(images)
    if len(dataset) != count or count < 10 or batch <= 0:
        raise ValueError(f"expected {count} images (at least 10), found {len(dataset)}")
    start, stop = count * rank() // world_size(), count * (rank() + 1) // world_size()
    loader = DataLoader(Subset(dataset, range(start, stop)), batch_size=batch, num_workers=workers)
    distances, score = {}, None
    for name, (filename, _) in IMAGENET_STATS.items():
        path = Path(stats) / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        encoder = EvaluationEncoder(name, weights, device)
        moments = FeatureMoments(encoder.dimension, device)
        logits = []
        for image, _ in loader:
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda") and name != "Inception",
            ):
                features, prediction = encoder(image.to(device))
            moments.update(features)
            if prediction is not None:
                logits.append(prediction.cpu())
        mean, covariance = moments.finalize(distributed=world_size() > 1)
        with np.load(path, allow_pickle=False) as reference:
            distances[name] = gaussian_distance(
                mean, covariance, reference["mu"], reference["sigma"]
            )
        if name == "Inception":
            local = torch.cat(logits) if logits else torch.empty(0, 1008)
            parts = [None] * world_size()
            if world_size() > 1:
                dist.all_gather_object(parts, local)
            else:
                parts = [local]
            score = inception_score(torch.cat(parts))
        del encoder, moments
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    result = imagenet_summary(distances, score, count)
    if rank() == 0:
        save_results(output, result)
    return result


def save_results(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
