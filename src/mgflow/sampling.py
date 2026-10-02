import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from .checkpoint import load_checkpoint
from .distributed import rank, world_size
from .models import PROFILES, build


def benchmark_directory(output):
    path = Path(output)
    error = None
    if rank() == 0:
        try:
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(path)
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exception:
            error = str(exception)
    if world_size() > 1:
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error is not None:
        raise FileExistsError(error)
    return path


@torch.inference_mode()
def imagenet(model_name, checkpoint, output, count=50000, batch=32, seed=1):
    if count <= 0 or batch <= 0:
        raise ValueError("sample count and batch size must be positive")
    path = benchmark_directory(output)
    temporary = path / "images.partial"
    temporary.touch(exist_ok=False)
    model = build(model_name).cuda().eval()
    load_checkpoint(checkpoint, model)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(count, 256, 256, 3)
    )
    profile = PROFILES[model_name]
    classes = torch.arange(count, device="cuda") % 1000
    labels = []
    for first in range(0, count, batch):
        n = min(batch, count - first)
        noise = (
            torch.randn(n, 3, 256, 256, generator=generator, device="cuda") * profile["noise_scale"]
        )
        y = classes[first : first + n]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_images = model(noise, y, profile).float()
        pixels = ((output_images * 0.5 + 0.5) * 255).round().clamp(0, 255).byte()
        images[first : first + n] = pixels.permute(0, 2, 3, 1).cpu().numpy()
        labels.extend(y.cpu().tolist())
    images.flush()
    del images
    (path / "sampling.json").write_text(
        json.dumps(dict(model=model_name, count=count, seed=seed, labels=labels))
    )
    temporary.rename(path / "images.npy")


@torch.inference_mode()
def text_to_image(base, checkpoint, prompts, output, seed=1):
    from .models.flux import load_pipeline

    pipeline, model = load_pipeline(base)
    load_checkpoint(checkpoint, model)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        destination = path / f"{index:06d}.png"
        if destination.exists():
            raise FileExistsError(destination)
        generator = torch.Generator(device="cuda").manual_seed(seed + index)
        noise = torch.randn(1, 128, 32, 32, generator=generator, device="cuda")
        render_text_to_image(pipeline, model, prompt, noise).save(destination)


@torch.inference_mode()
def render_text_to_image(pipeline, model, prompt, noise):
    from .models.flux import decode

    device = noise.device
    context, _ = pipeline.encode_prompt(prompt, device=device, max_sequence_length=512)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent = model(
            noise,
            context,
            pipeline._prepare_latent_ids(noise).to(device),
            pipeline._prepare_text_ids(context).to(device),
        )
        image = decode(pipeline, latent, resolution=512)
    pixels = (image[0].permute(1, 2, 0).float() * 255).round().clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(pixels)


@torch.inference_mode()
def sample_benchmark(kind, base, checkpoint, prompts, output, samples=4, seed=None):
    from .models.flux import load_pipeline
    from .t2i_evaluation import benchmark_rows

    if kind not in ("geneval", "pickscore") or samples <= 0:
        raise ValueError("benchmark must be geneval or pickscore with positive sample count")
    rows = benchmark_rows(prompts)
    path = benchmark_directory(output)
    pipeline, model = load_pipeline(base)
    load_checkpoint(checkpoint, model)
    seed = (46_000_000 if kind == "geneval" else 0) if seed is None else seed
    stream = torch.Generator(device="cuda").manual_seed(seed)
    noise_bank = (
        torch.randn(len(rows), 128, 32, 32, generator=stream, device="cuda")
        if kind == "pickscore"
        else None
    )
    for index in range(rank(), len(rows), world_size()):
        row = rows[index]
        if kind == "geneval":
            directory = path / f"{index:05d}"
            (directory / "samples").mkdir(parents=True)
            (directory / "metadata.jsonl").write_text(json.dumps(row) + "\n")
            for sample in range(samples):
                generator = torch.Generator(device="cuda").manual_seed(seed + index * 100 + sample)
                noise = torch.randn(1, 128, 32, 32, generator=generator, device="cuda")
                render_text_to_image(pipeline, model, row["prompt"], noise).save(
                    directory / "samples" / f"{sample:04d}.png"
                )
        else:
            render_text_to_image(
                pipeline, model, row["prompt"], noise_bank[index : index + 1]
            ).save(path / f"{index:05d}.png")
