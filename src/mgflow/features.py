from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ImageNetImages, T2IImages, prompts_from_file
from .encoders import SPECS, VisionEncoder


@torch.inference_mode()
def encode_text(prompts, output, model="timm/ViT-SO400M-16-SigLIP2-256", batch=64, device="cuda"):
    import open_clip

    prompts = prompts_from_file(prompts)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoder = open_clip.create_model(f"hf-hub:{model}")
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{model}")
    del encoder.visual
    encoder.to(device).eval().requires_grad_(False)
    bank = None
    for first in tqdm(range(0, len(prompts), batch), desc="Text features"):
        inputs = tokenizer(prompts[first : first + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            vectors = encoder.encode_text(inputs).float()
        vectors = F.normalize(vectors, dim=-1)
        if bank is None:
            bank = np.lib.format.open_memmap(
                destination, mode="w+", dtype=np.float32, shape=(len(prompts), vectors.shape[1])
            )
        bank[first : first + len(vectors)] = vectors.cpu().numpy()
    bank.flush()


@torch.inference_mode()
def encode_images(
    domain,
    images,
    output,
    weights,
    batch=32,
    workers=4,
    text_features=None,
    text_betas=None,
    device="cuda",
):
    dataset = ImageNetImages(images) if domain == "imagenet" else T2IImages(images)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    text = None
    if text_features:
        if domain != "t2i" or text_betas is None or len(text_betas) != len(SPECS):
            raise ValueError("joint features require T2I text features and three text scales")
        text = np.load(text_features, mmap_mode="r", allow_pickle=False)
        if text.dtype != np.float32 or text.shape != (len(dataset), 1152):
            raise ValueError("text features must be float32 [number of images, 1152]")
    paths = [destination / f"{name}.npy" for name in SPECS]
    if any(path.exists() for path in paths):
        raise FileExistsError("feature output already exists")
    encoders = [VisionEncoder(name, weights, device) for name in SPECS]
    banks = [
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), encoder.dimension + (1152 if text is not None else 0)),
        )
        for path, encoder in zip(paths, encoders)
    ]
    loader = DataLoader(
        dataset,
        batch_size=batch,
        num_workers=workers,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
    )
    for image, indices in tqdm(loader, desc="Image features"):
        image = image.to(device)
        for i, (encoder, bank) in enumerate(zip(encoders, banks)):
            feature = encoder(image).float().cpu().numpy()
            if text is not None:
                feature = np.concatenate([feature, text_betas[i] * text[indices.numpy()]], axis=1)
            bank[indices.numpy()] = feature
    for bank in banks:
        bank.flush()
