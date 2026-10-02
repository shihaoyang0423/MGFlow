import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def reference_rows(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows.sort(key=lambda row: int(row["id"]))
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate reference image IDs")
    return rows


def prompts_from_file(path):
    path = Path(path)
    if path.suffix == ".json":
        prompts = json.loads(path.read_text())
    else:
        prompts = [row["prompt"] for row in reference_rows(path)]
    if not prompts or any(not isinstance(value, str) or not value for value in prompts):
        raise ValueError("prompts must be a nonempty list of strings")
    return prompts


def center_crop(image, size=256):
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(value // 2 for value in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(
        tuple(round(value * scale) for value in image.size), Image.Resampling.BICUBIC
    )
    x, y = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((x, y, x + size, y + size))


def image_tensor(image):
    import torch

    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


class ImageNetImages:
    def __init__(self, root):
        root = Path(root)
        array = root if root.suffix == ".npy" else root / "images.npy"
        self.array = np.load(array, mmap_mode="r", allow_pickle=False) if array.is_file() else None
        if self.array is not None:
            if (
                self.array.dtype != np.uint8
                or self.array.ndim != 4
                or self.array.shape[-1] != 3
                or len(self.array) == 0
            ):
                raise ValueError("image arrays must be nonempty uint8 [N,H,W,3]")
            self.paths = []
            return
        self.paths = sorted(
            p
            for p in Path(root).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"no images in {root}")

    def __len__(self):
        return len(self.array) if self.array is not None else len(self.paths)

    def __getitem__(self, index):
        if self.array is not None:
            image = Image.fromarray(self.array[index])
            return image_tensor(center_crop(image)), index
        with Image.open(self.paths[index]) as image:
            return image_tensor(center_crop(image.convert("RGB"))), index


class T2IImages:
    def __init__(self, root):
        self.root = Path(root)
        self.rows = reference_rows(self.root / "metadata.jsonl")
        self.archive = None
        self.shard = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.shard != row["shard"]:
            if self.archive is not None:
                self.archive.close()
            self.archive = tarfile.open(self.root / row["shard"])
            self.shard = row["shard"]
        with self.archive.extractfile(row["image"]) as stream, Image.open(stream) as image:
            image = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
            return image_tensor(image), index
