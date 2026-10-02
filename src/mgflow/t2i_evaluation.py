import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .evaluation import save_results

GENEVAL_TASKS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")


def benchmark_rows(path):
    path = Path(path)
    if path.suffix == ".json":
        values = json.loads(path.read_text())
        rows = [dict(prompt=value) if isinstance(value, str) else value for value in values]
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("prompt"), str) or not row["prompt"] for row in rows):
        raise ValueError("benchmark metadata must contain nonempty prompts")
    return rows


def summarize_geneval(rows):
    categories = {name: [] for name in GENEVAL_TASKS}
    filenames = set()
    for row in rows:
        if row["tag"] not in categories or row["correct"] not in (True, False, 0, 1):
            raise ValueError("invalid GenEval task or correctness value")
        if row["filename"] in filenames:
            raise ValueError("duplicate GenEval image results")
        filenames.add(row["filename"])
        categories[row["tag"]].append(float(row["correct"]))
    if any(not values for values in categories.values()):
        raise ValueError("GenEval results must cover all six tasks")
    scores = {name: float(np.mean(values)) for name, values in categories.items()}
    return {"count": len(rows), **scores, "overall": float(np.mean(list(scores.values())))}


def geneval(images, repository, model_path, python=None, model_config=None, detector=None):
    images, repository = Path(images).resolve(), Path(repository).resolve()
    script = repository / "evaluation/evaluate_images.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    expected = {str(path.resolve()) for path in images.glob("*/samples/*.png")}
    if not expected:
        raise ValueError("GenEval images must use prompt/samples/*.png directories")
    for directory in sorted(images.iterdir()):
        if directory.is_dir() and not (directory / "metadata.jsonl").is_file():
            raise FileNotFoundError(directory / "metadata.jsonl")
    results = images / "results.jsonl"
    if results.exists():
        raise FileExistsError(results)
    command = [
        python or sys.executable,
        str(script),
        str(images),
        "--outfile",
        str(results),
        "--model-path",
        str(Path(model_path).resolve()),
    ]
    if model_config:
        command += ["--model-config", str(Path(model_config).resolve())]
    if detector:
        command += ["--options", f"model={detector}"]
    subprocess.run(command, check=True, cwd=repository)
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    actual = {str(Path(row["filename"]).resolve()) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("GenEval scorer did not return exactly one result for every image")
    return summarize_geneval(rows)


@torch.inference_mode()
def pickscore(images, prompts, device="cuda", batch=16, model="yuvalkirstain/PickScore_v1"):
    from transformers import AutoModel, AutoProcessor

    rows = benchmark_rows(prompts)
    files = list(Path(images).glob("*.png"))
    if len(files) != len(rows) or batch <= 0:
        raise ValueError("PickScore requires one image per prompt, in prompt order")
    if any(not path.stem.isdecimal() for path in files):
        raise ValueError("PickScore image filenames must be their prompt indices")
    files.sort(key=lambda path: int(path.stem))
    if [int(path.stem) for path in files] != list(range(len(rows))):
        raise ValueError("PickScore image indices must cover every prompt exactly once")
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    scorer = AutoModel.from_pretrained(model).to(device).eval().requires_grad_(False)
    scores = []
    for first in range(0, len(rows), batch):
        selected = rows[first : first + batch]
        pictures = []
        for path in files[first : first + len(selected)]:
            with Image.open(path) as picture:
                pictures.append(picture.convert("RGB"))
        image_inputs = processor(images=pictures, return_tensors="pt").to(device)
        text_inputs = processor(
            text=[row["prompt"] for row in selected],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        image_features = scorer.get_image_features(**image_inputs)
        text_features = scorer.get_text_features(**text_inputs)
        image_features = (
            image_features
            if isinstance(image_features, torch.Tensor)
            else image_features.pooler_output
        )
        text_features = (
            text_features
            if isinstance(text_features, torch.Tensor)
            else text_features.pooler_output
        )
        image_features, text_features = (
            F.normalize(image_features, dim=-1),
            F.normalize(text_features, dim=-1),
        )
        values = scorer.logit_scale.exp() * (image_features * text_features).sum(-1)
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite PickScore")
        scores.extend(values.cpu().tolist())
    return {"count": len(scores), "mean": float(np.mean(scores)), "per_prompt": scores}


def evaluate_t2i(
    output,
    *,
    geneval_images=None,
    geneval_repo=None,
    geneval_models=None,
    geneval_python=None,
    geneval_config=None,
    pickscore_images=None,
    pickscore_prompts=None,
    device="cuda",
    batch=16,
    geneval_detector=None,
):
    result = {}
    if geneval_images:
        if not geneval_repo or not geneval_models:
            raise ValueError("GenEval requires its official repository and detector weights")
        result["GenEval"] = geneval(
            geneval_images,
            geneval_repo,
            geneval_models,
            geneval_python,
            geneval_config,
            geneval_detector,
        )
    if pickscore_images:
        if not pickscore_prompts:
            raise ValueError("PickScore requires evaluation prompts")
        result["PickScore"] = pickscore(pickscore_images, pickscore_prompts, device, batch)
    if not result:
        raise ValueError("select GenEval and/or PickScore inputs")
    save_results(output, result)
    return result
