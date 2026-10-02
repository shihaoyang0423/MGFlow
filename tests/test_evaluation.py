import json

import numpy as np
import pytest
import torch

from mgflow.data import ImageNetImages
from mgflow.evaluation import (
    IMAGENET_STATS,
    FeatureMoments,
    fdr,
    gaussian_distance,
    imagenet_summary,
    inception_score,
)
from mgflow.t2i_evaluation import GENEVAL_TASKS, summarize_geneval


def test_image_array_input(tmp_path):
    pixels = np.random.default_rng(0).integers(0, 256, (12, 256, 256, 3), dtype=np.uint8)
    path = tmp_path / "images.npy"
    np.save(path, pixels)
    for source in (path, tmp_path):
        dataset = ImageNetImages(source)
        image, index = dataset[3]
        assert len(dataset) == 12 and index == 3
        torch.testing.assert_close(
            image, torch.from_numpy(pixels[3].copy()).permute(2, 0, 1).float() / 255
        )


def test_streaming_feature_moments():
    features = torch.randn(117, 5, dtype=torch.float64)
    moments = FeatureMoments(5)
    for part in features.split(13):
        moments.update(part)
    mean, covariance = moments.finalize()
    np.testing.assert_allclose(mean, features.mean(0))
    np.testing.assert_allclose(covariance, np.cov(features.numpy(), rowvar=False))
    assert gaussian_distance(mean, covariance, mean + 1, covariance) == pytest.approx(5, abs=1e-10)


def test_fdr_six_and_three():
    ratios = dict(Inception=10, MAE=20, SigLIP=30, ConvNeXt=1, DINOv2=2, CLIP=3)
    distances = {name: value * IMAGENET_STATS[name][1] for name, value in ratios.items()}
    result = imagenet_summary(distances, (300, 2), 50000)
    assert result["FDr6"] == 11
    assert result["FDr3"] == 2
    assert result["FID"] == 16.8
    with pytest.raises(ValueError):
        fdr([1, float("nan")], [1, 1])
    with pytest.raises(ValueError):
        imagenet_summary({"MAE": 0.1}, (1, 0), 10)


def test_inception_score():
    assert inception_score(torch.zeros(20, 5)) == (1, 0)
    mean, std = inception_score(torch.eye(5).repeat(20, 1) * 10)
    assert 1 < mean <= 5 and std >= 0


def test_geneval_task_mean():
    rows = [
        dict(tag=name, correct=True, filename=f"{i}.png") for i, name in enumerate(GENEVAL_TASKS)
    ]
    rows.extend(
        dict(tag="single_object", correct=False, filename=f"extra{i}.png") for i in range(3)
    )
    result = summarize_geneval(rows)
    assert result["overall"] == pytest.approx((5 + 0.25) / 6)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_geneval(rows + [rows[0]])


def test_evaluation_cli_help():
    import subprocess
    import sys

    for command in ("evaluate-imagenet", "evaluate-t2i", "sample-geneval", "sample-pickscore"):
        subprocess.run(
            [sys.executable, "-m", "mgflow.cli", command, "--help"], capture_output=True, check=True
        )


def test_cli_closes_distributed_group(monkeypatch):
    from mgflow import cli

    closed = []

    def fail():
        raise ValueError("test failure")

    monkeypatch.setattr(cli, "_main", fail)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: closed.append(True))
    with pytest.raises(ValueError, match="test failure"):
        cli.main()
    assert closed == [True]


def test_imagenet_evaluation_path(tmp_path, monkeypatch):
    import mgflow.evaluation as evaluation

    np.save(tmp_path / "images.npy", np.zeros((10, 256, 256, 3), dtype=np.uint8))
    for filename, _ in IMAGENET_STATS.values():
        np.savez(tmp_path / filename, mu=np.zeros(2), sigma=np.eye(2))

    class Encoder:
        dimension = 2

        def __init__(self, name, weights, device):
            self.name = name

        def __call__(self, image):
            return torch.zeros(len(image), 2), torch.zeros(
                len(image), 1008
            ) if self.name == "Inception" else None

    monkeypatch.setattr(evaluation, "EvaluationEncoder", Encoder)
    output = tmp_path / "result.json"
    result = evaluation.evaluate_imagenet(
        tmp_path / "images.npy", tmp_path, None, output, count=10, device="cpu", workers=0
    )
    assert result["IS"] == 1
    assert set(result["FD"]) == set(IMAGENET_STATS)
    assert json.loads(output.read_text()) == result


def test_benchmark_directory(tmp_path):
    from mgflow.sampling import benchmark_directory

    path = benchmark_directory(tmp_path / "samples")
    assert path.is_dir()
    (path / "sample.png").touch()
    with pytest.raises(FileExistsError):
        benchmark_directory(path)


def test_geneval_official_subprocess(tmp_path, monkeypatch):
    from mgflow.t2i_evaluation import geneval

    repo = tmp_path / "geneval"
    (repo / "evaluation").mkdir(parents=True)
    (repo / "evaluation/evaluate_images.py").touch()
    images = tmp_path / "images"
    rows = []
    for index, task in enumerate(GENEVAL_TASKS):
        directory = images / f"{index:05d}"
        (directory / "samples").mkdir(parents=True)
        (directory / "metadata.jsonl").write_text(json.dumps({"prompt": "a cat", "tag": task}))
        picture = directory / "samples/0000.png"
        picture.touch()
        rows.append({"filename": str(picture), "tag": task, "correct": True})

    def run(command, **kwargs):
        assert "--model-path" in command and "model=mask2former" in command
        (images / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))

    monkeypatch.setattr("mgflow.t2i_evaluation.subprocess.run", run)
    assert geneval(images, repo, tmp_path, detector="mask2former")["overall"] == 1


def test_pickscore_raw_scores_and_image_order(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from PIL import Image

    from mgflow.t2i_evaluation import pickscore

    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["red", "blue"]))
    for index, color in enumerate(("red", "blue")):
        Image.new("RGB", (4, 4), color).save(tmp_path / f"{index}.png")

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        @staticmethod
        def from_pretrained(name):
            return Processor()

        def __call__(self, images=None, text=None, **kwargs):
            if images is not None:
                vectors = [
                    [picture.getpixel((0, 0))[0], picture.getpixel((0, 0))[2]] for picture in images
                ]
            else:
                vectors = [[1, 0] if prompt == "red" else [0, 1] for prompt in text]
            return Inputs(values=torch.tensor(vectors, dtype=torch.float32))

    class Scorer(torch.nn.Module):
        logit_scale = torch.tensor(2.0).log()

        @staticmethod
        def from_pretrained(name):
            return Scorer()

        def get_image_features(self, values):
            return SimpleNamespace(pooler_output=values)

        def get_text_features(self, values):
            return values

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoModel=Scorer, AutoProcessor=Processor)
    )
    result = pickscore(tmp_path, prompts, device="cpu", batch=1)
    assert result == {"count": 2, "mean": 2, "per_prompt": [2, 2]}
    (tmp_path / "1.png").rename(tmp_path / "3.png")
    with pytest.raises(ValueError, match="indices"):
        pickscore(tmp_path, prompts, device="cpu")
