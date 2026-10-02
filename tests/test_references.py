import io
import json
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image

from mgflow import GaussianMixture
from mgflow.config import Config
from mgflow.data import T2IImages, prompts_from_file
from mgflow.distributions.fitting import FitOptions, check, feature_bank, fit


def test_single_gaussian(tmp_path):
    values = np.random.default_rng(0).normal(size=(83, 5)).astype(np.float32)
    reference = fit(values, 1, device="cpu", options=FitOptions(chunk=17))
    x = torch.from_numpy(values).double()
    torch.testing.assert_close(reference.mean[0], x.mean(0), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        reference.covariance[0], (x - x.mean(0)).T @ (x - x.mean(0)) / len(x)
    )
    path = tmp_path / "K1.npz"
    reference.save(path)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == {"pi", "mu", "sigma"}
        assert all(payload[key].dtype == np.float64 for key in payload.files)
    loaded = GaussianMixture.load(path, device="cpu")
    assert check(None, loaded) == {"components": 1, "dimension": 5}
    torch.testing.assert_close(loaded.covariance, reference.covariance, atol=0, rtol=0)


@pytest.mark.parametrize("k", [4, 16])
def test_full_data_em(k):
    rng = np.random.default_rng(0)
    centers = np.arange(k)[:, None] * np.array([[20, 30, 40]])
    values = np.concatenate([rng.normal(center, 0.5, size=(128, 3)) for center in centers]).astype(
        np.float32
    )
    options = FitOptions(seed=0, chunk=101, kmeans_min_iterations=2)
    reference = fit(values, k, device="cpu", options=options)
    result = check(values, reference, options)
    assert result["converged"]
    assert result["weight_ratio"] < 1.01
    distances = torch.cdist(reference.mean, torch.from_numpy(centers).double())
    assert distances.min(0).values.max() < 0.2
    assert (torch.linalg.eigvalsh(reference.covariance) > 0).all()


def test_feature_bank_validation(tmp_path):
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((16, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        feature_bank(path)


def test_t2i_image_prompt_alignment(tmp_path):
    rows = [
        dict(id="000001", prompt="blue", shard="data/images.tar", image="000001.png"),
        dict(id="000000", prompt="red", shard="data/images.tar", image="000000.png"),
    ]
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "data").mkdir()
    with tarfile.open(tmp_path / "data/images.tar", "w") as archive:
        for name, color in [("000000.png", "red"), ("000001.png", "blue")]:
            encoded = io.BytesIO()
            Image.new("RGB", (512, 512), color).save(encoded, format="PNG")
            content = encoded.getvalue()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    assert prompts_from_file(metadata) == ["red", "blue"]
    dataset = T2IImages(tmp_path)
    first, i = dataset[0]
    second, j = dataset[1]
    assert i == 0 and j == 1
    assert first.shape == second.shape == (3, 256, 256)
    assert first[0].mean() == 1 and second[2].mean() == 1
    dataset.archive.close()


def test_text_feature_alignment(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from mgflow.features import encode_text

    prompts = tmp_path / "metadata.jsonl"
    prompts.write_text(
        "\n".join(
            json.dumps(row) for row in [dict(id="1", prompt="bird"), dict(id="0", prompt="dog")]
        )
    )
    calls = []

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Identity()

        def encode_text(self, tokens):
            assert not hasattr(self, "visual")
            return tokens.float().repeat(1, 576)

    def create_model(name):
        calls.append(name)
        return Encoder()

    def get_tokenizer(name):
        calls.append(name)
        return lambda values: torch.tensor([[1 if value == "dog" else 2, 3] for value in values])

    monkeypatch.setitem(
        sys.modules,
        "open_clip",
        SimpleNamespace(create_model=create_model, get_tokenizer=get_tokenizer),
    )
    output = tmp_path / "text.npy"
    encode_text(prompts, output, batch=1, device="cpu")
    vectors = np.load(output, allow_pickle=False)
    assert calls == ["hf-hub:timm/ViT-SO400M-16-SigLIP2-256"] * 2
    assert vectors.shape == (2, 1152) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    assert vectors[0, 0] < vectors[1, 0]


def test_training_recipes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("imagenet-kl", "imagenet-w2", "t2i-joint", "t2i-image"):
        config = Config.load(root / "configs" / f"{name}.toml")
        if config.domain == "t2i":
            assert config.steps == 1000 and config.components == [1, 4]
        else:
            assert config.initial_weights.startswith("assets/Checkpoints/ImageNet/Base/")
        assert config.encoder_weights_dir == "assets/Encoders/encoders"
