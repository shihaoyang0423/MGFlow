import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from mgflow import sampling
from mgflow.data import ImageNetImages


@pytest.fixture
def imagenet_sampler(tmp_path, monkeypatch):
    output = tmp_path / "samples"

    class Model:
        calls = 0
        fail_after = None

        def cuda(self):
            return self

        def eval(self):
            return self

        def __call__(self, noise, *args):
            assert not (output / "images.npy").exists()
            if self.calls == self.fail_after:
                raise InterruptedError("test interrupted sampling")
            self.calls += 1
            return torch.zeros_like(noise)

    model = Model()
    generator, randn, arange = torch.Generator, torch.randn, torch.arange
    monkeypatch.setattr(sampling, "build", lambda name: model)
    monkeypatch.setattr(sampling, "load_checkpoint", lambda *args: None)
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: generator())
    monkeypatch.setattr(
        torch, "randn", lambda *args, **kwargs: randn(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(
        torch, "arange", lambda *args, **kwargs: arange(*args, **{**kwargs, "device": "cpu"})
    )
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    return model, output


def test_imagenet_publishes_only_complete_array(imagenet_sampler):
    model, output = imagenet_sampler
    sampling.imagenet("JiT-B", "checkpoint.pth", output, count=5, batch=2)
    assert model.calls == 3
    pixels = np.load(output / "images.npy")
    assert pixels.shape == (5, 256, 256, 3) and np.all(pixels == 128)
    assert not (output / "images.partial").exists()
    assert len(ImageNetImages(output)) == 5
    assert json.loads((output / "sampling.json").read_text())["labels"] == list(range(5))
    with pytest.raises(FileExistsError):
        sampling.imagenet("JiT-B", "checkpoint.pth", output, count=5, batch=2)
    assert model.calls == 3


def test_interrupted_array_is_not_accepted_for_evaluation(imagenet_sampler):
    model, output = imagenet_sampler
    model.fail_after = 1
    with pytest.raises(InterruptedError):
        sampling.imagenet("JiT-B", "checkpoint.pth", output, count=5, batch=2)
    assert not (output / "images.npy").exists()
    assert not (output / "sampling.json").exists()
    partial = output / "images.partial"
    pixels = np.load(partial)
    assert np.all(pixels[:2] == 128) and np.all(pixels[2:] == 0)
    for source in (output, partial, output / "images.npy"):
        with pytest.raises(FileNotFoundError):
            ImageNetImages(source)
    before = partial.read_bytes()
    with pytest.raises(FileExistsError):
        sampling.imagenet("JiT-B", "checkpoint.pth", output, count=5, batch=2)
    assert partial.read_bytes() == before


def test_failed_sampling_metadata_does_not_publish_array(imagenet_sampler, monkeypatch):
    _, output = imagenet_sampler

    def fail(*args, **kwargs):
        raise OSError("test metadata write failed")

    monkeypatch.setattr(type(output), "write_text", fail)
    with pytest.raises(OSError, match="metadata write failed"):
        sampling.imagenet("JiT-B", "checkpoint.pth", output, count=5, batch=2)
    assert not (output / "images.npy").exists()
    assert (output / "images.partial").is_file()
