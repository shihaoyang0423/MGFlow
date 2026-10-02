import random
import threading

import numpy as np
import pytest
import torch

from mgflow import Branch, GaussianMixture, MomentState
from mgflow.checkpoint import (
    CheckpointWriter,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    training_state,
)
from mgflow.config import Config
from mgflow.training.imagenet import ImageNetBatch
from mgflow.training.t2i import batch_randomness


def test_exact_optimizer_resume(tmp_path):
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    config = Config(steps=5, encoders=["SigLIP"]).validate()
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference = GaussianMixture(torch.ones(1), torch.zeros(1, 2), torch.eye(2)[None])
    route = Branch(reference, MomentState.from_reference(reference), ridge=0.1)
    adapter = ImageNetBatch(model, "JiT-B", "cpu")

    def step(model, opt, route):
        opt.zero_grad()
        x = torch.randn(16, 3) + np.random.rand() + random.random()
        features = model(x)
        gradient, _ = route.feature_gradient(features.detach(), 0.9)
        features.backward(gradient)
        opt.step()
        route.commit()

    for _ in range(3):
        step(model, opt, route)
    weights = tmp_path / "step_0000003.pth"
    save_checkpoint(weights, model, 3, opt, [[route]], adapter, config)
    for _ in range(2):
        step(model, opt, route)
    expected = [value.clone() for value in model.parameters()]
    expected_mean = route.state.mean.clone()
    restored = torch.nn.Linear(3, 2)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.5)
    restored_route = Branch(reference, MomentState.empty(reference), ridge=0.1)
    state, path = training_state(tmp_path, config)
    assert load_checkpoint(path, restored) == 3
    restore_training_state(state, restored_opt, [[restored_route]], adapter)
    for _ in range(2):
        step(restored, restored_opt, restored_route)
    for a, b in zip(expected, restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(expected_mean, restored_route.state.mean, rtol=0, atol=0)
    assert {"model", "step", "optimizer", "moments", "random"} <= set(
        torch.load(weights, weights_only=True)
    )
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_async_checkpoint_owns_snapshot(tmp_path, monkeypatch):
    import mgflow.checkpoint as checkpoint

    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(2, 3)).sum().backward()
    opt.step()
    ref = GaussianMixture(torch.ones(1), torch.zeros(1, 2), torch.eye(2)[None])
    route = Branch(ref, MomentState.from_reference(ref), ridge=0.1)
    adapter = ImageNetBatch(model, "JiT-B", "cpu")
    config = Config().validate()
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}
    expected_moment = opt.state[next(model.parameters())]["exp_avg"].clone()
    release = threading.Event()
    write = checkpoint._save

    def delayed(path, payload):
        assert release.wait(timeout=10)
        write(path, payload)

    monkeypatch.setattr(checkpoint, "_save", delayed)
    path = tmp_path / "step_0000001.pth"
    with CheckpointWriter() as writer:
        save_checkpoint(path, model, 1, opt, [[route]], adapter, config, writer)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
            route.state.mean.add_(1)
            opt.state[next(model.parameters())]["exp_avg"].add_(1)
        release.set()
    saved = torch.load(path, weights_only=True)
    for name, value in expected_model.items():
        torch.testing.assert_close(saved["model"][name], value, rtol=0, atol=0)
    torch.testing.assert_close(
        saved["moments"][0][0]["mean"], torch.zeros(1, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(saved["optimizer"][0]["state"][0]["exp_avg"], expected_moment)


def test_async_checkpoint_reports_write_failure(tmp_path, monkeypatch):
    import mgflow.checkpoint as checkpoint

    def fail(path, payload):
        raise OSError("disk write failed")

    monkeypatch.setattr(checkpoint, "_save", fail)
    with pytest.raises(RuntimeError, match="disk write failed"):
        with CheckpointWriter() as writer:
            writer.write(tmp_path / "step_0000001.pth", {})


@pytest.mark.parametrize("background", [False, True])
def test_checkpoint_does_not_overwrite_existing_file(tmp_path, background):
    path = tmp_path / "step_0000001.pth"
    model = torch.nn.Linear(3, 2)
    save_checkpoint(path, model, 1)
    previous = path.read_bytes()
    if background:
        with pytest.raises(RuntimeError, match="checkpoint write failed"):
            with CheckpointWriter() as writer:
                writer.write(path, {"model": {}, "step": 1})
    else:
        with pytest.raises(FileExistsError):
            save_checkpoint(path, model, 2)
    assert path.read_bytes() == previous
    assert not path.with_suffix(".partial").exists()


def test_t2i_seed_and_partition():
    outputs = []
    for seed in (1, 123):
        rng, noise = batch_randomness(seed, 5, False, 16, 0, 16, "cpu")
        outputs.append((rng.integers(0, 10000, 16), noise))
    assert not np.array_equal(outputs[0][0], outputs[1][0])
    assert not torch.equal(outputs[0][1], outputs[1][1])
    expected = torch.cat(
        [
            torch.randn(
                2, 128, 32, 32, generator=torch.Generator().manual_seed(73_000_000 + 500 + r)
            )
            for r in range(8)
        ]
    )
    torch.testing.assert_close(outputs[0][1], expected, rtol=0, atol=0)
    parts = [batch_randomness(1, 5, False, 16, r * 2, (r + 1) * 2, "cpu")[1] for r in range(8)]
    torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)


def test_encoder_subset():
    assert Config(encoders=["MAE"]).validate().encoders == ["MAE"]
    config = Config.load("t2i-joint", ['encoders=["Inception"]'])
    assert len(config.encoder_weights) == len(config.text_betas) == len(config.ridges) == 1


@pytest.mark.parametrize("components", [[1], [1, 4], [4, 1, 16], [1, 4, 16, 16]])
def test_imagenet_kl_requires_all_components(components):
    with pytest.raises(ValueError):
        Config(components=components).validate()


@pytest.mark.parametrize("recipe", ["imagenet-w2", "t2i-joint", "t2i-image"])
@pytest.mark.parametrize("components", ["[1]", "[4]", "[1,4,16]", "[4,1]"])
def test_pair_recipes_require_k1_k4(recipe, components):
    with pytest.raises(ValueError):
        Config.load(recipe, [f"components={components}"])


def test_k1_invalid_covariance():
    from mgflow.distributions.fitting import check

    invalid = GaussianMixture(torch.ones(1), torch.zeros(1, 3), -torch.eye(3)[None])
    with pytest.raises(ValueError, match="positive semidefinite"):
        check(None, invalid)
