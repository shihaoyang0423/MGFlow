import copy
import json

import pytest
import torch

from mgflow import GaussianMixture
from mgflow.checkpoint import save_checkpoint, training_state
from mgflow.config import Config
from mgflow.training import engine


@pytest.fixture
def training_setup(tmp_path, monkeypatch):
    reference = tmp_path / "references"
    for k in (1, 4, 16):
        GaussianMixture(torch.ones(k), torch.zeros(k, 2), torch.eye(2).repeat(k, 1, 1)).save(
            reference / f"SigLIP_K{k}.npz"
        )
    initial = tmp_path / "initial.pth"
    save_checkpoint(initial, torch.nn.Linear(3, 2), 0)
    warmups = []

    class Encoder(torch.nn.Identity):
        name, dimension = "SigLIP", 2

        def __init__(self, *args):
            super().__init__()

    class Batch:
        def __init__(self, model, *args):
            self.model = model

        def state_dict(self):
            return {}

        def load_state_dict(self, state):
            assert state == {}

        def warmup_collect(self, n, *args):
            warmups.append(n)
            return [torch.randn(n, 2)]

        def collect(self, n, *args):
            x = torch.randn(n, 3)
            image = self.model(x).detach()
            return image, [image], x

        def replay(self, x, gradient, *args):
            self.model(x).backward(gradient)

    monkeypatch.setattr(engine, "setup", lambda: torch.device("cpu"))
    monkeypatch.setattr(engine, "build", lambda name: torch.nn.Linear(3, 2))
    monkeypatch.setattr(engine, "VisionEncoder", Encoder)
    monkeypatch.setattr(engine, "ImageNetBatch", Batch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    config = Config(
        encoders=["SigLIP"],
        reference=str(reference),
        initial_weights=str(initial),
        global_batch=8,
        micro_batch=4,
        warmup_samples=8,
        warmup_local_batch=4,
        steps=3,
        lr_warmup=1,
        save_every=1,
        encoder_bf16=False,
        output=str(tmp_path / "complete"),
    ).validate()
    return config, warmups


def test_training_resume_skips_warmup_and_continues_schedule(tmp_path, monkeypatch, training_setup):
    config, warmups = training_setup
    engine.train(config)
    interrupted = copy.deepcopy(config)
    interrupted.output = str(tmp_path / "resumed")
    original_save = engine.save_checkpoint

    def stop_at_first_checkpoint(*args):
        original_save(*args)
        raise InterruptedError("test pause")

    monkeypatch.setattr(engine, "save_checkpoint", stop_at_first_checkpoint)
    with pytest.raises(InterruptedError):
        engine.train(interrupted)
    count = len(warmups)
    interrupted.resume = interrupted.output

    def crash_before_checkpoint(*args):
        raise InterruptedError("test incomplete step")

    monkeypatch.setattr(engine, "save_checkpoint", crash_before_checkpoint)
    for _ in range(2):
        with pytest.raises(InterruptedError):
            engine.train(interrupted)
    monkeypatch.setattr(engine, "save_checkpoint", original_save)
    engine.train(interrupted)
    assert len(warmups) == count
    expected = torch.load(tmp_path / "complete/step_0000003.pth", weights_only=True)
    actual = torch.load(tmp_path / "resumed/step_0000003.pth", weights_only=True)
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(tensor, actual["model"][name], rtol=0, atol=0)
    logs = [
        json.loads(line) for line in (tmp_path / "resumed/train.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in logs] == [1, 2, 3]
    assert [row["lr"] for row in logs] == [engine.learning_rate(config, i) for i in range(3)]
    assert len(list((tmp_path / "resumed").glob("train.before-resume-*.jsonl"))) == 2
    assert sorted(path.name for path in (tmp_path / "resumed").glob("*.pth")) == [
        "step_0000001.pth",
        "step_0000002.pth",
        "step_0000003.pth",
    ]
    assert not list((tmp_path / "resumed").glob(".checkpoint-*"))


@pytest.mark.parametrize("step", [1, 2])
def test_training_rejects_rollback_into_original_directory(tmp_path, training_setup, step):
    config, _ = training_setup
    engine.train(config)
    output = tmp_path / "complete"
    before = {path.name: path.read_bytes() for path in output.glob("*.pth")}
    log = (output / "train.jsonl").read_text()
    config.resume = str(output / f"step_{step:07d}.pth")
    with pytest.raises(ValueError, match="new output directory"):
        engine.train(config)
    assert {path.name: path.read_bytes() for path in output.glob("*.pth")} == before
    assert (output / "train.jsonl").read_text() == log
    assert not list(output.glob("train.before-resume-*.jsonl"))
    state, path = training_state(output, config)
    assert state["step"] == 3 and path.name == "step_0000003.pth"


def test_training_can_resume_older_checkpoint_in_new_directory(
    tmp_path, monkeypatch, training_setup
):
    config, _ = training_setup
    engine.train(config)
    source = tmp_path / "complete"
    original = {path.name: path.read_bytes() for path in source.glob("*.pth")}
    config.resume = str(source / "step_0000001.pth")
    config.output = str(tmp_path / "fork")
    save = engine.save_checkpoint

    def stop_after_save(*args):
        save(*args)
        raise InterruptedError("test pause")

    monkeypatch.setattr(engine, "save_checkpoint", stop_after_save)
    with pytest.raises(InterruptedError):
        engine.train(config)
    state, path = training_state(config.output, config)
    assert state["step"] == 2 and path.name == "step_0000002.pth"
    assert {path.name: path.read_bytes() for path in source.glob("*.pth")} == original
    assert list((tmp_path / "fork").glob("*.pth")) == [path]
