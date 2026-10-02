import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .distributed import gather_state, rank, world_size


def _model_state(model):
    state = {}
    for name, tensor in model.state_dict().items():
        name = name.replace("_orig_mod.", "")
        name = re.sub(r"((?:blocks|shared_blocks|u_heads|v_heads)\.\d+)\.block\.", r"\1.", name)
        state[name] = tensor.detach().to("cpu", copy=True).contiguous()
    return state


def _save(path, payload):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class CheckpointWriter:
    def __init__(self):
        self.group = dist.new_group(backend="gloo") if world_size() > 1 else None
        self.executor = ThreadPoolExecutor(max_workers=1) if rank() == 0 else None
        self.pending = None

    def wait(self):
        error = None
        if self.pending is not None:
            try:
                self.pending.result()
            except Exception as exception:
                error = str(exception)
            self.pending = None
        status = [error]
        if world_size() > 1:
            dist.broadcast_object_list(status, src=0, group=self.group)
        if status[0] is not None:
            raise RuntimeError(f"checkpoint write failed: {status[0]}")

    def write(self, path, payload):
        if self.executor is not None:
            self.pending = self.executor.submit(_save, path, payload)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.wait()
        finally:
            if self.executor is not None:
                self.executor.shutdown()
            if self.group is not None:
                dist.destroy_process_group(self.group)


def save_checkpoint(
    path, model, step, opt=None, routes=None, adapter=None, config=None, writer=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if opt is None:
        _save(path, {"model": _model_state(model), "step": int(step)})
        return
    if writer is not None:
        writer.wait()
    local = {
        "optimizer": getattr(opt, "optim", opt).state_dict(),
        "adapter": adapter.state_dict(),
        "random": random_state(),
    }
    local = _cpu(local)
    group = writer.group if writer is not None else None
    if world_size() > 1:
        if group is None:
            group = dist.new_group(backend="gloo")
        try:
            workers = gather_state(local, group)
        finally:
            if writer is None:
                dist.destroy_process_group(group)
    else:
        workers = [local]
    payload = None
    if rank() == 0:
        payload = {
            "model": _model_state(model),
            "step": int(step),
            "optimizer": [worker["optimizer"] for worker in workers],
            "adapter": [worker["adapter"] for worker in workers],
            "random": [worker["random"] for worker in workers],
            "moments": [
                [
                    dict(
                        mean=route.state.mean.to("cpu", copy=True),
                        second=route.state.second.to("cpu", copy=True),
                        count=route.state.count,
                    )
                    for route in group
                ]
                for group in routes
            ],
            "config": asdict(config),
            "world_size": world_size(),
        }
    if writer is not None:
        writer.write(path, payload)
    elif rank() == 0:
        _save(path, payload)
    if world_size() > 1:
        dist.barrier()


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model" in checkpoint:
        state, step = checkpoint["model"], int(checkpoint.get("step", 0))
    else:
        state, step = checkpoint, 0
    model.load_state_dict(state, strict=True)
    return step


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def random_state():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state().cpu() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def training_state(path, config):
    path = Path(path)
    if path.is_dir():
        checkpoints = sorted(path.glob("step_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints in {path}")
        path = checkpoints[-1]
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {
        "model",
        "optimizer",
        "moments",
        "adapter",
        "random",
        "config",
        "world_size",
        "step",
    }
    if not required <= payload.keys():
        raise ValueError("resume requires a checkpoint with complete training state")
    if payload["world_size"] != world_size():
        raise ValueError("exact resume requires the original process count")
    if any(len(payload[key]) != world_size() for key in ("optimizer", "adapter", "random")):
        raise ValueError("checkpoint worker count differs")
    ignored = {"resume", "output", "initial_weights", "save_every"}
    changed = [
        key
        for key, value in asdict(config).items()
        if key not in ignored and value != payload["config"].get(key)
    ]
    if changed:
        raise ValueError("resume configuration differs: " + ", ".join(changed))
    if not 0 <= payload["step"] <= config.steps:
        raise ValueError("resume step is outside the training schedule")
    if any(
        int(checkpoint.stem.removeprefix("step_")) > payload["step"]
        for checkpoint in Path(config.output).glob("step_*.pth")
    ):
        raise ValueError("output contains newer checkpoints; use a new output directory")
    return payload, path


def restore_training_state(payload, opt, routes, adapter):
    if len(payload["moments"]) != len(routes):
        raise ValueError("resume encoder count differs")
    for saved, group in zip(payload["moments"], routes):
        if len(saved) != len(group):
            raise ValueError("resume component branches differ")
        for values, route in zip(saved, group):
            if (
                values["mean"].shape != route.state.mean.shape
                or values["second"].shape != route.state.second.shape
            ):
                raise ValueError("resume moment shapes differ")
            route.state.mean.copy_(values["mean"])
            route.state.second.copy_(values["second"])
            route.state.count = values["count"]
    local_optimizer = getattr(opt, "optim", opt)
    local_optimizer.load_state_dict(payload["optimizer"][rank()])
    if local_optimizer is not opt:
        for shared, local in zip(opt.param_groups, local_optimizer.param_groups):
            shared.update({key: value for key, value in local.items() if key != "params"})
    adapter.load_state_dict(payload["adapter"][rank()])
    restore_random_state(payload["random"][rank()])
