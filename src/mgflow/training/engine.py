import fcntl
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..checkpoint import (
    CheckpointWriter,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    training_state,
)
from ..distributed import rank, setup, sum_gradients, world_size
from ..distributions import Branch, GaussianMixture, MomentState
from ..encoders import VisionEncoder
from ..models import PROFILES, build
from .imagenet import ImageNetBatch


def make_routes(config, device):
    groups = []
    for index, name in enumerate(config.encoders):
        routes = []
        for k in config.components:
            ref = GaussianMixture.load(Path(config.reference) / f"{name}_K{k}.npz", device)
            if ref.k != k:
                raise ValueError(f"{name}_K{k}: wrong reference component count")
            ridge = config.ridges[index] * config.ridge_multipliers.get(str(k), 1.0)
            weight = config.encoder_weights[index]
            routes.append(
                Branch(
                    ref,
                    MomentState.empty(ref),
                    config.objective,
                    ridge,
                    weight,
                    config.domain,
                    owner=index % world_size(),
                )
            )
        groups.append(routes)
    return groups


def optimizer(model, config):
    options = dict(
        lr=config.lr,
        betas=tuple(config.adam_betas),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        foreach=True,
    )
    if config.zero_optimizer and world_size() > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            model.parameters(), optimizer_class=torch.optim.AdamW, **options
        )
    return torch.optim.AdamW(model.parameters(), **options)


def learning_rate(config, step):
    if step < config.lr_warmup:
        return config.lr * step / max(1, config.lr_warmup)
    if config.lr_schedule == "constant":
        return config.lr
    progress = (step - config.lr_warmup) / max(1, config.steps - config.lr_warmup)
    return 0.5 * config.lr * (1 + math.cos(math.pi * progress))


def _local_rows(rows, n, micro, domain):
    if domain == "t2i":
        return rows[rank() * n : (rank() + 1) * n]
    pieces = []
    for start in range(0, n, micro):
        b = min(micro, n - start)
        offset = start * world_size() + rank() * b
        pieces.append(rows[offset : offset + b])
    return torch.cat(pieces)


def encoder_vjp(images, encoders, route_groups, features, config, step):
    n = len(images)
    image_gradient = torch.zeros_like(images)
    diagnostics = {}
    for index, (encoder, routes, full) in enumerate(zip(encoders, route_groups, features)):
        indices = _local_rows(
            torch.arange(len(full), device=full.device), n, config.micro_batch, config.domain
        )
        gradient = torch.zeros((n, full.shape[1]), device=full.device, dtype=full.dtype)
        t2i_kl = config.domain == "t2i" and config.objective == "kl"
        velocity_sum = torch.zeros((n, encoder.dimension), device=full.device) if t2i_kl else None
        for route in routes:
            if t2i_kl:
                velocity = route.velocity(full, config.beta(step), indices)
                component = velocity[:, : encoder.dimension].float()
                velocity_sum.add_(component, alpha=route.weight)
                loss = float(velocity.square().sum(1).mean())
            else:
                grad, loss = route.feature_gradient(full, config.beta(step), indices)
                gradient.add_(grad)
            diagnostics[f"{encoder.name}/K{route.reference.k}"] = loss
        gradient = (-2 / len(full)) * velocity_sum if t2i_kl else gradient[:, : encoder.dimension]
        for lo in range(0, n, config.encoder_micro_batch):
            image = images[lo : lo + config.encoder_micro_batch].detach().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.encoder_bf16):
                encoded = encoder(image).float()
            (vjp,) = torch.autograd.grad(
                encoded, image, grad_outputs=gradient[lo : lo + len(image)]
            )
            image_gradient[lo : lo + len(image)].add_(vjp)
    return image_gradient, diagnostics


@contextmanager
def run_directory(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    error = None
    if rank() == 0:
        try:
            lock = (output / "writer.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config_file = output / "config.json"
            if config_file.exists():
                if not config.resume:
                    raise FileExistsError(
                        "output already contains a run; use resume or a new directory"
                    )
                previous = json.loads(config_file.read_text())
                ignored = {"resume", "output", "initial_weights", "save_every"}
                if any(
                    previous.get(key) != value
                    for key, value in asdict(config).items()
                    if key not in ignored
                ):
                    raise ValueError(
                        "output directory belongs to a different training configuration"
                    )
            else:
                config_file.write_text(json.dumps(asdict(config), indent=2) + "\n")
        except (OSError, ValueError) as exception:
            error = str(exception)
    if world_size() > 1:
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error is not None:
        if lock:
            lock.close()
        raise RuntimeError(error)
    try:
        yield output
    finally:
        if lock:
            lock.close()


def train(config):
    device = setup()
    torch.manual_seed(config.seed + rank())
    np.random.seed(config.seed + rank())
    random.seed(config.seed + rank())
    with run_directory(config) as output, CheckpointWriter() as writer:
        _train(config, device, output, writer)


def _train(config, device, output, writer):
    if config.global_batch % world_size():
        raise ValueError("global batch must be divisible by world size")
    n = config.global_batch // world_size()
    if n % config.micro_batch:
        raise ValueError("local optimizer batch must be divisible by micro batch")
    saved, weights = training_state(config.resume, config) if config.resume else (None, None)
    if config.domain == "imagenet":
        if not config.lr:
            config.lr = PROFILES[config.model]["lr"]
        model = build(config.model).to(device)
        if not config.initial_weights and not saved:
            raise ValueError("initial_weights must point to model weights")
        loaded_step = load_checkpoint(weights if saved else config.initial_weights, model)
        if saved and loaded_step != saved["step"]:
            raise ValueError("resume weights and training state have different steps")
        if config.activation_checkpointing:
            from ..activation import checkpoint_blocks

            checkpoint_blocks(model.net)
        if config.compile:
            model.net = torch.compile(model.net)
        adapter = ImageNetBatch(model, config.model, device)
    else:
        from .t2i import TextToImageBatch

        adapter = TextToImageBatch(config, device)
        model = adapter.model
        if saved and load_checkpoint(weights, model) != saved["step"]:
            raise ValueError("resume weights and training state have different steps")
    model.train(config.domain == "t2i")
    encoders = [VisionEncoder(name, config.encoder_weights_dir, device) for name in config.encoders]
    routes = make_routes(config, device)
    opt = optimizer(model, config)
    first_step = 0
    if saved:
        restore_training_state(saved, opt, routes, adapter)
        first_step = saved["step"]
        if rank() == 0:
            log = output / "train.jsonl"
            if log.exists():
                previous = log.read_text().splitlines(keepends=True)
                kept = [line for line in previous if json.loads(line)["step"] <= first_step]
                if len(kept) != len(previous):
                    archive = (
                        output / f"train.before-resume-{first_step:07d}-{time.time_ns()}.jsonl"
                    )
                    log.rename(archive)
                    log.write_text("".join(kept))
            print(json.dumps({"resume_step": first_step}), flush=True)
    done = config.warmup_samples if saved else 0
    while done < config.warmup_samples:
        warmup_batch = (
            config.warmup_local_batch * world_size()
            if config.domain == "imagenet"
            else config.global_batch
        )
        count = min(warmup_batch, config.warmup_samples - done)
        if count % world_size():
            raise ValueError("warmup remainder must be divisible by world size")
        if config.domain == "imagenet":
            features = adapter.warmup_collect(count // world_size(), encoders, done, config.seed)
        else:
            images, features, _ = adapter.collect(
                count // world_size(), config.micro_batch, encoders, config.encoder_bf16
            )
            del images
        for group, rows in zip(routes, features):
            for route in group:
                route.warmup(rows)
        done += count
        del features
        if rank() == 0:
            print(json.dumps({"warmup_samples": done}), flush=True)
    if not saved:
        for group in routes:
            for route in group:
                route.finalize_warmup()
    model.train()
    for step in range(first_step, config.steps):
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        images, features, replay = adapter.collect(
            n, config.micro_batch, encoders, config.encoder_bf16
        )
        image_gradient, diagnostics = encoder_vjp(images, encoders, routes, features, config, step)
        del images, features
        adapter.replay(replay, image_gradient, config.micro_batch)
        del image_gradient
        sum_gradients(model.parameters())
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
        for group in opt.param_groups:
            group["lr"] = learning_rate(config, step + (config.domain == "t2i"))
        opt.step()
        for group in routes:
            for route in group:
                route.commit()
        torch.cuda.synchronize()
        if rank() == 0:
            record = {
                "step": step + 1,
                "seconds": time.perf_counter() - start,
                "lr": opt.param_groups[0]["lr"],
                "loss": diagnostics,
            }
            print(json.dumps(record), flush=True)
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if (step + 1) % config.save_every == 0 or step + 1 == config.steps:
            checkpoint = output / f"step_{step + 1:07d}.pth"
            save_checkpoint(checkpoint, model, step + 1, opt, routes, adapter, config, writer)
