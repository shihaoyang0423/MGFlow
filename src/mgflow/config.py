import math
import sysconfig
from dataclasses import dataclass, field, fields
from pathlib import Path

import tomllib


@dataclass
class Config:
    domain: str = "imagenet"
    objective: str = "kl"
    model: str = "JiT-B"
    components: list[int] = field(default_factory=lambda: [1, 4, 16])
    encoders: list[str] = field(default_factory=lambda: ["SigLIP", "MAE", "Inception"])
    encoder_weights: list[float] = field(default_factory=list)
    ridges: list[float] = field(default_factory=list)
    ridge_multipliers: dict = field(default_factory=lambda: {"1": 1.0, "4": 3.0, "16": 9.0})
    reference: str = "assets/Reference/ImageNet"
    initial_weights: str = ""
    resume: str = ""
    encoder_weights_dir: str = "assets/Encoders/encoders"
    output: str = "runs/mgflow"
    global_batch: int = 1024
    micro_batch: int = 16
    encoder_micro_batch: int = 16
    warmup_samples: int = 50000
    warmup_local_batch: int = 256
    steps: int = 125000
    seed: int = 1
    lr: float = 0.0
    lr_warmup: int = 6250
    lr_schedule: str = "cosine"
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    adam_epsilon: float = 1e-16
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    ema_beta: float = 0.995
    ema_beta_final: float = 0.999
    ema_start: int = 10000
    ema_end: int = 40000
    save_every: int = 12500
    compile: bool = False
    activation_checkpointing: bool = False
    encoder_bf16: bool = True
    zero_optimizer: bool = True
    joint: bool = False
    prompts: str = ""
    text_features: str = ""
    text_betas: list[float] = field(default_factory=list)
    text_dimension: int = 1152

    def validate(self):
        if self.domain not in ("imagenet", "t2i") or self.objective not in ("kl", "w2"):
            raise ValueError("domain must be imagenet/t2i; objective must be kl/w2")
        if self.domain == "t2i" and self.objective != "kl":
            raise ValueError("text-to-image training uses KL")
        required = [1, 4, 16] if self.domain == "imagenet" and self.objective == "kl" else [1, 4]
        if self.components != required or any(type(k) is not int for k in self.components):
            raise ValueError(f"{self.domain} {self.objective} requires components={required}")
        names = ["SigLIP", "MAE", "Inception"]
        if (
            not self.encoders
            or len(set(self.encoders)) != len(self.encoders)
            or any(name not in names for name in self.encoders)
        ):
            raise ValueError("encoders must be a nonempty subset of SigLIP, MAE and Inception")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.joint and self.domain != "t2i":
            raise ValueError("joint image-text features are used for text-to-image training")
        if self.domain == "t2i" and not self.prompts:
            raise ValueError("text-to-image training requires reference prompts")
        if self.joint and not self.text_features:
            raise ValueError("joint training requires cached text features")
        if self.global_batch <= 0 or self.micro_batch <= 0:
            raise ValueError("batch sizes must be positive")
        if self.domain == "t2i" and self.global_batch % 8:
            raise ValueError("T2I global_batch must be divisible by eight")
        if self.warmup_local_batch <= 0:
            raise ValueError("warmup batch size must be positive")
        if self.steps <= 0 or self.save_every <= 0 or self.encoder_micro_batch <= 0:
            raise ValueError("steps, save cadence and encoder batch size must be positive")
        if self.warmup_samples <= 0:
            raise ValueError("warmup_samples must be positive")
        if self.domain == "imagenet":
            from .models import PROFILES

            if self.model not in PROFILES:
                raise ValueError(f"unknown ImageNet model: {self.model}")
            if self.lr == 0:
                self.lr = PROFILES[self.model]["lr"]
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError("unknown learning-rate schedule")
        if not self.encoder_weights:
            default = (
                [0.3267037000166731, 0.2756730318072754, 0.11000585458566385]
                if self.objective == "kl"
                else [1 / 0.6246865359380966, 1 / 0.042119454490506913, 1 / 1.6795566170052427]
            )
            self.encoder_weights = [
                default[["SigLIP", "MAE", "Inception"].index(name)] for name in self.encoders
            ]
        if not self.ridges:
            self.ridges = (
                [{"SigLIP": 0.03, "MAE": 0.001, "Inception": 0.03}[name] for name in self.encoders]
                if self.objective == "kl"
                else [0.0] * len(self.encoders)
            )
        if len(self.encoders) != len(self.encoder_weights) or len(self.ridges) != len(
            self.encoders
        ):
            raise ValueError("encoder, weight and ridge lists must have equal lengths")
        if self.joint and len(self.text_betas) != len(self.encoders):
            raise ValueError("joint training needs a text beta for each encoder")
        if any(not math.isfinite(value) or value < 0 for value in self.ridges):
            raise ValueError("ridges must be finite and nonnegative")
        if any(not math.isfinite(value) or value <= 0 for value in self.encoder_weights):
            raise ValueError("encoder weights must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in self.text_betas):
            raise ValueError("text scales must be finite and nonnegative")
        return self

    @classmethod
    def load(cls, path, overrides=()):
        path = Path(path)
        if not path.exists() and len(path.parts) <= 2:
            name = path.name if path.suffix else path.name + ".toml"
            source = Path(__file__).resolve().parents[2] / "configs" / name
            installed = Path(sysconfig.get_path("data")) / "share/mgflow/configs" / name
            path = next((item for item in (source, installed) if item.is_file()), path)
        payload = tomllib.loads(path.read_text())
        previous_encoders = payload.get("encoders", ["SigLIP", "MAE", "Inception"])
        changed_keys = set()
        known = {item.name for item in fields(cls)}
        for override in overrides:
            key, value = override.split("=", 1)
            if key not in known:
                raise ValueError(f"unknown configuration key: {key}")
            payload[key] = tomllib.loads(f"value = {value}")["value"]
            changed_keys.add(key)
        if "encoders" in changed_keys:
            for key in ("encoder_weights", "ridges", "text_betas"):
                values = payload.get(key, [])
                if key not in changed_keys and len(values) == len(previous_encoders):
                    payload[key] = [
                        values[previous_encoders.index(name)]
                        for name in payload["encoders"]
                        if name in previous_encoders
                    ]
        return cls(**payload).validate()

    def beta(self, step):
        fraction = min(
            1.0, max(0.0, (step - self.ema_start) / max(1, self.ema_end - self.ema_start))
        )
        return self.ema_beta + (self.ema_beta_final - self.ema_beta) * fraction
