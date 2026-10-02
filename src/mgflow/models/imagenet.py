import torch

from .jit import JiT_models
from .mit import MiT_models


class ImageNetGenerator(torch.nn.Module):
    def __init__(self, family, size):
        super().__init__()
        self.family = family
        if family == "JiT":
            self.net = JiT_models[f"JiT-{size}"](
                input_size=256,
                in_channels=3,
                num_classes=1000,
                attn_drop=0.0,
                proj_drop=0.0,
                rope_2d=True,
                learned_pe=True,
            )
        elif family == "pMF":
            architecture = "MiT_B2" if size == "B" else f"MiT_{size}"
            self.net = MiT_models[architecture](
                input_size=256,
                in_channels=3,
                patch_size=16,
                num_classes=1000,
                aux_head_depth=8,
                num_class_tokens=8,
                num_time_tokens=4,
                num_cfg_tokens=4,
                num_interval_tokens=2,
                token_init_constant=1.0,
                embedding_init_constant=1.0,
                weight_init_constant=0.32,
                bottleneck_dim=256 if size == "H" else 128,
                output_type="x",
                rope_2d=True,
                learned_pe=True,
                disable_v_head=True,
                t_eps=0.05,
            )
        else:
            raise ValueError(f"unknown generator family: {family}")

    def forward(self, noise, labels, sampling_args=None):
        options = sampling_args or {}
        t = torch.ones(len(noise), device=noise.device)
        step = t.view(-1, 1, 1, 1)
        if self.family == "JiT":
            prediction = self.net(noise, torch.zeros_like(t), labels)
            velocity = (noise - prediction) / step
        else:
            cfg = torch.full_like(t, options.get("cfg", 1.0))
            low = torch.full_like(t, options.get("t_min", 0.4))
            high = torch.full_like(t, options.get("t_max", 0.65))
            velocity = self.net(noise, t, t, cfg, low, high, labels)[0]
        return noise - step * velocity
