from pathlib import Path

import torch
import torch.nn.functional as F

from .models.inception import INCEPTION_URL, InceptionV3

SPECS = {
    "SigLIP": ("vit_so400m_patch16_siglip_256.v2_webli", 1152, 224),
    "MAE": ("vit_large_patch16_224.mae", 1024, 224),
    "Inception": ("inception", 2048, 256),
}


class VisionEncoder(torch.nn.Module):
    def __init__(self, name, weights=None, device="cuda", spec=None):
        super().__init__()
        self.name = name
        model_name, self.dimension, self.size = spec or SPECS[name]
        if name == "Inception":
            self.model = InceptionV3(normalize=False)
            path = Path(weights) / "weights-inception-2015-12-05-6726825d.pth" if weights else None
            if path and not path.exists():
                path = Path(weights).parent / "torch/hub/checkpoints" / path.name
            state = (
                torch.load(path, map_location="cpu", weights_only=True)
                if path and path.exists()
                else torch.hub.load_state_dict_from_url(INCEPTION_URL, progress=True)
            )
            self.model.load_state_dict(state)
        else:
            import timm
            from timm.data import resolve_data_config

            options = {"pretrained": True, "num_classes": 0}
            if weights:
                folder = Path(weights) / model_name
                path = next(
                    (
                        folder / file
                        for file in ("model.safetensors", "pytorch_model.bin")
                        if (folder / file).exists()
                    ),
                    None,
                )
                if path is not None:
                    options["pretrained_cfg_overlay"] = {"file": str(path)}
            if not model_name.startswith("convnext"):
                options.update(dynamic_img_size=True, dynamic_img_pad=True)
            self.model = timm.create_model(model_name, **options)
            configuration = resolve_data_config(self.model.pretrained_cfg)
            self.register_buffer("mean", torch.tensor(configuration["mean"]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(configuration["std"]).view(1, 3, 1, 1))
        self.to(device).eval().requires_grad_(False)

    def forward(self, image):
        if self.name == "Inception":
            return self.model(image)[0]
        image = F.interpolate(
            image, size=(self.size, self.size), mode="bicubic", align_corners=False, antialias=True
        )
        tokens = self.model.forward_features((image - self.mean) / self.std)
        if tokens.ndim == 4:
            return tokens.mean((-2, -1))
        if getattr(self.model, "num_prefix_tokens", 0) > 0:
            return tokens[:, 0]
        if getattr(self.model, "attn_pool", None) is not None:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            return pool(tokens)
        return tokens.mean(1)


EVALUATION_SPECS = {
    **SPECS,
    "DINOv2": ("vit_large_patch14_dinov2.lvd142m", 1024, 256),
    "CLIP": ("vit_large_patch14_clip_224.openai", 1024, 256),
    "ConvNeXt": ("convnextv2_base.fcmae_ft_in22k_in1k", 1024, 224),
}


class EvaluationEncoder(VisionEncoder):
    def __init__(self, name, weights=None, device="cuda"):
        super().__init__(name, weights, device, EVALUATION_SPECS[name])

    def forward(self, image):
        if self.name == "Inception":
            return self.model(image)
        return super().forward(image), None
