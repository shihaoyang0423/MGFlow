import torch
import torch.nn.functional as F


class FluxGenerator(torch.nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.transformer = pipeline.transformer

    def forward(self, noise, context, image_ids, text_ids):
        from diffusers import Flux2KleinPipeline

        packed = Flux2KleinPipeline._pack_latents(noise)
        velocity = self.transformer(
            hidden_states=packed,
            encoder_hidden_states=context,
            timestep=torch.ones(len(noise), device=noise.device),
            guidance=None,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        return (
            (packed.float() - velocity.float())
            .transpose(1, 2)
            .reshape(len(noise), 128, noise.shape[-2], noise.shape[-1])
        )


def load_pipeline(path, device="cuda", training=False):
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    pipeline.to(device).set_progress_bar_config(disable=True)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.text_encoder.eval().requires_grad_(False)
    pipeline.transformer.float().requires_grad_(training)
    if training:
        pipeline.transformer.enable_gradient_checkpointing()
    return pipeline, FluxGenerator(pipeline)


def decode(pipeline, latent, resolution=256):
    from diffusers import Flux2KleinPipeline

    mean = pipeline.vae.bn.running_mean.reshape(1, -1, 1, 1)
    std = (
        pipeline.vae.bn.running_var.reshape(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
    ).sqrt()
    latent = Flux2KleinPipeline._unpatchify_latents(latent * std + mean)
    image = pipeline.vae.decode(latent.to(torch.bfloat16), return_dict=False)[0]
    image = (image.float() * 0.5 + 0.5).clamp(0, 1)
    if resolution != image.shape[-1]:
        image = F.interpolate(
            image, (resolution, resolution), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
    return image
