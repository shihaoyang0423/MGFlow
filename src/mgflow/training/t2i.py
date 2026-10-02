import numpy as np
import torch

from ..checkpoint import load_checkpoint
from ..data import prompts_from_file
from ..distributed import gather, rank, world_size
from ..models.flux import decode, load_pipeline


def batch_randomness(seed, step, warmup, total, lo, hi, device):
    offset = 2_000_000 if warmup else 0
    shift = (seed - 1) * 1_000_000_007
    prompt_seed = (1_000_003 + offset + step + shift) % (2**63 - 1)
    indices = np.random.default_rng(prompt_seed)
    if total <= 0 or total % 8 or not 0 <= lo < hi <= total:
        raise ValueError("T2I batches require eight nonempty logical noise streams")
    logical_n = total // 8
    parts = []
    for logical_rank in range(lo // logical_n, (hi - 1) // logical_n + 1):
        noise_seed = (73_000_000 + offset + step * 100 + logical_rank + shift) % (2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(
            logical_n, 128, 32, 32, device=device, generator=generator, dtype=torch.float32
        )
        parts.append(
            noise[
                max(0, lo - logical_rank * logical_n) : min(
                    logical_n, hi - logical_rank * logical_n
                )
            ]
        )
    return indices, torch.cat(parts)


class TextToImageBatch:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.pipeline, self.model = load_pipeline(config.model, device, training=True)
        if config.initial_weights:
            load_checkpoint(config.initial_weights, self.model)
        self.prompts = prompts_from_file(config.prompts)
        self.text = (
            np.load(config.text_features, mmap_mode="r", allow_pickle=False)
            if config.joint
            else None
        )
        if self.text is not None and (
            self.text.shape != (len(self.prompts), config.text_dimension)
            or self.text.dtype != np.float32
        ):
            raise ValueError("text features must be float32 [number of prompts, text dimension]")
        self.current = 0
        self.warmup_done = 0

    def state_dict(self):
        return {"current": self.current, "warmup_done": self.warmup_done}

    def load_state_dict(self, state):
        self.current = int(state["current"])
        self.warmup_done = int(state["warmup_done"])

    def _batch(self, n):
        warmup = self.warmup_done < self.config.warmup_samples
        step = self.warmup_done if warmup else self.current + 1
        total = n * world_size()
        lo, hi = rank() * n, (rank() + 1) * n
        rng, noise = batch_randomness(self.config.seed, step, warmup, total, lo, hi, self.device)
        indices = rng.integers(0, len(self.prompts), size=total)[lo:hi]
        contexts = []
        with torch.no_grad():
            for first in range(0, n, 8):
                context, _ = self.pipeline.encode_prompt(
                    [self.prompts[index] for index in indices[first : first + 8]],
                    device=self.device,
                    max_sequence_length=512,
                )
                contexts.append(context.cpu())
        text = (
            torch.tensor(np.array(self.text[indices]), device=self.device)
            if self.text is not None
            else None
        )
        if warmup:
            self.warmup_done += total
        else:
            self.current += 1
        return noise, torch.cat(contexts), text

    def _latent(self, noise, context):
        image_ids = self.pipeline._prepare_latent_ids(noise).to(self.device)
        text_ids = self.pipeline._prepare_text_ids(context).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(noise, context, image_ids, text_ids)

    def _image(self, noise, context):
        latent = self._latent(noise, context)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return torch.cat(
                [
                    decode(self.pipeline, latent[first : first + 4])
                    for first in range(0, len(latent), 4)
                ]
            )

    def collect(self, n, micro, encoders, encoder_bf16):
        noise, context, text = self._batch(n)
        with torch.no_grad():
            images = torch.cat(
                [
                    self._image(
                        noise[first : first + micro], context[first : first + micro].to(self.device)
                    )
                    for first in range(0, n, micro)
                ]
            )
            features = []
            for index, encoder in enumerate(encoders):
                rows = []
                for first in range(0, n, self.config.encoder_micro_batch):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=encoder_bf16):
                        feature = encoder(
                            images[first : first + self.config.encoder_micro_batch]
                        ).float()
                    if text is not None:
                        feature = torch.cat(
                            (
                                feature,
                                self.config.text_betas[index] * text[first : first + len(feature)],
                            ),
                            1,
                        )
                    rows.append(feature)
                features.append(gather(torch.cat(rows)))
        return images, features, (noise, context)

    def replay(self, batch, image_gradient, micro):
        noise, context = batch
        for first in range(0, len(noise), micro):
            latent = self._latent(
                noise[first : first + micro], context[first : first + micro].to(self.device)
            )
            leaf = latent.detach().requires_grad_(True)
            for offset in range(0, len(leaf), 4):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    image = decode(self.pipeline, leaf[offset : offset + 4])
                image.backward(image_gradient[first + offset : first + offset + len(image)])
            latent.backward(leaf.grad)
