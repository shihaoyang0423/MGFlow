import numpy as np
import torch

from ..distributed import gather, rank
from ..models import PROFILES


def _rng():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def _restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def warmup_labels(positions, seed):
    permutation = torch.as_tensor(
        np.random.default_rng(seed + 1729).permutation(1000),
        device=positions.device,
        dtype=torch.long,
    )
    return permutation[positions.remainder(1000)]


class ImageNetBatch:
    def __init__(self, model, name, device):
        self.model, self.device = model, device
        self.profile = PROFILES[name]

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state:
            raise ValueError("unexpected ImageNet adapter state")

    def sample(self, n, labels=None, bf16=True):
        noise = torch.randn(n, 3, 256, 256, device=self.device) * self.profile["noise_scale"]
        if labels is None:
            labels = torch.randint(0, 1000, (n,), device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            image = self.model(noise, labels, self.profile)
        return image.float() * 0.5 + 0.5

    @torch.no_grad()
    def warmup_collect(self, n, encoders, generated, seed):
        positions = generated + rank() * n + torch.arange(n, device=self.device)
        image = self.sample(n, labels=warmup_labels(positions, seed), bf16=False)
        return [gather(encoder(image).float()) for encoder in encoders]

    def collect(self, n, micro, encoders, encoder_bf16):
        features = [[] for _ in encoders]
        images, states = [], []
        for first in range(0, n, micro):
            states.append(_rng())
            # Replay must use the same grad mode when the model is compiled.
            with torch.enable_grad():
                image = self.sample(min(micro, n - first))
            images.append(image.detach())
            for index, encoder in enumerate(encoders):
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=encoder_bf16),
                ):
                    feature = encoder(image.detach()).float()
                features[index].append(gather(feature))
            del image
        return torch.cat(images), [torch.cat(rows) for rows in features], states

    def replay(self, states, image_gradient, micro):
        live = _rng()
        try:
            for index, state in enumerate(states):
                _restore_rng(state)
                gradient = image_gradient[index * micro : (index + 1) * micro]
                self.sample(len(gradient)).backward(gradient)
        finally:
            _restore_rng(live)
