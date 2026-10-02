from .imagenet import ImageNetGenerator

PROFILES = {
    "JiT-B": dict(cfg=1.0, noise_scale=1.0, t_min=0.1, t_max=1.0, lr=1e-5),
    "JiT-L": dict(cfg=1.0, noise_scale=1.0, t_min=0.1, t_max=1.0, lr=1e-5),
    "JiT-H": dict(cfg=1.0, noise_scale=1.0, t_min=0.1, t_max=1.0, lr=1e-5),
    "pMF-B": dict(cfg=8.5, noise_scale=1.0, t_min=0.1, t_max=0.7, lr=1e-6),
    "pMF-L": dict(cfg=7.0, noise_scale=1.0, t_min=0.2, t_max=0.7, lr=1e-6),
    "pMF-H": dict(cfg=7.0, noise_scale=2.0, t_min=0.2, t_max=0.6, lr=1e-6),
}


def build(name):
    family, size = name.split("-")
    return ImageNetGenerator(family, size)
