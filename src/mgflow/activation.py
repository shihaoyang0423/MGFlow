import torch
from torch.utils.checkpoint import checkpoint


class CheckpointBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(self.block, *args, use_reentrant=False, **kwargs)
        return self.block(*args, **kwargs)


def checkpoint_blocks(model):
    for name in ("blocks", "shared_blocks", "u_heads", "v_heads"):
        blocks = getattr(model, name, None)
        if blocks is not None:
            for index, block in enumerate(blocks):
                blocks[index] = CheckpointBlock(block)
