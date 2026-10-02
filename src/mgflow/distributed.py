import os

import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def setup(kind="cuda"):
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if kind == "cuda"
        else torch.device("cpu")
    )
    if kind == "cuda":
        torch.cuda.set_device(device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if kind == "cuda":
            dist.init_process_group("nccl", device_id=device)
        else:
            dist.init_process_group("gloo")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    return device


@torch.no_grad()
def gather(rows):
    if world_size() == 1:
        return rows
    result = torch.empty(
        (world_size() * len(rows), *rows.shape[1:]), device=rows.device, dtype=rows.dtype
    )
    dist.all_gather_into_tensor(result, rows.contiguous())
    return result


@torch.no_grad()
def sum_gradients(parameters, bucket_bytes=64 * 1024 * 1024):
    """Sum rank-local contributions to a loss already averaged over the global batch."""
    if world_size() == 1:
        return
    groups = {}
    for parameter in parameters:
        if parameter.grad is not None:
            groups.setdefault(parameter.grad.dtype, []).append(parameter.grad)
    for tensors in groups.values():
        bucket, size = [], 0
        for tensor in tensors:
            if bucket and size + tensor.numel() * tensor.element_size() > bucket_bytes:
                _reduce_bucket(bucket)
                bucket, size = [], 0
            bucket.append(tensor)
            size += tensor.numel() * tensor.element_size()
        if bucket:
            _reduce_bucket(bucket)


def _reduce_bucket(tensors):
    flat = torch.cat([tensor.reshape(-1) for tensor in tensors])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for tensor in tensors:
        tensor.copy_(flat[offset : offset + tensor.numel()].view_as(tensor))
        offset += tensor.numel()


@torch.no_grad()
def gather_state(value, group, chunk_bytes=64 * 1024 * 1024):
    """Collect tensor state on rank zero without serializing tensor data for communication."""
    tensors = []

    def pack(item):
        if isinstance(item, torch.Tensor):
            index = len(tensors)
            tensors.append(item.detach())
            return ("tensor", index, tuple(item.shape), str(item.dtype))
        if isinstance(item, dict):
            return ("dict", [(key, pack(value)) for key, value in item.items()])
        if isinstance(item, (list, tuple)):
            return (type(item).__name__, [pack(value) for value in item])
        return ("value", item)

    def unpack(tree, values):
        kind = tree[0]
        if kind == "tensor":
            return values[tree[1]].reshape(tree[2])
        if kind == "dict":
            return {key: unpack(value, values) for key, value in tree[1]}
        if kind in ("list", "tuple"):
            items = [unpack(value, values) for value in tree[1]]
            return tuple(items) if kind == "tuple" else items
        return tree[1]

    tree = pack(value)
    layout = {}
    for index, tensor in enumerate(tensors):
        if tensor.layout != torch.strided:
            raise ValueError("checkpoint transport requires dense tensors")
        layout.setdefault(str(tensor.dtype), []).append((index, tensor.numel()))
    metadata = [None] * world_size()
    dist.all_gather_object(metadata, (tree, layout), group=group)
    result = []
    for source, (source_tree, source_layout) in enumerate(metadata):
        values = {}
        for name, items in source_layout.items():
            dtype = getattr(torch, name.split(".")[-1])
            total = sum(count for _, count in items)
            capacity = max(1, chunk_bytes // torch.empty((), dtype=dtype).element_size())
            buffer = torch.empty(min(capacity, total), dtype=dtype)
            target = torch.empty(total, dtype=dtype) if rank() == 0 else None
            offset = item_index = item_offset = 0
            while offset < total:
                count = min(capacity, total - offset)
                if rank() == source:
                    used = 0
                    while used < count:
                        index, size = items[item_index]
                        amount = min(size - item_offset, count - used)
                        buffer[used : used + amount].copy_(
                            tensors[index].reshape(-1)[item_offset : item_offset + amount]
                        )
                        used += amount
                        item_offset += amount
                        if item_offset == size:
                            item_index += 1
                            item_offset = 0
                    if source != 0:
                        dist.send(buffer[:count], dst=0, group=group)
                if rank() == 0:
                    if source != 0:
                        dist.recv(buffer[:count], src=source, group=group)
                    target[offset : offset + count].copy_(buffer[:count])
                offset += count
            if rank() == 0:
                offset = 0
                for index, count in items:
                    values[index] = target[offset : offset + count]
                    offset += count
        if rank() == 0:
            result.append(unpack(source_tree, values))
        dist.barrier(group=group)
    return result
