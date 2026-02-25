import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


# -------------------------
# Helpers
# -------------------------

def shard_range(total: int, rank: int, world_size: int):
    """Even-ish split of [0, total) across ranks."""
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def all_gather_cat_dim1(y_shard: torch.Tensor) -> torch.Tensor:
    """
    Gather variable-size shards across ranks and concat along dim=1.
    Works even if shard sizes differ.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Each rank's local size along dim=1
    local_c = y_shard.size(1)
    sizes = [0 for _ in range(world_size)]
    dist.all_gather_object(sizes, local_c)

    max_c = max(sizes)

    # Pad to max_c so all_gather works with equal shapes
    if local_c < max_c:
        pad_shape = list(y_shard.shape)
        pad_shape[1] = max_c - local_c
        pad = torch.zeros(pad_shape, dtype=y_shard.dtype, device=y_shard.device)
        y_padded = torch.cat([y_shard, pad], dim=1)
    else:
        y_padded = y_shard

    gather_list = [torch.empty_like(y_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, y_padded)

    # Unpad and concat in rank order
    parts = []
    for r, t in enumerate(gather_list):
        c = sizes[r]
        parts.append(t[:, :c, ...] if c != t.size(1) else t)
    return torch.cat(parts, dim=1)


# -------------------------
# Output-sharded tensor-parallel layers
# (full input on every rank, shard by output units, then all_gather)
# -------------------------

class TPConv2dOut(nn.Module):
    """
    Output-channel sharded Conv2d:
    - Every rank receives FULL input x.
    - Shard weight/bias on out_channels.
    - Each rank computes y_shard for its out-channel slice.
    - all_gather + concat along channel dim to rebuild full y.
    """
    def __init__(self, conv: nn.Conv2d, rank: int, world_size: int):
        super().__init__()
        assert conv.groups == 1, "This TPConv2dOut assumes groups=1."

        self.rank = rank
        self.world_size = world_size

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        out_s, out_e = shard_range(self.out_channels, rank, world_size)
        self.out_start = out_s
        self.out_end = out_e

        # conv.weight: [out_channels, in_channels, kH, kW]
        w_shard = conv.weight[out_s:out_e, :, :, :].contiguous()
        self.weight = nn.Parameter(w_shard, requires_grad=False)

        if conv.bias is not None:
            b_shard = conv.bias[out_s:out_e].contiguous()
            self.bias = nn.Parameter(b_shard, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W] (FULL on every rank)
        y_shard = F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )  # [B, C_out_shard, H', W']

        # Rebuild full output for next layer
        y = all_gather_cat_dim1(y_shard)
        return y


class TPLinearOut(nn.Module):
    """
    Output-feature sharded Linear:
    - Every rank receives FULL input x.
    - Shard weight/bias on out_features.
    - Each rank computes y_shard for its out-feature slice.
    - all_gather + concat along feature dim to rebuild full y.
    """
    def __init__(self, linear: nn.Linear, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        out_s, out_e = shard_range(self.out_features, rank, world_size)
        self.out_start = out_s
        self.out_end = out_e

        # linear.weight: [out_features, in_features]
        w_shard = linear.weight[out_s:out_e, :].contiguous()
        self.weight = nn.Parameter(w_shard, requires_grad=False)

        if linear.bias is not None:
            b_shard = linear.bias[out_s:out_e].contiguous()
            self.bias = nn.Parameter(b_shard, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features] (FULL on every rank)
        y_shard = F.linear(x, self.weight, bias=self.bias)  # [B, out_shard]

        # Make it [B, out_shard, 1] so we can reuse cat-dim1 helper, then squeeze back
        y_shard_3d = y_shard.unsqueeze(-1)  # [B, out_shard, 1]
        y_full_3d = all_gather_cat_dim1(y_shard_3d)  # [B, out_features, 1]
        return y_full_3d.squeeze(-1)  # [B, out_features]


def tp_replace_layers(module: nn.Module, rank: int, world_size: int):
    """
    Recursively replace Conv2d -> TPConv2dOut and Linear -> TPLinearOut.
    BN/ReLU/etc remain replicated (fine in eval mode).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, TPConv2dOut(child, rank, world_size))
        elif isinstance(child, nn.Linear):
            setattr(module, name, TPLinearOut(child, rank, world_size))
        else:
            tp_replace_layers(child, rank, world_size)


# -------------------------
# Distributed worker
# -------------------------

def setup_dist(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_dist():
    dist.destroy_process_group()


def set_one_core_behavior():
    # Make each rank behave like ~one core worth of compute.
    # (Still not strict pinning; OS scheduler chooses exact core.)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def worker(rank: int, world_size: int, image_path: str):
    setup_dist(rank, world_size, master_addr="127.0.0.1", master_port="29501")
    set_one_core_behavior()

    # Load pretrained model on each rank (CPU)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()

    # Replace every conv/linear with output-sharded tensor-parallel versions
    tp_replace_layers(model, rank, world_size)

    labels = weights.meta["categories"]
    preprocess = weights.transforms()

    # Rank 0 loads image and broadcasts input to other ranks
    if rank == 0:
        img = Image.open(image_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).contiguous()  # [1,3,224,224]
    else:
        x = torch.empty((1, 3, 224, 224), dtype=torch.float32)

    dist.broadcast(x, src=0)

    # Warmup
    with torch.no_grad():
        _ = model(x)
        _ = model(x)

    # Timed run
    N = 20
    dist.barrier()
    t0 = time.perf_counter()

    outs = []
    with torch.no_grad():
        for _ in range(N):
            y = model(x)  # after each conv/linear: all_gather rebuilds full tensor
            outs.append(y)

    dist.barrier()
    t1 = time.perf_counter()

    if rank == 0:
        logits = outs[-1]
        pred = int(logits.argmax(1))
        print("pred idx:", pred, flush=True)
        print("pred label:", labels[pred], flush=True)
        print(f"Throughput: {N / (t1 - t0):.2f} images/sec", flush=True)
        #print(f"Avg latency per image: {(t1 - t0) * 1000 / N:.2f} ms", flush=True)

    cleanup_dist()


def main():
    mp.set_start_method("spawn", force=True)

    world_size = 2
    image_path = "bear.jpeg"  # change if needed

    mp.spawn(
        worker,
        args=(world_size, image_path),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()