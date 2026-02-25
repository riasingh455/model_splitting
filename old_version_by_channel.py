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
# Tensor-parallel layers
# (in-channel sharding + all_reduce)
# -------------------------

def shard_range(total: int, rank: int, world_size: int):
    """Even-ish split of [0, total) across ranks."""
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


class TPConv2d(nn.Module):
    """
    Tensor-parallel Conv2d:
    - Shard input channels across ranks.
    - Each rank computes partial output (no bias).
    - all_reduce SUM to get full output.
    - Add bias once (after reduce) if present (bias replicated).
    """
    def __init__(self, conv: nn.Conv2d, rank: int, world_size: int):
        super().__init__()
        assert conv.groups == 1, "This TPConv2d assumes groups=1 (ResNet18 satisfies this)."
        self.rank = rank
        self.world_size = world_size

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        in_s, in_e = shard_range(self.in_channels, rank, world_size)
        self.in_start = in_s
        self.in_end = in_e

        # Slice weights on input-channel dimension
        # conv.weight shape: [out_channels, in_channels, kH, kW]
        w_shard = conv.weight[:, in_s:in_e, :, :].contiguous()
        self.weight = nn.Parameter(w_shard, requires_grad=False)

        # Bias is applied after all_reduce (replicated)
        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.detach().contiguous(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x_shard = x[:, self.in_start:self.in_end, :, :].contiguous()

        # Partial conv output (no bias here to avoid double-add)
        y_partial = F.conv2d(
            x_shard,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Sum partial outputs across ranks to get full y
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)

        # Add bias once (after reduce); safe to do on all ranks since bias is replicated
        if self.bias is not None:
            y_partial = y_partial + self.bias.view(1, -1, 1, 1)

        return y_partial


class TPLinear(nn.Module):
    """
    Tensor-parallel Linear:
    - Shard input features across ranks.
    - Each rank computes partial output (no bias).
    - all_reduce SUM to get full output.
    - Add bias once (after reduce) if present (bias replicated).
    """
    def __init__(self, linear: nn.Linear, rank: int, world_size: int):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        in_s, in_e = shard_range(self.in_features, rank, world_size)
        self.in_start = in_s
        self.in_end = in_e

        # linear.weight shape: [out_features, in_features]
        w_shard = linear.weight[:, in_s:in_e].contiguous()
        self.weight = nn.Parameter(w_shard, requires_grad=False)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().contiguous(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        x_shard = x[:, self.in_start:self.in_end].contiguous()

        y_partial = F.linear(x_shard, self.weight, bias=None)  # [B, out_features]
        dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)

        if self.bias is not None:
            y_partial = y_partial + self.bias

        return y_partial


def tp_replace_layers(module: nn.Module, rank: int, world_size: int):
    """
    Recursively replace Conv2d -> TPConv2d and Linear -> TPLinear.
    BN/ReLU/etc remain replicated (fine in eval mode).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, TPConv2d(child, rank, world_size))
        elif isinstance(child, nn.Linear):
            setattr(module, name, TPLinear(child, rank, world_size))
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

    # Replace every conv/linear with tensor-parallel versions
    tp_replace_layers(model, rank, world_size)

    labels = weights.meta["categories"]
    preprocess = weights.transforms()

    # Rank 0 loads image and broadcasts input to other rank
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
            y = model(x)  # after each layer, all_reduce ensures full tensor is correct
            outs.append(y)

    dist.barrier()
    t1 = time.perf_counter()

    # Only rank 0 prints results
    if rank == 0:
        logits = outs[-1]
        pred = int(logits.argmax(1))
        print("pred idx:", pred, flush=True)
        print("pred label:", labels[pred], flush=True)
        print(f"Throughput: {N / (t1 - t0):.2f} images/sec", flush=True)
        print(f"Avg latency per image: {(t1 - t0) * 1000 / N:.2f} ms", flush=True)

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