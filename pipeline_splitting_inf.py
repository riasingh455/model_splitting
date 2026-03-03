import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------
# Model split helpers
# -------------------------

def split_resnet18_into_stages(stages: int) -> List[nn.Module]:
    """
    Split ResNet18 into N sequential stages for inference.
    N=2 or N=3 are most sensible for a "simple pipeline" demo.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()

    # ResNet18 logical blocks:
    # stem: conv1->bn1->relu->maxpool
    # then layer1, layer2, layer3, layer4
    # then avgpool->flatten->fc
    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    tail = nn.Sequential(model.avgpool, nn.Flatten(1), model.fc)

    if stages == 2:
        # Stage0: stem + layer1 + layer2
        # Stage1: layer3 + layer4 + tail
        s0 = nn.Sequential(stem, model.layer1, model.layer2)
        s1 = nn.Sequential(model.layer3, model.layer4, tail)
        return [s0, s1]

    if stages == 3:
        # Stage0: stem + layer1
        # Stage1: layer2 + layer3
        # Stage2: layer4 + tail
        s0 = nn.Sequential(stem, model.layer1)
        s1 = nn.Sequential(model.layer2, model.layer3)
        s2 = nn.Sequential(model.layer4, tail)
        return [s0, s1, s2]

    raise ValueError("stages must be 2 or 3 for this simple demo.")


# -------------------------
# Tensor send/recv utilities
# -------------------------

@dataclass
class TensorMeta:
    shape: Tuple[int, ...]
    dtype: torch.dtype

def _dtype_to_code(dt: torch.dtype) -> int:
    # small mapping to send dtype as int
    mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
    }
    if dt not in mapping:
        raise ValueError(f"Unsupported dtype for send/recv: {dt}")
    return mapping[dt]

def _code_to_dtype(code: int) -> torch.dtype:
    mapping = {
        0: torch.float32,
        1: torch.float16,
        2: torch.bfloat16,
        3: torch.int64,
        4: torch.int32,
    }
    if code not in mapping:
        raise ValueError(f"Unsupported dtype code: {code}")
    return mapping[code]

def send_tensor(x: torch.Tensor, dst: int):
    """
    Send tensor by first sending a small header: ndim + shape + dtypecode,
    then the contiguous payload.
    """
    x = x.contiguous()
    shape = torch.tensor([x.dim(), *_list_int(x.shape), _dtype_to_code(x.dtype)], dtype=torch.int64)
    dist.send(shape, dst=dst)
    dist.send(x, dst=dst)

def recv_tensor(src: int, device: torch.device) -> torch.Tensor:
    """
    Receive tensor header then payload.
    """
    header = torch.empty(1 + 32 + 1, dtype=torch.int64)  # oversized, we'll trim
    # But dist.send requires exact size match, so we instead do a 1st receive for ndim,
    # then receive shape+dtype with known length.
    ndim_t = torch.empty(1, dtype=torch.int64)
    dist.recv(ndim_t, src=src)
    ndim = int(ndim_t.item())

    shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
    dist.recv(shape_dtype, src=src)

    shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
    dtype = _code_to_dtype(int(shape_dtype[-1].item()))

    x = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(x, src=src)
    return x

def _list_int(shape) -> List[int]:
    return [int(s) for s in shape]


# -------------------------
# Image + labels
# -------------------------

def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
    return x

def top1_label(logits: torch.Tensor) -> str:
    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]
    idx = int(logits.argmax(dim=1).item())
    return f"{idx}: {categories[idx]}"


# -------------------------
# Main per-rank logic
# -------------------------

def maybe_pin_to_core(rank: int):
    # Optional core pinning (Linux/macOS support varies; harmless if it fails)
    try:
        import os
        import psutil
        p = psutil.Process()
        p.cpu_affinity([rank])  # pin each stage to core=rank
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="bear.jpeg")
    parser.add_argument("--stages", type=int, default=2, choices=[2, 3])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--pin", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if world != args.stages:
        if rank == 0:
            raise SystemExit(f"torchrun nproc_per_node={world} must equal --stages {args.stages}")
        return

    if args.pin:
        maybe_pin_to_core(rank)

    device = torch.device(args.device)
    stage_modules = split_resnet18_into_stages(args.stages)
    stage = stage_modules[rank].to(device).eval()

    prev_rank = rank - 1
    next_rank = rank + 1

    # Rank 0 loads the image once.
    x0 = None
    if rank == 0:
        x0 = load_image_tensor(args.image, device)

    # Warmup + timed iterations
    # NOTE: With a single image, pipeline "throughput" gains are limited;
    # this is mainly a correctness + plumbing demo.
    total_start = None
    outputs = None

    with torch.inference_mode():
        for i in range(args.warmup + args.iters):
            if i == args.warmup and rank == 0:
                dist.barrier()
                total_start = time.perf_counter()
            elif i == args.warmup:
                dist.barrier()

            if rank == 0:
                y = stage(x0)
                if next_rank < world:
                    # send y to next stage
                    # header: ndim, then (shape..., dtypecode)
                    dist.send(torch.tensor([y.dim()], dtype=torch.int64), dst=next_rank)
                    dist.send(torch.tensor([*y.shape, _dtype_to_code(y.dtype)], dtype=torch.int64), dst=next_rank)
                    dist.send(y.contiguous(), dst=next_rank)
                else:
                    outputs = y
            else:
                # recv from prev
                ndim_t = torch.empty(1, dtype=torch.int64)
                dist.recv(ndim_t, src=prev_rank)
                ndim = int(ndim_t.item())

                shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
                dist.recv(shape_dtype, src=prev_rank)
                shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
                dtype = _code_to_dtype(int(shape_dtype[-1].item()))

                x = torch.empty(shape, dtype=dtype, device=device)
                dist.recv(x, src=prev_rank)

                y = stage(x)

                if next_rank < world:
                    dist.send(torch.tensor([y.dim()], dtype=torch.int64), dst=next_rank)
                    dist.send(torch.tensor([*y.shape, _dtype_to_code(y.dtype)], dtype=torch.int64), dst=next_rank)
                    dist.send(y.contiguous(), dst=next_rank)
                else:
                    outputs = y

    dist.barrier()
    if rank == 0:
        total_end = time.perf_counter()
        elapsed = total_end - total_start
        avg = elapsed / max(1, args.iters)
        print(f"\n--- Pipeline Inference (stages={args.stages}, iters={args.iters}) ---")
        print(f"Total time (timed iters): {elapsed:.4f} s")
        print(f"Avg latency per image: {avg*1000:.2f} ms")
        print("Note: Single-image pipeline mostly demonstrates partitioning, not big speedups.")

    if rank == world - 1 and outputs is not None:
        print(f"[rank {rank}] pred {top1_label(outputs)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
