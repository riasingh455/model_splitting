import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision.models import resnet18
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import grad as nn_grad


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


def gather_sizes(local_c: int):
    """All ranks share their shard size (dim=1) so we can slice correctly."""
    world_size = dist.get_world_size()
    sizes = [0 for _ in range(world_size)]
    dist.all_gather_object(sizes, local_c)
    return sizes


def offsets_from_sizes(sizes):
    offs = [0]
    for s in sizes[:-1]:
        offs.append(offs[-1] + s)
    return offs


def all_gather_cat_dim1(y_shard: torch.Tensor):
    """
    Gather shards (possibly different sizes) and concat along dim=1.
    Pads to max size so all_gather works.
    Returns (y_full, sizes).
    """
    world_size = dist.get_world_size()
    local_c = y_shard.size(1)

    sizes = gather_sizes(local_c)
    max_c = max(sizes)

    if local_c < max_c:
        pad_shape = list(y_shard.shape)
        pad_shape[1] = max_c - local_c
        pad = torch.zeros(pad_shape, dtype=y_shard.dtype, device=y_shard.device)
        y_padded = torch.cat([y_shard, pad], dim=1)
    else:
        y_padded = y_shard

    gather_list = [torch.empty_like(y_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, y_padded)

    parts = []
    for r, t in enumerate(gather_list):
        c = sizes[r]
        parts.append(t[:, :c, ...] if c != t.size(1) else t)
    return torch.cat(parts, dim=1), sizes


# -------------------------
# Autograd Functions (the key training fix)
# -------------------------

class TPLinearOutFn(torch.autograd.Function):
    """
    Output-sharded Linear with correct backprop:
    forward:
      y_shard = x @ W_shard^T + b_shard
      y_full = all_gather + concat
    backward:
      grad_y_shard = slice(grad_y_full)
      grad_x_local = grad_y_shard @ W_shard
      all_reduce SUM grad_x_local -> grad_x_full
      grad_W_shard, grad_b_shard are local
    """

    @staticmethod
    def forward(ctx, x, w_shard, b_shard, out_start, out_end):
        # x: [B, in]
        # w_shard: [out_shard, in]
        y_shard = F.linear(x, w_shard, b_shard)  # [B, out_shard]

        # reuse dim=1 gather by making it [B, out_shard, 1]
        y_full_3d, sizes = all_gather_cat_dim1(y_shard.unsqueeze(-1))
        y_full = y_full_3d.squeeze(-1)  # [B, out_full]

        ctx.save_for_backward(x, w_shard)
        ctx.sizes = sizes
        ctx.rank = dist.get_rank()
        return y_full

    @staticmethod
    def backward(ctx, grad_y_full):
        x, w_shard = ctx.saved_tensors
        sizes = ctx.sizes
        rank = ctx.rank

        offs = offsets_from_sizes(sizes)
        start = offs[rank]
        end = start + sizes[rank]

        grad_y_shard = grad_y_full[:, start:end].contiguous()  # [B, out_shard]

        # Grad w.r.t input x
        grad_x_local = grad_y_shard @ w_shard  # [B, in]
        dist.all_reduce(grad_x_local, op=dist.ReduceOp.SUM)    # make it "full" for prev layer
        grad_x = grad_x_local

        # Grad w.r.t weights/bias (local shard only)
        grad_w = grad_y_shard.t() @ x  # [out_shard, in]
        grad_b = grad_y_shard.sum(dim=0) if grad_y_shard is not None else None

        # None for out_start/out_end (ints)
        return grad_x, grad_w, grad_b, None, None


class TPConv2dOutFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_shard, b_shard, stride, padding, dilation, groups):
        y_shard = F.conv2d(
            x, w_shard, bias=b_shard,
            stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        y_full, sizes = all_gather_cat_dim1(y_shard)

        ctx.save_for_backward(x, w_shard, b_shard if b_shard is not None else torch.tensor([], device=x.device))
        ctx.has_bias = (b_shard is not None)

        ctx.sizes = sizes
        ctx.rank = dist.get_rank()
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return y_full

    @staticmethod
    def backward(ctx, grad_y_full):
        x, w_shard, b_saved = ctx.saved_tensors
        b_shard = b_saved if ctx.has_bias else None

        sizes = ctx.sizes
        rank = ctx.rank
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        offs = offsets_from_sizes(sizes)
        start = offs[rank]
        end = start + sizes[rank]
        grad_y_shard = grad_y_full[:, start:end, :, :].contiguous()

        # dX (sum across ranks)
        grad_x_local = nn_grad.conv2d_input(
            input_size=x.shape,
            weight=w_shard,
            grad_output=grad_y_shard,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        )
        dist.all_reduce(grad_x_local, op=dist.ReduceOp.SUM)
        grad_x = grad_x_local

        # dW (local shard)
        grad_w = nn_grad.conv2d_weight(
            input=x,
            weight_size=w_shard.shape,
            grad_output=grad_y_shard,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
        )

        # db (local shard) if bias exists
        grad_b = grad_y_shard.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # IMPORTANT: return None for non-tensor args
        return grad_x, grad_w, grad_b, None, None, None, None

class TPConv2dOutTrain(nn.Module):
    def __init__(self, conv: nn.Conv2d, rank: int, world_size: int):
        super().__init__()
        assert conv.groups == 1, "This demo assumes groups=1."

        out_s, out_e = shard_range(conv.out_channels, rank, world_size)

        w_shard = conv.weight[out_s:out_e].contiguous()
        self.weight = nn.Parameter(w_shard)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias[out_s:out_e].contiguous())
        else:
            self.bias = None

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x):
        return TPConv2dOutFn.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups
        )


class TPLinearOutTrain(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, world_size: int):
        super().__init__()

        out_s, out_e = shard_range(linear.out_features, rank, world_size)

        w_shard = linear.weight[out_s:out_e].contiguous()
        self.weight = nn.Parameter(w_shard)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias[out_s:out_e].contiguous())
        else:
            self.bias = None

        self.out_start = out_s
        self.out_end = out_e

    def forward(self, x):
        return TPLinearOutFn.apply(x, self.weight, self.bias, self.out_start, self.out_end)


def tp_replace_layers_train(module: nn.Module, rank: int, world_size: int):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, TPConv2dOutTrain(child, rank, world_size))
        elif isinstance(child, nn.Linear):
            setattr(module, name, TPLinearOutTrain(child, rank, world_size))
        else:
            tp_replace_layers_train(child, rank, world_size)


def freeze_batchnorm_stats(model: nn.Module):
    # Keeps BN from doing running-stat updates (stability)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


# -------------------------
# Distributed setup
# -------------------------

def setup_dist(rank: int, world_size: int, master_addr: str, master_port: str):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_dist():
    dist.destroy_process_group()


def set_one_core_behavior():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


# -------------------------
# Worker: training
# -------------------------

def worker(rank: int, world_size: int, master_port: str):
    setup_dist(rank, world_size, master_addr="127.0.0.1", master_port="29502")
    set_one_core_behavior()

    # Use a simpler transform (more stable than ImageNet normalize for random data)
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    num_classes = 10

    # Model (train from scratch for stability in this demo)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.train()

    # Replace conv/linear with TP versions
    tp_replace_layers_train(model, rank, world_size)

    # Freeze BN running stats for stability
    freeze_batchnorm_stats(model)

    # Dataset
    ds = FakeData(
        size=256,
        image_size=(3, 224, 224),
        num_classes=num_classes,
        transform=tfm,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    steps = 20
    step = 0

    dist.barrier()
    t0 = time.perf_counter()

    for images, targets in loader:
        # Tensor-parallel demo: all ranks must use the SAME batch
        if rank != 0:
            images = torch.empty((16, 3, 224, 224), dtype=torch.float32)
            targets = torch.empty((16,), dtype=torch.int64)

        dist.broadcast(images, src=0)
        dist.broadcast(targets, src=0)

        logits = model(images)

        # quick NaN check
        if torch.isnan(logits).any():
            if rank == 0:
                print("NaN detected in logits; stopping.", flush=True)
            break

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient clipping prevents blow-ups
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if rank == 0 and step % 5 == 0:
            print(f"step {step:02d}  loss={loss.item():.4f}", flush=True)

        step += 1
        if step >= steps:
            break

    dist.barrier()
    t1 = time.perf_counter()

    if rank == 0:
        print(f"Finished {step} steps in {t1 - t0:.2f}s", flush=True)

    cleanup_dist()


def main():
    mp.set_start_method("spawn", force=True)
    world_size = 8
    master_port = "29502"  # any free port is fine
    mp.spawn(worker, args=(world_size, master_port), nprocs=world_size, join=True)
    # mp.set_start_method("spawn", force=True)
    # world_size = 2

    # runs = 10
    # times = []

    # for r in range(runs):
    #     print(f"\nRun {r+1}/{runs}")

    #     t0 = time.perf_counter()
    #     mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    #     t1 = time.perf_counter()

    #     run_time = t1 - t0
    #     times.append(run_time)
    #     print(f"Run {r+1} time: {run_time:.2f}s")

    # avg_time = sum(times) / len(times)
    # print("\n==============================")
    # print(f"Average time over {runs} runs: {avg_time:.2f}s")

    # Optional useful metrics
    # steps = 20
    # batch_size = 16
    # avg_step_time = avg_time / steps
    # images_per_sec = batch_size / avg_step_time

    # print(f"Avg step time: {avg_step_time*1000:.2f} ms")
    # print(f"Throughput: {images_per_sec:.2f} images/sec")


if __name__ == "__main__":
    main()