import argparse
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


def split_resnet18_into_stages(stages: int) -> List[nn.Module]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()

    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    tail = nn.Sequential(model.avgpool, nn.Flatten(1), model.fc)
    #loads the pretrained resnet, and divides into two helper chunks so its easier to split cleanly

    if stages == 2:
        s0 = nn.Sequential(stem, model.layer1, model.layer2)
        s1 = nn.Sequential(model.layer3, model.layer4, tail)
        return [s0, s1]

    if stages == 3:
        s0 = nn.Sequential(stem, model.layer1)
        s1 = nn.Sequential(model.layer2, model.layer3)
        s2 = nn.Sequential(model.layer4, tail)
        return [s0, s1, s2]

    raise ValueError("stages must be 2 or 3")
    #choose whether to split 2 or 3 times (# of stages)


def dtype_to_code(dt: torch.dtype) -> int:
    mapping = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.int64: 3,
        torch.int32: 4,
    }
    if dt not in mapping:
        raise ValueError(f"Unsupported dtype: {dt}")
    return mapping[dt]
#gives each pytorch datatype a number (each receiver needs to know dim, shape, datatype)


def code_to_dtype(code: int) -> torch.dtype:
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
#does the reverse as previous, these functions are just used for communication (why exactly??)


def send_tensor(x: torch.Tensor, dst: int):
    x = x.contiguous()
    dist.send(torch.tensor([x.dim()], dtype=torch.int64), dst=dst)
    dist.send(torch.tensor([*x.shape, dtype_to_code(x.dtype)], dtype=torch.int64), dst=dst)
    dist.send(x.cpu(), dst=dst)
    #the goal is to send enough information so the other process can reconstruct the tensor exactly
    #the receiving process must know the num of dimensions of x, shape, datatype, values


def recv_tensor(src: int) -> torch.Tensor:
    ndim_t = torch.empty(1, dtype=torch.int64)
    dist.recv(ndim_t, src=src)
    ndim = int(ndim_t.item())

    shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
    dist.recv(shape_dtype, src=src)

    shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
    dtype = code_to_dtype(int(shape_dtype[-1].item()))

    x = torch.empty(shape, dtype=dtype)
    dist.recv(x, src=src)
    return x
#receive the tensor from prev function sending

def load_image_tensor(path: str) -> torch.Tensor:
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    return x
#loads the image + applies the exact preprocessing ResNet-18 expects
#unsqueezes the image --> batch of 1 image 


def duplicate_into_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return x.repeat(batch_size, 1, 1, 1)
#duplicating the image into a batch (basically repeating the image) --> to test throughput and batching without needing many image files


def split_microbatches(x: torch.Tensor, num_microbatches: int) -> List[torch.Tensor]:
    b = x.size(0)
    if num_microbatches < 1 or num_microbatches > b:
        raise ValueError("microbatches must be between 1 and batch size")
    sizes = [b // num_microbatches] * num_microbatches
    for i in range(b % num_microbatches):
        sizes[i] += 1
    return list(torch.split(x, sizes, dim=0))
#check if the num of microbatches makes sense and then creates chunk sizes that are as even as possible
#this splits along batch dimension so if the full batch is 8 images and microbatches is 4, each chunk may have 2 images


def top1_labels(logits: torch.Tensor) -> List[str]:
    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]
    preds = logits.argmax(dim=1).tolist()
    return [f"{idx}: {categories[idx]}" for idx in preds]
#turn the model output into labels

def maybe_pin_to_core(rank: int):
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([rank])
    except Exception:
        pass
#optional: This tries to bind each process to one CPU core. meaning: rank 0 on core 0, rank 1 on core 1, rank 2 on core 2


def run_gpipe_inference(stage: nn.Module, rank: int, world: int, microbatches: int, x_full: torch.Tensor | None):
    prev_rank = rank - 1
    next_rank = rank + 1
    outputs = []

    with torch.inference_mode():
        if rank == 0:
            chunks = split_microbatches(x_full, microbatches)
            for chunk in chunks:
                y = stage(chunk)
                if next_rank < world:
                    send_tensor(y, next_rank)
                else:
                    outputs.append(y)
        elif rank < world - 1:
            for _ in range(microbatches):
                x = recv_tensor(prev_rank)
                y = stage(x)
                send_tensor(y, next_rank)
        else:
            for _ in range(microbatches):
                x = recv_tensor(prev_rank)
                y = stage(x)
                outputs.append(y)

    if rank == world - 1:
        return torch.cat(outputs, dim=0)
    return None
#rank 0 splits the input into microbatches and starts the pipeline
#middle ranks receive tensors, run their model stage, and forward results
#the last rank runs the final stage and concatenates outputs to produce the final batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="bear.jpeg")
    parser.add_argument("--stages", type=int, default=2, choices=[2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
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

    stage_modules = split_resnet18_into_stages(args.stages)
    stage = stage_modules[rank].eval()

    x_full = None
    if rank == 0:
        x = load_image_tensor(args.image)
        x_full = duplicate_into_batch(x, args.batch_size)

    timed_outputs = None
    total_start = None

    for i in range(args.warmup + args.iters):
        dist.barrier()
        if i == args.warmup and rank == 0:
            total_start = time.perf_counter()

        out = run_gpipe_inference(
            stage=stage,
            rank=rank,
            world=world,
            microbatches=args.microbatches,
            x_full=x_full,
        )

        if rank == world - 1:
            timed_outputs = out

    dist.barrier()

    if rank == 0:
        total_end = time.perf_counter()
        elapsed = total_end - total_start
        total_imgs = args.iters * args.batch_size
        throughput = total_imgs / elapsed
        avg_per_image = elapsed / total_imgs * 1000

        print(f"\n--- GPipe-style Inference ---")
        print(f"stages={args.stages}")
        print(f"batch_size={args.batch_size}")
        print(f"microbatches={args.microbatches}")
        print(f"iters={args.iters}")
        print(f"Total timed time: {elapsed:.4f} s")
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Avg time per image: {avg_per_image:.2f} ms")

    if rank == world - 1 and timed_outputs is not None:
        labels = top1_labels(timed_outputs)
        print(f"[rank {rank}] predictions:")
        for i, label in enumerate(labels):
            print(f"  sample {i}: {label}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()