import argparse
import time

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torch.distributed.pipelining import SplitPoint, ScheduleGPipe, pipeline


def load_batch(batch_size: int) -> torch.Tensor:
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    #makes sure the images are in the right format for ResNet-18

    bear = preprocess(Image.open("bear.jpeg").convert("RGB"))
    penguin = preprocess(Image.open("penguin.jpeg").convert("RGB"))
    #converts the images to pytorch tensors

    images = []
    for i in range(batch_size):
        if i % 2 == 0:
            images.append(bear)
        else:
            images.append(penguin)
    return torch.stack(images)
#^^ builds the list of images 
#this function makes a batch of images (input = batch_size) and outputs a tensor containing that many images 


def top1_labels(logits: torch.Tensor):
    categories = ResNet18_Weights.DEFAULT.meta["categories"]
    preds = logits.argmax(dim=1).tolist()
    return [f"{idx}: {categories[idx]}" for idx in preds]
#this function converts raw model outputs into readable class names


def get_split_spec(stages: int):
    if stages == 2:
        return {"layer3": SplitPoint.BEGINNING}
    if stages == 3:
        return {
            "layer2": SplitPoint.BEGINNING,
            "layer4": SplitPoint.BEGINNING,
        }
    raise ValueError("stages must be 2 or 3")
#decides whether to split the model 2 or 3 times and where to split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=2, choices=[2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    #arguments to pass in when running the file 

    if args.batch_size % args.microbatches != 0:
        raise SystemExit("batch-size must be divisible by microbatches for static-shape GPipe in this script")

    microbatch_size = args.batch_size // args.microbatches
    #determines the size of the microbatch - batch size must be divisible by microbatches

    dist.init_process_group(backend="gloo") #this starts the distributed environment - all processes launched by torchrun can join the same group 
    #gloo is the backend being used here which works on CPU

    rank = dist.get_rank() #rank = which process am I?
    world = dist.get_world_size() #world = total num of processes 
    

    if world != args.stages:
        if rank == 0:
            raise SystemExit(
                f"torchrun --nproc-per-node={world} must match --stages {args.stages}"
            )
        dist.destroy_process_group()
        return
    #checks that the total num of processes matches the num of stages 

    device = torch.device("cpu") #script runs on the CPU
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(device) #creates pretrained model, puts it in eval mode bc we are doing inference, moves it to CPU


    example_mb = load_batch(microbatch_size).to(device) #creates an example microbatch in order to trace the model and learn tensor shapes

    pipe = pipeline(
        module=model,
        mb_args=(example_mb,),
        split_spec=get_split_spec(args.stages),
    ) #this converts the normal model into a pipelined model - pytorch knows what the model is, the input shape, where to split into stages

    stage = pipe.build_stage(rank, device, dist.group.WORLD) #give this process its stage 
    schedule = ScheduleGPipe(stage, n_microbatches=args.microbatches) #creates the GPipe schedule - works by sending microbatches through the stages

    x_full = None
    if rank == 0:
        x_full = load_batch(args.batch_size).to(device)
    #only rank 0 loads the full batch 

    timed_output = None
    total_start = None
    #variables for timing

    with torch.inference_mode(): 
        for i in range(args.warmup + args.iters): #2 warmup iterations (bc first runs are slower bc of initialization) and 10 real timed iterations 
            if i == args.warmup and rank == 0:
                total_start = time.perf_counter() #start timer when warmup ends 

            if rank == 0: #rank0 starts the pipeline by giving it the whole batch 
                schedule.step(x_full)
            elif rank == world - 1: #last rank receives the input data from the prev stage. we store final output tensor in timed output 
                timed_output = schedule.step()
            else:
                schedule.step() #middle stages just run their stage and pass data along 


    if rank == 0:
        total_end = time.perf_counter()
        elapsed = total_end - total_start
        total_imgs = args.iters * args.batch_size
        throughput = total_imgs / elapsed
        avg_per_image_ms = elapsed / total_imgs * 1000

        print("\n--- ResNet18 GPipe Inference ---")
        print(f"stages={args.stages}")
        print(f"batch_size={args.batch_size}")
        print(f"microbatches={args.microbatches}")
        print(f"microbatch_size={microbatch_size}")
        print(f"iters={args.iters}")
        print(f"Total timed time: {elapsed:.4f} s")
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Avg time per image: {avg_per_image_ms:.2f} ms")

    if rank == world - 1 and timed_output is not None:
        labels = top1_labels(timed_output)
        print(f"[rank {rank}] predictions:")
        for i, label in enumerate(labels):
            print(f"  sample {i}: {label}")
    #collecting the data

    dist.destroy_process_group() #clean up distributed state


if __name__ == "__main__":
    main()



# command to run it:
# GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=3 inference.py \
#  --stages 3 \
#  --batch-size 9 \
#  --microbatches 3 \
#  --iters 10
