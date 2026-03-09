'''
Code to simulate GPipe pipelining for ResNet18 using PyTorch
'''
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, pipeline, PipelineStage, SplitPoint

NUM_CHUNKS = 8  # number of GPipe micro-batches
BATCH_SIZE = 16  # must be divisible by NUM_CHUNKS

def run_pipelining_gpipe():
    # 1. Initialize Distributed Environment
    # if not dist.is_initialized():
    dist.init_process_group(backend='gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cpu")

    print(f"[Rank {rank}/{world_size}] Initializing...")

    # 2. Setup Model
    model = resnet18(num_classes=10)
    model.train()

    # 3. Create the Pipeline with explicit split points
    # We split ResNet18 into `world_size` stages.
    # ResNet18 has: conv1/bn1/relu/maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    # With 2 ranks: split after layer2 (stage 0: input->layer2, stage 1: layer3->output)
    # With 4 ranks: split after layer1, layer2, layer3
    # Adjust split_spec based on your world_size.
    if world_size == 2:
        split_spec = {
            "layer2": SplitPoint.END,
        }
    elif world_size == 4:
        split_spec = {
            "layer1": SplitPoint.END,
            "layer2": SplitPoint.END,
            "layer3": SplitPoint.END,
        }
    else:
        raise ValueError(f"This example supports world_size=2 or 4, got {world_size}")

    # example_args is used for tracing the model graph only (single micro-batch shape)
    example_input = torch.randn(BATCH_SIZE // NUM_CHUNKS, 3, 224, 224)

    pipe = pipeline(
        model,
        mb_args=(example_input,),       # FIX: use mb_args (micro-batch args), not example_args
        split_spec=split_spec,          # FIX: define explicit split points for partitioning
    )

    print(f"[Rank {rank}] Pipeline has {pipe.num_stages} stages")

    # 4. Extract this rank's stage from the pipe
    # FIX: correct PipelineStage signature — (pipe, stage_index, num_stages, device)
    stage = PipelineStage(
        pipe,
        rank,           # stage_index: this rank handles this stage
        world_size,     # num_stages: total number of pipeline stages
        device,
    )

    # 5. Define the GPipe Schedule
    # FIX: must pass n_microbatches explicitly to ScheduleGPipe
    schedule = ScheduleGPipe(stage, n_microbatches=NUM_CHUNKS)

    # 6. Training Data & Optimizer
    # Only rank 0 provides inputs; only the last rank provides targets
    inputs  = torch.randn(BATCH_SIZE, 3, 224, 224) if rank == 0 else None
    targets = torch.randint(0, 10, (BATCH_SIZE,))  if rank == world_size - 1 else None

    optimizer = torch.optim.SGD(stage.submod.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 7. Execute Training Step
    print(f"[Rank {rank}] Executing GPipe Schedule...")
    optimizer.zero_grad()

    # schedule.step() runs fwd + bwd for all micro-batches in GPipe order.
    # - First rank receives `args=(inputs,)`
    # - Last rank receives `target` and `loss_fn` to compute loss
    # - Middle ranks receive neither
    if rank == 0:
        output = schedule.step(inputs)
    elif rank == world_size - 1:
        output = schedule.step(target=targets, loss_fn=criterion)
    else:
        output = schedule.step()

    optimizer.step()

    if output is not None:
        print(f"[Rank {rank}] Step complete. Loss: {output:.4f}")
    else:
        print(f"[Rank {rank}] Step complete. Gradients accumulated.")

if __name__ == "__main__":
    run_pipelining_gpipe()
    # python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 gpipe.py