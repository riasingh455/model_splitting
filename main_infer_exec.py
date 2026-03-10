import tyro
from dataclasses import dataclass
import pipeline_library 
import torch
import torch.multiprocessing as mp
from torchvision.models import resnet18, ResNet18_Weights
from torch.fx.passes.split_module import split_module
import torch.distributed as dist
import numpy as np

@dataclass
class Args:
    # Path to instruction profile results
    # stages: int
    # assigned_rank:int
    # local_procs: int
    cores:int = 1
    device: str = "cpu"
    image: str = "./bear.jpeg"
    images: tuple[str,...] = () #if multiple images provided, each image run an iters+warmups number of times 
    iters:int = 10 #number of times to inference a single image and measures time
    warmup: int = 0 #number of times to not measure time to warmup the inference
    # local_pipeline: bool = False

    @property
    def backend(self):
        return "gloo" if self.device=="cpu" else "nccl"

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = tyro.cli(Args)
    device = torch.device(args.device)
    #import model and do splits before hand
    pipeline_library.set_core_behavior(args.cores)
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights).eval()
    pretrained_model.share_memory()
    pre_proc = weights.transforms()
    data_labels = weights.meta['categories'] #subject to change depending on fine-tuning and training
    pipe_mod = pipeline_library.FBModel(pretrained_model, device, data_labels)
    #early gc potential
    weights=None
    pretrained_model=None 
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    example_input = torch.randn(1, 3, 224, 224)
    # print(f"world{world}")
    pipe_mod.split(example_input, rank, world, input_count=len(args.images))
    # print(pipe_mod.exec_pipe)
    # pipe_mod.split(rank, world)
    # exit()
    x0=None
    x1=None
    l = 1 if len(args.images)==0 else len(args.images)
    proc_images = [None]*len(args.images)
    if rank==0:
        for ind, i in enumerate(args.images):
            proc_images[ind] = pipe_mod.load_image_tensor(i, pre_proc)

    #for _ in range(l):
    time_sets = []
    net_sets = []
    count_flop=True
    for it in range(args.warmup + args.iters):
                #x0 = pipe_mod.load_image_tensor(args.images[0], pre_proc) 
        #x1 = pipe_mod.load_image_tensor(args.images[1], pre_proc) 
    # output = pipe_mod.pipeline_inference(world, rank, args.warmup, args.iters, x0)
    # print(x0.shape)
    #for it in range(args.warmup+args.iters):
        outputs, times, nets = pipe_mod.custom_pipeline_inf(world, rank, proc_images, count_flop)
        count_flop=False
        if len(times)>0:
            time_sets.extend(times)
        if len(nets)>0:
            net_sets.extend(nets)
        if len(outputs)>0:
            for output in outputs:
                res = pipe_mod.top1_label(data_labels, output)
                #print(res)
    if len(time_sets)>0:
        print(f"rank {rank} FLOP count: {pipe_mod.total_flops} and "+
              f"full time*10**9 s/op: {( (10**9)*(np.mean(time_sets)/len(args.images))/pipe_mod.total_flops ):.4f}")

        print(f"Time taken by rank:{rank} in total(avg): {np.mean(time_sets):.4f}s "+
              f"on avg per image: {(np.mean(time_sets)/len(args.images)):.4f}s "+ 
              f"with std: {(np.std(time_sets)/len(args.images)):.4f}s "+
              f"network time: {np.mean(net_sets):.4f}s and network std: {np.std(net_sets):.4f}s "+
              f"compute time: { ((np.mean(time_sets)/len(args.images)) - np.mean(net_sets)):.4f}s "+
              f"compute/network ratio: "+
              f"{( ((np.mean(time_sets)/len(args.images)) - np.mean(net_sets))/np.mean(net_sets) ):.4f} "
              f"and throughput {(len(args.images)/np.mean(time_sets)):.4f} img/s")


    dist.destroy_process_group()
    




