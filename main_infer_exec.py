import tyro
from PIL import Image
from dataclasses import dataclass
import pipeline_library 
import torch
import torch.multiprocessing as mp
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights# mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.fx.passes.split_module import split_module
import torch.distributed as dist
import numpy as np
from datetime import datetime

@dataclass
class Args:
    # Path to instruction profile results
    # stages: int
    # assigned_rank:int
    # local_procs: int
    rank: int = 0
    world: int = 1
    ip: str = "127.0.0.1"
    port: int = 8123
    batch_size: int = 5
    batch_num: int = 1
    copy: int = 1
    cores:int = 1
    device: str = "cpu"
    model_type:str = "resnet18"
    model_split_type:str = "children"
    image: str = "./bear.jpeg"
    images: tuple[str,...] = () #if multiple images provided, each image run an iters+warmups number of times 
    iters:int = 5 #number of times to inference a single image and measures time
    warmup: int = 5 #number of times to not measure time to warmup the inference
    # local_pipeline: bool = False

    @property
    def backend(self):
        return "gloo" if self.device=="cpu" else "nccl"


def worker(world, rank, batch_size, num_batches, backend, ip, port, warmup, iters, inps, model_type, model_split_type, device):
    init_method = f"tcp://{ip}:{port}"
    print(f"Model {model_type} and split {model_split_type}")
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world, rank=rank)
    
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights).eval()
    if model_type=="resnet18":
        weights = ResNet18_Weights.DEFAULT
        pretrained_model = resnet18(weights=weights).eval()
    elif model_type=="mbv3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        pretrained_model = mobilenet_v3_small(weights=weights).eval()
    elif model_type=="eb0":
        weights = EfficientNet_B0_Weights.DEFAULT
        pretrained_model = efficientnet_b0(weights=weights).eval()
    
    # c_names = [c for c,_ in pretrained_model.named_children()]
    # print(c_names, [ [c, d] for c,d in pretrained_model.named_modules() if c!='' and c not in c_names])
    # exit()
    # print(len([n for n,_ in pretrained_model.named_modules()]))
    pretrained_model.share_memory()
    pre_proc = weights.transforms()
    data_labels = weights.meta['categories'] #subject to change depending on fine-tuning and training
    pipe_mod = pipeline_library.FBModel(pretrained_model, device, data_labels)
    #example_input = torch.randn(1, 3, 224, 224)
    #batch size has to be multiple of inps
    old_b = batch_size
    if batch_size%len(inps)!=0:
        batch_size= (batch_size//len(inps))*len(inps)
    if batch_size < len(inps):
        batch_size = len(inps)
    if batch_size!=old_b:
        print(f"Correcting batch size from {old_b} to {batch_size}", flush=True)

    example_input = torch.randn(1*batch_size, 3, 224, 224)

    pipe_mod.split(example_input, rank, world, model_split_type=model_split_type, input_count=num_batches)
    print(f"{datetime.now()} Split done -> model sync start", flush=True)
    #rank = dist.get_rank()
    #world = dist.get_world_size()
    # print(f"world{world}")
    #pipe_mod.split(example_input, rank, world, input_count=len(args.images))
    # print(pipe_mod.exec_pipe)
    # pipe_mod.split(rank, world)
    # exit()
    #x0=None
    #x1=None
    l = 1 if len(inps)==0 else len(inps)
    proc_images = [None]*len(inps)
    if rank==0:
        for ind, i in enumerate(inps):
            proc_images[ind] = pre_proc(Image.open(i).convert("RGB"))
 # pipe_mod.load_image_tensor(i, pre_proc)
        
        temp_images = proc_images*(batch_size//len(inps))#14 images total 
        proc_images = [torch.stack(temp_images) for i in range(num_batches)]


    #for _ in range(l):
    time_sets = []
    batch_sets = []
    net_sets = []
    raw_nets = []
    count_flop=True
    full_iter = warmup+iters
    for it in range(full_iter): #full_iter = warmup+iters
                #x0 = pipe_mod.load_image_tensor(args.images[0], pre_proc) 
        #x1 = pipe_mod.load_image_tensor(args.images[1], pre_proc) 
    # output = pipe_mod.pipeline_inference(world, rank, args.warmup, args.iters, x0)
    # print(x0.shape)
    #for it in range(args.warmup+args.iters):
        outputs, times, nets, bts, send_bytes, recv_bytes = pipe_mod.custom_pipeline_inf(world, rank, proc_images, count_flop)
        count_flop=False
        if it > warmup:
            if len(times)>0:
                time_sets.extend(times)
            if len(nets)>0:
                net_sets.extend(nets)
                raw_nets.append([round(i,4) for i in nets])
            if len(bts) > 0:
                batch_sets.append([round(i,4) for i in bts])
            #if len(outputs)>0:
                #for output in outputs:
                    #res = pipe_mod.top1_label(data_labels, output)
                    #print(res)
    num_imgs = num_batches*batch_size
    if len(time_sets)>0:
        print(f"rank {rank} FLOP count: {pipe_mod.total_flops} and "+
              f"full time*10**9 s/op: {( (10**9)*(np.mean(time_sets)/num_imgs)/pipe_mod.total_flops ):.4f}", flush=True)

        print(f"Time taken by rank:{rank} in total(avg): {np.mean(time_sets):.4f}s "+
            #   f"Time sets raw:{[time_sets[r:r+world] for r in range(0, len(time_sets), world)]} "+
            #   f"Batch time sets raw:{[batch_sets[r:r+num_batches] for r in range(0, len(batch_sets), num_batches)]} "+
              f"Batch time sets raw:{batch_sets} "+
            #   f"Network sets raw:{[net_sets[r:r+world] for r in range(0, len(net_sets), world)]} "+
              f"Network sets raw:{raw_nets} "+
              f"bytes sent:{send_bytes} "+
              f"bytes recv:{recv_bytes} "+
              f"on avg per image: {(np.mean(time_sets)/num_imgs):.4f}s "+ 
              f"with std: {(np.std(time_sets)/num_imgs):.4f}s "+
              f"network time: {np.mean(net_sets):.4f}s and network std: {np.std(net_sets):.4f}s "+
              f"network time per img: {(np.mean(net_sets)/num_imgs):.4f} "+
              f"compute time: { ((np.mean(time_sets)/num_imgs) - np.mean(net_sets)/num_imgs):.4f}s "+
              f"compute/network ratio: "+
              f"{( ((np.mean(time_sets)/num_imgs) - (np.mean(net_sets)/num_imgs))/np.mean(net_sets) ):.4f} "
              f"and throughput {(num_imgs/np.mean(time_sets)):.4f} img/s", flush=True)


    dist.destroy_process_group()




if __name__ == "__main__":

    #TODO
    #first: find the flop limit on the rpis such that below that it doesn't count as "intense", 
    # think it might be a percentage of the best flop rate possible on the device? currently assuming 10% of maximum flop rate of model but this is model specific
    #second: See how well we can predict the chunk splits and their performance across the pis
    #third: (part of second actually) come up with a chunk combo that can be faster if split into more layers rather than less

    #mp.set_start_method('spawn') 
    #mp.set_start_method("spawn", force=True)
    args = tyro.cli(Args)
    device = torch.device(args.device)
    #import model and do splits before hand
    pipeline_library.set_core_behavior(args.cores)
    
    #weights = ResNet18_Weights.DEFAULT
    #pretrained_model = resnet18(weights=weights).eval()
    #pretrained_model.share_memory()
    #pre_proc = weights.transforms()
    #data_labels = weights.meta['categories'] #subject to change depending on fine-tuning and training
    #pipe_mod = pipeline_library.FBModel(pretrained_model, device, data_labels)
    
    #early gc potential
    
    #weights=None
    #pretrained_model=None

    #example_input = torch.randn(1, 3, 224, 224)
    #pipe_mod.split(example_input, args.rank, args.world, input_count=len(args.images))
    print("Code start -> moving to subprocess", flush=True) 
    worker(args.world, args.rank, args.batch_size, args.batch_num, args.backend, args.ip, 
           args.port, args.warmup, args.iters, args.images, args.model_type, args.model_split_type, device)

  #contexts = []
    #for off in range(args.copy):
    #    ctx = mp.spawn(worker, 
    #             args=(args.world, args.rank, args.backend, args.ip, 
    #                   args.port+off, args.warmup+args.iters, args.images, device), 
    #             nprocs=1, join=False) 
    #    contexts.append(ctx)
#
 #   for ctx in contexts:
 #       ctx.join()




#bramble-4-5 middling
#[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 20, 24, 25, 26, 27, 30, 31, 32]