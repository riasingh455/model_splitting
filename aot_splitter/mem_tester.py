import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        # tracemalloc.Filter(False, "frozen"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()

import tyro
from PIL import Image
from dataclasses import dataclass
from ast import literal_eval 
import time
from datetime import datetime
import gc
# import dill as pickle 

import torch
import torch.export as export
import torch.distributed as dist
import torch.multiprocessing as mp
from executorch.runtime import Runtime
# import pathos.multiprocessing as mp

@dataclass
class Args:
    world: int = 1
    rank: int = 0
    batch_size: int = 4
    batch_num: int = 10
    model_type: str = "resnet18"
    model_split_type:str = "children"
    image: str = "./bear.jpeg"
    ip: str = "127.0.0.1"
    port: int = 9123
    cores: int = 4
    iters: int = 15
    warmup: int = 1
    backend: str = "gloo"

def mp_proc(model, input_tensor, output_tensor, no_star, store, iters, w_iters, warmup):
    w_tracks=[]
    for i in range(iters+w_iters):
        w_start = time.perf_counter()
        # print(f"check {iters}")
        with torch.inference_mode():
            if store:
                output = model.forward(input_tensor) if no_star else model.forward(*input_tensor)
            else:
                model.forward(input_tensor) if no_star else model.forward(*input_tensor)
        w_end = time.perf_counter()
        w_tracks.append(w_end-w_start)

    if store:
        if type(output)!=type(output_tensor):
            output = torch.cat([o.flatten() for o in output]).flatten()
        output_tensor.copy_(output)
        warmup[0] = sum(w_tracks[:w_iters])
    output=None

def str_to_dtype(code):
    mapping = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        }
    if code not in mapping:
        raise ValueError(f"Unsupported dtype code: {code}")
    return mapping[code]


def custom_pipeline(aot_dir, batch_num, world, rank, cores, iters=1, w_iters=1, inputs=None):
    stage_dict = open(f"{aot_dir}/stages.dict")
    stage_dict = literal_eval(stage_dict.readlines()[0].strip())
    #expected communications
    exp_recvs = []
    exp_recvs_tensors = [] if rank!=0 else inputs
    #tensor used to collect send request
    mp_collect_tensor = torch.zeros(stage_dict[rank+1][0], dtype=str_to_dtype(stage_dict[rank+1][1])) #if rank+1 < world else []
    if len(mp_collect_tensor) > 0:
        mp_collect_tensor.share_memory_() 
    no_star=True
    star_track = []
    if rank!=0:
        for b in range(batch_num):
            placeholder=torch.empty(stage_dict[rank][0], dtype=str_to_dtype(stage_dict[rank][1]))
            if rank-1 > 0 and len(stage_dict[rank-1]) > 2:
                star_track = [stage_dict[rank-1][2], [k for k in stage_dict[rank-1][3]]]
                no_star = False
            exp_recvs_tensors.append(placeholder)
            recv_op=dist.P2POp(dist.irecv, placeholder, rank-1)
            exp_recvs.append(recv_op)    
    
    # split_pt = f"{aot_dir}/split_{rank}.pt2"
    split_pt = f"{aot_dir}/exe_split_{rank}.pte.exp"
    mod = export.load(split_pt).module()
    # mod = torch._inductor.aoti_load_package("/Users/animeshnd/model_splitting/aot_splitter/resnet18_children_1_1/ind_split_0.pt2")
    
    # mod.eval()
    mod.share_memory()
    total_times=[]
    comms=[]
    net_times=[]
    comp_times=[]
    warmup_times=[]
    #sync point to get timings right for everyone
    gc.collect()
    dist.barrier()
    total_start = time.perf_counter()
    ts = datetime.now()
    print(f"{ts} Sync done -> model run start", flush=True)

    #can actually just use the communication from the stage dict!
    
    while len(exp_recvs_tensors) > 0:
        if len(exp_recvs)>0:
            #assumes no broadcast/multi-recvs for now
            comms.append(exp_recvs.pop(0))
        if len(comms)>0:
            works = dist.batch_isend_irecv(comms)
            net_start = time.perf_counter()
            for w in works:
                w.wait()
            net_end= time.perf_counter()
            net_times.append(net_end-net_start)

        processes = []
        recv_tensor = exp_recvs_tensors.pop(0)
        if no_star:
            recv_tensor.share_memory_()
        else:
            slices=[] 
            split_t=[]
            #for each tuple shape get full tuple slice
            for k in star_track[1]:
                c = 1
                for l in k:
                    c*=l
                slices.append(c)
            counter=0 #to include the first slice
            for sl in range(len(slices)):
                split_t.append(recv_tensor[counter:counter+slices[sl]].reshape(tuple(star_track[1][sl])))
                counter+=slices[sl]
            recv_tensor = tuple([t.share_memory_() for t in split_t])

        warm_up = torch.empty(1).share_memory_()
        comp_start = time.perf_counter()
        #parallelising here
        for c in range(cores):
            store=False if c!=0 else True
            p = mp.Process(target=mp_proc, args=(mod, recv_tensor, mp_collect_tensor, no_star, store, iters, w_iters, warm_up ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        comp_end = time.perf_counter()
        comp_times.append(comp_end-comp_start)
        warmup_times.append(warm_up[0].item())
        #reset comms for next iteration
        comms=[]
        if rank+1 < world:
            mp_collect_tensor = mp_collect_tensor.contiguous()
            send_op = dist.P2POp(dist.isend, mp_collect_tensor, rank+1)
            comms.append(send_op)
    #if leftover sends
    if len(comms)>0:
        net_start=time.perf_counter()
        works = dist.batch_isend_irecv(comms)
        for w in works:
            w.wait()
        net_end=time.perf_counter()
        net_times.append(net_end-net_start)
    mod=None
    gc.collect()
    total_end = time.perf_counter()
    total_times.append(total_end-total_start)
    #NOTE warmuptimes need to be excluded from their current compute time AND from the next layer's network time for true time!
    #NOTE same for total times since it includes both network and compute so two different warmup corrections!
    return mp_collect_tensor, comp_times, warmup_times, net_times, total_times

def data_loader(model_type, batch_num, batch_size, image_path):
    weights, transforms= None, None

    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        transforms=weights.transforms()
    
    img_batch=[transforms(Image.open(image_path).convert("RGB"))]*batch_size
    batches = [torch.stack(img_batch)]*batch_num
    return batches

def set_core_behavior(n:int=1):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)

def top1_label(model_type, output):
    categories=None
    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        categories = weights.meta["categories"]
    idx=0
    try:
        idx = int(output.argmax(dim=1).item())
    except Exception as e:
        idx = output.argmax(dim=1).tolist()
    return f"{idx}: {categories[idx]}" if type(idx)!=list else ", ".join([str(categories[i]) for i in idx])

if __name__ == "__main__":
    mp.set_start_method("fork")
    set_core_behavior(1)
    args = tyro.cli(Args)
    print(f"Model {args.model_type} and split {args.model_split_type}", flush=True)
    
    init_method = f"tcp://{args.ip}:{args.port}"
    dist.init_process_group(backend=args.backend, init_method=init_method, 
    world_size=args.world, rank=args.rank)
    aot_dir = f"./{args.model_type}_{args.model_split_type}_{args.world}_{args.batch_size}"
    inputs = data_loader(args.model_type, args.batch_num, args.batch_size, args.image) if args.rank==0 else []

    op, comp_times, warmup_times, net_times, total_times = custom_pipeline(aot_dir, args.batch_num, args.world, args.rank, args.cores, iters=args.iters, w_iters=args.warmup, inputs=inputs)

    #NOTE difference parsing strat -> literal_eval my goat
    log_dict = {"rank":args.rank, "world":args.world, "comp_times":comp_times, "warmup": warmup_times ,"net_times":net_times, "total_times":total_times}
    print(log_dict, flush=True)
    # if args.rank+1==args.world:
    #     s = top1_label(args.model_type, op)
        # print(s)
    snapshot = tracemalloc.take_snapshot()
    print()
    display_top(snapshot)




