import tyro
from PIL import Image
from dataclasses import dataclass
from ast import literal_eval 
import time
from datetime import datetime
import dill as pickle

import torch
import torch.export as export
import torch.distributed as dist
import torch.multiprocessing as mp
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

def mp_proc(model, input_tensor, output_tensor, no_star, store):
    # model = pickle.loads(model)
    st=time.perf_counter()
    model = export.load(model).module()
    et=time.perf_counter()
    print(et-st)
    with torch.no_grad():
        output = model.forward(input_tensor) if no_star else model.forward(*input_tensor)
    #fit into output tensor shape
    if store:
        if type(output)!=type(output_tensor):
            output = torch.cat([o.flatten() for o in output]).flatten()
        output_tensor.copy_(output)

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


def custom_pipeline(aot_dir, batch_num, world, rank, cores, inputs=None):
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
    
    split_pt = f"{aot_dir}/split_{rank}.pt2"
    # mod = export.load(split_pt).module()
    # mod.eval()
    # mod.share_memory()
    total_times=[]
    comms=[]
    net_times=[]
    comp_times=[]

    #sync point to get timings right for everyone
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

        comp_start = time.perf_counter()
        #parallelising here
        # pool = mp.Pool(cores)
        # store=True
        # mp_res = pool.map_async(mp_proc, [(mod,recv_tensor, mp_collect_tensor, no_star, store ) for i in range(cores)])

        # # while not mp_res.ready():
        # #     time.sleep(1)
        # res = mp_res.get()
        # print(res)
        # mp_res.wait(300)
        for c in range(cores):
            store=False if c!=0 else True
            p = mp.Process(target=mp_proc, args=(split_pt, recv_tensor, mp_collect_tensor, no_star, store, ))
            # p = mp.Process(target=mp_proc, args=(pickle.dumps(mod, recurse=True, byref=True), recv_tensor, mp_collect_tensor, no_star, store, ))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        comp_end = time.perf_counter()
        comp_times.append(comp_end-comp_start)
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
    
    total_end = time.perf_counter()
    total_times.append(total_end-total_start)
    return mp_collect_tensor

def data_loader(model_type, batch_num, batch_size, image_path):
    weights, transforms= None, None

    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        transforms=weights.transforms()
    
    img_batch=[transforms(Image.open(image_path).convert("RGB"))]*batch_size
    batches = [torch.stack(img_batch)]*batch_num
    return batches


def top1_label(model_type, output):
    categories=None
    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        categories = weights.meta["categories"]

    #weights = ResNet18_Weights.DEFAULT
    #categories = weights.meta["categories"]
    idx=0
    try:
        idx = int(output.argmax(dim=1).item())
    except Exception as e:
        idx = output.argmax(dim=1).tolist()
    return f"{idx}: {categories[idx]}" if type(idx)!=list else ", ".join([str(categories[i]) for i in idx])

if __name__ == "__main__":
    args = tyro.cli(Args)
    print(f"Model {args.model_type} and split {args.model_split_type}")
    
    init_method = f"tcp://{args.ip}:{args.port}"
    dist.init_process_group(backend=args.backend, init_method=init_method, 
    world_size=args.world, rank=args.rank)
    aot_dir = f"./{args.model_type}_{args.model_split_type}_{args.world}_{args.batch_size}"
    inputs = data_loader(args.model_type, args.batch_num, args.batch_size, args.image) if args.rank==0 else []

    op = custom_pipeline(aot_dir, args.batch_num, args.world, args.rank, args.cores, inputs)
    if args.rank+1==args.world:
        s = top1_label(args.model_type, op)
        print(s)




