from __future__ import annotations

import time
from PIL import Image
from pathlib import Path
import networkx as nx
from tqdm import tqdm
from datetime import datetime


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.fx.passes.split_module import split_module
from torch.distributed.pipelining import pipeline, SplitPoint

from torch.utils.flop_counter import FlopCounterMode

import torch.distributed as dist

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List 

torch.manual_seed(3)
np.random.seed(3)

def set_core_behavior(n:int=1):
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)

@dataclass
class DataWrap:
    train_data: Any 
    test_data: Any
    set_train_loader: Any 
    set_test_loader: Any 
    data_transform: List[Any] #train_transform, test_transform

    @classmethod
    def dataloader_gen(cls, train_transform:Any, test_transform: Any, set_name: str = "cifar10", subset_size:int = 1000):
        full_train:Any = []
        test_dataset:Any = []
        if set_name == "cifar10":
            full_train = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=test_transform)
        
        indices:Any = torch.randperm(len(full_train))[:subset_size]
        small_train_dataset = Subset(full_train, indices)
        print(f"Training with only {len(small_train_dataset)} images")
        print(f"Testing with {len(test_dataset)} images")
        train_loader = DataLoader(small_train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        cls.train_data = full_train
        cls.test_data = test_dataset
        cls.set_train_loader = train_loader
        cls.set_test_loader = test_loader
        return cls

    def make_train_subset(self, subset_len:int):
        indices:Any = torch.randperm(len(self.train_data))[:subset_len]
        small_train_dataset = Subset(self.train_data, indices)
        self.train_data = DataLoader(small_train_dataset, batch_size=32, shuffle=True)

@dataclass 
class GenModel:
    model: nn.Module
    # split_model: Dict[int, nn.Module]
    device: torch.device
    total_flops = 0
    exec_pipe: CustomPipeline|Any = ""

    def dtype_to_code(self, dt: torch.dtype):
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

    def code_to_dtype(self, code: int) -> torch.dtype:
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


    def freeze_layers_until(self, layer_depth:int=1, freeze_layer_names:List[str] = []):
        model = self.model 
        #freeze appropriare layers
        l_counter = -1
        
        for l, layer in self.model.named_children():
            l_counter+=1
            if l_counter==layer_depth and len(freeze_layer_names)==0:
                break
            if (len(freeze_layer_names)==0) or (str(l) in freeze_layer_names):
                for param in layer.parameters():
                    param.requires_grad = False
        self.model = model 
    
    def num_classes(self, num: int=10):
        model = self.model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num)
        self.model = model

    def split(self, example_input, rank:int, split_num:int = 2, input_count:int =1, specific_chunks:List[int] = []):
        #use split policy/spec to implement specific chunks?
        #limits to top-level layers only -> anything below and we need module instead of children here
        #don't really see why it can't be modules/graph modules?
        
        #forcefully add the stupid flatten layer
        # children = []
        # for k, v in self.model.named_children():
        #     if k.startswith("fc"):
        #         children.append(["flat", nn.Flatten(1)])
        #     children.append([k ,v])

        # split_model = {k:v for k,v in children}
        # if split_num > len(split_model):
        #     split_num = len(split_model)
        
        # sizes = [len(split_model) // split_num + (1 if i < len(split_model) % split_num else 0) for i in range(split_num)] if len(specific_chunks)==0 else specific_chunks
        # r=0
        # counter=0
        # split_names = list(split_model.keys())
        # splits = {}
        # while counter < len(sizes):
        #     if len(split_names[r:r+sizes[counter]]) > 0:
        #         splits[counter] = nn.Sequential(*[ split_model[m] for m in split_names[r:r+sizes[counter]]])
        #     r+=sizes[counter]
        #     counter+=1

        # self.model = splits[rank]

        #Same problem as before, can't trace when more than one input to next layer :/
        # traced = torch.fx.symbolic_trace(self.model)
        # nodes = [n for n in traced.graph.nodes if n.op not in ['placeholder', 'output']]
        # if split_num > len(nodes):
        #     split_num = len(nodes)
        # num_nodes = len(nodes)
        
        # sizes = [num_nodes // split_num + (1 if i < num_nodes % split_num else 0) for i in range(split_num)] if len(specific_chunks)==0 else specific_chunks
        # acc_sizes = np.cumsum(sizes)
        # #TODO add flop counter here?
        # def split_callback(node):
        #     if node.op == 'placeholder': return 0
        #     if node.op == 'output': return split_num - 1
            
        #     try:
        #         idx = nodes.index(node)
        #         node_idx = split_num - 1
        #         for t_ind, t in enumerate(acc_sizes):
        #             if idx < t:
        #                 node_idx = t_ind
        #                 break
        #         return node_idx
        #     except ValueError:
        #         return 0

        # split_gm = split_module(traced, self.model, split_callback)
        # self.split_model = {int(k.split("_")[-1]): v for k,v in split_gm.named_children()}
        # #somewhat forced Garbage Collection
        # # print(self.split_model)
        # self.model = PipelineStage(self.split_model[rank], rank, split_num, self.device)
        # self.split_model={}

        #Using default pipeline too memory heavy? -> Nope, but kinda slower? 
        #we need to stick with this to do the scheduling correctly though unfortunately : /
        #with this we also create the stage communications as well
        #and then trigger early gc by overwriting variables hopefully
        split_model = {k:v for k,v in self.model.named_children()}
        if split_num > len(split_model):
            split_num = len(split_model)
        
        sizes = [len(split_model) // split_num + (1 if i < len(split_model) % split_num else 0) for i in range(split_num)] if len(specific_chunks)==0 else specific_chunks
        split_spec = {}
        split_names = list(split_model.keys())
        r=0
        counter=0
        while counter < len(sizes) and r+sizes[counter] < len(split_model):
            temp_key = split_names[r+sizes[counter]]
            split_spec[temp_key] = SplitPoint.END
            r+=sizes[counter]
            counter+=1
        if split_names[-1] not in split_spec:
            split_spec[split_names[-1]] = SplitPoint.END

        pipe = pipeline(self.model, mb_args=(example_input,), split_spec=split_spec )
        #hard code for now
        #TODO this is where the dag maker code shows up or csv reader whichever is easier
        #input count is number of micro-batches
        #based on input count and mb count create a pipeline/dag 
        # custom_pipe = CustomPipeline(nx.DiGraph(), [])
        exec_pipe = nx.DiGraph()
        stage_pipe = nx.DiGraph()
        #only fwds
        stage_list = []
        unit_map = {}
        if split_num==1:
            temp_stage = CustomStage(pipe.get_stage_module(rank), f"fw_s_{rank}", rank, rank)
            stage_pipe.add_node(temp_stage)
            stage_list = [temp_stage]
            unit_map[rank] = rank
        else:
            for s in range(split_num-1):
                temp_stage = CustomStage(pipe.get_stage_module(s), f"fw_s_{s}", s, s)
                temp_stage_next = CustomStage(pipe.get_stage_module(s+1), f"fw_s_{s+1}", s+1, s+1)
                exec_pipe.add_node(temp_stage)
                exec_pipe.add_node(temp_stage_next)
                exec_pipe.add_edge(temp_stage, temp_stage_next)
                #arrival time (proxy with stage for now)
                unit_map[s]=s
                unit_map[s+1]=s+1
                if s == rank:
                    stage_pipe.add_node(temp_stage)
                    stage_list.append(temp_stage)
                    for n in exec_pipe.successors(temp_stage):
                        stage_pipe.add_edge(temp_stage, n)
                    for p in exec_pipe.predecessors(temp_stage):
                        stage_pipe.add_edge(p, temp_stage)
            
            #add the last stage
            if rank == split_num-1:
                temp_stage = CustomStage(pipe.get_stage_module(rank), f"fw_s_{rank}", rank, rank)
                stage_pipe.add_node(temp_stage)
                stage_list.append(temp_stage)
                for n in exec_pipe.successors(temp_stage):
                    stage_pipe.add_edge(temp_stage, n)
                for p in exec_pipe.predecessors(temp_stage):
                    stage_pipe.add_edge(p, temp_stage)



        custom_pipe = CustomPipeline(exec_dag=stage_pipe, stage_list=stage_list, unit_map=unit_map)
        self.exec_pipe = custom_pipe
            
        self.model = pipe.get_stage_module(rank)
        #can implement compile or jit here potentially to further speed up inference
    
    def load_image_tensor(self, path: str, preprocess: Any) -> torch.Tensor:
        #weights = ResNet18_Weights.DEFAULT
        #preprocess = weights.transforms()

        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(self.device)  # [1,3,224,224]
        return x

    def top1_label(self, categories:Any, logits: torch.Tensor) -> str:
        #weights = ResNet18_Weights.DEFAULT
        #categories = weights.meta["categories"]
        idx = int(logits.argmax(dim=1).item())
        return f"{idx}: {categories[idx]}"

    def save_model(self):
        pass

@dataclass
class CustomStage:
    op: nn.Module
    stage_id: str
    rank: int
    arrival_time: float

    def __hash__(self):
        return hash(self.stage_id)
    def __eq__(self, other):
        return self.stage_id == other.stage_id
    def __repr__(self):
        return self.stage_id

@dataclass
class CustomP2POp:
    op: dist.P2POp #tells source, dest, vector etc, initially placeholder so some info is repeated
    src: int
    dst: int
    blocking: bool #if communication needs force wait
    local: bool #if communication can happen via ipc

@dataclass 
class CustomP2PCommunication:
    #fwd means communication going from s -> s/s+1 only
    #no scenario for broadcast here so we just don't add it
    fwd_send_ops: Dict[str, List[CustomP2POp]] = field(default_factory=dict)
    fwd_recv_ops: Dict[str, List[CustomP2POp]] = field(default_factory=dict)
    #bwd can be both SGD p2p and ZOO broadcast
    #bwd just means communication going from higher rank to lower rank 
    #doesn't mean s+1 -> s but any si -> sj where i >= j
    bwd_send_ops: Dict[str, List[CustomP2POp]] = field(default_factory=dict)
    bwd_recv_ops: Dict[str, List[CustomP2POp]] = field(default_factory=dict)
    # stage_to_rank: Dict[str, int] = field(default_factory=dict)
    rank: int=-1
    #str key -> "stage_timeunit/batchunit"
    #legal stage communications -> s->s/s+1 for fwd, si -> sj where i >=j 
    #rank -> let's us optimize comms a little bit (see TODO below) 
    
    #TODO small optimization possible during send/recv
    #if p2pop and ranks of tasks the same (i.e same stage) 
    #can transfer as ipc rather than over the network

    def punch_out_comms(self, dag: nx.DiGraph, stage_list: List[CustomStage], unit_map: Any):
        #comms just needs the stage node and (and full dag inherently)
        #get successors -> map to sends -> never blocking
        #get predeccessors -> map to recv, recv almost always blocking -> depends on unit_map
        #e.g
        # StageA: f1, f2, f3, f1+, , f2+, , f3+ -> one iteration w/ microbatch for zoo -> either force f1+ as occuring in unit 4
        # or
        # StageA: f1, , , f1+ -> one iteration w/o microbatch for zoo -> still force f1+ as occuring in unit 4
        #unit is technically time but using it as a proxy of order of events than anything
        #stage_list: {nx.Nodes......}} nx.Nodes contain stage info(name, duration, arrival_time), successors, predecessors,
        #arrival_time used as proxy to calculate "unit" -> unit here is index of arrival time in unit_list
        #successors of node defined fwd send list events -> all sends non-blocking -> batched with blocking recvs
        #predecessors of node defined fwd recv list events -> all recvs usually blocking -> if an event forced to run at some unit recv has to be blocking 
        #for f1+ events -> bwd list filled (w/ same reasoning as above)
        for src in stage_list:
            # print(src)
            if "fw" in src.stage_id:
                raw_fwd_events_list = dag.successors(src)
                for raw_fwd_event in raw_fwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank #self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_fwd_event.rank #self.stage_to_rank[raw_fwd_event.stage_id]
                    fwd_event = CustomP2POp(None, src_rank, dst, False, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.fwd_send_ops:
                        self.fwd_send_ops[f"{src}_{src_unit}"] = []
                    self.fwd_send_ops[f"{src}_{src_unit}"].append(fwd_event)
                
                raw_fwd_events_list = dag.predecessors(src)
                for raw_fwd_event in raw_fwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank #self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_fwd_event.rank #self.stage_to_rank[raw_fwd_event.stage_id]
                    fwd_event = CustomP2POp(None, src_rank, dst, False, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.fwd_recv_ops:
                        self.fwd_recv_ops[f"{src}_{src_unit}"] = []
                    self.fwd_recv_ops[f"{src}_{src_unit}"].append(fwd_event)

                
            else:
                raw_bwd_events_list = dag.successors(src)
                for raw_bwd_event in raw_bwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank #self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_bwd_event.rank #self.stage_to_rank[raw_bwd_event.stage_id]
                    bwd_event = CustomP2POp(None, src_rank, dst, False, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.bwd_send_ops:
                        self.bwd_send_ops[f"{src}_{src_unit}"] = []
                    self.bwd_send_ops[f"{src}_{src_unit}"].append(bwd_event)
                raw_bwd_events_list = dag.predecessors(src)
                for raw_bwd_event in raw_bwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank#self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_bwd_event.rank #self.stage_to_rank[raw_bwd_event.stage_id]
                    bwd_event = CustomP2POp(None, src_rank, dst, False, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.bwd_recv_ops:
                        self.bwd_recv_ops[f"{src}_{src_unit}"] = []
                    self.bwd_recv_ops[f"{src}_{src_unit}"].append(bwd_event)

    def simulate_exec(self, fname:str = "sim_exec.csv", additional_inputs:int = 0):
        #write into csv 
        #headers are the units 
        #additional units just means how many times to punch out the pattern
        if self.rank == 0:
            f=open(fname, "w")
            #write the ops per "unit"
            sorted_fwd_sends = sorted(list(self.fwd_send_ops.keys()))
            sorted_bwd_recvs = sorted(list(self.bwd_recv_ops.keys()))
            set_units = set(sorted_fwd_sends + sorted_bwd_recvs)

            total_units = sorted([i for i in set_units])
            f_str = ""
            for unit in total_units:
                num = int(unit.split("_")[-1])
                unit_commas = ''.join([',']*num)
                if unit in sorted_fwd_sends:
                    unit_tasks  = "_".join([str(i.src) for i in self.fwd_send_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.fwd_send_ops[unit]])
                    f_str += f"{unit_commas}get input+run({unit_tasks})+batch_send_fwd({unit_comms})"
                if unit in sorted_bwd_recvs:
                    unit_tasks  = "_".join([str(i.src) for i in self.bwd_recv_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.bwd_recv_ops[unit]])
                    f_str += f"{unit_commas}batch_bwd_recv_block({unit_comms})+run({unit_tasks})"

            f.write(f"{f_str}\n")
            f.close()

                
        else:
            f=open(fname)
            while len(f.readlines())!=self.rank:
                time.sleep(5)
                f=open(fname)

            f=open(fname, "a+")
            
            #write the ops per "unit"
            sorted_fwd_recvs = sorted(list(self.fwd_recv_ops.keys()))
            sorted_fwd_sends = sorted(list(self.fwd_send_ops.keys()))
            sorted_bwd_recvs = sorted(list(self.bwd_recv_ops.keys()))
            sorted_bwd_sends = sorted(list(self.bwd_send_ops.keys()))
            total_units = sorted_bwd_recvs + sorted_bwd_sends + sorted_fwd_recvs + sorted_fwd_sends
            total_units = sorted([i for i in set(total_units)])
            
            f_str = ""
            for unit in total_units:
                num = int(unit.split("_")[-1])
                unit_commas = ' '.join([',']*num)
                if unit in sorted_fwd_recvs:
                    unit_tasks = "_".join([str(i.src) for i in self.fwd_recv_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.fwd_recv_ops[unit]])
                    f_str += f"{unit_commas}batch_fwd_recv_block({unit_comms})+run({unit_tasks})"
                if unit in sorted_fwd_sends:
                    unit_tasks = "_".join([str(i.src) for i in self.fwd_send_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.fwd_send_ops[unit]])
                    f_str += f"+batch_send({unit_comms})"
                if unit in sorted_bwd_recvs:
                    unit_tasks = "_".join([str(i.src) for i in self.bwd_recv_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.bwd_recv_ops[unit]])
                    f_str += f"{unit_commas}batch_bwd_recv_block({unit_comms})+run({unit_tasks})"
                if unit in sorted_bwd_sends:
                    unit_tasks = "_".join([str(i.src) for i in self.bwd_send_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.bwd_send_ops[unit]])
                    f_str += f"+batch_send_bwd({unit_comms})"
                f.write(f"{f_str}\n")
            f.close()

            # if len(sorted_bwd_recvs)>0:
            #     f=open(fname, "r") #read first
            #     lines = f.readlines()
            #     f_str=''
            #     line_to_edit = lines[self.rank]
            #     for unit in sorted_fwd_sends:
            #         unit_commas = ' '.join([',']*unit)
            #         unit_tasks = "_".join([str(i.src) for i in sorted_fwd_sends[unit]])
            #         unit_comms = "_".join([str(i.dst) for i in sorted_fwd_sends[unit]])
            #         f_str = f"{unit_commas}batch_recv_block({unit_comms})+run({unit_tasks})"
            #         line_to_edit+=f_str
            #     lines[self.rank] = line_to_edit
            #     f = open(fname, "w") #overwrite with corrected info
            #     f.writelines(lines)
            #     f.close()

            



@dataclass
class CustomPipeline:
    exec_dag: nx.DiGraph
    stage_list: List[Any]
    unit_map: Any

    def exec_line(self):
        pass
    #needs to do three things 
    #first: set up comms and waits  (p2pop or rpc call) -> p2pop lower level, easier to manipulate
    #second: represent input as micro-batches if required 
    #third: take in stage, rank and world size, schedule stage based on this


    # dag: Any = None 
    # #networkx dag graph here, dag contains specific node object 
    # #containing stage information which is 
    # #akin to rank here, mb_id and action type
    # metrics: Any = None
    # rank: Any = None
    # math_start_map = field(default_factory=dict)
    # math_duration_map = field(default_factory=dict)

    # def add_hooks(self):
    #     def pre_hook(module, input):
    #         self.math_start = time.perf_counter_ns()
        
    #     def post_hook(module, input, output):
    #         self.math_duration = (time.perf_counter_ns() - self.math_start) / 1e6

    #     self.stage.submod.register_forward_pre_hook(pre_hook)
    #     self.stage.submod.register_forward_hook(post_hook)
    
    # def _get_schedule(self):
    #     #so for a particular rank in the stage,
    #     #define the pipeline/tasks/action it needs to run 
    #     #actions are "forward" and "backward"
    #     #but we only care about "forward" for inference for now
         
    #     actions = []
    #     global_order = list(nx.topological_sort(self.dag))
    #     for node_data in global_order:            
    #         if node_data.rank == self.rank:
    #             actions.append(Action(
    #                 mb_id=node_data.mb_id, 
    #                 step_type=node_data.type
    #             ))
                
    #     return actions



@dataclass
class FBModel(GenModel):

    def custom_pipeline_inf(self, world, rank):
        # we have a stage module from the split operation 
        # now we need to "schedule" this stage
        # each stage only schedules it's tasks from the dag 
        # pytorch usually handles this within PipelineStage itself
        # but unfortunately for us, given that we have some custom communications 
        # we can't really use this 

        # custom_stage = CustomStage(self.model, rank, rank)
        custom_comms = CustomP2PCommunication(rank=rank)
        
        custom_comms.punch_out_comms(self.exec_pipe.exec_dag, self.exec_pipe.stage_list, 
        self.exec_pipe.unit_map)
        custom_comms.simulate_exec()

        pass





    
    def pipeline_inference(self, world, rank, warmup, iters, x0=None):
        #TODO eventually replace with CustomSchedule and CustomSteps

        #essentially the main from pipeline_splitting_inf.py
        #we include assumed rank from args to garbage collect the rest of the model quickly
        total_start = 0
        outputs = None
        #rank = args.assigned_rank
        next_rank = rank+1
        prev_rank = rank-1
        stage=self.model.to(self.device).eval()
        stage_start = 0
        stage_end = 0
        print(f"Staring inference for rank={rank}: {datetime.strftime(datetime.now(), '%H:%M:%S.%f')}")
        compute_time = []
        comm_time = []
        with torch.inference_mode():
            # stage_start = time.perf_counter()
            for i in range(warmup + iters):
                recv_start = 0
                recv_end = 0
                send_start_time = 0
                send_end_time = 0
                if i == warmup and rank == 0:
                    dist.barrier()
                    total_start = time.perf_counter()
                elif i == warmup:
                    dist.barrier()
                    stage_start = time.perf_counter()

                if rank == 0:
                    x=x0
                else:
                    # recv from prev
                    recv_start = time.perf_counter()
                    ndim_t = torch.empty(1, dtype=torch.int64)
                    dist.recv(ndim_t, src=prev_rank)
                    ndim = int(ndim_t.item())

                    shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
                    dist.recv(shape_dtype, src=prev_rank)
                    shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
                    dtype = self.code_to_dtype(int(shape_dtype[-1].item()))

                    x = torch.empty(shape, dtype=dtype, device=self.device)
                    dist.recv(x, src=prev_rank)
                    recv_end = time.perf_counter()

                y=[]
                if i == warmup-1:
                    flop_counter = FlopCounterMode(display=False, depth=None)
                    with flop_counter:
                            y=stage(x)
                    self.total_flops =  flop_counter.get_total_flops()                    
                else:
                    compute_start = time.perf_counter()
                    y=stage(x)
                    compute_end = time.perf_counter()
                    compute_time.append(compute_end-compute_start)
                
                send_start_time = time.perf_counter()
                if next_rank < world:
                    dist.send(torch.tensor([y.dim()], dtype=torch.int64), dst=next_rank)
                    dist.send(torch.tensor([*y.shape, self.dtype_to_code(y.dtype)], dtype=torch.int64), dst=next_rank)
                    dist.send(y.contiguous(), dst=next_rank)
                else:
                    outputs = y
                send_end_time = time.perf_counter()
                comm_time.append([recv_end-recv_start, send_end_time-send_start_time])

            stage_end = time.perf_counter()
            # print(f"Stage elapsed time per iter rank {rank}:{stage_end-stage_start}")
        dist.barrier()
        # stage_end = time.perf_counter()
        stage_elapsed = (stage_end - stage_start) if rank!=0 else (stage_end - total_start)
        print(f"\n--- Pipeline Inference (stages={world}, iters={iters}, rank={rank}) ---")
        # print(f"Ending inference for rank={rank}: {datetime.strftime(datetime.now(), '%H:%M:%S.%f')}")
        print(f"Only compute time per rank={rank} : {np.mean(compute_time[warmup:]):.4f} s")
        print(f"Only comm time per rank={rank} : {np.max([i[0]+i[1] for i in comm_time[warmup:]]):.4f} s recv_time: {np.max([i[0] for i in comm_time[warmup:]]):.4f} s send_time:{np.max([i[1] for i in comm_time[warmup:]]):.4f} s")
        print(f"Time (timed iters) stages={world} iters={iters}, rank={rank} : {stage_elapsed:.4f} s")
        print(f"Avg latency per image stages={world} iters={iters}, rank={rank}, warmup={warmup}: {(stage_elapsed/max(1,iters))*1000:.2f} ms")
        print(f"FLOP count stages={world} iters={iters}, rank={rank}, warmup={warmup} : {self.total_flops} ")
        print(f"FLOP s/op rate stages={world} iters={iters}, rank={rank}, warmup={warmup} : {(stage_elapsed/(max(1,iters)*max(1,self.total_flops)))*10**9} Gs/op")
        if rank == 0:
            total_end = time.perf_counter()
            elapsed = total_end - total_start
            avg = elapsed / max(1, iters)
            print(f"\n--- Pipeline Inference (stages={world}, iters={iters}) total ---")
            print(f"Time (timed iters) total: {elapsed:.4f} s")
            print(f"Avg latency per image total: {avg*1000:.2f} ms")
            # print("Note: Single-image pipeline mostly demonstrates partitioning, not big speedups.")

        #if rank == world - 1 and outputs is not None:
        #    print(f"[rank {rank}] pred {self.top1_label(outputs)}")

        return outputs

    def train_model(self, train_loader: Any, lr=1e-3, num_epochs:int=10):
        model=self.model
        # Optimizer setup
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        epoch = -1
        while epoch < num_epochs:
            epoch+=1
            model.train()
            pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch} Training")
            s=time.time()
            total=0
            correct = 0
            
            for i, (inputs, labels) in enumerate(pbar):
            # Benchmark training with frozen layers
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix_str(f"Lr {lr:.2e} Loss: {round(loss.item(),3)} Acc {100*(correct/total):.2f}%")
            e = time.time()
            epoch_time = e-s
            pbar.set_postfix_str(f"Epoch time: {epoch_time}")
    
    def evaluate_model(self, test_loader: Any):
        model = self.model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        return accuracy


