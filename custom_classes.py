from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
import networkx as nx
import time 

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

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
                #fwd sends -> non-blocking 
                raw_fwd_events_list = dag.successors(src)
                # print(self.rank, src, [r for r in raw_fwd_events_list])
                # exit()
                for raw_fwd_event in raw_fwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank #self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_fwd_event.rank #self.stage_to_rank[raw_fwd_event.stage_id]
                    fwd_event = CustomP2POp(None, src_rank, dst, False, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.fwd_send_ops:
                        self.fwd_send_ops[f"{src}_{src_unit}"] = []
                    self.fwd_send_ops[f"{src}_{src_unit}"].append(fwd_event)
                
                #fwd recvs -> blocking
                raw_fwd_events_list = dag.predecessors(src)
                for raw_fwd_event in raw_fwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank #self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_fwd_event.rank #self.stage_to_rank[raw_fwd_event.stage_id]
                    fwd_event = CustomP2POp(None, src_rank, dst, True, (src_rank==dst))
                    if f"{src}_{src_unit}" not in self.fwd_recv_ops:
                        self.fwd_recv_ops[f"{src}_{src_unit}"] = []
                    self.fwd_recv_ops[f"{src}_{src_unit}"].append(fwd_event)

                
            else:
                #bwd sends -> non-blocking
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
                #bwd recvs -> blocking
                raw_bwd_events_list = dag.predecessors(src)
                for raw_bwd_event in raw_bwd_events_list:
                    #placeholder for dist.p2pop for now
                    src_rank = src.rank#self.stage_to_rank[src.stage_id] 
                    src_unit = unit_map[src.arrival_time]
                    dst = raw_bwd_event.rank #self.stage_to_rank[raw_bwd_event.stage_id]
                    bwd_event = CustomP2POp(None, src_rank, dst, True, (src_rank==dst))
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
            # num = int(unit.split("_")[-1])
            # unit_commas = ' '.join([',']*num)
            # f_str = f"{unit_commas}"
            prev_num = 0
            for unit in total_units:
                num = int(unit.split("_")[-1]) + int(unit.split("_")[1]) - prev_num
                unit_commas = ''.join([',']*num)
                if unit in sorted_fwd_sends:
                    unit_tasks  = "_".join([str(i.src) for i in self.fwd_send_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.fwd_send_ops[unit]])
                    f_str += f"get input+run({unit_tasks})+batch_send_fwd({unit_comms})"
                if unit in sorted_bwd_recvs:
                    unit_tasks  = "_".join([str(i.src) for i in self.bwd_recv_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.bwd_recv_ops[unit]])
                    f_str += f"{unit_commas}batch_bwd_recv_block({unit_comms})+run({unit_tasks})"
                prev_num=num
                f_str+="," #only for rank=0
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
            # print(total_units)
            f_str = ""
            # num = int(unit.split("_")[-1])
            # unit_commas = ' '.join([',']*num)
            # f_str = f"{unit_commas}"
            prev_num = 0
            for unit in total_units:
                num = int(unit.split("_")[-1]) + int(unit.split("_")[1])
                unit_commas = ' '.join([',']*(num-prev_num))
                # print(self.rank, num, prev_num)
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
                    f_str += f"batch_bwd_recv_block({unit_comms})+run({unit_tasks})"
                if unit in sorted_bwd_sends:
                    unit_tasks = "_".join([str(i.src) for i in self.bwd_send_ops[unit]])
                    unit_comms = "_".join([str(i.dst) for i in self.bwd_send_ops[unit]])
                    f_str += f"+batch_send_bwd({unit_comms})"
                prev_num=num
                # f_str+=","    
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

            

# def recv_tensor(src: int, device: torch.device) -> torch.Tensor:
#     """
#     Receive tensor header then payload.
#     """
#     header = torch.empty(1 + 32 + 1, dtype=torch.int64)  # oversized, we'll trim
#     # But dist.send requires exact size match, so we instead do a 1st receive for ndim,
#     # then receive shape+dtype with known length.
#     ndim_t = torch.empty(1, dtype=torch.int64)
#     dist.recv(ndim_t, src=src)
#     ndim = int(ndim_t.item())

#     shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
#     dist.recv(shape_dtype, src=src)

#     shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
#     dtype = _code_to_dtype(int(shape_dtype[-1].item()))

#     x = torch.empty(shape, dtype=dtype, device=device)
#     dist.recv(x, src=src)
#     return x

@dataclass
class CustomPipeline:
    exec_dag: nx.DiGraph
    stage_list: List[Any]
    unit_map: Any
    inp_shape: Tuple
    inp_dtype: Any
    device: Any

    def __repr__(self):
        return "_".join([str(i) for i in self.stage_list])
    def __hash__(self):
        return hash("_".join([str(i) for i in self.stage_list]))
    def __eq__(self, other):
        #return 
        return "_".join([str(i) for i in self.stage_list]) == "_".join([str(i) for i in other.stage_list])

    def stager(self, stage_ind, t, q):
        #q.put(4*4)
        print("CHECKING INSIDE")
        #q.put(stage_ind.forward(t))
        q.put(self.stage_list[stage_ind].op.forward(t))
        print("CHECKING INSIDE AFTER")
        #q.put(self.stage_list[stage_ind].op.forward(t))

    def exec_line(self, exepected_inp_count:int, rank:int , world:int, comms: CustomP2PCommunication, inputs = None):
        #first check if any fwd recvs
        # dist.batch_isend_irecv()
        batched_recvs: List[dist.P2POp] = []
        batched_recv_tensors: List[torch.Tensor] = []
        batched_sends: List[dist.P2POp] = []
        sorted_fwd_recvs = sorted(list(comms.fwd_recv_ops.keys()))
        sorted_fwd_sends = sorted(list(comms.fwd_send_ops.keys()))
        sorted_bwd_recvs = sorted(list(comms.bwd_recv_ops.keys()))
        sorted_bwd_sends = sorted(list(comms.bwd_send_ops.keys()))
        output_labels_and_times = []
        copy_stage_list = [i for i in range(len(self.stage_list))]

        for ind in range(exepected_inp_count):
            #wait for fwd first
            # blocking = False
            if len(sorted_fwd_recvs)>0:
                recv_op = sorted_fwd_recvs.pop(0)
                for recv in comms.fwd_recv_ops[recv_op]:
                    # print(comms.fwd_recv_ops[recv_op])
                    src = recv.dst
                    batched_recv_tensors.append(torch.empty(self.inp_shape, dtype=self.inp_dtype))
                    # blocking = recv.blocking
                    recv_p2p = dist.P2POp(dist.irecv, 
                    batched_recv_tensors[-1], src)
                    batched_recvs.append(recv_p2p)
                    # print(recv_p2p)

                    # dist.recv(batched_recv_tensors[-1], src)
            net_start,net_end=0,0
            if len(batched_recvs) > 0 or len(batched_sends)>0:
                # print(f"Trying recv for {rank} with {ind}")
                #print(rank, ind , batched_sends)
                #print(rank, ind , batched_recvs)
                works = dist.batch_isend_irecv(batched_sends+batched_recvs)
                # print(f"rank {rank} and {len(works)} with {ind}")
                net_start = time.perf_counter()
                for w in works:
                    # print(f"{rank} {ind} {w}")
                    w.wait()
                    # print(batched_recv_tensors)
                net_end = time.perf_counter()
                #print(f"rank:{rank} ind:{ind} network time:{(net_end-net_start)}s")
                #print(f"Successful recv for {rank}")
            batched_sends = []            

            output_labels_and_times.append([ind, [], net_end-net_start ])
            #because we are grouping recv and send, sends are automatically async lmao 
            #so blocking as a term is pointless : / -> remove from class
            if rank==0 and inputs!=None:
                # print(inputs[ind].shape)
                # exit()
                batched_recv_tensors.append(inputs[ind])
            #use recv tensors
            batched_send_tensors = []
            #print("CHECKING BEFORE")
            for t in batched_recv_tensors:
                stage_ind = copy_stage_list.pop(0)
                stage = self.stage_list[stage_ind]

                #stage.op.share_memory()
                #list if broadcasted/recomposed?
                # for s in stages:
                #output=[]
                #procs = []
                #q=mp.Queue()
                #for k in range(1):
                #    p = mp.Process(target=self.stager, args=(stage_ind, t, q, ))
                #    p.start()
                #    procs.append(p)
                #for p in procs:
                #    p.join()
                #output = q.get()
                #print("BIG LETTERS HERE")
                #exit()
                #print("Outpout", output)

                #with Pool(processes=4) as pool:
                #    res = list(pool.imap_unordered(self.stager, [(stage, [0])]*4))
                #    output = res[0]
                    #for i in res:
                    #    output = i #change to stack later



                output = stage.op.forward(t)
                batched_send_tensors.append(output)
                #print(f"Successful recv and forward for {rank} with {output.shape}")
            batched_recvs = []
            batched_recv_tensors= []

            if len(sorted_fwd_sends)>0:
                recv_op = sorted_fwd_sends.pop(0)
                # to_send = batched_send_tensors.pop(0)
                for recv in comms.fwd_send_ops[recv_op]:
                    dst = recv.dst
                    #batched_send_tensors.append(torch.empty(self.inp_shape, dtype=self.inp_dtype, device=self.device))
                    # blocking = recv.blocking
                    for to_send in batched_send_tensors:
                        to_send = to_send.contiguous()
                    #assumes stage list and inputs in expected order?
                        send_p2p = dist.P2POp(dist.isend, 
                        to_send, dst)
                        batched_sends.append(send_p2p)

                        # dist.send(to_send, dst)
                        # print(f"Sent! w/ rank {rank} and ind {ind}")
            
            elif len(batched_send_tensors)>0:
                #this means it's final layer
                #return output label tensors
                #TODO for training there might be left-over tasks make sure to add them here too if required
                output_labels_and_times[-1]=[ind, batched_send_tensors[0], net_end-net_start ]
                
            #else:
            #TODO check bwds and repeat above processes
            #TODO be careful of leftover tasks if any
            
        
        #left over sends fired off
        if len(batched_sends)>0:
            works = dist.batch_isend_irecv(batched_sends)
            # print(f"leftovers for rank {rank}")
            for w in works:
                w.wait()

        return output_labels_and_times


        
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
