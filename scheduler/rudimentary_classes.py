from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
import model_splitter
import numpy as np

@dataclass
class ModelSplitWrapper:
    @staticmethod
    def split(splits, model_name, export, flop_w, comm_w):
        if model_name=="resnet18":
            from torchvision.models import resnet18
            model = resnet18(weights=None).eval()
            splitter = model_splitter.FlopAwareResNet18PipelineSplitter(model, input_shape=(2, 3, 224, 224))

            result = splitter.split_by_flops_pipeline(
                flop_percentages=splits,
                lookahead=5,
                out_dir="./resnet18_splits",
                meta_name="resnet_meta.json",
                w_flop=flop_w,
                w_net=0,
                w_comm=comm_w,
                export=export
            )
            return result

        elif model_name=="vit":
            from torchvision.models import vision_transformer

            model = vision_transformer.vit_b_16(weights=None).eval()
            splitter = model_splitter.FlopAwareViTPipelineSplitter(model, input_shape=(2, 3, 224, 224))

            result = splitter.split_by_flops_pipeline(
                flop_percentages=splits,
                lookahead=5,
                out_dir="./vit_splits",
                meta_name="meta.json",
                w_flop=flop_w,
                w_net=0,
                w_comm=comm_w,
                export=export
            )
            return result

        if model_name=="tcn":
            import tcn_library as tcn
            model = tcn.SensorTCN(
            num_channels=8,
            hidden_channels=256,
            levels=8,
            kernel_size=5,
            output_channels=8,
            ).eval()

            splitter = model_splitter.FlopAwareTCNPipelineSplitter(model, input_shape=(2, 2048, 8))

            result = splitter.split_by_flops_pipeline(
                flop_percentages=splits,
                lookahead=5,
                out_dir="./tcn_splits",
                meta_name="meta.json",
                w_flop=flop_w,
                w_net=0,
                w_comm=comm_w,
                export=export
            )
            return result

@dataclass
class Job:
    id: str
    arrival: int
    wait: int
    model: str
    input_size: int
    best_rate: float
    tasks: List[Task] = None

    def even_split(self, num_splits, bs=1, export=False, bw=240):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        self.tasks=[]

        for r_ind, r in enumerate(result["splits"]):
            task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)
    
    def comm_split(self, num_splits, bs=1, export=False, bw=240):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=0, comm_w=1)
        self.tasks=[]

        for r_ind, r in enumerate(result["splits"]):
            task = task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)

    def custom_split(self, splits, bs=1, export=False, bw=240):
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        self.tasks=[]
        
        for r_ind, r in enumerate(result["splits"]):
            task = task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)


    def total_flops(self):
        if self.tasks==None:
            return sum([task.flop for task in self.tasks])

    #TODO:
    #with jobs, iterate over batch size, decide input and time taken based on interference
    #also use valid wait time list per batch size, batch num combo
    #long iterations but it's okay, after add to device and update all relevant tasks
    #add a tick_tock function either in Device or Subcluster
    def cost_function_explorer(self, subcluster:Subcluster, flag="even", custom=None, bw=240, k=5):
        assignment_info=[]
        #wait time, splits of tasks, bs, bn per iteration
        for bs in range(1, self.input_size+1):
            bn = np.ceil(self.input_size/bs)
            if bn*bs != self.input_size:
                continue
            #comp time
            old_len = 0
            # for splits in range(1, subcluster.total_devs+1):
            for splits in [5]:
                if flag=="even":
                    self.even_split(splits, bs=bs, bw=bw)
                elif flag=="comm":
                    self.comm_split(splits, bs=bs, bw=bw)
                elif flag=="custom" and custom!=None:
                    self.custom_split(custom, bs=bs, bw=bw)
                else:
                    raise Exception("Incorrect options for cost function exploration please ensure all appropriate options filled")
                if len(self.tasks) <= old_len:
                    break
                print([str(i) for i in self.tasks], [i.flop for i in self.tasks], len(self.tasks))
                old_len = len(self.tasks)
                #with the current tasks now in job, figure out best possible throughput 
                #either through running immediately or running after waiting 
                #wait time counts for throughput calculation
                # print(bn)
                m = subcluster.assign_exploration(tasks=self.tasks, batch_num=int(bn))
                # print(m)
                # sorted_m = {k: m[k] for k in sorted(list(m.keys()))}
                sorted_map = {k:round(v,3) for k,v in sorted(m.items(), key=lambda x: x[1]+x[0])}

                best_throughput = round(float(bn*bs / sorted_map[list(sorted_map.keys())[0]]),3)
                info = {"bs":bs, "bn":int(bn), "splits":splits, "best_throughput": best_throughput, "top_k": {k: v for k,v in list(sorted_map.items())[:k]}}
                print(info)
                assignment_info.append(info)
        assignment_info = sorted(assignment_info, key=lambda x: x["best_throughput"], reverse=True)
        return assignment_info


@dataclass
class Task:
    id: str
    job: Job
    flop: int
    peak_rate: float
    output_bytes: int
    task_arrival: int
    task_wait: int
    task_remaining_time: int #includes both wait+run time
    cur_tf: float #hard stopped at 0.5, if greater than 0.5, bring down to 0.5
    device: Device
    peak_bw: float

    
    def __str__(self):
        return self.id
    
    def __eq__(self, other):
        return self.task_remaining_time == other.task_remaining_time
    
    def __lt__(self, other):
        return self.task_remaining_time < other.task_remaining_time
    
    def __gt__(self, other):
        return self.task_remaining_time > other.task_remaining_time



@dataclass
class Subcluster:
    name: str
    total_devs: int
    devices: List[Device] = None

    @classmethod
    def setup_subcluster(cls, name, ut, t, v):
        #make this specific ids instead of just number?
        # cls.name = name
        devices=[]
        for d in range(ut+t+v):
            if d<ut:
                dev = Device(cls, d, "ut", [])
                devices.append(dev)
            elif d<ut+v:
                dev = Device(cls, d, "v", [])
                devices.append(dev)
            elif d<ut+v+t:
                dev = Device(cls, d, "t", [])
                devices.append(dev)
        total_devs=ut+t+v
        return cls(name, total_devs, devices)
    
    def instance_map(self, wait=0):
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        device_map = {"ut":0, "t":0, "v":0}
        for d in self.devices:
            if len(d.tasks)==0:
                device_map[d.dev_type]+=1
                continue
            t = sorted(d.tasks)[-1] 
            #last running task on device, 
            #tasks binpacked so we only care about the last one
            if wait >= t.task_remaining_time:
                #if last task's remaining time is <= wait time
                #device free within wait duration
                device_map[d.dev_type]+=1
        return device_map

    def interferences(self, wait=0, overlap_duration=0):
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        interfering_tasks = {}
        for d in self.devices:
            if d.dev_type=="t":
                continue
            tasks=sorted(d.tasks)
            for t in tasks:
                if wait >= t.task_remaining_time:
                    #doesn't interfere if wait time exceed remaining time
                    continue
                if t.task_arrival <= wait+overlap_duration:
                    #if within overlap duration, will interfere
                    if d.device_name() not in interfering_tasks:
                        interfering_tasks[d.device_name()] = []
                    interfering_tasks[d.device_name()].append(t)
                    
        return interfering_tasks

    def valid_wait_times(self):
        #return list of minimum wait times required for any change in device compositions
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        wait_times = []
        for d in self.devices:
            if len(d.tasks)==0:
                continue
            t=sorted(d.tasks)[-1]
            wait_times.append(t.task_remaining_time)
        return sorted(wait_times)


    def tf_calc(self, w, n):
        y=0
        if w < 8:
            y = 0.157 - 0.004*n + 0.075*w
        else:
            y = 0.667 - 0.008*n + 0.023*w
        return y
    
    def time_predictor(self, peak_times, fp, bg_load=0):
        t = len(self.devices)
        maybe_gnt = [0]*len(peak_times)
        achieved_fps = [i for i in peak_times]
        if len(peak_times)==1 and bg_load==0:
            return achieved_fps
        elif len(peak_times)==1:
            val = self.tf_calc(len(peak_times)+bg_load, t)
            val = val if val < 0.3 else 0.3 #fixed upper limit
            gnt = (peak_times[0]-maybe_gnt[0])/(1-val)
            achieved_fps = [gnt]
            return achieved_fps

        for ind, i in enumerate(peak_times):
            i = i - peak_times[ind-1] if ind > 0 else i
            if i>0 and len(peak_times)-ind>0:
                val = self.tf_calc(len(peak_times)-ind+bg_load, t)
                val = val if val < 0.3 else 0.3 #fixed upper limit
                # gnt = (i-maybe_gnt[ind])/(1-val) #if tf = (t-p)/t
                gnt = (i)/(1-val) #if i > 0 else i / (i-val) #if tf = (t-p)/t
                for k in range(ind, len(maybe_gnt)):
                    maybe_gnt[k]=maybe_gnt[k]+gnt
                    achieved_fps[k] = maybe_gnt[k] + (fp[k]-fp[ind])*peak_times[k]/fp[k]
        return achieved_fps
    
    def assign_exploration(self, tasks: List[Task], batch_num: int):
        #explores assignments for all possible wait times and returns assignment with highest throughput
        wait_times = [0] + self.valid_wait_times() 
        # intra_interference = [] #the pipeline shape for the job, given all the tasks it interferes within itself

        rank_to_time_step_map={r:[0]*r+[tasks[r]]*batch_num+[0]*(len(tasks)-r-1) for r in range(len(tasks))}
        # for time_step in range(len(rank_to_time_step_map[0])): #always has rank 0 -> single ml model, no splits
        #     slice_fp = [rank_to_time_step_map[r][time_step] for r in rank_to_time_step_map]
        #     slice_fp = [s for s in slice_fp if s!=0]
        #     intra_interference.append([i for i in slice_fp])
        time_map = {}
        # print(wait_times, len(tasks))
        for w in wait_times:
            if w not in time_map:
                time_map[w]=np.inf
            else:
                continue
            device_comp = self.instance_map(w)
            if len(tasks) <= sum(device_comp.values()):
                #temporarily assign tasks to device types
                #greedily assign longest flop task to best available device type ut -> v -> t
                bg_load = self.total_devs - sum(device_comp.values()) #other busy devices
                #get the list of flops available, remove ones on ut and t devices since they're always the worst
                #but count them as bg_load instead
                fp = []
                #sort tasks and map them to device types
                flop_sorted_tasks = sorted(tasks, key = lambda x: x.flop, reverse=True)
                task_mapping = {f"{t}":0 for t in flop_sorted_tasks}
                r=0
                while r<len(flop_sorted_tasks) and task_mapping[f"{flop_sorted_tasks[r]}"]==0:
                    if device_comp['ut']>0:
                        task_mapping[f"{flop_sorted_tasks[r]}"]="ut"
                        device_comp['ut']-=1
                    elif device_comp['v']>0:
                        task_mapping[f"{flop_sorted_tasks[r]}"]="v"
                        device_comp['v']-=1
                    elif device_comp['t']>0:
                        task_mapping[f"{flop_sorted_tasks[r]}"]="t"
                        device_comp['t']-=1
                    r+=1
                    
                

                # for u in range(device_comp["ut"]):
                #     if task_mapping[flop_sorted_tasks[u]] == 0:
                        # task_mapping[flop_sorted_tasks[u]] = "ut"
                # for v in range(device_comp[])
                accumulated_time = 0
                # print(device_comp, task_mapping)
                print([str(i) for i in rank_to_time_step_map[0]])
                # print( len(rank_to_time_step_map[0]), batch_num)
                for time_slice in range(len(rank_to_time_step_map[0])):

                    fp = [rank_to_time_step_map[r][time_slice].flop for r in rank_to_time_step_map 
                    if str(rank_to_time_step_map[r][time_slice])!='0' and task_mapping[f"{rank_to_time_step_map[r][time_slice]}"]!="ut" ] #do we count t devices as well?
                    # print(fp, task_mapping) #, rank_to_time_step_map)
                    non_fp = [rank_to_time_step_map[r][time_slice].flop for r in rank_to_time_step_map 
                    if str(rank_to_time_step_map[r][time_slice])!='0']

                    temp_bg_load=bg_load+len(non_fp)-len(fp)
                    all_fp_nw = [rank_to_time_step_map[r][time_slice].output_bytes/rank_to_time_step_map[r][time_slice].peak_bw for r in rank_to_time_step_map 
                    if str(rank_to_time_step_map[r][time_slice])!='0' and r!=len(tasks)-1]
                    # print(non_fp, task_mapping, time_slice) #, rank_to_time_step_map)
                    # print(fp, task_mapping, rank_to_time_step_map.keys())
                    peak_r = tasks[0].peak_rate
                    print(max([f*10**-9/peak_r for f in non_fp]))
                    # print(self.time_predictor([f*10**-9/2 for f in fp], fp, bg_load), fp)
                    accumulated_time+=max(self.time_predictor([f*10**-9/peak_r for f in fp], fp, temp_bg_load)+[f*10**-9/peak_r for f in non_fp] ) + max(all_fp_nw+[0])

                time_map[w]=accumulated_time
        return {k:v for k,v in sorted(time_map.items(), key=lambda x: x[1])}

@dataclass
class Device:
    subcluster: Subcluster
    device_id: int
    dev_type: str #"ut, t, v"
    tasks: List[Task] 

    def device_name(self):
        return f"{self.subcluster.name}-{self.device_id}"
    
    def assigned(self, tasks:List[Task]):
        for t in tasks:
            t.device = self
        self.tasks.extend(tasks)

if __name__=="__main__":
    import time
    subcluster = Subcluster.setup_subcluster("dummy", 5, 5, 5)
    # print(subcluster.total_devs)
    j = Job("j0", 0, 0, "resnet18", 10, 3, [] )
    s=time.time()
    comp_a = j.cost_function_explorer(subcluster, flag="even")
    print(comp_a)

    print(time.time() - s)
    print()
    exit()

    s=time.time()
    comm_a = j.cost_function_explorer(subcluster, flag="comm")
    print(comm_a)
    print(time.time() - s)

