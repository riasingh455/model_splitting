from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
import model_splitter
import numpy as np
from pathlib import Path
from multiprocessing import Pool

@dataclass
class ModelSplitWrapper:
    @staticmethod
    def split(splits, model_name, export, flop_w, comm_w):
        add = 1 if flop_w==1 else 2
        if model_name=="resnet18":
            from torchvision.models import resnet18
            model = resnet18(weights=None).eval()
            splitter = model_splitter.FlopAwareResNet18PipelineSplitter(model, input_shape=(2, 3, 224, 224))

            result = splitter.split_by_flops_pipeline(
                flop_percentages=splits,
                lookahead=5,
                out_dir=f"./resnet18_splits_{add}",
                meta_name="meta.json",
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
                out_dir=f"./vit_splits_{add}",
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
                out_dir=f"./tcn_splits_{add}",
                meta_name="meta.json",
                w_flop=flop_w,
                w_net=0,
                w_comm=comm_w,
                export=export
            )
            return result

        if model_name=="eff":
            from torchvision.models import efficientnet_b0

            model = efficientnet_b0(weights=None).eval()
            splitter = model_splitter.FlopAwareEfficientNetB0PipelineSplitter(model, input_shape=(2, 3, 224, 224))

            result = splitter.split_by_flops_pipeline(
                flop_percentages=splits,
                lookahead=5,
                out_dir=f"./eff_splits_{add}",
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
    bn: int = 0
    bs: int = 0

    def even_split(self, num_splits, bs=1, export=False, bw=240):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        self.tasks=[]
        self.bn = self.input_size//bs
        self.bs = bs

        for r_ind, r in enumerate(result["splits"]):
            task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*8*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)
        # for i in self.tasks:
        #     print(i.output_bytes)
    
    def comm_split(self, num_splits, bs=1, export=False, bw=240):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=0, comm_w=1)
        self.tasks=[]
        self.bn = self.input_size//bs
        self.bs = bs

        for r_ind, r in enumerate(result["splits"]):
            task = task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*8*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)
        # for i in self.tasks:
        #     print(i.output_bytes)

    def custom_split(self, splits, bs=1, export=False, bw=240):
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        self.tasks=[]
        self.bn = self.input_size//bs
        self.bs = bs

        for r_ind, r in enumerate(result["splits"]):
            task = task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"]*bs, 
            self.best_rate, r["boundary_transfer_bytes"]*bs*8*10**-6, self.arrival, self.wait, np.inf, 0, None, bw  )
            if r["actual_flops"]!=0:
                self.tasks.append(task)


    def total_flops(self):
        if self.tasks==None:
            return sum([task.flop for task in self.tasks])


    def multi_proc_func(self, subcluster, flag, bs, bn, bw, custom, lag_pri, k):
        # for bs in range(1, self.input_size+1):
        #     bn = np.ceil(self.input_size/bs)
        #     if bn*bs != self.input_size:
        #         continue
        #comp time
        old_len = 0
        # for splits in range(1, subcluster.total_devs+1):
        temp_assign = []
        # for splits in range(1, 7):
        export=False if Path(f"{self.model}_splits_{1 if flag=='even' else 2}").is_dir()==True else True
        # for splits in [5]:
        for splits in [3,4,5,6]:
        # for splits in [3]:
            if flag=="even":
                self.even_split(splits, bs=bs, bw=bw, export=export)
            elif flag=="comm":
                self.comm_split(splits, bs=bs, bw=bw, export=export)
            elif flag=="custom" and custom!=None:
                self.custom_split(custom, bs=bs, bw=bw, export=export)
            else:
                raise Exception("Incorrect options for cost function exploration please ensure all appropriate options filled")
            if len(self.tasks) <= old_len:
                break
            # print([str(i) for i in self.tasks], [i.flop for i in self.tasks], len(self.tasks))
            old_len = len(self.tasks)
            #with the current tasks now in job, figure out best possible throughput 
            #either through running immediately or running after waiting 
            #wait time counts for throughput calculation
            # print(bn)
            m = subcluster.assign_exploration(tasks=self.tasks, batch_num=int(bn))
            if len(m)==0:
                continue
            # print(m)
            # sorted_m = {k: m[k] for k in sorted(list(m.keys()))}
            sorted_m = {}
            if lag_pri:
                #latency sort first
                sorted_map = {k:[round(v[0],3), round(v[1],3), round(v[2],3)] for k,v in sorted(m.items(), key=lambda x: x[1][0]+x[0])}
                # print(sorted_map)
                #top k sort on lag now
                sorted_map = {k:[round(v[0],3), round(v[1],3), round(v[2],3)] for k,v in sorted(list(m.items())[:k], key=lambda x: x[1][2])}
            else:
                #top k sort on lag now
                sorted_map = {k:[round(v[0],3), round(v[1],3), round(v[2],3)] for k,v in sorted(m.items(), key=lambda x: x[1][2])}
                #latency sort first
                sorted_map = {k:[round(v[0],3), round(v[1],3), round(v[2],3)] for k,v in sorted(list(m.items())[:k], key=lambda x: x[1][0]+x[0])}
                # print(sorted_map)
            #can change the order based on lag or throughput priority
            best_throughput = round(float(bn*bs / (sorted_map[list(sorted_map.keys())[0]][0] + list(sorted_map.keys())[0]) ),3)
            
            info = {"bs":bs, "bn":int(bn), "splits":splits, "best_throughput": best_throughput, "tf":sorted_map[list(sorted_map.keys())[0]][1],
                "wait":list(sorted_map.keys())[0], "best_runtime":sorted_map[list(sorted_map.keys())[0]][0],
                "top_k": {k: v for k,v in list(sorted_map.items())[:k]}}
            
            # print(info)
            temp_assign.append(info)
        temp_assign = sorted(temp_assign, key=lambda x: x["best_throughput"], reverse=True)
        return temp_assign[0]
    #TODO:
    #with jobs, iterate over batch size, decide input and time taken based on interference
    #also use valid wait time list per batch size, batch num combo
    #long iterations but it's okay, after add to device and update all relevant tasks
    #add a tick_tock function either in Device or Subcluster
    def cost_function_explorer(self, subcluster:Subcluster, flag="even", custom=None, bw=240, k=5, lag_pri=False):
        assignment_info=[]
        #wait time, splits of tasks, bs, bn per iteration
        with Pool(processes=4) as pool:
            # map() blocks until all processes complete and returns a standard list
            # for bs in range(1, self.input_size+1):
        #     bn = np.ceil(self.input_size/bs)
        #     if bn*bs != self.input_size:
        #         continue
            assignment_info = pool.starmap(self.multi_proc_func, [ (subcluster,flag, bs, np.ceil(self.input_size/bs), bw, custom, lag_pri, k,) for bs in [1,2,3,5,10] if bs*np.ceil(self.input_size/bs)==self.input_size ])
        # self.multi_proc_func(flag, bw, custom, lag_pri, k)
        
        assignment_info = sorted(assignment_info, key=lambda x: x["best_throughput"], reverse=True)
        # print(assignment_info)
        return assignment_info

    def assign_subcluster(self, subcluster:Subcluster, assign_info, flag="even", custom=None, bw=240):
        splits = assign_info["splits"]
        runtime = assign_info["best_runtime"]
        bn = assign_info["bn"]
        bs = assign_info["bs"]
        wait = assign_info["wait"]
        tf= assign_info["tf"]
        if flag=="even":
            self.even_split(splits, bs=bs, bw=bw)
        elif flag=="comm":
            self.comm_split(splits, bs=bs, bw=bw)
        elif flag=="custom" and custom!=None:
            self.custom_split(custom, bs=bs, bw=bw)
        
        subcluster.assign(tasks=self.tasks, runtime=runtime, wait=wait, cur_tf=tf)


        



@dataclass
class Task:
    id: str
    job: Job
    flop: int
    peak_rate: float
    output_bytes: int
    task_arrival: int
    task_wait: int
    run_time: int #includes both wait+run time
    cur_tf: float #hard stopped at 0.5, if greater than 0.5, bring down to 0.5
    device: Device
    peak_bw: float

    @property
    def task_remaining_time(self):
        return self.task_wait + self.run_time

    def tick_tock(self, tick):
        if tick <= self.task_wait:
            self.task_wait-=tick
        elif tick > self.task_wait:
            if self.task_wait!=0:
                self.task_wait-=tick
                self.run_time= self.run_time + self.task_wait
                self.task_wait=0
                self.run_time = 0 if self.run_time<0 else self.run_time
            else:
                self.run_time= (self.run_time - tick) if tick <= self.run_time else 0

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
        if not Path(f"./subcluster_stats/{name}.ut.op").is_file():
            total_devs=ut+t+v
            sub = cls(name, total_devs, devices)
            for d in range(ut+t+v):
                if d<ut:
                    dev = Device(sub, d, "ut", [])
                    sub.devices.append(dev)
                elif d<ut+v:
                    dev = Device(sub, d, "v", [])
                    sub.devices.append(dev)
                elif d<ut+v+t:
                    dev = Device(sub, d, "t", [])
                    sub.devices.append(dev)
        else:
            un_f=open(f"./subcluster_stats/{name}.ut.op")
            un_lines=un_f.readlines()
            v_f = open(f"./subcluster_stats/{name}.v.op")
            v_lines = v_f.readlines()
            t_f = open(f"./subcluster_stats/{name}.t.op")
            t_lines = t_f.readlines()
            total_devs=len(un_lines) + len(v_lines) + len(t_lines)
            sub = cls(name, total_devs, devices)
            for d in un_lines:
                d=d.strip()
                id = d.split("-")[-1]
                dev = Device(sub, id, "ut", [])
                sub.devices.append(dev)
            for d in v_lines:
                d=d.strip()
                id = d.split("-")[-1]
                dev = Device(sub, id, "v", [])
                sub.devices.append(dev)
            for d in t_lines:
                d=d.strip()
                id = d.split("-")[-1]
                dev = Device(sub, id, "t", [])
                sub.devices.append(dev)
        return sub
    
    def tick_tock(self, tick):
        for d in self.devices:
            for t in d.tasks:
                t.tick_tock(tick)
            d.cleanup()


    def instance_map(self, wait=0, w_dev=False):
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        device_map = {"ut":0, "t":0, "v":0} if w_dev==False else {"ut":[], "t":[], "v":[]}
        for d in self.devices:
            if len(d.tasks)==0:
                device_map[d.dev_type]+=1 if w_dev==False else [d]
                continue
            t = sorted(d.tasks)[-1] 
            #last running task on device, 
            #tasks binpacked so we only care about the last one
            if wait >= t.task_remaining_time:
                #if last task's remaining time is <= wait time
                #device free within wait duration
                device_map[d.dev_type]+=1 if w_dev==False else [d]
        return device_map

    def interferences(self, wait=0, overlap_duration=0):
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        interfering_tasks = {}
        for d in self.devices:
            if d.dev_type=="ut":
                continue
            tasks=sorted(d.tasks)
            for t in tasks:
                if wait >= t.task_remaining_time:
                    #doesn't interfere if wait time exceed remaining time
                    continue
                if t.task_arrival <= wait+overlap_duration:
                    #if within overlap duration, will interfere
                    # if d.device_name() not in interfering_tasks:
                    if t.job.id not in interfering_tasks:
                        interfering_tasks[t.job.id] = t.job
                    # interfering_tasks[d.device_name()].append(t)
                    
        return list(interfering_tasks.values())

    def valid_wait_times(self):
        #return list of minimum wait times required for any change in device compositions
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        wait_times = []
        for d in self.devices:
            if len(d.tasks)==0:
                continue
            t=sorted(d.tasks)[-1]
            if t.task_remaining_time not in wait_times:
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
            # val = val if val < 0.9 else 0.9 #fixed upper limit
            val = val if val < 0.3 else 0.3 #fixed upper limit
            gnt = (peak_times[0]-maybe_gnt[0])/(1-val)
            achieved_fps = [gnt]
            return achieved_fps

        # for ind, i in enumerate(peak_times):
        val = self.tf_calc(len(peak_times)+bg_load, t)
        val = val if val < 0.3 else 0.3 #fixed upper limit
        achieved_fps = [(peak_times[i]-maybe_gnt[i])/(1-val) for i in range(len(peak_times))]

            # i = i - peak_times[ind-1] if ind > 0 else i
            # if i>0 and len(peak_times)-ind>0:
            #     val = self.tf_calc(len(peak_times)-ind+bg_load, t)
            #     # val = val if val < 0.9 else 0.9 #fixed upper limit
            #     val = val if val < 0.3 else 0.3 #fixed upper limit
            #     # gnt = (i-maybe_gnt[ind])/(1-val) #if tf = (t-p)/t
            #     gnt = (i)/(1-val) #if i > 0 else i / (i-val) #if tf = (t-p)/t
            #     for k in range(ind, len(maybe_gnt)):
            #         maybe_gnt[k]=maybe_gnt[k]+gnt
            #         achieved_fps[k] = maybe_gnt[k] + (fp[k]-fp[ind])*peak_times[k]/fp[k]
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
            # print(wait_times)
            device_comp = self.instance_map(w)
            # print(device_comp)
            if len(tasks) <= sum(device_comp.values()):
                if w not in time_map:
                    time_map[w]=np.inf
                else:
                    continue
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
                peak_accumulated_time=0
                # print(device_comp, task_mapping)
                # print([str(i) for i in rank_to_time_step_map[0]])
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
                    # print("ass_ex", peak_r, tasks[0].peak_bw, [i*tasks[0].peak_bw for i in all_fp_nw])
                    # print(max([f*10**-9/peak_r for f in non_fp]))
                    # print(self.time_predictor([f*10**-9/2 for f in fp], fp, bg_load), fp)
                    accumulated_time+=max(self.time_predictor([f*10**-9/peak_r for f in fp], fp, temp_bg_load)+[f*10**-9/peak_r for f in non_fp] ) + max(all_fp_nw+[0])
                    peak_accumulated_time += max([f*10**-9/peak_r for f in non_fp])+max(all_fp_nw+[0])
                # time_map[w]=[accumulated_time, task_mapping]

                #impact on other running jobs if any?
                #collect all tasks that are on volatile/throttled devices
                jobs_at_wait_time = self.interferences(w, accumulated_time)
                #get time passed (current arrival+wait time) and subtract flops achieved from tasks
                time_passed = tasks[0].job.arrival+w
                
                #redo interference for tasks with lower flops and bg load from new tasks
                # new_fp_per_job = [ [t.flop-(time_passed*t.flop/t.run_time)*j.bn for t in j.tasks if t.cur_tf < 0.9] for j in jobs_at_wait_time]
                new_fp_per_job = [ [t.flop-(time_passed*t.flop/t.run_time)*j.bn for t in j.tasks if t.cur_tf < 0.3] for j in jobs_at_wait_time]
                lags = [0]
                for fp_entry in new_fp_per_job:
                    #take the accumulated remaining flops as an approximation 
                    # -> all of lags is approximations since we don't maintain batch num implementations -> might be a TODO moment tbh
                    fp = [sum([f for f in fp_entry if f>0])]
                    temp_bg_load = bg_load - 1 + len(tasks) 
                    # or -1 to exclude current running task but bg load is being saturated anyway ? 
                    temp_lag = max(self.time_predictor([f*10**-9/peak_r for f in fp], fp, temp_bg_load)+[0])
                    lags.append(temp_lag)
                    # if max(fp_entry) <= 0:
                        # continue
                    # fp = [max(fp_entry)] 
                    #[f for f in fp if f>0]
                    # temp_bg_load = bg_load - 1 + len(tasks) #worst case assumption of lag




                #get throughput difference as lag and add it to the sorting criteria -> best throughput sort first, top k, then sort by lowest lag
                # or vice versa depending on what we greedily prioritize!

                time_map[w]=[accumulated_time, (accumulated_time - peak_accumulated_time)/accumulated_time, max(lags)]
        # return {k:v[0] for k,v in sorted(time_map.items(), key=lambda x: x[1][0])}#, {k:v[1] for k,v in sorted(time_map.items(), key=lambda x: x[1][0])}
        # print(time_map)
        return {k:v for k,v in sorted(time_map.items(), key=lambda x: x[1][0])}#, {k:v[1] for k,v in sorted(time_map.items(), key=lambda x: x[1][0])}


    def assign(self, tasks: List[Task], runtime:float, wait: float, cur_tf:float):
        #explores assignments for all possible wait times and returns assignment with highest throughput
        device_comp = self.instance_map(wait, w_dev=True)
        tasks[0].job.wait=wait
        if len(tasks) <= sum([len(i) for i in device_comp.values()]):
            jobs_at_wait_time = self.interferences(wait, runtime)
            #get time passed (current arrival+wait time) and subtract flops achieved from tasks
            time_passed = tasks[0].job.arrival+wait
            
            #redo interference for tasks with lower flops and bg load from new tasks
            # new_fp_per_job = [ [t.flop-(time_passed*t.flop/t.run_time)*j.bn for t in j.tasks if t.cur_tf < 0.9] for j in jobs_at_wait_time]
            new_fp_per_job = [ [t.flop-(time_passed*t.flop/t.run_time)*j.bn for t in j.tasks if t.cur_tf < 0.3] for j in jobs_at_wait_time]
            # lags = [0]
            bg_load = self.total_devs
            peak_r = tasks[0].peak_rate

            for fp_ind, fp_entry in enumerate(new_fp_per_job):
                #take the accumulated remaining flops as an approximation 
                # -> all of lags is approximations since we don't maintain batch num implementations -> might be a TODO moment tbh
                fp = [sum([f for f in fp_entry if f>0])]
                temp_bg_load = bg_load - 1 + len(tasks) 
                # or -1 to exclude current running task but bg load is being saturated anyway ? 
                temp_lag = max(self.time_predictor([f*10**-9/peak_r for f in fp], fp, temp_bg_load)+[0])
                cur_tasks = jobs_at_wait_time[fp_ind].tasks
                for t in cur_tasks:
                    # if t.cur_tf<0.9:
                    if t.cur_tf<0.3:
                        temp =  ((t.run_time+temp_lag) - (t.flop*10**-9/peak_r))/(t.run_time+temp_lag)
                        # temp = 0.9 if temp > 0.9 else temp
                        temp = 0.3 if temp > 0.3 else temp
                        # new_cur_tfs.apend(temp)
                        t.cur_tf = temp
                        t.run_time = t.run_time+temp_lag
            
            #temporarily assign tasks to device types
            #greedily assign longest flop task to best available device type ut -> v -> t
            
            #sort tasks and map them to device types
            flop_sorted_tasks = sorted(tasks, key = lambda x: x.flop, reverse=True)
            task_mapping = {f"{t}":0 for t in flop_sorted_tasks}
            r=0
            while r<len(flop_sorted_tasks) and task_mapping[f"{flop_sorted_tasks[r]}"]==0:
                if len(device_comp['ut'])>0:
                    task_mapping[f"{flop_sorted_tasks[r]}"]="ut"
                    flop_sorted_tasks[r].device = device_comp['ut'].pop(0)
                    flop_sorted_tasks[r].cur_tf=cur_tf
                    flop_sorted_tasks[r].task_wait = wait
                    flop_sorted_tasks[r].run_time = runtime
                    flop_sorted_tasks[r].device.tasks.append(flop_sorted_tasks[r])
                    # device_comp['ut']-=1
                
                elif len(device_comp['v'])>0:
                    task_mapping[f"{flop_sorted_tasks[r]}"]="v"
                    flop_sorted_tasks[r].device = device_comp['v'].pop(0)
                    flop_sorted_tasks[r].cur_tf=cur_tf
                    flop_sorted_tasks[r].task_wait = wait
                    flop_sorted_tasks[r].run_time = runtime
                    flop_sorted_tasks[r].device.tasks.append(flop_sorted_tasks[r])
                    # device_comp['v']-=1

                elif len(device_comp['t'])>0:
                    task_mapping[f"{flop_sorted_tasks[r]}"]="t"
                    flop_sorted_tasks[r].device = device_comp["t"].pop(0)
                    flop_sorted_tasks[r].cur_tf=cur_tf
                    flop_sorted_tasks[r].task_wait = wait
                    flop_sorted_tasks[r].run_time = runtime
                    flop_sorted_tasks[r].device.tasks.append(flop_sorted_tasks[r])
                    # device_comp['t']-=1
                r+=1

        
        
@dataclass
class Device:
    subcluster: Subcluster
    device_id: int
    dev_type: str #"ut, t, v"
    tasks: List[Task] 
    
    @property
    def device_name(self):
        return f"{self.subcluster.name}-{self.device_id}"
    
    def cleanup(self):
        self.tasks = [i for i in self.tasks if i.task_remaining_time!=0]



if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt
    import sys
    # subcluster = Subcluster.setup_subcluster("dummy", 10, 10, 10)
    dummy = "dummy" if len(sys.argv)<3 else sys.argv[2]
    bash_writer = 0 if len(sys.argv)<4 else int(sys.argv[3])
    subcluster = Subcluster.setup_subcluster(dummy, 0, 0, 10)
    # run_flag = 
    # print(subcluster.total_devs)
    # uniform_arrival_times = [0, 0.5, 1]
    model_type = "resnet18" if len(sys.argv)<2 else sys.argv[1]
    pr=3 if len(sys.argv)<6 else float(sys.argv[5])
    #fp rate = 0.35 for eff
    #fp rate = 2.8 for res
    #fp rate = 2.5 for vit
    if model_type=="resnet18":
        # pr=2.8
        pr= 3 if len(sys.argv)<6 else float(sys.argv[5])
        # pr=28
    if model_type=="vit":
        pr=3 if len(sys.argv)<6 else float(sys.argv[5])
    template = open("schedule_template.sh")
    template_lines = template.readlines()

    job_array_list = []
    full_lines=[]
    inter_arrival_time = 0.5 if len(sys.argv)<7 else float(sys.argv[6])
    uniform_arrival_times = [inter_arrival_time*i for i in range(0,6)]

    prev_time = uniform_arrival_times[0]
    bw=240 if len(sys.argv)<5 else int(sys.argv[4])

    for u_ind, u in enumerate(uniform_arrival_times):
        subcluster.tick_tock(u)

        #peak rate for eff is 3?
        #peak rate for resnet is -> 30
        #peak rate for vit somehow 15????
        j = Job(f"j{u_ind}", u, 0, model_type, 10, pr, [] )
        job_array_list.append(j)
        # s=time.time()
        # comp_a = j.cost_function_explorer(subcluster, flag="even", bw=bw)
        # print(comp_a[0])
        # print(time.time() - s)
        # print()
        comp_a = [{"best_throughput":0}]
        s=time.time()
        comm_a = j.cost_function_explorer(subcluster, flag="comm", bw=bw)
        print(comm_a[0])
        print(time.time() - s)
        # comm_a = [{"best_throughput":0}]

        if comp_a[0]["best_throughput"] > comm_a[0]["best_throughput"]:
            j.assign_subcluster(subcluster, assign_info=comp_a[0], bw=bw)
            t_flag=1
        else:
            j.assign_subcluster(subcluster, assign_info=comm_a[0], bw=bw)
            t_flag=2

        print([ [str(i) for i in d.tasks ] for d in subcluster.devices])
        print("full device picture", [ [ [f"{i} {i.task_remaining_time}"] for i in d.tasks ] for d in subcluster.devices])
        #final job times and impact throughout the system
        # print([ [ f"{t}, {t.task_wait}, {t.run_time}, {t.task_remaining_time}" for t in l.tasks ] for l in job_array_list])
        # lines_to_write = [ ' '.join([ f"task:{t},{t.task_wait},{t.run_time},{t.task_remaining_time}" for t in l.tasks ]) + "\n" for l in job_array_list]
        csv_lines = []
        for d in subcluster.devices:
            for t in d.tasks:
                t_str = f"{u},{len(t.job.tasks)},{t.job.bs},{t.job.bn},{t.id},{t.task_arrival},{t.task_wait},{t.run_time},{t.task_remaining_time}\n"
                csv_lines.append(t_str)


        # lines_to_write = [ '\n'.join([f"task:{i},wait:{i.task_wait},runtime:{i.run_time},full_time:{i.task_remaining_time}" for i in d.tasks]) for d in subcluster.devices]
        full_lines.extend(csv_lines)
        
        print([f"{i} {i.task_remaining_time}" for i in j.tasks])
        print()

        #10 devices, 5ut, 5v, schedule 20 jobs, uniform interval -> 0.1s, 1s, 10s
        #keep inputs 10, bn, bs should change ideally as we progress
        #3 scripts, each script pick 10 devices at random from red_blue.op file, repeat 10 times
        #loop for device selection and nodes is templated in schedule_template.sh



        #construct command and write to hardcoded bash script : )
        #blurb to add per job
        #"dt=$(date -d '+60 seconds' +%s)"
        #"command = "+f'"python3 onnxtest_w_json_bs.py 30 ./{model_type}_splits_{len(j.tasks)}_{1 or 0} {rank based on bn, bs} 4 {bs} '+ '${dt}"'
        #'timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@${nodes[$n]} "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/speed_chronos${nodes[$n]}_${fake_world}_${fake_rank}_${inter}.log &" &'
        #"wait" -> so all tasks that need to run together run together based on batch num
        
        add_lines= []
        bn = j.bn
        bs = j.bs
        sleep_str = f"    sleep {u+j.wait if j.wait!=0 else prev_time+u}\n"
        prev_time = u+j.wait

        rank_to_time_step_map={r:[0]*r+[f"{j.tasks[r]}"]*bn+[0]*(len(j.tasks)-r-1) for r in range(len(j.tasks))}
        full_commands=[]
        metadata_dt = "    echo $(date '+TIME:%H:%M:%S.%3N') >> ${path_dst}/${repeat}/metadata_"+f"{u_ind}.log\n"
        metadata_vals = "    echo 'minutes to subtract "+ f"{len(rank_to_time_step_map[0])}"+" splits " +f"{len(j.tasks)}" + " batch_num "+f"{bn} batch_size {bs}' >> "+"${path_dst}/${repeat}/metadata_"+f"{u_ind}.log\n"
        # print(t_flag)
        for time_slice in range(len(rank_to_time_step_map[0])):
            dt_str = "    dt=$(date -d '+60 seconds' +%s)\n"
            commands = []
            for r in rank_to_time_step_map:
                if rank_to_time_step_map[r][time_slice]!=0:
                    # pre_command_str = "for i in $(ps -aex | grep onnxtest | grep ${nodes["+str(j.tasks[r].device.device_id)+"]} | awk '{print $1}'); do while [[ -d /proc/${i} ]]; do sleep 0.1; done; done;\n"
                    pre_command_str = "for i in $(ps -aex | grep onnxtest | grep "+str(j.tasks[r].device.device_name)+" | awk '{print $1}'); do while [[ -d /proc/${i} ]]; do sleep 0.1; done; done;\n"
                    command_str = "    command="+f'"python3 onnxtest_w_json_bs.py 1 ./{model_type}_splits_{len(j.tasks)}_{t_flag} {r} 4 {bs} '+ '${dt}"\n'
                    ssh_str = '    timeout 10m ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no animesh@'+str(j.tasks[r].device.device_name)+' "pushd $path_prefix/aot_splitter; source /home/animesh/model_splitting/pi-torch/bin/activate; ${command} > ${path_dst}/${repeat}/speed_chronos'+str(j.tasks[r].device.device_name)+'_' + f'{len(j.tasks)}_{r}_{t_flag}_{time_slice}_{u}.log &" &\n'
                    commands.extend([pre_command_str, command_str, ssh_str])
            wait_str = f"    wait\n{dt_str}"
            commands.append(wait_str)
            full_commands.extend(commands)
        function_lines = [f"f_{u_ind} () "+"{\n"] + [metadata_dt, metadata_vals] + [dt_str] + full_commands + [metadata_dt] + ["}\n"]
        template_lines = template_lines[0:1]+["\n"]+function_lines+["\n"]+template_lines[1:]
        template_lines.extend(["\n", sleep_str, f"    f_{u_ind} &\n"])
        
        # template_lines.extend(full_commands)
        # template_lines.extend([ssh_str, wait_str])
    final_wait = "\n    wait\n"
    done_str = "done"
    template_lines.append(final_wait)
    template_lines.append(done_str)
    if bash_writer==1:
        write_file = open(f"{model_type}_uniform_exp_0.5.sh", "w")
        write_file.writelines(template_lines)
        write_file.close()
    csv_header="clock,world,bs,bn,task_id,task_arrival,task_wait,run_time,task_remaining_time\n"
    simulation_f = open(f"sim.op.{dummy}.{model_type}.{bw}.{inter_arrival_time}.only_comm", "w")
    simulation_f.write(csv_header)
    simulation_f.writelines(full_lines)
    simulation_f.close()

    raise Exception("thanks for all the fish")
    j = Job("j0", 0, 0, "resnet18", 10, 3, [] )
    s=time.time()
    comp_a = j.cost_function_explorer(subcluster, flag="even")
    print(comp_a)

    print(time.time() - s)
    print()
    # exit()

    s=time.time()
    comm_a = j.cost_function_explorer(subcluster, flag="comm")
    print(comm_a)
    print(time.time() - s)

    #graph results, pick best throughput regardless of split
    import sys
    graph=0
    if len(sys.argv) > 1:
        graph = int(sys.argv[1])
    if graph!=0:
        fig, axs = plt.subplots(figsize=(25,10))
        axs.set_xlabel("machine used (splits in model)")
        axs.set_ylabel("throughput")
        split_sorted_comp = sorted(comp_a, key=lambda x: x["splits"])
        split_sorted_comm = sorted(comm_a, key=lambda x: x["splits"])
        axs.scatter( [i["splits"] for i in split_sorted_comp], [i["best_throughput"] for i in split_sorted_comp], marker="^", label="balanced comp split")
        axs.scatter( [i["splits"] for i in split_sorted_comm], [i["best_throughput"] for i in split_sorted_comm], marker="x", label="balanced comm split")
        axs.legend()
        fig.savefig("test_throughput.png")

    if comp_a[0]["best_throughput"] < comm_a[0]["best_throughput"]:
        j.assign_subcluster(subcluster, assign_info=comp_a[0])
    else:
        j.assign_subcluster(subcluster, assign_info=comm_a[0])
    
    # print(j)
    # print([ i.device.device_name for i in j.tasks])
    print([ [str(i) for i in d.tasks ] for d in subcluster.devices])

    print([f"{i} {i.task_remaining_time}" for i in j.tasks])
    print()
    subcluster.tick_tock(10)
    print([f"{i} {i.task_remaining_time}" for i in j.tasks])
    print([ [str(i) for i in d.tasks ] for d in subcluster.devices])
    