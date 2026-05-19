from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple
import model_splitter

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

    def even_split(self, num_splits, export=False):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        
        for r_ind, r in enumerate(result["splits"]):
            task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"])
            self.tasks.append(task)
    
    def comm_split(self, num_splits, export=False):
        splits = [100/num_splits]*num_splits
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=0, comm_w=1)
        
        for r_ind, r in enumerate(result["splits"]):
            task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"])
            self.tasks.append(task)

    def custom_split(self, splits, export=False):
        result = ModelSplitWrapper.split(splits, self.model, export, flop_w=1, comm_w=0)
        
        for r_ind, r in enumerate(result["splits"]):
            task = Task(f"{self.id}.{r_ind}", self, r["actual_flops"])
            self.tasks.append(task)


    def total_flops(self):
        if self.tasks==None:
            return sum([task.flop for task in self.tasks])

    #TODO:
    #with jobs, iterate over batch size, decide input and time taken based on interference
    #also use valid wait time list per batch size, batch num combo
    #long iterations but it's okay, after add to device and update all relevant tasks
    #add a tick_tock function either in Device or Subcluster
    


@dataclass
class Task:
    id: str
    job: Job
    flop: int
    peak_rate: float
    output_bytes: int
    device: Device
    task_arrival: int
    task_wait: int
    task_remaining_time: int #includes both wait+run time
    
    def __eq__(self, other):
        return self.task_remaining_time == other.task_remaining_time
    
    def __lt__(self, other):
        return self.task_remaining_time < other.task_remaining_time
    
    def __gt__(self, other):
        return self.task_remaining_time > other.task_remaining_time



@dataclass
class Subcluster:
    name: str
    devices: List[Device] = None

    @classmethod
    def setup_subcluster(cls, name, ut, t, v):
        #make this specific ids instead of just number?
        cls.name = name
        cls.devices=[]
        for d in range(ut+t+v):
            if d<ut:
                dev = Device(cls, d, "ut", [])
                cls.devices.append(dev)
            elif d<ut+v:
                dev = Device(cls, d, "v", [])
                cls.devices.append(dev)
            elif d<ut+v+t:
                dev = Device(cls, d, "t", [])
                cls.devices.append(dev)
        return cls
    
    def instance_map(self, wait=0):
        if self.devices==None:
            return TypeError("Uninitialized subcluster! Call setup first!")
        device_map = {"ut":0, "t":0, "v":0}
        for d in self.devices:
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
            t=sorted(d.tasks)[-1]
            wait_times.append(t.task_remaining_time)
        return sorted(wait_times)

@dataclass
class Device:
    subcluster: Subcluster
    device_id: int
    dev_type: str #"ut, t, v"
    tasks: List[Task] 

    def device_name(self):
        return f"{self.subcluster.name}-{self.device_id}"




