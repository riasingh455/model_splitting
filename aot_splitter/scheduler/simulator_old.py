#simulate cluster setup
#simulate job arrives (hard code for now)
#simulate the additional slowdown 
#simulate waiting for "best" conditions
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict
import ast 
from scipy.special import binom
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(3)

def drift_mean(drift_arr:List[float]) -> float:
    remove_zero_arr = [i for i in drift_arr if i!=0]
    if len(remove_zero_arr)==0:
        remove_zero_arr+=[0]
    return np.mean(remove_zero_arr)

def drift_std(drift_arr:List[float]) -> float:
    remove_zero_arr = [i for i in drift_arr if i!=0]
    if len(remove_zero_arr)==0:
        remove_zero_arr+=[0]
    return np.std(remove_zero_arr)

@dataclass
class Job:
    name: str
    subcluster: Subcluster = None
    
    ops: float = 0.0
    best_rate: float = 20/(2*(600)**3-(600)**2) #s/op
    min_device_count: int = 0#memory spillover impact -> always schedule for number of devices >= min_device_count
    actual_device_count: int = 0  #how many actually used, execution is best_time/device count -> assuming perfectly parallelisable
    throttled: int = 0
    unthrottled: int = 0
    volatile: int = 0

    
    parallel_time: float = 0 #time that can be parallel
    # best_time: float = 0 #if on ut or v without throttling, no parallelism

    spillover_time: float = 0 #if on ut or v or without throttling but not enough device counts
    impacted_time: float = 0 #ephemeral value, changes depending on what job is being potentially scheduled
    wait_impacted_time: float = 0 #ephemeral value, added to shift potential overlapping times 
    arrival_time:float = 0
    wait_time:float = 0
    time_spent_waiting:float = 0
    time_spent_running:float = 0 
    #difference between expected_execution_time_when arrived and final expected_execution time 
    #shows impact of jobs that come after scheduling -> see error_in_execution_time
    expected_execution_time_when_arrived:float = 0
    expected_execution_time:float = 0
    #whether we can get away with putting only on throttled devs as it has minimal impact on every other running dev
    # or get away with waiting for resources -> see deadline_violation function
    # or take spillover hit? 
    deadline:float = 0 
    drift_limit: float = 0 #this is the expected time from assigned subcluster regardless of n -> can be thought of as deadline? 


    def __eq__(self, other:Job):
        return self.remaining_time == other.remaining_time
    def __lt__(self, other:Job):
        return self.remaining_time < other.remaining_time
    def __gt__(self, other:Job):
        return self.remaining_time > other.remaining_time
    def __ge__(self, other:Job):
        return self.remaining_time >= other.remaining_time
    def __le__(self, other:Job):
        return self.remaining_time <= other.remaining_time
    def __ne__(self, other:Job):
        return self.remaining_time != other.remaining_time

    @property
    def serial_time(self):
        return self.ops*self.best_rate
    
    @property
    def remaining_time(self):
        return self.expected_execution_time - self.time_spent_running + self.wait_time - self.time_spent_waiting
    
    @property
    def turnaround_time(self):
        return self.time_spent_running+self.wait_time
    
    @property
    def deadline_difference(self):
        return self.deadline - self.expected_execution_time - self.wait_time - self.wait_impacted_time
    
    @property
    def drift_in_execution_time(self):
        return self.expected_execution_time + self.impacted_time - self.expected_execution_time_when_arrived

    def best_time(self, device_count):
        return self.serial_time + self.parallel_time/device_count
    
    def assign(self, subcluster:Subcluster, device_count:int = 0, wait_time:float = 0,
    with_ut:bool = False, with_ut_and_v:bool = False, with_any:bool = True,
    with_t:bool = False, force:bool = False, dry:bool = False, composition:Any=None) -> Any:
        #if with_ut, assign job to only ut devices
        #if with_ut_and_v, assign job to ut_and_v only, if ut not available, assign to just v only
        #if with_any, assign job to any random available device set, start from ut, to v then to t
        #if with_t, assign job to t devices only 
        #force allows to break min_device_count, device_count constraint
        #all of this assumes a greedy assignment composition, i.e if ut exists, use it, never save it for anyone later
        #dry: if true treat as dry run and provide expected throttle fraction and existing job impact instead of assignment
        
        ut, t, v = subcluster.avail_resources(wait_time)
        # print(wait_time, ut, t, v)
        device_count = self.min_device_count if device_count==0 else device_count
        sel_composition = {}
        expected_tf = -1
        self_expected_tf = -1
        best_time = 0
        if composition!=None:
            # print(composition)
            sel_composition = composition
            best_time  = self.best_time(device_count)
            if composition["unthrottled"] == device_count:
                self_expected_tf=0
                expected_tf = self.calc_expectation(ut, t, v, device_count, subcluster)

            if composition["throttled"]!=0:
                if composition["throttled"]!=device_count:
                    self_expected_tf=1
                    expected_tf = self.calc_expectation(ut, t, v, device_count-composition["throttled"], subcluster)
                elif composition["throttled"]==device_count:
                    self_expected_tf=1
                    expected_tf=0

            if composition["throttled"]==0:
                self_expected_tf = self.calc_expectation(ut, t, v, device_count, subcluster)
                expected_tf = self_expected_tf
            # if self.name=="job_15":
            #     print(expected_tf, self_expected_tf)
            
            
        if composition==None:
            if with_ut:
                if ut >= device_count or force:
                    self_expected_tf = 0
                    expected_tf = self.calc_expectation(ut, t, v, device_count, subcluster)
                    sel_composition = {"unthrottled": min(ut, device_count), "throttled":0, "volatile":0}
                    best_time = self.best_time(min(ut, device_count)) if ut >= device_count else self.spillover_time
                
                else:
                    raise Exception(f"AssignError: ut count {ut} but job device_count {device_count}")
            
            elif with_ut_and_v:
                if ut+v >= device_count or force:
                    expected_tf = self.calc_expectation(ut, t, v, device_count, subcluster)
                    self_expected_tf = 0 if ut >= device_count else expected_tf
                    sel_composition = {"unthrottled": min(ut, device_count), "throttled":0, 
                    "volatile":min( max(device_count-ut, 0), v)}
                    best_time = self.best_time(min(ut+v, device_count)) if ut+v >= device_count else self.spillover_time

                else:
                    raise Exception(f"AssignError: ut, v count {ut}, {v} but job device_count {device_count}")
                
            elif with_t:
                if t >= device_count or force:
                    expected_tf = 0
                    self_expected_tf = 1
                    sel_composition = {"unthrottled": 0, "throttled":min(t, device_count), "volatile":0}
                    best_time = self.best_time(min(t, device_count)) if t >= device_count else self.spillover_time
                else:
                    raise Exception(f"AssignError: t count {t} but job device_count {device_count}")
                
            elif with_any:
                if ut+v+t >= device_count or force:
                    expected_tf = self.calc_expectation(ut, t, v, device_count, subcluster) if ut+v >= device_count else 1
                    # print("calc exp tf", expected_tf)
                    self_expected_tf = 0 if ut >= device_count else expected_tf
                    sel_composition = {"unthrottled": min(ut, device_count), "throttled":min(max(device_count-ut-v, 0), t), 
                    "volatile":min( max(device_count-ut, 0), v)}
                    best_time = self.best_time(min(sum(sel_composition.values()), device_count)) if sum(sel_composition.values()) >= device_count else self.spillover_time

                else:
                    raise Exception(f"AssignError: ut, v, t count {ut}, {v}, {t} but job device_count {device_count}")
        expected_self_job_time = (best_time)*self_expected_tf + best_time
        if expected_tf>0 and sel_composition["unthrottled"]==0 and sel_composition["volatile"]==0:
            expected_tf = 0
        # print(job.name, sel_composition, expected_tf)
        subcluster.assign(self, expected_tf, wait_time=wait_time, expected_cur_job_time = expected_self_job_time, dry=dry)
        if not dry:
            self.subcluster = subcluster
            self.actual_device_count = device_count
            self.wait_time = wait_time
            self.unthrottled = sel_composition["unthrottled"]
            self.throttled = sel_composition["throttled"]
            self.volatile = sel_composition["volatile"]
            self.expected_execution_time_when_arrived = (best_time)*self_expected_tf + best_time
            self.expected_execution_time = self.expected_execution_time_when_arrived
            #expected time throughout subcluster with any n after current set
            extended_tf = [(best_time)*self_expected_tf + best_time]
            temp_tf = -1
            if ut+v < device_count and sel_composition["throttled"]==0:
                for ext_count in range(device_count, ut+v):
                    temp_tf = self.calc_expectation(ut, t, v, ext_count, subcluster)
                    extended_tf.append(self.best_time(device_count)*temp_tf+ self.best_time(device_count))
            else:
                for ext_count in range(device_count, ut+v+t):
                    # print(ext_count, subcluster.total_devices)
                    temp_tf = self.calc_expectation(ut, t ,v , ext_count, subcluster, with_t=True)
                    extended_tf.append(self.best_time(device_count)*temp_tf+ self.best_time(device_count))
            self.drift_limit = np.mean(extended_tf)
            
                    
                    

        
        return [expected_tf, sel_composition]
    @staticmethod
    def calc_expectation(ut, t, v, parallel_size:int, subcluster:Subcluster, with_t:bool = False) -> float:
        pool_composition = {"unthrottled":ut, 
        "throttled":t, 
        "volatile":v}
        dev_pool_size = sum( list(pool_composition.values()) )

        pick_ut_and_v = binom(pool_composition["unthrottled"]+pool_composition["throttled"], parallel_size)
        pick_ut_and_v = 1 if pick_ut_and_v==0 else pick_ut_and_v

        pick_ut = binom(pool_composition["unthrottled"], parallel_size)
        pick_ut = 1 if pick_ut==0 else pick_ut

        pick_total = binom(dev_pool_size, parallel_size)
        pick_total = 1 if pick_total == 0 else pick_total

        # if self.name=="job_15":
        #     print(pool_composition, pick_ut_and_v, pick_total, dev_pool_size, parallel_size)
        expected_val_with_t = (1-(pick_ut_and_v/pick_total))*1 + ((pick_ut_and_v-pick_ut)/pick_total)*subcluster.proj_tf_calc(parallel_size)
        expected_val_without_t = ((pick_ut_and_v-pick_ut)/pick_ut_and_v)*subcluster.proj_tf_calc(parallel_size) if pick_ut_and_v > 0 else np.inf
        # print("inside func", expected_val_without_t, (pick_ut_and_v-pick_ut), pick_ut_and_v )
        #brute forced expectation?
        
        
        
        return expected_val_with_t if with_t else expected_val_without_t


@dataclass
class Subcluster:
    name: str 
    throttled: int
    unthrottled: int
    volatile: int
    jobs: List[Job]
    scale: int
    # time_map: Any
    @property
    def total_devices(self):
        return self.unthrottled + self.throttled + self.volatile

    @property
    def eq(self):
        m3_map = {1:[0.157, -0.004, 0.075], 2:[0.667 ,  -0.008, 0.023]}
        return {k:[i/self.scale for i in v] for k,v in m3_map.items()}
    # eq: Any = field(
    #     default_factory=lambda: {1:[0.157, -0.004, 0.075], 2:[0.667 ,  -0.008, 0.023]} #min-med-max change_point=8
    # )
    #private method for funsies
    def _cleanup(self, wait_time:float = 0) -> Any:
        #maybe multi-threadable? 
        cleaned_jobs = []
        ut, t, v = self.unthrottled, self.throttled, self.volatile
        for job_ind in range(len(self.jobs)):
            job = self.jobs[job_ind]
            # print(job.name, job.expected_execution_time , job.time_spent_running , job.wait_time , job.time_spent_waiting)
            # print(job.remaining_time, wait_time)
            if job.remaining_time - wait_time <= 0:
                continue
            cleaned_jobs.append(job)
            ut-=job.unthrottled
            v-=job.volatile
            t-=job.throttled
        if wait_time == 0:
            self.jobs = [j for j in cleaned_jobs]

        return [ut if ut>=0 else 0, t if t >=0 else 0, v if v>=0 else 0]
    
    def composition_cal(self, dev_pool_size, run_flip, skip_list) -> None:
        # print(dev_pool_size, run_flip, len(run_flip), skip_list, len(skip_list))
        self.throttled =  dev_pool_size - len(run_flip) #+ len(skip_list)
        self.unthrottled =  sum([1 for i in run_flip if run_flip[i]==-1 and i not in skip_list])
        self.volatile = sum([1 for i in run_flip if run_flip[i]!=-1 and i not in skip_list])


    def proj_tf_calc(self, device_count, eq=None, change=8) -> float:
        eq = self.eq if eq==None else eq
        pool = self.unthrottled + self.volatile + self.throttled
        n_counts = [job.unthrottled + job.volatile for job in self.jobs if job.time_spent_running!=0]+[device_count]
        n = sum(n_counts)
        val_1 = eq[1][0] + eq[1][1]*pool + eq[1][2]*n
        val_2 = eq[2][0] + eq[2][1]*pool + eq[2][2]*n 
        val_2 = val_2 if val_2 <= 1 else 1

        return val_1 if n<change else val_2
    
    def existing_jobs_impact(self, expected_tf, expected_cur_job_time:float = 0, wait_time:float=0, ) -> None:
        # proj_job_times = {}
        for job in self.jobs:
            # job.impacted_time=0
            # job.wait_impacted_time=0
            # proj_job_times[job.name] = [job, (time_left*expected_tf + time_left)]
            if job.remaining_time <= wait_time or job.actual_device_count == job.unthrottled or job.actual_device_count == job.throttled:
                continue
            if job.time_spent_running > 0 and expected_tf!=np.inf:
                # print(f"expected tf:{expected_tf}")
                job.impacted_time = (job.remaining_time-wait_time)*expected_tf
                job.impacted_time = 0 if job.impacted_time < 0 else job.impacted_time
                # print(job.name, expected_tf, job.impacted_time)

            elif job.time_spent_running==0 and expected_tf!=0 and (job.wait_time) <= expected_cur_job_time+wait_time:
                # job.impacted_time = abs(job.wait_time - expected_cur_job_time+wait_time)
                job.wait_impacted_time += expected_cur_job_time+wait_time-(job.wait_time)
    
    def tick_tock(self, add_time=1) -> None:
        # print(add_time)
        exclude_jobs = []
        for job in self.jobs:
            if job.time_spent_waiting+add_time <= job.wait_time:
                job.time_spent_waiting+=add_time
            elif job.time_spent_waiting+add_time >= job.wait_time:
                add_time = job.time_spent_waiting + add_time - job.wait_time 
                job.time_spent_waiting = job.wait_time
                job.time_spent_running+=add_time
                if job.time_spent_running >= job.expected_execution_time:
                    exclude_jobs.append(job.name)
                    # print(job)
        self.jobs = [job for job in self.jobs if job.name not in exclude_jobs]
        # if len(exclude_jobs)>0:
        #     print(exclude_jobs)
        #     print(self)
            # raise Exception("clean the damn jobs!")

    def avail_resources(self, wait_time:float = 0) -> Any:
        #composition of unthrottled, volatile, throttled remaining
        ut, t, v = self._cleanup(wait_time=wait_time)
        return [ut, t, v] 
    
    def assign(self, job:Job, expected_tf: float, expected_cur_job_time:float=0, wait_time:float = 0, dry:bool = False) -> None:
        # if dry:
        for j in self.jobs:
            j.impacted_time = 0
            j.wait_impacted_time = 0
        self.existing_jobs_impact(expected_tf, wait_time=wait_time, expected_cur_job_time=expected_cur_job_time)
        if not dry:
            for j in self.jobs:
                # print(j.name, j.impacted_time, j.expected_execution_time, j.expected_execution_time_when_arrived, j.arrival_time)
                j.expected_execution_time += j.impacted_time
                j.wait_time += j.wait_impacted_time
                j.wait_impacted_time = 0
                j.impacted_time = 0
            self.jobs.append(job)
            

    def rank_resource_waiting_list(self) -> Any:
        #ranks how quickly a resource can be free-d depending on the jobs running currently
        #does not include projected time if job-to-be-scheduled is added to subcluster -> see existing_job_impact
        #returns {waiting_time: {ut:x, t:y, v:z}}

        #does not allow partial reclaiming -> if job running on ut, v, t order of completion would be ut, v, t
        #but cannot claim ut, v earlier than last part of job at t. 
        ut, t, v = self.avail_resources()
        time_to_resources = {0:{"unthrottled":ut, "throttled":t, "volatile":v}}
        sorted_jobs = sorted(self.jobs)
        # last_ind = 0
        for sj in sorted_jobs:
            new_wait =  sj.remaining_time+0.1
            new_ut, new_t, new_v = self.avail_resources(wait_time = new_wait)
            # print(new_ut, new_t, new_v)
            sj_resources = {"unthrottled":new_ut, "throttled":new_t, "volatile":new_v}
            time_to_resources[new_wait] = {k:sj_resources[k] for k in time_to_resources[0]}
            # last_ind = new_wait
        return time_to_resources


@dataclass
class SystemState:
    subclusters: Dict[str, Subcluster]
    scale:int = 1
    skip_list: Any = field(
        default_factory=lambda: {"bramble-1-1":[], "bramble-1-2":[5], "bramble-1-3":[19], 
        "bramble-1-4":[3, 14, 31, 32, 36, 37, 41], "bramble-2-1":[], "bramble-2-5":[], 
        "bramble-2-6":[12, 22, 26], "bramble-4-1":[], "bramble-4-2":[], 
        "bramble-4-3":[12, 20, 22, 24, 28, 34, 36, 38], "bramble-4-5":[], "bramble-4-6":[42]
        }
    )
    
    total_devs: Any = field( default_factory=lambda: {
        'bramble-1-1': 40, 
        'bramble-1-2': 28, 
        'bramble-1-3': 39, 
        'bramble-1-4': 39, 
        'bramble-2-1': 25, 
        'bramble-2-5': 41, 
        'bramble-2-6': 39, 
        'bramble-4-1': 42, 
        'bramble-4-2': 41, 
        'bramble-4-3': 41, 
        'bramble-4-5': 41, 
        'bramble-4-6': 36
        })

    full_map: Any = field(default_factory=lambda: {
        "bramble-1-1":range(1,13),
        "bramble-1-2":range(1,11), 
        "bramble-1-3":range(1,22), 
        "bramble-1-4":range(1,28),
        "bramble-2-1":range(1,18),
        "bramble-2-5":range(1,20),
        "bramble-2-6":range(1,18),
        "bramble-4-1":range(1,29),
        "bramble-4-2":range(1,26),
        "bramble-4-3":range(1,32),
        "bramble-4-5":range(1,23),
        "bramble-4-6":range(1,25)
        })

    def init_setup(self, subcluster_spotlight=None, scale=1) -> None:
        subcluster_map = {}
        f=open("./dataset/global_time_rank_dump.txt")
        lines = f.readlines()
        global_time_rank = ast.literal_eval(lines[0].strip())
        global_run_flip = ast.literal_eval(lines[1].strip())
        for i in self.total_devs:
            if subcluster_spotlight!=None and i not in subcluster_spotlight:
                continue
            # subcluster = Subcluster(i, 0,0,0,[], global_time_rank[i])
            subcluster = Subcluster(i, 0,0,0,[], scale=self.scale)
            subcluster.composition_cal(self.total_devs[i], global_run_flip[i], self.skip_list[i])
            subcluster_map[subcluster.name] = subcluster
        self.subclusters = subcluster_map

    def tick_tock(self, add_time=1) -> None:
        # print(add_time)
        for subcluster_name in self.subclusters:
            self.subclusters[subcluster_name].tick_tock(add_time)
    
    def dev_count_trials(self, subcluster:Subcluster, job:Job, ut:int, t:int, v:int,  waiting_time:float, trial_runs:Any, 
    drift_enforced:bool = False, deadline_enforced:bool=False, with_any:bool = False, force_min:bool = False):
        continue_flag = False
        current_resource_size = ut+t+v if with_any else t
        current_resource_size = current_resource_size if force_min!=True else job.min_device_count
        for device_count in range(job.min_device_count, current_resource_size+1):
            if (with_any==False and device_count > t) or (with_any==True and device_count > ut+v+t):
                break
            expected_tf, composition = job.assign(subcluster, device_count=device_count, wait_time=waiting_time, 
            with_t=not with_any, with_any=with_any, dry=True)
            # print("expected tf", expected_tf)
            if composition["unthrottled"] == device_count:
                expected_tf = 0
            if composition["throttled"] != 0:
                expected_tf = 1
            cur_job_impact = [0]
            cur_wait_impact= [0]
            cur_job_deadline = [job.deadline - waiting_time - (job.best_time(device_count))*expected_tf]
            #if we exceed est drift then continue 
            est_drift_arr = []
            impact_arr = []
            deadline_arr = []
            waiting_arr = []
            for j in subcluster.jobs:
                if j.drift_in_execution_time > j.drift_limit and drift_enforced:
                    continue_flag = True
                    break
                if j.deadline_difference - j.impacted_time < 0 and deadline_enforced:
                    continue_flag = True
                    break
                est_drift_arr.append(j.drift_in_execution_time)
                impact_arr.append(j.impacted_time)
                deadline_arr.append(j.deadline_difference-j.impacted_time)
                waiting_arr.append(j.wait_time+j.wait_impacted_time)
            if continue_flag:
                continue_flag = False
                # if with_any==False:
                # print(with_any)
                # print(waiting_time)
                # print(cur_job_deadline)
                # print([[j.name, j.wait_time, j.deadline, j.drift_in_execution_time, j.expected_execution_time, j.expected_execution_time_when_arrived, j.unthrottled, j.throttled, j.volatile] for j in subcluster.jobs])
                # print([j.deadline_difference for j in subcluster.jobs])
                # print([j.impacted_time for j in subcluster.jobs])
                continue
            trial_runs[f"{device_count}_{with_any}"] = {"est_drifts":est_drift_arr+cur_job_impact, 
            "deadline_violations":deadline_arr+cur_job_deadline, 
            "waiting_times":waiting_arr+cur_wait_impact, 
            "waiting_time_key":waiting_time, "composition":{k:v for k,v in  composition.items()}, 
            "individual_job_time": waiting_time + job.best_time(device_count) + expected_tf*job.best_time(device_count)}
        return trial_runs


    def schedule_on_subcluster(self, subcluster:Subcluster, job:Job, force_min:bool = False, deadline_driven:bool = False, 
    est_drift_driven:bool = True, flip:bool = False, drift_enforced:bool = False, wait_tie_break:bool = False, 
    deadline_enforced:bool = False, ind_tie_break:bool=False):
        #returns best option for scheduling on given subcluster
        #we do not assume subcluster spanning tasks -> incurs wireless penalty and we do not calculate that just yet
        #either
        # - wait for resource
        # - run immediately 
        # - run with force
        # all ranked based on deadline violation(if true) or estimated time drift(if true) 
        # final score (see simulator function) is how many jobs were scheduled, time to completion of all jobs, number of times had to wait
        # no tie breaking just yet
        resource_wait_map = subcluster.rank_resource_waiting_list()
        assignment_score_map = {}
        first_trial, first_device_count = None,  -1
        # print(resource_wait_map)
        for waiting_time in resource_wait_map:
            # print(waiting_time)
            current_resource_size = sum(list(resource_wait_map[waiting_time].values())) if not force_min else job.min_device_count
            ut, t, v = resource_wait_map[waiting_time]["unthrottled"], resource_wait_map[waiting_time]["throttled"], resource_wait_map[waiting_time]["volatile"]
            if job.min_device_count > current_resource_size:
                # use force only if minimum not met, never spillover memory unless you have to ???
                #reason being prediction accuracy drops hard
                continue #for now wait longer, but here is where we would add memmory spillover
            else:
                trial_runs = {}
                

                trial_runs = self.dev_count_trials(subcluster, job, ut, t, v, waiting_time, trial_runs, 
                drift_enforced=drift_enforced, deadline_enforced=deadline_enforced, with_any=True, force_min=force_min)
                #bigger the deadline miss -> more negative the value
                #bigger the est drift -> more positive the value
                trial_runs = self.dev_count_trials(subcluster, job, ut, t, v, waiting_time, trial_runs, 
                drift_enforced=drift_enforced, deadline_enforced=deadline_enforced, with_any=False, force_min=force_min) 
                #with_any = false, then with_t= true
                #also include entries where it's only throttled devices 
                #sort map based on deadline or est_drift
                if len(trial_runs)==0:
                    continue
                
                sorted_trials = {}
                tie_trials = {}
                if est_drift_driven:
                    # est drifts include 0s which heavily drops mean! bad thing! exclude 0s when calc mean
                    sorted_trials = {k:v for k,v in sorted(trial_runs.items(), 
                    key=lambda x: drift_mean(x[1]["est_drifts"]), reverse=flip )}
                    # print(job.min_device_count, current_resource_size)
                    # print(sorted_trials)
                    first_device_count = list(sorted_trials.keys())[0]
                    first_trial = sorted_trials[ first_device_count ]
                    for t in sorted_trials:
                        if drift_mean(sorted_trials[ first_device_count ]["est_drifts"]) == drift_mean(sorted_trials[t]["est_drifts"]):
                            tie_trials[t] = sorted_trials[t]
                    
                elif deadline_driven:
                    sorted_trials = {k:v for k,v in sorted(trial_runs.items(), 
                    key=lambda x: sum([1 for i in x[1]["deadline_violations"] if i<0]), reverse=flip)}
                    first_device_count = list(sorted_trials.keys())[0]
                    first_trial = sorted_trials[ first_device_count ]
                    for t in sorted_trials:
                        if sum([1 for i in sorted_trials[first_device_count]["deadline_violations"] if i<0]) ==sum([1 for i in sorted_trials[t]["deadline_violations"] if i<0]):
                            tie_trials[t] = sorted_trials[t]
                 
                # raise Exception("stop here thanks")

                #tie break on best individual job time first
                sorted_tie_trials={}
                if ind_tie_break:
                    sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: x[1]["individual_job_time"], reverse=flip )}
                elif wait_tie_break:
                    sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: drift_mean(x[1]["waiting_times"]), reverse=flip )}
                first_device_count = list(sorted_tie_trials.keys())[0]
                first_trial = sorted_trials[ first_device_count ]
                # if "False" in first_device_count:
                #     raise Exception("At leaste one False!")
                # print(sorted_tie_trials)
                # raise Exception("stop here thanks")

                # f = open(f"./logs/subcluster-{subcluster.name}.log","a")
                # f.write(f"{datetime.now()} with {device_count} using {'est_drive' if est_drift_driven else 'deadline_drive'} sorted:{sorted_trials}\n")
                # f.write(f"{datetime.now()} with {device_count} using {'est_drive' if est_drift_driven else 'deadline_drive'} selected:{first_trial}\n")
                # f.close()

                #one layer deep copy trial information
                assignment_score_map[f"{first_device_count}_{first_trial['waiting_time_key']}"] = {k:v for k,v in trial_runs[first_device_count].items()}
            
        sorted_trials = {}
        tie_trials = {}
        # if job.name=="job_1":
        #     print(assignment_score_map)
        
        if len(assignment_score_map)==0:
            # print("when assignment failed?", trial_runs)
            return subcluster, np.inf, None
        if est_drift_driven:
            sorted_trials = {k:v for k,v in sorted(assignment_score_map.items(), key=lambda x: drift_mean(x[1]["est_drifts"]), reverse=flip )}
            first_device_count = list(sorted_trials.keys())[0]
            first_trial = sorted_trials[ first_device_count ]
            for t in sorted_trials:
                if drift_mean(sorted_trials[ first_device_count ]["est_drifts"]) == drift_mean(sorted_trials[t]["est_drifts"]):
                    tie_trials[t] = sorted_trials[t]
            
        elif deadline_driven:
            sorted_trials = {k:v for k,v in sorted(assignment_score_map.items(), 
            key=lambda x: sum([1 for i in x[1]["deadline_violations"] if i<0]), reverse=flip)}
            first_device_count = list(sorted_trials.keys())[0]
            first_trial = sorted_trials[ first_device_count ]
            for t in sorted_trials:
                if sum([1 for i in sorted_trials[first_device_count]["deadline_violations"] if i<0]) ==sum([1 for i in sorted_trials[t]["deadline_violations"] if i<0]):
                    tie_trials[t] = sorted_trials[t]
        
        #tie break on individually best performing jobs
        # print(tie_trials)
        sorted_tie_trials={}
        if ind_tie_break:
            sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: x[1]["individual_job_time"], reverse=flip )}
        elif wait_tie_break:
            sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: drift_mean(x[1]["waiting_times"]), reverse=flip )}
        first_waiting_and_device_count = list(sorted_tie_trials.keys())[0]
        first_trial = sorted_trials[ first_waiting_and_device_count ]
        # if job.name=="job_1":
        #     print(assignment_score_map)
        #     print(sorted_trials[ first_waiting_and_device_count ])


        return subcluster, first_waiting_and_device_count, first_trial

    def subcluster_assignment_rank(self, best_trials:Any, deadline_driven:bool = False, est_drift_driven:bool = True, 
    flip:bool = False, ind_tie_break:bool = False, wait_tie_break:bool = False):
        #best trial is subcluster_name:{est_drift:[], deadline_violations:[], waiting_time:[], composition:{}}
        tie_trials = {}
        sorted_trials = {}
        if est_drift_driven:
            sorted_trials = {k:v for k,v in sorted(best_trials.items(), key=lambda x: drift_mean(x[1]["est_drifts"]), reverse=flip )}
            first_device_count = list(sorted_trials.keys())[0]
            first_trial = sorted_trials[ first_device_count ]
            for t in sorted_trials:
                if drift_mean(sorted_trials[ first_device_count ]["est_drifts"]) == drift_mean(sorted_trials[t]["est_drifts"]):
                    tie_trials[t] = sorted_trials[t]
        elif deadline_driven:
            sorted_trials = {k:v for k,v in sorted(best_trials.items(), 
            key=lambda x: sum([1 for i in x[1]["deadline_violations"] if i<0]), reverse=flip)}
            first_device_count = list(sorted_trials.keys())[0]
            first_trial = sorted_trials[ first_device_count ]
            for t in sorted_trials:
                if sum([1 for i in sorted_trials[first_device_count]["deadline_violations"] if i<0]) ==sum([1 for i in sorted_trials[t]["deadline_violations"] if i<0]):
                    tie_trials[t] = sorted_trials[t]
        
        #tie break on best individual job time
        sorted_tie_trials={}
        if ind_tie_break:
            sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: x[1]["individual_job_time"], reverse=flip )}
        elif wait_tie_break:
            sorted_tie_trials = {k:v for k,v in sorted(tie_trials.items(), key=lambda x: drift_mean(x[1]["waiting_times"]), reverse=flip )}
        first_subcluster_name = list(sorted_tie_trials.keys())[0]
        first_trial = sorted_trials[ first_subcluster_name ]
        # if job.name=="job_0": 
        #     print(first_trial)# = sorted_trials[ first_subcluster_name ]

        return first_subcluster_name, first_trial
    
    def assign(self, subcluster_name:str, job:Job, trial:Any):
        subcluster = self.subclusters[subcluster_name]
        device_count = sum(list(trial["composition"].values()))
        wait_time = trial["waiting_time_key"]
        # if job.name=="job_15":
        #     print(subcluster.name, device_count, wait_time, trial["composition"])
        job.assign(subcluster, device_count, wait_time, with_any=True, dry=False, 
        composition=trial["composition"])





if __name__ == "__main__":
    system = SystemState([])
    scale=10**3
    # system.init_setup(subcluster_spotlight=["bramble-2-1", "bramble-1-3", "bramble-4-1"])
    # system.init_setup(subcluster_spotlight=["bramble-2-1"])
    # print(system.subclusters["bramble-4-3"])
    # exit()
    # print([system.subclusters[i].name for i in system.subclusters])
    #schedule a single job, total time period 24hr, job shows up even intervals
    # timeline = 86400 #s in a day
    timeline = 3600 #s in an hour
    job_nums = 46 #min count for bramble-2-1 and azure-fn
    job_nums = 38 #min count for bramble-2-1 and poisson for timeline=3600s
    arrival_times = []
    ops = [2*(600)**3-(600)**2]*job_nums
    serial_time = [20*scale]*job_nums
    parallel_time=[0*scale]*job_nums
    deadline_time = [2*(serial_time[j]+parallel_time[j]) for j in range(job_nums)]
    # arr_style="azure_fn"
    # arr_style="poisson"
    arr_style="custom"
    if arr_style == "uniform":
        stagger = timeline/job_nums
        arrival_times = [i+stagger for i in range(job_nums)] #change later 
    elif arr_style=="custom":
        stagger = 0
        job_nums=4 #one pipeline
        arrival_times = [0 for i in range(job_nums)]
        rate = 20/(2*(600)**3-(600)**2) #s/op
        ops = [236027904,1746927616,1644167168,1024000]
        serial_time = [i*rate for i in ops]
        parallel_time = [0]*job_nums
        deadline_time = [2*(serial_time[j]+parallel_time[j]) for j in range(job_nums)]

    elif arr_style == "poisson":
        avg_rate = job_nums/timeline #jobs per time scale
        lm = 1/avg_rate
        print(avg_rate, lm/scale)
        inter_arrival_times = poisson.rvs(lm, size=job_nums)
        arrival_times = np.cumsum(inter_arrival_times)#[i for i in np.cumsum(inter_arrival_times) if i < timeline]
    elif arr_style == "azure_fn":
        inv_df = pd.read_csv("~/Downloads/azurefunctions-dataset2019/invocations_per_function_md.anon.d01.csv")
        # print(inv_df.columns)
        tim_df = pd.read_csv("~/Downloads/azurefunctions-dataset2019/function_durations_percentiles.anon.d01.csv")
        hash_fn_columns = [c for c in inv_df["HashFunction"]]
        # print(len(hash_fn_columns), len(set(hash_fn_columns)))
        hash_fn_other_columns = [c for c in tim_df["HashFunction"]]
        # print(len(hash_fn_other_columns), len(set(hash_fn_other_columns)), len(set(hash_fn_other_columns).intersection(set(hash_fn_columns))))
        fn_hashes = sorted([i for i in set(hash_fn_other_columns).intersection(set(hash_fn_columns))])
        tim_df.set_index("HashFunction", inplace=True)
        inv_df.set_index("HashFunction", inplace=True)
        # print(tim_df.columns)
        arrival_to_job_name = {}
        # print("filtering")
        #can combine with hashapp to get fn counts and combinations 
        counter=0
        break_flag=False
        for fn in fn_hashes:
            temp_time_arr = inv_df.loc[fn, '1':'1440']
            for t_ind, t in enumerate(temp_time_arr):
                if t==0:
                    continue
                if counter >= job_nums:
                    break_flag=True
                    break
                t_key = (t_ind+1)*60#*scale
                if t_key not in arrival_to_job_name:
                    arrival_to_job_name[t_key] = []
                for r in range(t):
                    # arrival_to_job_name[t_key].append([fn, tim_df.loc[fn, 'Average'], tim_df.loc[fn, 'Maximum']])
                    arrival_to_job_name[t_key].append([fn, serial_time[0], parallel_time[0], deadline_time[0]] )
                
                counter+=t

            if break_flag:
                break
                #can use count here from tim_df to combine fns or keep them separate instead of app?
        sorted_arrivals_to_jobs = {k:[i for i in v] for k,v in sorted(arrival_to_job_name.items(), key= lambda x: x[0])}
        arrival_times = []
        serial_time = []
        parallel_time = []
        deadline_time = []
        # break_flag=False
        for k_ind, k in enumerate(sorted_arrivals_to_jobs):
            for v_ind, v in enumerate(sorted_arrivals_to_jobs[k]):
                # if k_ind+v_ind >= job_nums:
                #     break_flag=True
                #     break
                arrival_times.append(k)
                if len(sorted_arrivals_to_jobs[k][v_ind]) == 3:
                    serial_time.append(sorted_arrivals_to_jobs[k][v_ind][1])
                    deadline_time.append(sorted_arrivals_to_jobs[k][v_ind][2])
                elif len(sorted_arrivals_to_jobs[k][v_ind]) == 4:
                    serial_time.append(sorted_arrivals_to_jobs[k][v_ind][1])
                    parallel_time.append(sorted_arrivals_to_jobs[k][v_ind][2])
                    deadline_time.append(sorted_arrivals_to_jobs[k][v_ind][3])
            # if break_flag:
            #     break
        # print([k for k in inv_df.loc[hash_fn_columns[0], '1':'1440']])

        # print(tim_df.loc[hash_fn_columns[0],'Average'], tim_df.loc[hash_fn_columns[0],'Maximum'])
        arrival_times = [i-arrival_times[0] for i in arrival_times][:job_nums]
        serial_time = serial_time[:job_nums]
        deadline_time = deadline_time[:job_nums]
        parallel_time = parallel_time[:job_nums]
        # print(arrival_times, len(arrival_times), len(parallel_time), len(deadline_time))
        # print(len(serial_time))
        # parallel_time=[0]*job_nums
        print("deadline stats")
        print(max(deadline_time)/scale, min(deadline_time)/scale, np.mean(deadline_time)/scale, np.median(deadline_time)/scale, np.std(deadline_time)/scale)
        diff_arr = np.diff(arrival_times)
        temp_max = [0]
        for r_ind, r in enumerate(diff_arr):
            if r == 0:
                temp_max.append(serial_time[r_ind]+parallel_time[r_ind])
            if r!=0:
                trivial_time = max(temp_max)+r
                temp_max=[serial_time[r_ind]+parallel_time[r_ind]]
        # trivial_time = max(diff_arr) + sum(serial_time)
        print(f"trivial serial tasks if {len(arrival_times)} devs available: { trivial_time/scale},  {job_nums*scale/trivial_time} (job/s)")
        # trivial_time = max(arrival_times)+sum(serial_time) + sum(parallel_time)
        # print(f"trivial serial tasks if one devs available: { trivial_time/scale}, {job_nums*scale/trivial_time} (job/s)")
        # trivial_time = sum([max(diff_arr[i:i+115]) for i in range(0, len(diff_arr), 115)]) + sum([max(serial_time[i:i+115]) for i in range(0, len(serial_time), 115)])
        # print(f"trivial serial tasks if 115 devs available: { trivial_time/scale},  {job_nums*scale/trivial_time} (job/s)")


        # temp_max = [0]
        # for r_ind, r in enumerate(diff_arr):
        #     if r == 0:
        #         temp_max.append(serial_time[r_ind])
        #     if r!=0 or r_ind==115:
        #         trivial_time = max(temp_max)+r
        #         temp_max=[serial_time[r_ind]]
        
        # exit()

    elif arr_style == "azure_vm":
        pass
    # print(arrival_times)

    # job_nums = 117 #if yes force min
    stagger=0
    force_min=True
    drift_enforced= False
    deadline_enforced= True
    drift_driven= True
    deadline_driven=not drift_driven
    wait_tie_break = False
    ind_tie_break = not wait_tie_break
    
    # arrival_times = [i+stagger for i in range(job_nums)] #change later 
    jobs:List[Job] = []
    # print(arrival_times[-1])
    cluster_util = {}
    throughput_map = {}
    avg_drift_map = {}
    avg_job_time_map = {}
    avg_exec_map = {}
    total_time_map = {}

    for min_count in [1]:#range(1, 10):
        jobs=[]
        system=SystemState([])
        system.init_setup(["bramble-2-1"])
        big_break_flag=False
        for j in range(job_nums):
            # print(parallel_time[j])
            job = Job(f"job_{j}", subcluster=None, arrival_time=arrival_times[j], min_device_count=min_count, ops = ops[j], #serial_time=serial_time[j], 
            parallel_time=parallel_time[j], deadline=deadline_time[j], 
            impacted_time=0, wait_impacted_time=0)
            jobs.append(job)
            best_trials = {}
            
            if j != 0:
                system.tick_tock(arrival_times[j] - arrival_times[j-1])
            
            for subcluster_name in system.subclusters:
                # print(subcluster_name)
                # print(system.subclusters[subcluster_name].unthrottled, 
                # system.subclusters[subcluster_name].throttled, 
                # system.subclusters[subcluster_name].volatile)

                # force_min=False
                subcluster, wait_and_count, trial_info = system.schedule_on_subcluster(system.subclusters[subcluster_name], 
                job, force_min=force_min, drift_enforced=drift_enforced, ind_tie_break=ind_tie_break, wait_tie_break=wait_tie_break, 
                deadline_enforced=deadline_enforced, est_drift_driven=drift_driven, deadline_driven=deadline_driven)
                if wait_and_count == np.inf:
                    continue
                best_trials[subcluster_name] = trial_info
            
            # if j >= 0:
            #     print(best_trials)
            #     print()
            if len(best_trials)==0:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("PARTIAL SCHEDULE POSSIBLE ONLY WITH CURRENT CONSTRAINTS")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                big_break_flag=True
                break
            first_subcluster_name, first_trial = system.subcluster_assignment_rank(best_trials, est_drift_driven=drift_driven, deadline_driven=deadline_driven,
            ind_tie_break=ind_tie_break, wait_tie_break=wait_tie_break)
            system.assign(first_subcluster_name, job, first_trial)
            # print(system.subclusters[first_subcluster_name])
            if j >= np.inf:
                # print(best_trials)
                print(job.name, job.subcluster.name, job.actual_device_count, job.wait_time/scale, job.time_spent_running/scale, 
            job.expected_execution_time_when_arrived/scale, job.expected_execution_time/scale)
                print()
        # print(system.subclusters[first_subcluster_name])
        if big_break_flag:
            break
        #full schedule stats:
        #cluster utilization stats:
        total_ut, total_v, total_t = 0,0,0
        used_ut, used_v, used_t = 0,0,0
        cur_ut, cur_v, cur_t = 0,0,0 
        usage_per_arrival_time = {round(job.arrival_time/scale,3):{"unthrottled":0, "throttled":0, "volatile":0} for job in jobs}
        running_jobs = [j.name for j in jobs if j.time_spent_running>0 and j.time_spent_running < j.expected_execution_time]
        j_count=0
        not_running_yet = [j.name for j in jobs if j.time_spent_running==0]
        # print(not_running_yet)
        fin_jobs = [j.name for j in jobs if j.time_spent_running>0 and j.time_spent_running >= j.expected_execution_time]
        for subcluster_name in system.subclusters:
            total_ut += system.subclusters[subcluster_name].unthrottled
            total_v += system.subclusters[subcluster_name].volatile
            total_t += system.subclusters[subcluster_name].throttled
            for j in system.subclusters[subcluster_name].jobs:
                if j.time_spent_running>0 and j.time_spent_running < j.expected_execution_time:
                    j_count+=1
                    # print(j.time_spent_running, j.expected_execution_time, j.drift_limit, j.drift_in_execution_time )
                    cur_ut+=j.unthrottled
                    cur_v+=j.volatile
                    cur_t+=j.throttled
                    # continue
                # elif j.time_spent_running >= j.expected_execution_time:
                #     fin_jobs.append(j.name)
                # else:
                #     not_running_yet.append(j.name)

        print(len(running_jobs), len(not_running_yet), len(fin_jobs))
        # print("job count:",j_count, len(jobs))
        # print(usage_per_arrival_time.keys())
        another_running_jobs = []
        completion_times = [job.expected_execution_time + job.wait_time for job in jobs]
        wait_times = [job.wait_time for job in jobs]
        para_per_job = [job.actual_device_count for job in jobs]
        print(f"min, max, mean, median, std_dev of device count per job:{min(para_per_job), max(para_per_job), np.mean(para_per_job), np.median(para_per_job), np.std(para_per_job)}")
        

        usage_per_arrival_time[round( (arrival_times[-1]+max(wait_times))/scale,3)] = {"unthrottled":0, "throttled":0, "volatile":0}
        for arrival in usage_per_arrival_time:
            ti = arrival
            j_count=0
            for job in jobs:
                arr_ti = round(job.arrival_time/scale, 3)
                run_ti = round((job.arrival_time+job.wait_time)/scale, 3)
                tot_ti = round((job.arrival_time+job.wait_time+job.expected_execution_time)/scale, 3)
                # print(arr_ti, ti, tot_ti)
                if ti!="random" and arr_ti <= ti and run_ti <= ti and tot_ti >= ti:
                    usage_per_arrival_time[ti]["unthrottled"]+=job.unthrottled
                    usage_per_arrival_time[ti]["throttled"]+=job.throttled
                    usage_per_arrival_time[ti]["volatile"]+=job.volatile
                    j_count+=1
                    if job.name not in another_running_jobs:
                        another_running_jobs.append(job.name)

        # print(usage_per_arrival_time[round( (arrival_times[-1]+max(wait_times))/scale,3)])
        # print(another_running_jobs, len(another_running_jobs))
                # else:
                #     # print("random")
                #     usage_per_arrival_time["random"]["unthrottled"]+=job.unthrottled
                #     usage_per_arrival_time["random"]["throttled"]+=job.throttled
                #     usage_per_arrival_time["random"]["volatile"]+=job.volatile


        # for job in jobs:
        #     ti = round(job.arrival_time/scale,3)
        #     usage_per_arrival_time[ti]["unthrottled"]+=job.unthrottled
        #     usage_per_arrival_time[ti]["throttled"]+=job.throttled
        #     usage_per_arrival_time[ti]["volatile"]+=job.volatile

        usage_per_arrival_time  = {k:{"unthrottled":usage_per_arrival_time[k]["unthrottled"]/total_ut, 
        "throttled":usage_per_arrival_time[k]["throttled"]/total_t , "volatile": usage_per_arrival_time[k]["volatile"]/total_v } for k in usage_per_arrival_time}
        sorted_usage_per_arrival_time = {k:v for k,v in sorted(usage_per_arrival_time.items(), key = lambda x: sum(x[1].values()), reverse=True)}
        print("total counts (ut, v, t):", total_ut, total_v, total_t)
        print(cur_ut/total_ut, cur_v/total_v, cur_t/total_t)
        # print(usage_per_arrival_time)
        peak_util_arr_time = list(sorted_usage_per_arrival_time.keys())[0]
        print(f"peak utilization @ {peak_util_arr_time} :{sorted_usage_per_arrival_time[peak_util_arr_time]}")
        # if min_count not in cluster_util:
        #     cluster_util[min_count] = [] 
        cluster_util[min_count] = np.mean(list(sorted_usage_per_arrival_time[peak_util_arr_time].values()))
        print(cluster_util)


        #avg arrival time
        arr_times = np.diff([job.arrival_time for job in jobs])
        print(f"min, max, mean, median, std_dev arrival time:{round(min(arr_times)/scale,3), round(max(arr_times)/scale,3), round(np.mean(arr_times)/scale,3), round(np.median(arr_times)/scale,3), round(np.std(arr_times)/scale,3)}")

        # print number of jobs scheduled, average job completion time, longest running job, shortes running job, most used subcluster
        # print also total time taken to finish all jobs, drift in jobs, worst drift in jobs, wait times, worst wait time
        print(f"# of jobs scheduled:{len(jobs)}, # running jobs {len(running_jobs)}, # waiting jobs {len(not_running_yet)} # fin jobs {len(fin_jobs)}")
        print(f"min, max, mean, median, std_dev completion time:{round(min(completion_times)/scale,3), round(max(completion_times)/scale,3), round(np.mean(completion_times)/scale,3), round(np.median(completion_times)/scale,3), round(np.std(completion_times)/scale,3)}")
        avg_job_time_map[min_count] = [np.mean(completion_times), np.std(completion_times)]
        # print(f"min time jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.expected_execution_time + job.wait_time == min(completion_times)]}")
        # print(f"max time jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.expected_execution_time + job.wait_time == max(completion_times)]}")

        drift_times = [job.drift_in_execution_time for job in jobs]
        # print(drift_times)
        print(f"min, max, (drift) mean, median, std_dev drift times:{ round(min(drift_times)/scale,3), round(max(drift_times)/scale,3), round(drift_mean(drift_times)/scale,3), round(np.median(drift_times)/scale,3), round(np.std(drift_times)/scale,3)}")
        # if min_count not in avg_drift_map:
        #     avg_drift_map[min_count] = 0
        avg_drift_map[min_count] = [drift_mean(drift_times), drift_std(drift_times)]
        # print(f"min drift jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.drift_in_execution_time == min(drift_times)]}")
        # print(f"max drift jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.drift_in_execution_time == max(drift_times)]}")

        print(f"min, max, mean, median, std_dev wait times:{ round(min(wait_times)/scale,3), round(max(wait_times)/scale,3), round(drift_mean(wait_times)/scale,3), round(np.median(wait_times)/scale,3), round(np.std(wait_times)/scale,3)}")
        # print(f"min wait time jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.wait_time == min(drift_times)]}")
        # print(f"max wait time jobs:{[ [job.name, job.subcluster.name] for job in jobs if job.wait_time == max(drift_times)]}")
        
        #print execution times only
        exec_times = [job.expected_execution_time for job in jobs]
        print(f"min, max, mean, median, std_dev exec times:{ round(min(exec_times)/scale,3), round(max(exec_times)/scale,3), round(np.mean(exec_times)/scale,3), round(np.median(exec_times)/scale,3), round(np.std(exec_times)/scale,3)}")
        avg_exec_map[min_count] = [np.mean(exec_times), np.std(exec_times)]
        #group jobs by arrival times 
        arrival_times_maps = {}
        for job in jobs:
            if job.arrival_time not in arrival_times_maps:
                arrival_times_maps[job.arrival_time] = []
            arrival_times_maps[job.arrival_time].append(job.wait_time + job.expected_execution_time)
        
        # for at in arrival_times_maps:
        #     ma
        # TODO fix this calc, it doesn't account for overlaps from previous jobs? -> technically doesn't matter job would get pushed in bins anyway?
        full_schedule_latency = max(arrival_times_maps.keys()) + max(arrival_times_maps[max(arrival_times_maps.keys())])#sum([max(arrival_times_maps[at]) for at in arrival_times_maps])
        total_time_map[min_count] = full_schedule_latency
        
        print(f"total time taken (latency):{round(full_schedule_latency/scale,3)}s and # of jobs/latency (throughput):{round(len(jobs)*scale/full_schedule_latency,3)}(jobs/s)")
        #best time serially
        # bt = 2*(jobs[0].best_time(42)/(scale))
        # # print(bt)
        # serial_new_jobs = [bt*ind+bt for ind in range(len(jobs))]
        # total_time = sum(serial_new_jobs)

        # print(f"trivial serial time+perfect parallelisation:{round(total_time,3)}s and trivial serial throughput:{ round(len(jobs)/total_time, 3)}(job/s)")
        print(f"next min count {min_count}")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    fig, axs = plt.subplots(figsize=(25,10))
    # axs.errorbar([i[0]/scale for i in avg_drift_map.values()],    cluster_util.values(), marker=".", linestyle="--", xerr=[i[1]/scale for i in avg_drift_map.values()] ,label="drift")
    # axs.errorbar([i[0]/scale for i in avg_job_time_map.values()], cluster_util.values(), marker=".", linestyle="--", xerr=[i[1]/scale for i in avg_job_time_map.values()] ,label="avg job time")
    # axs.errorbar([i[0]/scale for i in avg_exec_map.values()],     cluster_util.values(), marker=".", linestyle="--", xerr=[i[1]/scale for i in avg_exec_map.values()] ,label="avg exec time")
    axs.plot([i[0]/scale for i in avg_drift_map.values()],    [i*100 for i in cluster_util.values()], marker=".", linestyle="--",label="drift")
    axs.plot([i[0]/scale for i in avg_job_time_map.values()], [i*100 for i in cluster_util.values()], marker=".", linestyle="--",label="avg job time")
    axs.plot([i[0]/scale for i in avg_exec_map.values()],     [i*100 for i in cluster_util.values()], marker=".", linestyle="--",label="avg exec time")
    axs.plot([i/scale for i in total_time_map.values()],      [i*100 for i in cluster_util.values()], marker=".", linestyle="--",label="total time")
    counts = list(cluster_util.keys())
    for i in range(len(counts)):
        axs.text(y=cluster_util[counts[i]]*100,x=avg_drift_map[counts[i]][0]/scale,s=counts[i])
        axs.text(y=cluster_util[counts[i]]*100,x=avg_job_time_map[counts[i]][0]/scale,s=counts[i])
        axs.text(y=cluster_util[counts[i]]*100,x=avg_exec_map[counts[i]][0]/scale,s=counts[i])
        axs.text(y=cluster_util[counts[i]]*100,x=total_time_map[counts[i]]/scale,s=counts[i])
    axs.legend()
    axs.set_ylabel("Utilization of devices in cluster (%)")
    axs.set_xlabel("Time (s)")
    axs.set_yticks(np.arange(30, 90, 10))
    axs.set_xticks(np.arange(0, 100, 20))
    fig.savefig(f"util_vs_avg_drift_{force_min}.pdf")

    fig, axs = plt.subplots(figsize=(25,10))
    axs.scatter(cluster_util.keys(), [i[0]/scale for i in avg_drift_map.values()], label="drift")
    axs.scatter(cluster_util.keys(), [i[0]/scale for i in avg_job_time_map.values()], label="avg job time")
    axs.scatter(cluster_util.keys(), [i[0]/scale for i in avg_exec_map.values()], label="avg exec time")
    axs.scatter(cluster_util.keys(), [i/scale for i in total_time_map.values()], label="total time")
    counts = list(cluster_util.keys())
    for i in range(len(counts)):
        axs.text(x=counts[i],y=avg_drift_map[counts[i]][0]/scale,s=cluster_util[counts[i]])
        axs.text(x=counts[i],y=avg_job_time_map[counts[i]][0]/scale,s=cluster_util[counts[i]])
        axs.text(x=counts[i],y=avg_exec_map[counts[i]][0]/scale,s=cluster_util[counts[i]])
        axs.text(x=counts[i],y=total_time_map[counts[i]]/scale,s=cluster_util[counts[i]])
    axs.legend()
    fig.savefig(f"count_vs_times_{force_min}.pdf")
    

