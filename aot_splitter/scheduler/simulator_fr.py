from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Dict
import ast 
from scipy.special import binom
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import pandas as pd

np.random.seed(3)

@dataclass
class Task:
    name: str
    job: Job
    glops: float
    

@dataclass
class Job:
    name: str
    tasks: List[Task]
    subcluster: Subcluster = None
    arrival_time: float = 0
    wait_time: float = 0
    deadline: float = 0

    


@dataclass
class Subcluster:
    name: str
    jobs: List[Job]
    throttled: int
    unthrottled: int
    volatile: int
    scale: int
    glops: float

    @property
    def total_devices(self):
        return self.unthrottled + self.throttled + self.volatile
    
    def _cleanup(self, wait_time:float = 0) -> Any:
        #remove jobs from global system dictionary 
        #wait time 
        pass
    
    def avail_resources(self, wait_time:float = 0) -> Any:
        #composition of unthrottled, volatile, throttled remaining
        #
        pass
    
    def tick_tock(self, add_time=1) -> None:
        #removes time from runtime/wait time
        #cleans up resources 
        pass

    def assign(self, job:Job, expected_tf: float, expected_cur_job_time:float=0, wait_time:float = 0, dry:bool = False) -> None:
        #assign job to subcluster
        #takes care of updating times for each task/job
        pass

            

    def rank_resource_waiting_list(self) -> Any:
        #ranks how quickly a resource can be free-d depending on the jobs running currently
        #returns {waiting_time: {ut:x, t:y, v:z}}
        pass
        

@dataclass
class SystemState:
    subclusters: Dict[str, Subcluster]
    scale: int = 1
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
        #sets up subcluster statistics
        pass

    def tick_tock(self, add_time=1) -> None:
        # print(add_time)
        for subcluster_name in self.subclusters:
            self.subclusters[subcluster_name].tick_tock(add_time)
    

