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
class Job:
    name: str
    subcluster: Subcluster = None


@dataclass
class Subcluster:
    name: str
    jobs: List[Job]
    throttled: int
    unthrottled: int
    volatile: int

    @property
    def total_devices(self):
        return self.unthrottled + self.throttled + self.volatile
    
    