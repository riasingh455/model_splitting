from __future__ import annotations

import time
from PIL import Image
from pathlib import Path

from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import torch.distributed as dist

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List 

torch.manual_seed(3)
np.random.seed(3)

def set_one_core_behavior():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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
    split_model: Dict[int, nn.Module]
    device: torch.device

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
    
    def split(self, rank:int, split_num:int = 2, split_layers:List[List[str]] = []):
        #splits model sequentially 
        #if split not even, main node takes additional split
        l = split_num if len(split_layers) == 0 else len(split_layers)
        state_dict = {k:v for k,v in self.model.named_modules() if k!=''}
        counter=0
        for k in range(l):
            temp_list=[]
            l_names = list(state_dict.keys())
            for l_ind in range(counter, len(state_dict)):
                l_name = l_names[l_ind]
                if (len(split_layers)==0 and l_ind > counter+np.ceil(len(state_dict)/l)) or (len(split_layers)!=0 and l_name not in split_layers[k]):
                    break
                if len(split_layers)==0 or l_name in split_layers[k]:
                    temp_list.append(state_dict[l_name])
            counter=len(temp_list) 
            self.split_model[k] = nn.Sequential(*temp_list)
        
        #somewhat forced Garbage Collection
        self.model = self.split_model[rank]
        self.split_model={}
    
    def load_image_tensor(self, path: str, preprocess: Any, device: torch.device) -> torch.Tensor:
        #weights = ResNet18_Weights.DEFAULT
        #preprocess = weights.transforms()

        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
        return x

    def top1_label(self, categories:Any, logits: torch.Tensor) -> str:
        #weights = ResNet18_Weights.DEFAULT
        #categories = weights.meta["categories"]
        idx = int(logits.argmax(dim=1).item())
        return f"{idx}: {categories[idx]}"

    def save_model(self):
        pass

@dataclass
class FBModel(GenModel):
    
    def pipeline_inference(self, backend, world, rank, warmup, iters, x0=None):
        #essentially the main from pipeline_splitting_inf.py
        #we include assumed rank from args to garbage collect the rest of the model quickly
        total_start = 0
        outputs = None
        #rank = args.assigned_rank
        next_rank = rank+1
        prev_rank = rank-1
        stage=self.model
        dist.init_process_group(backend=backend)

        with torch.inference_mode():
            for i in range(warmup + iters):
                if i == warmup and rank == 0:
                    dist.barrier()
                    total_start = time.perf_counter()
                elif i == warmup:
                    dist.barrier()

                if rank == 0:
                    y = stage(x0)
                    if next_rank < world:
                        # send y to next stage
                        # header: ndim, then (shape..., dtypecode)
                        dist.send(torch.tensor([y.dim()], dtype=torch.int64), dst=next_rank)
                        dist.send(torch.tensor([*y.shape, self.dtype_to_code(y.dtype)], dtype=torch.int64), dst=next_rank)
                        dist.send(y.contiguous(), dst=next_rank)
                    else:
                        outputs = y
                else:
                    # recv from prev
                    ndim_t = torch.empty(1, dtype=torch.int64)
                    dist.recv(ndim_t, src=prev_rank)
                    ndim = int(ndim_t.item())

                    shape_dtype = torch.empty(ndim + 1, dtype=torch.int64)
                    dist.recv(shape_dtype, src=prev_rank)
                    shape = tuple(int(v) for v in shape_dtype[:-1].tolist())
                    dtype = self.code_to_dtype(int(shape_dtype[-1].item()))

                    x = torch.empty(shape, dtype=dtype, device=self.device)
                    dist.recv(x, src=prev_rank)

                    y = stage(x)

                    if next_rank < world:
                        dist.send(torch.tensor([y.dim()], dtype=torch.int64), dst=next_rank)
                        dist.send(torch.tensor([*y.shape, self.dtype_to_code(y.dtype)], dtype=torch.int64), dst=next_rank)
                        dist.send(y.contiguous(), dst=next_rank)
                    else:
                        outputs = y

        dist.barrier()
        if rank == 0:
            total_end = time.perf_counter()
            elapsed = total_end - total_start
            avg = elapsed / max(1, iters)
            print(f"\n--- Pipeline Inference (stages={world}, iters={iters}) ---")
            print(f"Total time (timed iters): {elapsed:.4f} s")
            print(f"Avg latency per image: {avg*1000:.2f} ms")
            print("Note: Single-image pipeline mostly demonstrates partitioning, not big speedups.")

        #if rank == world - 1 and outputs is not None:
        #    print(f"[rank {rank}] pred {self.top1_label(outputs)}")

        #dist.destroy_process_group()
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


