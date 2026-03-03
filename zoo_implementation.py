#simple ZOO implementation
from __future__ import annotations

import time
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset

import torch.distributed
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
        full_train = []
        test_dataset = []
        if set_name == "cifar10":
            full_train = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=test_transform)
        
        indices = torch.randperm(len(full_train))[:subset_size]
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
        indices = torch.randperm(len(self.train_data))[:subset_len]
        small_train_dataset = Subset(self.train_data, indices)
        self.train_data = DataLoader(small_train_dataset, batch_size=32, shuffle=True)

@dataclass 
class GenModel:
    model: nn.Module
    def freeze_layers_until(self, layer_depth:int=1, freeze_layer_names:List[str] = None):
        model = self.model 
        #freeze appropriare layers
        l_counter = -1
        
        for l, layer in self.model.named_children():
            l_counter+=1
            if l_counter==layer_depth and freeze_layer_names==None:
                break
            if (freeze_layer_names==None) or (str(l) in freeze_layer_names):
                for param in layer.parameters():
                    param.requires_grad = False
        self.model = model 
    
    def num_classes(self, num: int=10):
        model = self.model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num)
        self.model = model
    
    def save_model():
        pass

@dataclass
class FBModel(GenModel):

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


# @dataclass
# class ZOOModel(GenModel):
#     def train_model(self, train_loader: Any, lr=1e-3, num_epochs:int=10):
#         model=self.model
#         # Optimizer setup
#         #TODO do we stick with Adam? Or switch to SGD that DeepZero does?
#         #TODO do we add a learning rate scheduler? is that required for fine-tuning/transfer learning?
#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#         criterion = torch.nn.CrossEntropyLoss()
#         epoch = -1
#         while epoch < num_epochs:
#             #TODO what does cge_weight_to_allocate_process do here?

#             model.train() #happens after cge allocate for some reason?
#             epoch+=1
#             pbar = tqdm(train_loader, total=len(train_loader),
#                     desc=f"Epoch {epoch} Training")
#             s=time.time()
#             total=0
#             correct = 0
#             for i, (inputs, labels) in enumerate(pbar):
#             # Benchmark training with frozen layers
#                 #TODO might need to assign x, y to device?
#                 optimizer.zero_grad()
#                 #replacing loss.backwards()
#                 with torch.no_grad():
#                     #TODO might need to modify how forward works?
#                     fx = model(inputs)
#                     loss_batch = F.cross_entropy(fx, labels).cpu()
#                 self.cge_calc(lr)


#                 optimizer.step()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 pbar.set_postfix_str(f"Lr {lr:.2e} Loss: {round(loss.item(),3)} Acc {100*(correct/total):.2f}%")
#             e = time.time()
#             epoch_time = e-s
#             pbar.set_postfix_str(f"Epoch time: {epoch_time}")

#     def cge_calc(self, lr:float):
#         params_dict = {name: p for name, p in self.model.named_parameters() if p.requires_grad}







if __name__ == "__main__":
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights)
    print({k:v for k,v in pretrained_model.named_modules() if k!=''})
    exit()
    pretrained_transforms = weights.transforms()  # built-in preprocessing transforms

    # Augmentation + Preprocessing for Training
    train_transform = transforms.Compose([
        transforms.Resize(224),  # ensure size is correct
        transforms.RandomHorizontalFlip(p=0.5),  # augmentation
        transforms.RandomRotation(15),  # augmentation
        transforms.ColorJitter(0.2, 0.2, 0.2),  # augmentation
        pretrained_transforms  # built-in normalization
    ])

    test_transform = pretrained_transforms
    dl = DataWrap.dataloader_gen(train_transform=train_transform, test_transform=test_transform)

    bp_mod = FBModel(pretrained_model)
    bp_mod.freeze_layers_until(1)
    bp_mod.num_classes(10)
    bp_mod.train_model(dl.set_train_loader)
    acc = bp_mod.evaluate_model(dl.set_test_loader)
    print(acc)

    # bp_mod.num_classes(100)
    # print(bp_mod.model.fc.in_features, bp_mod.model.fc.out_features)
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
    # bp_mod.freeze_layers_until(5)
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
    # bp_mod.freeze_layers_until(freeze_layer_names=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
