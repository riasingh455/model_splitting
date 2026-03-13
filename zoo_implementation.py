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


@dataclass
class ZOOModel(GenModel):
    def train_model(self, train_loader: Any, lr=1e-3, num_epochs:int=10):
        model=self.model
        # Optimizer setup
        #TODO do we stick with Adam? Or switch to SGD that DeepZero does?
        #TODO do we add a learning rate scheduler? is that required for fine-tuning/transfer learning?
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        epoch = -1
        while epoch < num_epochs:

            #depending on sparsity and pruning -> from deepzero, can be done here
            #TODO pruning for us is equivalent to "unfreezing" some layers/parts of layers
            #also called mask -> if param has a mask skip training, else perturb 

            #next we do cge_weight_to_allocate -> 
            #dict of params and masked params
            #tracks params to be perturbed (for some reason also changes original to mask?)
            #makes a tensor list to keep track of indices of params to be manipulated -> not entirely sure why flatten?
            #these are the params to be perturbed, sent to respective devices there (we don't necessarily need to do this)
            #TODO what does weighted allocate do?
            #when we sent params to be perturbed we set the instruction here too  
            #TODO what does cge_weight_to_allocate_process do here?

            model.train() #happens after cge allocate for some reason?
            epoch+=1
            pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch} Training")
            s=time.time()
            total=0
            correct = 0
            for i, (inputs, labels) in enumerate(pbar):
            # Benchmark training with frozen layers
                #TODO might need to assign x, y to device?
                #TODO warmups here

                optimizer.zero_grad()
                #replacing loss.backwards()
                with torch.no_grad():
                    #TODO might need to modify how forward works?
                    outputs = model(inputs)
                    loss_batch = F.cross_entropy(outputs, labels).cpu()
                #TODO if an LR schedulers, get lr here
                #here's the cge calculation
                #sends the x(inputs),y(labels) to each remote device (not necessariy for us)
                #then run calculate_grads -> 
                #once again specify torch.no_grad() here within calculate_grads?
                #pass through the network again, add it to the input list
                #so input list = [x] -> becomes -> [x, passed_through_x] -> might not need this? could be part of feature-reuse?
                #get loss here too -> base loss
                #have an intruction ??? -> TODO where assigned? ->set_params_to_be_perturbed -> happens during cge_weight_to_allocate_process above
                #per instruction -> have an index
                #loop starts here, per instruction to be perturbed
                    #index is used to match and perturb a parameter -> TODO what is perturb_a_param? -> literally adds or subtracts(reset) the lr to the weights/params per index specified (from instruction)
                    #pass through the model when perturbed 
                    #reset pertubration
                    #get loss from petrubed_pass_through_x, y(labels)
                    #add this to a list of calculated gradient losses -> grad_list ??? -> isn't this a list of lists?
                    #repeat loop
                #end loop
                #get difference between grads_list and base, divided by lr -> again how is this difference the right size? grad_list should be a list of lists no?
                #concatenate grads to instruction -> TODO how is grads decided here and how is the difference between base used?
                #return instr_grad concat
                #concat all the received instr_grads from every device (we don't necessarily need to do this)
                #assign each gradient to it's param index? -> TODO figure out this last part?
                #TODO what does network synthesis actually do? 
                #seems like it's regenerating the next params to update? 

                #okay think I got it
                #essentially, for each parallel model calculate the loss, upgrade the gradients for the appropriate parameters with that loss (when on the machine with the single model)
                #run optimizer.step() -> this updates the data/parameters 
                #then send the updated parameters to the parallel models in network_synchronize!
                self.cge_calc(model, inputs, labels, lr)


                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix_str(f"Lr {lr:.2e} Loss: {round(loss.item(),3)} Acc {100*(correct/total):.2f}%")
            e = time.time()
            epoch_time = e-s
            pbar.set_postfix_str(f"Epoch time: {epoch_time}")

        #TODO add test set run here?

    def cge_calc(self, model:nn.Module, inputs, outputs, lr:float):
        params_dict = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        print(params_dict.keys())





if __name__ == "__main__":
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights)
    # print({k:v for k,v in pretrained_model.named_modules() if k!=''})
    # exit()
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

    zoo_mod = ZOOModel(pretrained_model)
    zoo_mod.cge_calc(pretrained_model, [], [], 0.1)
    zoo_mod.freeze_layers_until(7)
    zoo_mod.cge_calc(zoo_mod.model, [], [], 0.1)
    #Regular SGD
    # bp_mod = FBModel(pretrained_model)

    # bp_mod.freeze_layers_until(1)
    # bp_mod.num_classes(10)
    # bp_mod.train_model(dl.set_train_loader)
    # acc = bp_mod.evaluate_model(dl.set_test_loader)
    # print(acc)

    # bp_mod.num_classes(100)
    # print(bp_mod.model.fc.in_features, bp_mod.model.fc.out_features)
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
    # bp_mod.freeze_layers_until(5)
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
    # bp_mod.freeze_layers_until(freeze_layer_names=['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc'])
    # print([ [m, [p.requires_grad for p in ml.parameters()]] for m,ml in bp_mod.model.named_children()])
