
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test_1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, 'bn1', torch.load(r'resnet18_children_10_1/test_folder_1/bn1.pt', weights_only=False)) # Module()
        self.load_state_dict(torch.load(r'resnet18_children_10_1/test_folder_1/state_dict.pt'))



    def forward(self, conv2d):
        conv2d, = fx_pytree.tree_flatten_spec(([conv2d], {}), self._in_spec)
        bn1_weight = self.bn1.weight
        bn1_bias = self.bn1.bias
        bn1_running_mean = self.bn1.running_mean
        bn1_running_var = self.bn1.running_var
        batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, False, 0.1, 1e-05, False);  conv2d = bn1_weight = bn1_bias = bn1_running_mean = bn1_running_var = None
        return pytree.tree_unflatten((batch_norm,), self._out_spec)

