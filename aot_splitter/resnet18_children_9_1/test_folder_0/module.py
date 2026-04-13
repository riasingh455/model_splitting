
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, 'conv1', torch.load(r'resnet18_children_9_1/test_folder_0/conv1.pt', weights_only=False)) # Module()
        setattr(self, 'bn1', torch.load(r'resnet18_children_9_1/test_folder_0/bn1.pt', weights_only=False)) # Module()
        self.load_state_dict(torch.load(r'resnet18_children_9_1/test_folder_0/state_dict.pt'))



    def forward(self, x):
        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        conv1_weight = self.conv1.weight
        bn1_weight = self.bn1.weight
        bn1_bias = self.bn1.bias
        bn1_running_mean = self.bn1.running_mean
        bn1_running_var = self.bn1.running_var
        conv2d = torch.ops.aten.conv2d.default(x, conv1_weight, None, [2, 2], [3, 3]);  x = conv1_weight = None
        batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, False, 0.1, 1e-05, False);  conv2d = bn1_weight = bn1_bias = bn1_running_mean = bn1_running_var = None
        return pytree.tree_unflatten((batch_norm,), self._out_spec)

