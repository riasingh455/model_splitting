
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test_4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, 'fc', torch.load(r'resnet18_children_5_1/test_folder_4/fc.pt', weights_only=False)) # Module()
        self.load_state_dict(torch.load(r'resnet18_children_5_1/test_folder_4/state_dict.pt'))



    def forward(self, relu__16):
        relu__16, = fx_pytree.tree_flatten_spec(([relu__16], {}), self._in_spec)
        fc_weight = self.fc.weight
        fc_bias = self.fc.bias
        adaptive_avg_pool2d = torch.ops.aten.adaptive_avg_pool2d.default(relu__16, [1, 1]);  relu__16 = None
        flatten = torch.ops.aten.flatten.using_ints(adaptive_avg_pool2d, 1);  adaptive_avg_pool2d = None
        linear = torch.ops.aten.linear.default(flatten, fc_weight, fc_bias);  flatten = fc_weight = fc_bias = None
        return pytree.tree_unflatten((linear,), self._out_spec)

