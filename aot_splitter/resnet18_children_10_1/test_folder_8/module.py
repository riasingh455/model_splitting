
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test_8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict(torch.load(r'resnet18_children_10_1/test_folder_8/state_dict.pt'))



    def forward(self, relu__16):
        relu__16, = fx_pytree.tree_flatten_spec(([relu__16], {}), self._in_spec)
        adaptive_avg_pool2d = torch.ops.aten.adaptive_avg_pool2d.default(relu__16, [1, 1]);  relu__16 = None
        return pytree.tree_unflatten((adaptive_avg_pool2d,), self._out_spec)

