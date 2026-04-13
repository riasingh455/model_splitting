
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
        self.load_state_dict(torch.load(r'resnet18_children_9_1/test_folder_1/state_dict.pt'))



    def forward(self, batch_norm):
        batch_norm, = fx_pytree.tree_flatten_spec(([batch_norm], {}), self._in_spec)
        relu_ = torch.ops.aten.relu_.default(batch_norm);  batch_norm = None
        return pytree.tree_unflatten((relu_,), self._out_spec)

