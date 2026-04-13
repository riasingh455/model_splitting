
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
        setattr(self, 'layer3', torch.load(r'resnet18_children_8_1/test_folder_4/layer3.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        self.load_state_dict(torch.load(r'resnet18_children_8_1/test_folder_4/state_dict.pt'))



    def forward(self, relu__8):
        relu__8, = fx_pytree.tree_flatten_spec(([relu__8], {}), self._in_spec)
        layer3_0_conv1_weight = getattr(self.layer3, "0").conv1.weight
        layer3_0_bn1_weight = getattr(self.layer3, "0").bn1.weight
        layer3_0_bn1_bias = getattr(self.layer3, "0").bn1.bias
        layer3_0_conv2_weight = getattr(self.layer3, "0").conv2.weight
        layer3_0_bn2_weight = getattr(self.layer3, "0").bn2.weight
        layer3_0_bn2_bias = getattr(self.layer3, "0").bn2.bias
        layer3_0_downsample_0_weight = getattr(getattr(self.layer3, "0").downsample, "0").weight
        layer3_0_downsample_1_weight = getattr(getattr(self.layer3, "0").downsample, "1").weight
        layer3_0_downsample_1_bias = getattr(getattr(self.layer3, "0").downsample, "1").bias
        layer3_1_conv1_weight = getattr(self.layer3, "1").conv1.weight
        layer3_1_bn1_weight = getattr(self.layer3, "1").bn1.weight
        layer3_1_bn1_bias = getattr(self.layer3, "1").bn1.bias
        layer3_1_conv2_weight = getattr(self.layer3, "1").conv2.weight
        layer3_1_bn2_weight = getattr(self.layer3, "1").bn2.weight
        layer3_1_bn2_bias = getattr(self.layer3, "1").bn2.bias
        layer3_0_bn1_running_mean = getattr(self.layer3, "0").bn1.running_mean
        layer3_0_bn1_running_var = getattr(self.layer3, "0").bn1.running_var
        layer3_0_bn2_running_mean = getattr(self.layer3, "0").bn2.running_mean
        layer3_0_bn2_running_var = getattr(self.layer3, "0").bn2.running_var
        layer3_0_downsample_1_running_mean = getattr(getattr(self.layer3, "0").downsample, "1").running_mean
        layer3_0_downsample_1_running_var = getattr(getattr(self.layer3, "0").downsample, "1").running_var
        layer3_1_bn1_running_mean = getattr(self.layer3, "1").bn1.running_mean
        layer3_1_bn1_running_var = getattr(self.layer3, "1").bn1.running_var
        layer3_1_bn2_running_mean = getattr(self.layer3, "1").bn2.running_mean
        layer3_1_bn2_running_var = getattr(self.layer3, "1").bn2.running_var
        conv2d = torch.ops.aten.conv2d.default(relu__8, layer3_0_conv1_weight, None, [2, 2], [1, 1]);  layer3_0_conv1_weight = None
        batch_norm = torch.ops.aten.batch_norm.default(conv2d, layer3_0_bn1_weight, layer3_0_bn1_bias, layer3_0_bn1_running_mean, layer3_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d = layer3_0_bn1_weight = layer3_0_bn1_bias = layer3_0_bn1_running_mean = layer3_0_bn1_running_var = None
        relu_ = torch.ops.aten.relu_.default(batch_norm);  batch_norm = None
        conv2d_1 = torch.ops.aten.conv2d.default(relu_, layer3_0_conv2_weight, None, [1, 1], [1, 1]);  relu_ = layer3_0_conv2_weight = None
        batch_norm_1 = torch.ops.aten.batch_norm.default(conv2d_1, layer3_0_bn2_weight, layer3_0_bn2_bias, layer3_0_bn2_running_mean, layer3_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_1 = layer3_0_bn2_weight = layer3_0_bn2_bias = layer3_0_bn2_running_mean = layer3_0_bn2_running_var = None
        conv2d_2 = torch.ops.aten.conv2d.default(relu__8, layer3_0_downsample_0_weight, None, [2, 2]);  relu__8 = layer3_0_downsample_0_weight = None
        batch_norm_2 = torch.ops.aten.batch_norm.default(conv2d_2, layer3_0_downsample_1_weight, layer3_0_downsample_1_bias, layer3_0_downsample_1_running_mean, layer3_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_2 = layer3_0_downsample_1_weight = layer3_0_downsample_1_bias = layer3_0_downsample_1_running_mean = layer3_0_downsample_1_running_var = None
        add_ = torch.ops.aten.add_.Tensor(batch_norm_1, batch_norm_2);  batch_norm_1 = batch_norm_2 = None
        relu__1 = torch.ops.aten.relu_.default(add_);  add_ = None
        conv2d_3 = torch.ops.aten.conv2d.default(relu__1, layer3_1_conv1_weight, None, [1, 1], [1, 1]);  layer3_1_conv1_weight = None
        batch_norm_3 = torch.ops.aten.batch_norm.default(conv2d_3, layer3_1_bn1_weight, layer3_1_bn1_bias, layer3_1_bn1_running_mean, layer3_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_3 = layer3_1_bn1_weight = layer3_1_bn1_bias = layer3_1_bn1_running_mean = layer3_1_bn1_running_var = None
        relu__2 = torch.ops.aten.relu_.default(batch_norm_3);  batch_norm_3 = None
        conv2d_4 = torch.ops.aten.conv2d.default(relu__2, layer3_1_conv2_weight, None, [1, 1], [1, 1]);  relu__2 = layer3_1_conv2_weight = None
        batch_norm_4 = torch.ops.aten.batch_norm.default(conv2d_4, layer3_1_bn2_weight, layer3_1_bn2_bias, layer3_1_bn2_running_mean, layer3_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_4 = layer3_1_bn2_weight = layer3_1_bn2_bias = layer3_1_bn2_running_mean = layer3_1_bn2_running_var = None
        add__1 = torch.ops.aten.add_.Tensor(batch_norm_4, relu__1);  batch_norm_4 = relu__1 = None
        relu__3 = torch.ops.aten.relu_.default(add__1);  add__1 = None
        return pytree.tree_unflatten((relu__3,), self._out_spec)

