
import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class test_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, 'layer3', torch.load(r'resnet18_children_4_1/test_folder_2/layer3.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'layer4', torch.load(r'resnet18_children_4_1/test_folder_2/layer4.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        self.load_state_dict(torch.load(r'resnet18_children_4_1/test_folder_2/state_dict.pt'))



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
        layer4_0_conv1_weight = getattr(self.layer4, "0").conv1.weight
        layer4_0_bn1_weight = getattr(self.layer4, "0").bn1.weight
        layer4_0_bn1_bias = getattr(self.layer4, "0").bn1.bias
        layer4_0_conv2_weight = getattr(self.layer4, "0").conv2.weight
        layer4_0_bn2_weight = getattr(self.layer4, "0").bn2.weight
        layer4_0_bn2_bias = getattr(self.layer4, "0").bn2.bias
        layer4_0_downsample_0_weight = getattr(getattr(self.layer4, "0").downsample, "0").weight
        layer4_0_downsample_1_weight = getattr(getattr(self.layer4, "0").downsample, "1").weight
        layer4_0_downsample_1_bias = getattr(getattr(self.layer4, "0").downsample, "1").bias
        layer4_1_conv1_weight = getattr(self.layer4, "1").conv1.weight
        layer4_1_bn1_weight = getattr(self.layer4, "1").bn1.weight
        layer4_1_bn1_bias = getattr(self.layer4, "1").bn1.bias
        layer4_1_conv2_weight = getattr(self.layer4, "1").conv2.weight
        layer4_1_bn2_weight = getattr(self.layer4, "1").bn2.weight
        layer4_1_bn2_bias = getattr(self.layer4, "1").bn2.bias
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
        layer4_0_bn1_running_mean = getattr(self.layer4, "0").bn1.running_mean
        layer4_0_bn1_running_var = getattr(self.layer4, "0").bn1.running_var
        layer4_0_bn2_running_mean = getattr(self.layer4, "0").bn2.running_mean
        layer4_0_bn2_running_var = getattr(self.layer4, "0").bn2.running_var
        layer4_0_downsample_1_running_mean = getattr(getattr(self.layer4, "0").downsample, "1").running_mean
        layer4_0_downsample_1_running_var = getattr(getattr(self.layer4, "0").downsample, "1").running_var
        layer4_1_bn1_running_mean = getattr(self.layer4, "1").bn1.running_mean
        layer4_1_bn1_running_var = getattr(self.layer4, "1").bn1.running_var
        layer4_1_bn2_running_mean = getattr(self.layer4, "1").bn2.running_mean
        layer4_1_bn2_running_var = getattr(self.layer4, "1").bn2.running_var
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
        conv2d_5 = torch.ops.aten.conv2d.default(relu__3, layer4_0_conv1_weight, None, [2, 2], [1, 1]);  layer4_0_conv1_weight = None
        batch_norm_5 = torch.ops.aten.batch_norm.default(conv2d_5, layer4_0_bn1_weight, layer4_0_bn1_bias, layer4_0_bn1_running_mean, layer4_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_5 = layer4_0_bn1_weight = layer4_0_bn1_bias = layer4_0_bn1_running_mean = layer4_0_bn1_running_var = None
        relu__4 = torch.ops.aten.relu_.default(batch_norm_5);  batch_norm_5 = None
        conv2d_6 = torch.ops.aten.conv2d.default(relu__4, layer4_0_conv2_weight, None, [1, 1], [1, 1]);  relu__4 = layer4_0_conv2_weight = None
        batch_norm_6 = torch.ops.aten.batch_norm.default(conv2d_6, layer4_0_bn2_weight, layer4_0_bn2_bias, layer4_0_bn2_running_mean, layer4_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_6 = layer4_0_bn2_weight = layer4_0_bn2_bias = layer4_0_bn2_running_mean = layer4_0_bn2_running_var = None
        conv2d_7 = torch.ops.aten.conv2d.default(relu__3, layer4_0_downsample_0_weight, None, [2, 2]);  relu__3 = layer4_0_downsample_0_weight = None
        batch_norm_7 = torch.ops.aten.batch_norm.default(conv2d_7, layer4_0_downsample_1_weight, layer4_0_downsample_1_bias, layer4_0_downsample_1_running_mean, layer4_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_7 = layer4_0_downsample_1_weight = layer4_0_downsample_1_bias = layer4_0_downsample_1_running_mean = layer4_0_downsample_1_running_var = None
        add__2 = torch.ops.aten.add_.Tensor(batch_norm_6, batch_norm_7);  batch_norm_6 = batch_norm_7 = None
        relu__5 = torch.ops.aten.relu_.default(add__2);  add__2 = None
        conv2d_8 = torch.ops.aten.conv2d.default(relu__5, layer4_1_conv1_weight, None, [1, 1], [1, 1]);  layer4_1_conv1_weight = None
        batch_norm_8 = torch.ops.aten.batch_norm.default(conv2d_8, layer4_1_bn1_weight, layer4_1_bn1_bias, layer4_1_bn1_running_mean, layer4_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_8 = layer4_1_bn1_weight = layer4_1_bn1_bias = layer4_1_bn1_running_mean = layer4_1_bn1_running_var = None
        relu__6 = torch.ops.aten.relu_.default(batch_norm_8);  batch_norm_8 = None
        conv2d_9 = torch.ops.aten.conv2d.default(relu__6, layer4_1_conv2_weight, None, [1, 1], [1, 1]);  relu__6 = layer4_1_conv2_weight = None
        batch_norm_9 = torch.ops.aten.batch_norm.default(conv2d_9, layer4_1_bn2_weight, layer4_1_bn2_bias, layer4_1_bn2_running_mean, layer4_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_9 = layer4_1_bn2_weight = layer4_1_bn2_bias = layer4_1_bn2_running_mean = layer4_1_bn2_running_var = None
        add__3 = torch.ops.aten.add_.Tensor(batch_norm_9, relu__5);  batch_norm_9 = relu__5 = None
        relu__7 = torch.ops.aten.relu_.default(add__3);  add__3 = None
        return pytree.tree_unflatten((relu__7,), self._out_spec)

