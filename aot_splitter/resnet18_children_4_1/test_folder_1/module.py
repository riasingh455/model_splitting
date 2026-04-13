
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
        setattr(self, 'layer1', torch.load(r'resnet18_children_4_1/test_folder_1/layer1.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'layer2', torch.load(r'resnet18_children_4_1/test_folder_1/layer2.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        self.load_state_dict(torch.load(r'resnet18_children_4_1/test_folder_1/state_dict.pt'))



    def forward(self, relu_):
        relu_, = fx_pytree.tree_flatten_spec(([relu_], {}), self._in_spec)
        layer1_0_conv1_weight = getattr(self.layer1, "0").conv1.weight
        layer1_0_bn1_weight = getattr(self.layer1, "0").bn1.weight
        layer1_0_bn1_bias = getattr(self.layer1, "0").bn1.bias
        layer1_0_conv2_weight = getattr(self.layer1, "0").conv2.weight
        layer1_0_bn2_weight = getattr(self.layer1, "0").bn2.weight
        layer1_0_bn2_bias = getattr(self.layer1, "0").bn2.bias
        layer1_1_conv1_weight = getattr(self.layer1, "1").conv1.weight
        layer1_1_bn1_weight = getattr(self.layer1, "1").bn1.weight
        layer1_1_bn1_bias = getattr(self.layer1, "1").bn1.bias
        layer1_1_conv2_weight = getattr(self.layer1, "1").conv2.weight
        layer1_1_bn2_weight = getattr(self.layer1, "1").bn2.weight
        layer1_1_bn2_bias = getattr(self.layer1, "1").bn2.bias
        layer2_0_conv1_weight = getattr(self.layer2, "0").conv1.weight
        layer2_0_bn1_weight = getattr(self.layer2, "0").bn1.weight
        layer2_0_bn1_bias = getattr(self.layer2, "0").bn1.bias
        layer2_0_conv2_weight = getattr(self.layer2, "0").conv2.weight
        layer2_0_bn2_weight = getattr(self.layer2, "0").bn2.weight
        layer2_0_bn2_bias = getattr(self.layer2, "0").bn2.bias
        layer2_0_downsample_0_weight = getattr(getattr(self.layer2, "0").downsample, "0").weight
        layer2_0_downsample_1_weight = getattr(getattr(self.layer2, "0").downsample, "1").weight
        layer2_0_downsample_1_bias = getattr(getattr(self.layer2, "0").downsample, "1").bias
        layer2_1_conv1_weight = getattr(self.layer2, "1").conv1.weight
        layer2_1_bn1_weight = getattr(self.layer2, "1").bn1.weight
        layer2_1_bn1_bias = getattr(self.layer2, "1").bn1.bias
        layer2_1_conv2_weight = getattr(self.layer2, "1").conv2.weight
        layer2_1_bn2_weight = getattr(self.layer2, "1").bn2.weight
        layer2_1_bn2_bias = getattr(self.layer2, "1").bn2.bias
        layer1_0_bn1_running_mean = getattr(self.layer1, "0").bn1.running_mean
        layer1_0_bn1_running_var = getattr(self.layer1, "0").bn1.running_var
        layer1_0_bn2_running_mean = getattr(self.layer1, "0").bn2.running_mean
        layer1_0_bn2_running_var = getattr(self.layer1, "0").bn2.running_var
        layer1_1_bn1_running_mean = getattr(self.layer1, "1").bn1.running_mean
        layer1_1_bn1_running_var = getattr(self.layer1, "1").bn1.running_var
        layer1_1_bn2_running_mean = getattr(self.layer1, "1").bn2.running_mean
        layer1_1_bn2_running_var = getattr(self.layer1, "1").bn2.running_var
        layer2_0_bn1_running_mean = getattr(self.layer2, "0").bn1.running_mean
        layer2_0_bn1_running_var = getattr(self.layer2, "0").bn1.running_var
        layer2_0_bn2_running_mean = getattr(self.layer2, "0").bn2.running_mean
        layer2_0_bn2_running_var = getattr(self.layer2, "0").bn2.running_var
        layer2_0_downsample_1_running_mean = getattr(getattr(self.layer2, "0").downsample, "1").running_mean
        layer2_0_downsample_1_running_var = getattr(getattr(self.layer2, "0").downsample, "1").running_var
        layer2_1_bn1_running_mean = getattr(self.layer2, "1").bn1.running_mean
        layer2_1_bn1_running_var = getattr(self.layer2, "1").bn1.running_var
        layer2_1_bn2_running_mean = getattr(self.layer2, "1").bn2.running_mean
        layer2_1_bn2_running_var = getattr(self.layer2, "1").bn2.running_var
        max_pool2d = torch.ops.aten.max_pool2d.default(relu_, [3, 3], [2, 2], [1, 1]);  relu_ = None
        conv2d = torch.ops.aten.conv2d.default(max_pool2d, layer1_0_conv1_weight, None, [1, 1], [1, 1]);  layer1_0_conv1_weight = None
        batch_norm = torch.ops.aten.batch_norm.default(conv2d, layer1_0_bn1_weight, layer1_0_bn1_bias, layer1_0_bn1_running_mean, layer1_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d = layer1_0_bn1_weight = layer1_0_bn1_bias = layer1_0_bn1_running_mean = layer1_0_bn1_running_var = None
        relu__1 = torch.ops.aten.relu_.default(batch_norm);  batch_norm = None
        conv2d_1 = torch.ops.aten.conv2d.default(relu__1, layer1_0_conv2_weight, None, [1, 1], [1, 1]);  relu__1 = layer1_0_conv2_weight = None
        batch_norm_1 = torch.ops.aten.batch_norm.default(conv2d_1, layer1_0_bn2_weight, layer1_0_bn2_bias, layer1_0_bn2_running_mean, layer1_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_1 = layer1_0_bn2_weight = layer1_0_bn2_bias = layer1_0_bn2_running_mean = layer1_0_bn2_running_var = None
        add_ = torch.ops.aten.add_.Tensor(batch_norm_1, max_pool2d);  batch_norm_1 = max_pool2d = None
        relu__2 = torch.ops.aten.relu_.default(add_);  add_ = None
        conv2d_2 = torch.ops.aten.conv2d.default(relu__2, layer1_1_conv1_weight, None, [1, 1], [1, 1]);  layer1_1_conv1_weight = None
        batch_norm_2 = torch.ops.aten.batch_norm.default(conv2d_2, layer1_1_bn1_weight, layer1_1_bn1_bias, layer1_1_bn1_running_mean, layer1_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_2 = layer1_1_bn1_weight = layer1_1_bn1_bias = layer1_1_bn1_running_mean = layer1_1_bn1_running_var = None
        relu__3 = torch.ops.aten.relu_.default(batch_norm_2);  batch_norm_2 = None
        conv2d_3 = torch.ops.aten.conv2d.default(relu__3, layer1_1_conv2_weight, None, [1, 1], [1, 1]);  relu__3 = layer1_1_conv2_weight = None
        batch_norm_3 = torch.ops.aten.batch_norm.default(conv2d_3, layer1_1_bn2_weight, layer1_1_bn2_bias, layer1_1_bn2_running_mean, layer1_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_3 = layer1_1_bn2_weight = layer1_1_bn2_bias = layer1_1_bn2_running_mean = layer1_1_bn2_running_var = None
        add__1 = torch.ops.aten.add_.Tensor(batch_norm_3, relu__2);  batch_norm_3 = relu__2 = None
        relu__4 = torch.ops.aten.relu_.default(add__1);  add__1 = None
        conv2d_4 = torch.ops.aten.conv2d.default(relu__4, layer2_0_conv1_weight, None, [2, 2], [1, 1]);  layer2_0_conv1_weight = None
        batch_norm_4 = torch.ops.aten.batch_norm.default(conv2d_4, layer2_0_bn1_weight, layer2_0_bn1_bias, layer2_0_bn1_running_mean, layer2_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_4 = layer2_0_bn1_weight = layer2_0_bn1_bias = layer2_0_bn1_running_mean = layer2_0_bn1_running_var = None
        relu__5 = torch.ops.aten.relu_.default(batch_norm_4);  batch_norm_4 = None
        conv2d_5 = torch.ops.aten.conv2d.default(relu__5, layer2_0_conv2_weight, None, [1, 1], [1, 1]);  relu__5 = layer2_0_conv2_weight = None
        batch_norm_5 = torch.ops.aten.batch_norm.default(conv2d_5, layer2_0_bn2_weight, layer2_0_bn2_bias, layer2_0_bn2_running_mean, layer2_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_5 = layer2_0_bn2_weight = layer2_0_bn2_bias = layer2_0_bn2_running_mean = layer2_0_bn2_running_var = None
        conv2d_6 = torch.ops.aten.conv2d.default(relu__4, layer2_0_downsample_0_weight, None, [2, 2]);  relu__4 = layer2_0_downsample_0_weight = None
        batch_norm_6 = torch.ops.aten.batch_norm.default(conv2d_6, layer2_0_downsample_1_weight, layer2_0_downsample_1_bias, layer2_0_downsample_1_running_mean, layer2_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_6 = layer2_0_downsample_1_weight = layer2_0_downsample_1_bias = layer2_0_downsample_1_running_mean = layer2_0_downsample_1_running_var = None
        add__2 = torch.ops.aten.add_.Tensor(batch_norm_5, batch_norm_6);  batch_norm_5 = batch_norm_6 = None
        relu__6 = torch.ops.aten.relu_.default(add__2);  add__2 = None
        conv2d_7 = torch.ops.aten.conv2d.default(relu__6, layer2_1_conv1_weight, None, [1, 1], [1, 1]);  layer2_1_conv1_weight = None
        batch_norm_7 = torch.ops.aten.batch_norm.default(conv2d_7, layer2_1_bn1_weight, layer2_1_bn1_bias, layer2_1_bn1_running_mean, layer2_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_7 = layer2_1_bn1_weight = layer2_1_bn1_bias = layer2_1_bn1_running_mean = layer2_1_bn1_running_var = None
        relu__7 = torch.ops.aten.relu_.default(batch_norm_7);  batch_norm_7 = None
        conv2d_8 = torch.ops.aten.conv2d.default(relu__7, layer2_1_conv2_weight, None, [1, 1], [1, 1]);  relu__7 = layer2_1_conv2_weight = None
        batch_norm_8 = torch.ops.aten.batch_norm.default(conv2d_8, layer2_1_bn2_weight, layer2_1_bn2_bias, layer2_1_bn2_running_mean, layer2_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_8 = layer2_1_bn2_weight = layer2_1_bn2_bias = layer2_1_bn2_running_mean = layer2_1_bn2_running_var = None
        add__3 = torch.ops.aten.add_.Tensor(batch_norm_8, relu__6);  batch_norm_8 = relu__6 = None
        relu__8 = torch.ops.aten.relu_.default(add__3);  add__3 = None
        return pytree.tree_unflatten((relu__8,), self._out_spec)

