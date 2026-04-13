
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
        setattr(self, 'conv1', torch.load(r'resnet18_children_1_1/test_folder_0/conv1.pt', weights_only=False)) # Module()
        setattr(self, 'bn1', torch.load(r'resnet18_children_1_1/test_folder_0/bn1.pt', weights_only=False)) # Module()
        setattr(self, 'layer1', torch.load(r'resnet18_children_1_1/test_folder_0/layer1.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'layer2', torch.load(r'resnet18_children_1_1/test_folder_0/layer2.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'layer3', torch.load(r'resnet18_children_1_1/test_folder_0/layer3.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'layer4', torch.load(r'resnet18_children_1_1/test_folder_0/layer4.pt', weights_only=False)) # Module(   (0): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()     (downsample): Module(       (0): Module()       (1): Module()     )   )   (1): Module(     (conv1): Module()     (bn1): Module()     (conv2): Module()     (bn2): Module()   ) )
        setattr(self, 'fc', torch.load(r'resnet18_children_1_1/test_folder_0/fc.pt', weights_only=False)) # Module()
        self.load_state_dict(torch.load(r'resnet18_children_1_1/test_folder_0/state_dict.pt'))



    def forward(self, x):
        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        conv1_weight = self.conv1.weight
        bn1_weight = self.bn1.weight
        bn1_bias = self.bn1.bias
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
        fc_weight = self.fc.weight
        fc_bias = self.fc.bias
        bn1_running_mean = self.bn1.running_mean
        bn1_running_var = self.bn1.running_var
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
        conv2d = torch.ops.aten.conv2d.default(x, conv1_weight, None, [2, 2], [3, 3]);  x = conv1_weight = None
        batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, False, 0.1, 1e-05, False);  conv2d = bn1_weight = bn1_bias = bn1_running_mean = bn1_running_var = None
        relu_ = torch.ops.aten.relu_.default(batch_norm);  batch_norm = None
        max_pool2d = torch.ops.aten.max_pool2d.default(relu_, [3, 3], [2, 2], [1, 1]);  relu_ = None
        conv2d_1 = torch.ops.aten.conv2d.default(max_pool2d, layer1_0_conv1_weight, None, [1, 1], [1, 1]);  layer1_0_conv1_weight = None
        batch_norm_1 = torch.ops.aten.batch_norm.default(conv2d_1, layer1_0_bn1_weight, layer1_0_bn1_bias, layer1_0_bn1_running_mean, layer1_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_1 = layer1_0_bn1_weight = layer1_0_bn1_bias = layer1_0_bn1_running_mean = layer1_0_bn1_running_var = None
        relu__1 = torch.ops.aten.relu_.default(batch_norm_1);  batch_norm_1 = None
        conv2d_2 = torch.ops.aten.conv2d.default(relu__1, layer1_0_conv2_weight, None, [1, 1], [1, 1]);  relu__1 = layer1_0_conv2_weight = None
        batch_norm_2 = torch.ops.aten.batch_norm.default(conv2d_2, layer1_0_bn2_weight, layer1_0_bn2_bias, layer1_0_bn2_running_mean, layer1_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_2 = layer1_0_bn2_weight = layer1_0_bn2_bias = layer1_0_bn2_running_mean = layer1_0_bn2_running_var = None
        add_ = torch.ops.aten.add_.Tensor(batch_norm_2, max_pool2d);  batch_norm_2 = max_pool2d = None
        relu__2 = torch.ops.aten.relu_.default(add_);  add_ = None
        conv2d_3 = torch.ops.aten.conv2d.default(relu__2, layer1_1_conv1_weight, None, [1, 1], [1, 1]);  layer1_1_conv1_weight = None
        batch_norm_3 = torch.ops.aten.batch_norm.default(conv2d_3, layer1_1_bn1_weight, layer1_1_bn1_bias, layer1_1_bn1_running_mean, layer1_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_3 = layer1_1_bn1_weight = layer1_1_bn1_bias = layer1_1_bn1_running_mean = layer1_1_bn1_running_var = None
        relu__3 = torch.ops.aten.relu_.default(batch_norm_3);  batch_norm_3 = None
        conv2d_4 = torch.ops.aten.conv2d.default(relu__3, layer1_1_conv2_weight, None, [1, 1], [1, 1]);  relu__3 = layer1_1_conv2_weight = None
        batch_norm_4 = torch.ops.aten.batch_norm.default(conv2d_4, layer1_1_bn2_weight, layer1_1_bn2_bias, layer1_1_bn2_running_mean, layer1_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_4 = layer1_1_bn2_weight = layer1_1_bn2_bias = layer1_1_bn2_running_mean = layer1_1_bn2_running_var = None
        add__1 = torch.ops.aten.add_.Tensor(batch_norm_4, relu__2);  batch_norm_4 = relu__2 = None
        relu__4 = torch.ops.aten.relu_.default(add__1);  add__1 = None
        conv2d_5 = torch.ops.aten.conv2d.default(relu__4, layer2_0_conv1_weight, None, [2, 2], [1, 1]);  layer2_0_conv1_weight = None
        batch_norm_5 = torch.ops.aten.batch_norm.default(conv2d_5, layer2_0_bn1_weight, layer2_0_bn1_bias, layer2_0_bn1_running_mean, layer2_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_5 = layer2_0_bn1_weight = layer2_0_bn1_bias = layer2_0_bn1_running_mean = layer2_0_bn1_running_var = None
        relu__5 = torch.ops.aten.relu_.default(batch_norm_5);  batch_norm_5 = None
        conv2d_6 = torch.ops.aten.conv2d.default(relu__5, layer2_0_conv2_weight, None, [1, 1], [1, 1]);  relu__5 = layer2_0_conv2_weight = None
        batch_norm_6 = torch.ops.aten.batch_norm.default(conv2d_6, layer2_0_bn2_weight, layer2_0_bn2_bias, layer2_0_bn2_running_mean, layer2_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_6 = layer2_0_bn2_weight = layer2_0_bn2_bias = layer2_0_bn2_running_mean = layer2_0_bn2_running_var = None
        conv2d_7 = torch.ops.aten.conv2d.default(relu__4, layer2_0_downsample_0_weight, None, [2, 2]);  relu__4 = layer2_0_downsample_0_weight = None
        batch_norm_7 = torch.ops.aten.batch_norm.default(conv2d_7, layer2_0_downsample_1_weight, layer2_0_downsample_1_bias, layer2_0_downsample_1_running_mean, layer2_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_7 = layer2_0_downsample_1_weight = layer2_0_downsample_1_bias = layer2_0_downsample_1_running_mean = layer2_0_downsample_1_running_var = None
        add__2 = torch.ops.aten.add_.Tensor(batch_norm_6, batch_norm_7);  batch_norm_6 = batch_norm_7 = None
        relu__6 = torch.ops.aten.relu_.default(add__2);  add__2 = None
        conv2d_8 = torch.ops.aten.conv2d.default(relu__6, layer2_1_conv1_weight, None, [1, 1], [1, 1]);  layer2_1_conv1_weight = None
        batch_norm_8 = torch.ops.aten.batch_norm.default(conv2d_8, layer2_1_bn1_weight, layer2_1_bn1_bias, layer2_1_bn1_running_mean, layer2_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_8 = layer2_1_bn1_weight = layer2_1_bn1_bias = layer2_1_bn1_running_mean = layer2_1_bn1_running_var = None
        relu__7 = torch.ops.aten.relu_.default(batch_norm_8);  batch_norm_8 = None
        conv2d_9 = torch.ops.aten.conv2d.default(relu__7, layer2_1_conv2_weight, None, [1, 1], [1, 1]);  relu__7 = layer2_1_conv2_weight = None
        batch_norm_9 = torch.ops.aten.batch_norm.default(conv2d_9, layer2_1_bn2_weight, layer2_1_bn2_bias, layer2_1_bn2_running_mean, layer2_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_9 = layer2_1_bn2_weight = layer2_1_bn2_bias = layer2_1_bn2_running_mean = layer2_1_bn2_running_var = None
        add__3 = torch.ops.aten.add_.Tensor(batch_norm_9, relu__6);  batch_norm_9 = relu__6 = None
        relu__8 = torch.ops.aten.relu_.default(add__3);  add__3 = None
        conv2d_10 = torch.ops.aten.conv2d.default(relu__8, layer3_0_conv1_weight, None, [2, 2], [1, 1]);  layer3_0_conv1_weight = None
        batch_norm_10 = torch.ops.aten.batch_norm.default(conv2d_10, layer3_0_bn1_weight, layer3_0_bn1_bias, layer3_0_bn1_running_mean, layer3_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_10 = layer3_0_bn1_weight = layer3_0_bn1_bias = layer3_0_bn1_running_mean = layer3_0_bn1_running_var = None
        relu__9 = torch.ops.aten.relu_.default(batch_norm_10);  batch_norm_10 = None
        conv2d_11 = torch.ops.aten.conv2d.default(relu__9, layer3_0_conv2_weight, None, [1, 1], [1, 1]);  relu__9 = layer3_0_conv2_weight = None
        batch_norm_11 = torch.ops.aten.batch_norm.default(conv2d_11, layer3_0_bn2_weight, layer3_0_bn2_bias, layer3_0_bn2_running_mean, layer3_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_11 = layer3_0_bn2_weight = layer3_0_bn2_bias = layer3_0_bn2_running_mean = layer3_0_bn2_running_var = None
        conv2d_12 = torch.ops.aten.conv2d.default(relu__8, layer3_0_downsample_0_weight, None, [2, 2]);  relu__8 = layer3_0_downsample_0_weight = None
        batch_norm_12 = torch.ops.aten.batch_norm.default(conv2d_12, layer3_0_downsample_1_weight, layer3_0_downsample_1_bias, layer3_0_downsample_1_running_mean, layer3_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_12 = layer3_0_downsample_1_weight = layer3_0_downsample_1_bias = layer3_0_downsample_1_running_mean = layer3_0_downsample_1_running_var = None
        add__4 = torch.ops.aten.add_.Tensor(batch_norm_11, batch_norm_12);  batch_norm_11 = batch_norm_12 = None
        relu__10 = torch.ops.aten.relu_.default(add__4);  add__4 = None
        conv2d_13 = torch.ops.aten.conv2d.default(relu__10, layer3_1_conv1_weight, None, [1, 1], [1, 1]);  layer3_1_conv1_weight = None
        batch_norm_13 = torch.ops.aten.batch_norm.default(conv2d_13, layer3_1_bn1_weight, layer3_1_bn1_bias, layer3_1_bn1_running_mean, layer3_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_13 = layer3_1_bn1_weight = layer3_1_bn1_bias = layer3_1_bn1_running_mean = layer3_1_bn1_running_var = None
        relu__11 = torch.ops.aten.relu_.default(batch_norm_13);  batch_norm_13 = None
        conv2d_14 = torch.ops.aten.conv2d.default(relu__11, layer3_1_conv2_weight, None, [1, 1], [1, 1]);  relu__11 = layer3_1_conv2_weight = None
        batch_norm_14 = torch.ops.aten.batch_norm.default(conv2d_14, layer3_1_bn2_weight, layer3_1_bn2_bias, layer3_1_bn2_running_mean, layer3_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_14 = layer3_1_bn2_weight = layer3_1_bn2_bias = layer3_1_bn2_running_mean = layer3_1_bn2_running_var = None
        add__5 = torch.ops.aten.add_.Tensor(batch_norm_14, relu__10);  batch_norm_14 = relu__10 = None
        relu__12 = torch.ops.aten.relu_.default(add__5);  add__5 = None
        conv2d_15 = torch.ops.aten.conv2d.default(relu__12, layer4_0_conv1_weight, None, [2, 2], [1, 1]);  layer4_0_conv1_weight = None
        batch_norm_15 = torch.ops.aten.batch_norm.default(conv2d_15, layer4_0_bn1_weight, layer4_0_bn1_bias, layer4_0_bn1_running_mean, layer4_0_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_15 = layer4_0_bn1_weight = layer4_0_bn1_bias = layer4_0_bn1_running_mean = layer4_0_bn1_running_var = None
        relu__13 = torch.ops.aten.relu_.default(batch_norm_15);  batch_norm_15 = None
        conv2d_16 = torch.ops.aten.conv2d.default(relu__13, layer4_0_conv2_weight, None, [1, 1], [1, 1]);  relu__13 = layer4_0_conv2_weight = None
        batch_norm_16 = torch.ops.aten.batch_norm.default(conv2d_16, layer4_0_bn2_weight, layer4_0_bn2_bias, layer4_0_bn2_running_mean, layer4_0_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_16 = layer4_0_bn2_weight = layer4_0_bn2_bias = layer4_0_bn2_running_mean = layer4_0_bn2_running_var = None
        conv2d_17 = torch.ops.aten.conv2d.default(relu__12, layer4_0_downsample_0_weight, None, [2, 2]);  relu__12 = layer4_0_downsample_0_weight = None
        batch_norm_17 = torch.ops.aten.batch_norm.default(conv2d_17, layer4_0_downsample_1_weight, layer4_0_downsample_1_bias, layer4_0_downsample_1_running_mean, layer4_0_downsample_1_running_var, False, 0.1, 1e-05, False);  conv2d_17 = layer4_0_downsample_1_weight = layer4_0_downsample_1_bias = layer4_0_downsample_1_running_mean = layer4_0_downsample_1_running_var = None
        add__6 = torch.ops.aten.add_.Tensor(batch_norm_16, batch_norm_17);  batch_norm_16 = batch_norm_17 = None
        relu__14 = torch.ops.aten.relu_.default(add__6);  add__6 = None
        conv2d_18 = torch.ops.aten.conv2d.default(relu__14, layer4_1_conv1_weight, None, [1, 1], [1, 1]);  layer4_1_conv1_weight = None
        batch_norm_18 = torch.ops.aten.batch_norm.default(conv2d_18, layer4_1_bn1_weight, layer4_1_bn1_bias, layer4_1_bn1_running_mean, layer4_1_bn1_running_var, False, 0.1, 1e-05, False);  conv2d_18 = layer4_1_bn1_weight = layer4_1_bn1_bias = layer4_1_bn1_running_mean = layer4_1_bn1_running_var = None
        relu__15 = torch.ops.aten.relu_.default(batch_norm_18);  batch_norm_18 = None
        conv2d_19 = torch.ops.aten.conv2d.default(relu__15, layer4_1_conv2_weight, None, [1, 1], [1, 1]);  relu__15 = layer4_1_conv2_weight = None
        batch_norm_19 = torch.ops.aten.batch_norm.default(conv2d_19, layer4_1_bn2_weight, layer4_1_bn2_bias, layer4_1_bn2_running_mean, layer4_1_bn2_running_var, False, 0.1, 1e-05, False);  conv2d_19 = layer4_1_bn2_weight = layer4_1_bn2_bias = layer4_1_bn2_running_mean = layer4_1_bn2_running_var = None
        add__7 = torch.ops.aten.add_.Tensor(batch_norm_19, relu__14);  batch_norm_19 = relu__14 = None
        relu__16 = torch.ops.aten.relu_.default(add__7);  add__7 = None
        adaptive_avg_pool2d = torch.ops.aten.adaptive_avg_pool2d.default(relu__16, [1, 1]);  relu__16 = None
        flatten = torch.ops.aten.flatten.using_ints(adaptive_avg_pool2d, 1);  adaptive_avg_pool2d = None
        linear = torch.ops.aten.linear.default(flatten, fc_weight, fc_bias);  flatten = fc_weight = fc_bias = None
        return pytree.tree_unflatten((linear,), self._out_spec)

