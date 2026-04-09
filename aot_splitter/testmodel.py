import torch 

class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[1, 3, 224, 224]"; 

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        # No stacktrace found for following nodes
        conv1: "f32[1, 64, 112, 112]" = self.conv1(x);  x = None
        bn1: "f32[1, 64, 112, 112]" = self.bn1(conv1);  conv1 = None
        relu: "f32[1, 64, 112, 112]" = self.relu(bn1);  bn1 = None
        maxpool: "f32[1, 64, 56, 56]" = self.maxpool(relu);  relu = None
        layer1: "f32[1, 64, 56, 56]" = self.layer1(maxpool);  maxpool = None
        layer2: "f32[1, 128, 28, 28]" = self.layer2(layer1);  layer1 = None
        layer3: "f32[1, 256, 14, 14]" = self.layer3(layer2);  layer2 = None
        layer4: "f32[1, 512, 7, 7]" = self.layer4(layer3);  layer3 = None
        avgpool: "f32[1, 512, 1, 1]" = self.avgpool(layer4);  layer4 = None

        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:285 in forward, code: return self._forward_impl(x)
        flatten_using_ints: "f32[1, 512]" = torch.ops.aten.flatten.using_ints(avgpool, 1);  avgpool = None

        # No stacktrace found for following nodes
        fc: "f32[1, 1000]" = self.fc(flatten_using_ints);  flatten_using_ints = None
        return fc

    class conv1(torch.nn.Module):
        def forward(self, x: "f32[1, 3, 224, 224]"):
            # No stacktrace found for following nodes
            weight: "f32[64, 3, 7, 7]" = self.weight

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
            conv2d_default: "f32[1, 64, 112, 112]" = torch.ops.aten.conv2d.default(x, weight, None, [2, 2], [3, 3]);  x = weight = None
            return conv2d_default

    class bn1(torch.nn.Module):
        def forward(self, conv2d_default: "f32[1, 64, 112, 112]"):
            # No stacktrace found for following nodes
            weight: "f32[64]" = self.weight
            bias: "f32[64]" = self.bias
            running_mean: "f32[64]" = self.running_mean
            running_var: "f32[64]" = self.running_var

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
            batch_norm_default: "f32[1, 64, 112, 112]" = torch.ops.aten.batch_norm.default(conv2d_default, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default = weight = bias = running_mean = running_var = None
            return batch_norm_default

    class relu(torch.nn.Module):
        def forward(self, batch_norm_default: "f32[1, 64, 112, 112]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
            relu__default: "f32[1, 64, 112, 112]" = torch.ops.aten.relu_.default(batch_norm_default);  batch_norm_default = None
            return relu__default

    class maxpool(torch.nn.Module):
        def forward(self, relu__default: "f32[1, 64, 112, 112]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/pooling.py:224 in forward, code: return F.max_pool2d(
            max_pool2d_default: "f32[1, 64, 56, 56]" = torch.ops.aten.max_pool2d.default(relu__default, [3, 3], [2, 2], [1, 1]);  relu__default = None
            return max_pool2d_default

    class layer1(torch.nn.Module):
        def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 64, 56, 56]" = getattr(self, "0")(max_pool2d_default);  max_pool2d_default = None
            _1: "f32[1, 64, 56, 56]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 64, 56, 56]" = self.conv1(max_pool2d_default)
                bn1: "f32[1, 64, 56, 56]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 64, 56, 56]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 64, 56, 56]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 64, 56, 56]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor: "f32[1, 64, 56, 56]" = torch.ops.aten.add_.Tensor(bn2, max_pool2d_default);  bn2 = max_pool2d_default = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 64, 56, 56]" = getattr(self, "relu@1")(add__tensor);  add__tensor = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(max_pool2d_default, weight, None, [1, 1], [1, 1]);  max_pool2d_default = weight = None
                    return conv2d_default_1

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_1: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_1, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_1 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_1

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_1: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(batch_norm_default_1);  batch_norm_default_1 = None
                    return relu__default_1

            class conv2(torch.nn.Module):
                def forward(self, relu__default_1: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_1, weight, None, [1, 1], [1, 1]);  relu__default_1 = weight = None
                    return conv2d_default_2

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_2: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_2, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_2 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_2

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(add__tensor);  add__tensor = None
                    return relu__default_2

        class 1(torch.nn.Module):
            def forward(self, relu__default_2: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 64, 56, 56]" = self.conv1(relu__default_2)
                bn1: "f32[1, 64, 56, 56]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 64, 56, 56]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 64, 56, 56]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 64, 56, 56]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_1: "f32[1, 64, 56, 56]" = torch.ops.aten.add_.Tensor(bn2, relu__default_2);  bn2 = relu__default_2 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 64, 56, 56]" = getattr(self, "relu@1")(add__tensor_1);  add__tensor_1 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_2: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_2, weight, None, [1, 1], [1, 1]);  relu__default_2 = weight = None
                    return conv2d_default_3

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_3: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_3, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_3 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_3

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_3: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(batch_norm_default_3);  batch_norm_default_3 = None
                    return relu__default_3

            class conv2(torch.nn.Module):
                def forward(self, relu__default_3: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_3, weight, None, [1, 1], [1, 1]);  relu__default_3 = weight = None
                    return conv2d_default_4

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_4, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_4 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_4

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_1: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(add__tensor_1);  add__tensor_1 = None
                    return relu__default_4

    class layer2(torch.nn.Module):
        def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 128, 28, 28]" = getattr(self, "0")(relu__default_4);  relu__default_4 = None
            _1: "f32[1, 128, 28, 28]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 128, 28, 28]" = self.conv1(relu__default_4)
                bn1: "f32[1, 128, 28, 28]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 128, 28, 28]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 128, 28, 28]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 128, 28, 28]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 128, 28, 28]" = self.downsample(relu__default_4);  relu__default_4 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_2: "f32[1, 128, 28, 28]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 128, 28, 28]" = getattr(self, "relu@1")(add__tensor_2);  add__tensor_2 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_4, weight, None, [2, 2], [1, 1]);  relu__default_4 = weight = None
                    return conv2d_default_5

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_5: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_5, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_5 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_5

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_5: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(batch_norm_default_5);  batch_norm_default_5 = None
                    return relu__default_5

            class conv2(torch.nn.Module):
                def forward(self, relu__default_5: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_5, weight, None, [1, 1], [1, 1]);  relu__default_5 = weight = None
                    return conv2d_default_6

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_6: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_6, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_6 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_6

            class downsample(torch.nn.Module):
                def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 128, 28, 28]" = getattr(self, "0")(relu__default_4);  relu__default_4 = None
                    _1: "f32[1, 128, 28, 28]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                        # No stacktrace found for following nodes
                        weight: "f32[128, 64, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_4, weight, None, [2, 2]);  relu__default_4 = weight = None
                        return conv2d_default_7

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_7: "f32[1, 128, 28, 28]"):
                        # No stacktrace found for following nodes
                        weight: "f32[128]" = self.weight
                        bias: "f32[128]" = self.bias
                        running_mean: "f32[128]" = self.running_mean
                        running_var: "f32[128]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_7, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_7 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_7

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_2: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(add__tensor_2);  add__tensor_2 = None
                    return relu__default_6

        class 1(torch.nn.Module):
            def forward(self, relu__default_6: "f32[1, 128, 28, 28]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 128, 28, 28]" = self.conv1(relu__default_6)
                bn1: "f32[1, 128, 28, 28]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 128, 28, 28]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 128, 28, 28]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 128, 28, 28]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_3: "f32[1, 128, 28, 28]" = torch.ops.aten.add_.Tensor(bn2, relu__default_6);  bn2 = relu__default_6 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 128, 28, 28]" = getattr(self, "relu@1")(add__tensor_3);  add__tensor_3 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_6: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_6, weight, None, [1, 1], [1, 1]);  relu__default_6 = weight = None
                    return conv2d_default_8

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_8, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_8 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_8

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_8: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(batch_norm_default_8);  batch_norm_default_8 = None
                    return relu__default_7

            class conv2(torch.nn.Module):
                def forward(self, relu__default_7: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_9: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_7, weight, None, [1, 1], [1, 1]);  relu__default_7 = weight = None
                    return conv2d_default_9

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_9: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_9: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_9, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_9 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_9

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_3: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(add__tensor_3);  add__tensor_3 = None
                    return relu__default_8

    class layer3(torch.nn.Module):
        def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 256, 14, 14]" = getattr(self, "0")(relu__default_8);  relu__default_8 = None
            _1: "f32[1, 256, 14, 14]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 256, 14, 14]" = self.conv1(relu__default_8)
                bn1: "f32[1, 256, 14, 14]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 256, 14, 14]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 256, 14, 14]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 256, 14, 14]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 256, 14, 14]" = self.downsample(relu__default_8);  relu__default_8 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_4: "f32[1, 256, 14, 14]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 256, 14, 14]" = getattr(self, "relu@1")(add__tensor_4);  add__tensor_4 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_8, weight, None, [2, 2], [1, 1]);  relu__default_8 = weight = None
                    return conv2d_default_10

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_10: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_10, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_10 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_10

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_10: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_9: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(batch_norm_default_10);  batch_norm_default_10 = None
                    return relu__default_9

            class conv2(torch.nn.Module):
                def forward(self, relu__default_9: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_9, weight, None, [1, 1], [1, 1]);  relu__default_9 = weight = None
                    return conv2d_default_11

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_11: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_11, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_11 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_11

            class downsample(torch.nn.Module):
                def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 256, 14, 14]" = getattr(self, "0")(relu__default_8);  relu__default_8 = None
                    _1: "f32[1, 256, 14, 14]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                        # No stacktrace found for following nodes
                        weight: "f32[256, 128, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_8, weight, None, [2, 2]);  relu__default_8 = weight = None
                        return conv2d_default_12

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_12: "f32[1, 256, 14, 14]"):
                        # No stacktrace found for following nodes
                        weight: "f32[256]" = self.weight
                        bias: "f32[256]" = self.bias
                        running_mean: "f32[256]" = self.running_mean
                        running_var: "f32[256]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_12, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_12 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_12

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_4: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(add__tensor_4);  add__tensor_4 = None
                    return relu__default_10

        class 1(torch.nn.Module):
            def forward(self, relu__default_10: "f32[1, 256, 14, 14]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 256, 14, 14]" = self.conv1(relu__default_10)
                bn1: "f32[1, 256, 14, 14]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 256, 14, 14]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 256, 14, 14]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 256, 14, 14]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_5: "f32[1, 256, 14, 14]" = torch.ops.aten.add_.Tensor(bn2, relu__default_10);  bn2 = relu__default_10 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 256, 14, 14]" = getattr(self, "relu@1")(add__tensor_5);  add__tensor_5 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_10: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_13: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_10, weight, None, [1, 1], [1, 1]);  relu__default_10 = weight = None
                    return conv2d_default_13

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_13: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_13: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_13, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_13 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_13

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_13: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(batch_norm_default_13);  batch_norm_default_13 = None
                    return relu__default_11

            class conv2(torch.nn.Module):
                def forward(self, relu__default_11: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_14: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_11, weight, None, [1, 1], [1, 1]);  relu__default_11 = weight = None
                    return conv2d_default_14

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_14: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_14: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_14, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_14 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_14

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_5: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(add__tensor_5);  add__tensor_5 = None
                    return relu__default_12

    class layer4(torch.nn.Module):
        def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 512, 7, 7]" = getattr(self, "0")(relu__default_12);  relu__default_12 = None
            _1: "f32[1, 512, 7, 7]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 512, 7, 7]" = self.conv1(relu__default_12)
                bn1: "f32[1, 512, 7, 7]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 512, 7, 7]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 512, 7, 7]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 512, 7, 7]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 512, 7, 7]" = self.downsample(relu__default_12);  relu__default_12 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_6: "f32[1, 512, 7, 7]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 512, 7, 7]" = getattr(self, "relu@1")(add__tensor_6);  add__tensor_6 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_12, weight, None, [2, 2], [1, 1]);  relu__default_12 = weight = None
                    return conv2d_default_15

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_15: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_15, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_15 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_15

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_15: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_13: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(batch_norm_default_15);  batch_norm_default_15 = None
                    return relu__default_13

            class conv2(torch.nn.Module):
                def forward(self, relu__default_13: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_13, weight, None, [1, 1], [1, 1]);  relu__default_13 = weight = None
                    return conv2d_default_16

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_16: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_16, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_16 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_16

            class downsample(torch.nn.Module):
                def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 512, 7, 7]" = getattr(self, "0")(relu__default_12);  relu__default_12 = None
                    _1: "f32[1, 512, 7, 7]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                        # No stacktrace found for following nodes
                        weight: "f32[512, 256, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_17: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_12, weight, None, [2, 2]);  relu__default_12 = weight = None
                        return conv2d_default_17

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_17: "f32[1, 512, 7, 7]"):
                        # No stacktrace found for following nodes
                        weight: "f32[512]" = self.weight
                        bias: "f32[512]" = self.bias
                        running_mean: "f32[512]" = self.running_mean
                        running_var: "f32[512]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_17: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_17, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_17 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_17

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_6: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_14: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(add__tensor_6);  add__tensor_6 = None
                    return relu__default_14

        class 1(torch.nn.Module):
            def forward(self, relu__default_14: "f32[1, 512, 7, 7]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 512, 7, 7]" = self.conv1(relu__default_14)
                bn1: "f32[1, 512, 7, 7]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 512, 7, 7]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 512, 7, 7]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 512, 7, 7]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_7: "f32[1, 512, 7, 7]" = torch.ops.aten.add_.Tensor(bn2, relu__default_14);  bn2 = relu__default_14 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 512, 7, 7]" = getattr(self, "relu@1")(add__tensor_7);  add__tensor_7 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_14: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_18: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_14, weight, None, [1, 1], [1, 1]);  relu__default_14 = weight = None
                    return conv2d_default_18

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_18: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_18: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_18, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_18 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_18

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_18: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(batch_norm_default_18);  batch_norm_default_18 = None
                    return relu__default_15

            class conv2(torch.nn.Module):
                def forward(self, relu__default_15: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_19: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_15, weight, None, [1, 1], [1, 1]);  relu__default_15 = weight = None
                    return conv2d_default_19

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_19: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_19: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_19, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_19 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_19

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_7: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(add__tensor_7);  add__tensor_7 = None
                    return relu__default_16

    class avgpool(torch.nn.Module):
        def forward(self, relu__default_16: "f32[1, 512, 7, 7]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/pooling.py:1510 in forward, code: return F.adaptive_avg_pool2d(input, self.output_size)
            adaptive_avg_pool2d_default: "f32[1, 512, 1, 1]" = torch.ops.aten.adaptive_avg_pool2d.default(relu__default_16, [1, 1]);  relu__default_16 = None
            return adaptive_avg_pool2d_default

    class fc(torch.nn.Module):
        def forward(self, flatten_using_ints: "f32[1, 512]"):
            # No stacktrace found for following nodes
            weight: "f32[1000, 512]" = self.weight
            bias: "f32[1000]" = self.bias

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_default: "f32[1, 1000]" = torch.ops.aten.linear.default(flatten_using_ints, weight, bias);  flatten_using_ints = weight = bias = None
            return linear_default

class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[1, 3, 224, 224]"; 

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        # No stacktrace found for following nodes
        conv1: "f32[1, 64, 112, 112]" = self.conv1(x);  x = None
        bn1: "f32[1, 64, 112, 112]" = self.bn1(conv1);  conv1 = None
        relu: "f32[1, 64, 112, 112]" = self.relu(bn1);  bn1 = None
        maxpool: "f32[1, 64, 56, 56]" = self.maxpool(relu);  relu = None
        layer1: "f32[1, 64, 56, 56]" = self.layer1(maxpool);  maxpool = None
        layer2: "f32[1, 128, 28, 28]" = self.layer2(layer1);  layer1 = None
        layer3: "f32[1, 256, 14, 14]" = self.layer3(layer2);  layer2 = None
        layer4: "f32[1, 512, 7, 7]" = self.layer4(layer3);  layer3 = None
        avgpool: "f32[1, 512, 1, 1]" = self.avgpool(layer4);  layer4 = None

        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:285 in forward, code: return self._forward_impl(x)
        flatten_using_ints: "f32[1, 512]" = torch.ops.aten.flatten.using_ints(avgpool, 1);  avgpool = None

        # No stacktrace found for following nodes
        fc: "f32[1, 1000]" = self.fc(flatten_using_ints);  flatten_using_ints = None
        return fc

    class conv1(torch.nn.Module):
        def forward(self, x: "f32[1, 3, 224, 224]"):
            # No stacktrace found for following nodes
            weight: "f32[64, 3, 7, 7]" = self.weight

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
            conv2d_default: "f32[1, 64, 112, 112]" = torch.ops.aten.conv2d.default(x, weight, None, [2, 2], [3, 3]);  x = weight = None
            return conv2d_default

    class bn1(torch.nn.Module):
        def forward(self, conv2d_default: "f32[1, 64, 112, 112]"):
            # No stacktrace found for following nodes
            weight: "f32[64]" = self.weight
            bias: "f32[64]" = self.bias
            running_mean: "f32[64]" = self.running_mean
            running_var: "f32[64]" = self.running_var

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
            batch_norm_default: "f32[1, 64, 112, 112]" = torch.ops.aten.batch_norm.default(conv2d_default, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default = weight = bias = running_mean = running_var = None
            return batch_norm_default

    class relu(torch.nn.Module):
        def forward(self, batch_norm_default: "f32[1, 64, 112, 112]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
            relu__default: "f32[1, 64, 112, 112]" = torch.ops.aten.relu_.default(batch_norm_default);  batch_norm_default = None
            return relu__default

    class maxpool(torch.nn.Module):
        def forward(self, relu__default: "f32[1, 64, 112, 112]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/pooling.py:224 in forward, code: return F.max_pool2d(
            max_pool2d_default: "f32[1, 64, 56, 56]" = torch.ops.aten.max_pool2d.default(relu__default, [3, 3], [2, 2], [1, 1]);  relu__default = None
            return max_pool2d_default

    class layer1(torch.nn.Module):
        def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 64, 56, 56]" = getattr(self, "0")(max_pool2d_default);  max_pool2d_default = None
            _1: "f32[1, 64, 56, 56]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 64, 56, 56]" = self.conv1(max_pool2d_default)
                bn1: "f32[1, 64, 56, 56]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 64, 56, 56]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 64, 56, 56]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 64, 56, 56]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor: "f32[1, 64, 56, 56]" = torch.ops.aten.add_.Tensor(bn2, max_pool2d_default);  bn2 = max_pool2d_default = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 64, 56, 56]" = getattr(self, "relu@1")(add__tensor);  add__tensor = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, max_pool2d_default: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(max_pool2d_default, weight, None, [1, 1], [1, 1]);  max_pool2d_default = weight = None
                    return conv2d_default_1

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_1: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_1, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_1 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_1

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_1: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_1: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(batch_norm_default_1);  batch_norm_default_1 = None
                    return relu__default_1

            class conv2(torch.nn.Module):
                def forward(self, relu__default_1: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_1, weight, None, [1, 1], [1, 1]);  relu__default_1 = weight = None
                    return conv2d_default_2

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_2: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_2, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_2 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_2

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_2: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(add__tensor);  add__tensor = None
                    return relu__default_2

        class 1(torch.nn.Module):
            def forward(self, relu__default_2: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 64, 56, 56]" = self.conv1(relu__default_2)
                bn1: "f32[1, 64, 56, 56]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 64, 56, 56]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 64, 56, 56]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 64, 56, 56]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_1: "f32[1, 64, 56, 56]" = torch.ops.aten.add_.Tensor(bn2, relu__default_2);  bn2 = relu__default_2 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 64, 56, 56]" = getattr(self, "relu@1")(add__tensor_1);  add__tensor_1 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_2: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_2, weight, None, [1, 1], [1, 1]);  relu__default_2 = weight = None
                    return conv2d_default_3

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_3: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_3, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_3 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_3

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_3: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_3: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(batch_norm_default_3);  batch_norm_default_3 = None
                    return relu__default_3

            class conv2(torch.nn.Module):
                def forward(self, relu__default_3: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.conv2d.default(relu__default_3, weight, None, [1, 1], [1, 1]);  relu__default_3 = weight = None
                    return conv2d_default_4

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[64]" = self.weight
                    bias: "f32[64]" = self.bias
                    running_mean: "f32[64]" = self.running_mean
                    running_var: "f32[64]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.batch_norm.default(conv2d_default_4, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_4 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_4

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_1: "f32[1, 64, 56, 56]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_4: "f32[1, 64, 56, 56]" = torch.ops.aten.relu_.default(add__tensor_1);  add__tensor_1 = None
                    return relu__default_4

    class layer2(torch.nn.Module):
        def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 128, 28, 28]" = getattr(self, "0")(relu__default_4);  relu__default_4 = None
            _1: "f32[1, 128, 28, 28]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 128, 28, 28]" = self.conv1(relu__default_4)
                bn1: "f32[1, 128, 28, 28]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 128, 28, 28]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 128, 28, 28]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 128, 28, 28]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 128, 28, 28]" = self.downsample(relu__default_4);  relu__default_4 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_2: "f32[1, 128, 28, 28]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 128, 28, 28]" = getattr(self, "relu@1")(add__tensor_2);  add__tensor_2 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 64, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_4, weight, None, [2, 2], [1, 1]);  relu__default_4 = weight = None
                    return conv2d_default_5

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_5: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_5, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_5 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_5

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_5: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_5: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(batch_norm_default_5);  batch_norm_default_5 = None
                    return relu__default_5

            class conv2(torch.nn.Module):
                def forward(self, relu__default_5: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_5, weight, None, [1, 1], [1, 1]);  relu__default_5 = weight = None
                    return conv2d_default_6

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_6: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_6, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_6 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_6

            class downsample(torch.nn.Module):
                def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 128, 28, 28]" = getattr(self, "0")(relu__default_4);  relu__default_4 = None
                    _1: "f32[1, 128, 28, 28]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_4: "f32[1, 64, 56, 56]"):
                        # No stacktrace found for following nodes
                        weight: "f32[128, 64, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_4, weight, None, [2, 2]);  relu__default_4 = weight = None
                        return conv2d_default_7

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_7: "f32[1, 128, 28, 28]"):
                        # No stacktrace found for following nodes
                        weight: "f32[128]" = self.weight
                        bias: "f32[128]" = self.bias
                        running_mean: "f32[128]" = self.running_mean
                        running_var: "f32[128]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_7, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_7 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_7

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_2: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_6: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(add__tensor_2);  add__tensor_2 = None
                    return relu__default_6

        class 1(torch.nn.Module):
            def forward(self, relu__default_6: "f32[1, 128, 28, 28]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 128, 28, 28]" = self.conv1(relu__default_6)
                bn1: "f32[1, 128, 28, 28]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 128, 28, 28]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 128, 28, 28]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 128, 28, 28]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_3: "f32[1, 128, 28, 28]" = torch.ops.aten.add_.Tensor(bn2, relu__default_6);  bn2 = relu__default_6 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 128, 28, 28]" = getattr(self, "relu@1")(add__tensor_3);  add__tensor_3 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_6: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_6, weight, None, [1, 1], [1, 1]);  relu__default_6 = weight = None
                    return conv2d_default_8

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_8, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_8 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_8

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_8: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_7: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(batch_norm_default_8);  batch_norm_default_8 = None
                    return relu__default_7

            class conv2(torch.nn.Module):
                def forward(self, relu__default_7: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_9: "f32[1, 128, 28, 28]" = torch.ops.aten.conv2d.default(relu__default_7, weight, None, [1, 1], [1, 1]);  relu__default_7 = weight = None
                    return conv2d_default_9

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_9: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[128]" = self.weight
                    bias: "f32[128]" = self.bias
                    running_mean: "f32[128]" = self.running_mean
                    running_var: "f32[128]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_9: "f32[1, 128, 28, 28]" = torch.ops.aten.batch_norm.default(conv2d_default_9, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_9 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_9

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_3: "f32[1, 128, 28, 28]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_8: "f32[1, 128, 28, 28]" = torch.ops.aten.relu_.default(add__tensor_3);  add__tensor_3 = None
                    return relu__default_8

    class layer3(torch.nn.Module):
        def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 256, 14, 14]" = getattr(self, "0")(relu__default_8);  relu__default_8 = None
            _1: "f32[1, 256, 14, 14]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 256, 14, 14]" = self.conv1(relu__default_8)
                bn1: "f32[1, 256, 14, 14]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 256, 14, 14]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 256, 14, 14]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 256, 14, 14]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 256, 14, 14]" = self.downsample(relu__default_8);  relu__default_8 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_4: "f32[1, 256, 14, 14]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 256, 14, 14]" = getattr(self, "relu@1")(add__tensor_4);  add__tensor_4 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 128, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_8, weight, None, [2, 2], [1, 1]);  relu__default_8 = weight = None
                    return conv2d_default_10

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_10: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_10, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_10 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_10

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_10: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_9: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(batch_norm_default_10);  batch_norm_default_10 = None
                    return relu__default_9

            class conv2(torch.nn.Module):
                def forward(self, relu__default_9: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_9, weight, None, [1, 1], [1, 1]);  relu__default_9 = weight = None
                    return conv2d_default_11

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_11: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_11, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_11 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_11

            class downsample(torch.nn.Module):
                def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 256, 14, 14]" = getattr(self, "0")(relu__default_8);  relu__default_8 = None
                    _1: "f32[1, 256, 14, 14]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_8: "f32[1, 128, 28, 28]"):
                        # No stacktrace found for following nodes
                        weight: "f32[256, 128, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_8, weight, None, [2, 2]);  relu__default_8 = weight = None
                        return conv2d_default_12

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_12: "f32[1, 256, 14, 14]"):
                        # No stacktrace found for following nodes
                        weight: "f32[256]" = self.weight
                        bias: "f32[256]" = self.bias
                        running_mean: "f32[256]" = self.running_mean
                        running_var: "f32[256]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_12, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_12 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_12

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_4: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_10: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(add__tensor_4);  add__tensor_4 = None
                    return relu__default_10

        class 1(torch.nn.Module):
            def forward(self, relu__default_10: "f32[1, 256, 14, 14]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 256, 14, 14]" = self.conv1(relu__default_10)
                bn1: "f32[1, 256, 14, 14]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 256, 14, 14]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 256, 14, 14]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 256, 14, 14]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_5: "f32[1, 256, 14, 14]" = torch.ops.aten.add_.Tensor(bn2, relu__default_10);  bn2 = relu__default_10 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 256, 14, 14]" = getattr(self, "relu@1")(add__tensor_5);  add__tensor_5 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_10: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_13: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_10, weight, None, [1, 1], [1, 1]);  relu__default_10 = weight = None
                    return conv2d_default_13

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_13: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_13: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_13, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_13 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_13

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_13: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_11: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(batch_norm_default_13);  batch_norm_default_13 = None
                    return relu__default_11

            class conv2(torch.nn.Module):
                def forward(self, relu__default_11: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_14: "f32[1, 256, 14, 14]" = torch.ops.aten.conv2d.default(relu__default_11, weight, None, [1, 1], [1, 1]);  relu__default_11 = weight = None
                    return conv2d_default_14

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_14: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[256]" = self.weight
                    bias: "f32[256]" = self.bias
                    running_mean: "f32[256]" = self.running_mean
                    running_var: "f32[256]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_14: "f32[1, 256, 14, 14]" = torch.ops.aten.batch_norm.default(conv2d_default_14, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_14 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_14

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_5: "f32[1, 256, 14, 14]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_12: "f32[1, 256, 14, 14]" = torch.ops.aten.relu_.default(add__tensor_5);  add__tensor_5 = None
                    return relu__default_12

    class layer4(torch.nn.Module):
        def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
            # No stacktrace found for following nodes
            _0: "f32[1, 512, 7, 7]" = getattr(self, "0")(relu__default_12);  relu__default_12 = None
            _1: "f32[1, 512, 7, 7]" = getattr(self, "1")(_0);  _0 = None
            return _1

        class 0(torch.nn.Module):
            def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 512, 7, 7]" = self.conv1(relu__default_12)
                bn1: "f32[1, 512, 7, 7]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 512, 7, 7]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 512, 7, 7]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 512, 7, 7]" = self.bn2(conv2);  conv2 = None
                downsample: "f32[1, 512, 7, 7]" = self.downsample(relu__default_12);  relu__default_12 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_6: "f32[1, 512, 7, 7]" = torch.ops.aten.add_.Tensor(bn2, downsample);  bn2 = downsample = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 512, 7, 7]" = getattr(self, "relu@1")(add__tensor_6);  add__tensor_6 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 256, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_12, weight, None, [2, 2], [1, 1]);  relu__default_12 = weight = None
                    return conv2d_default_15

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_15: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_15, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_15 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_15

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_15: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_13: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(batch_norm_default_15);  batch_norm_default_15 = None
                    return relu__default_13

            class conv2(torch.nn.Module):
                def forward(self, relu__default_13: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_13, weight, None, [1, 1], [1, 1]);  relu__default_13 = weight = None
                    return conv2d_default_16

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_16: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_16, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_16 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_16

            class downsample(torch.nn.Module):
                def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                    # No stacktrace found for following nodes
                    _0: "f32[1, 512, 7, 7]" = getattr(self, "0")(relu__default_12);  relu__default_12 = None
                    _1: "f32[1, 512, 7, 7]" = getattr(self, "1")(_0);  _0 = None
                    return _1

                class 0(torch.nn.Module):
                    def forward(self, relu__default_12: "f32[1, 256, 14, 14]"):
                        # No stacktrace found for following nodes
                        weight: "f32[512, 256, 1, 1]" = self.weight

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                        conv2d_default_17: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_12, weight, None, [2, 2]);  relu__default_12 = weight = None
                        return conv2d_default_17

                class 1(torch.nn.Module):
                    def forward(self, conv2d_default_17: "f32[1, 512, 7, 7]"):
                        # No stacktrace found for following nodes
                        weight: "f32[512]" = self.weight
                        bias: "f32[512]" = self.bias
                        running_mean: "f32[512]" = self.running_mean
                        running_var: "f32[512]" = self.running_var

                        # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                        batch_norm_default_17: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_17, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_17 = weight = bias = running_mean = running_var = None
                        return batch_norm_default_17

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_6: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_14: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(add__tensor_6);  add__tensor_6 = None
                    return relu__default_14

        class 1(torch.nn.Module):
            def forward(self, relu__default_14: "f32[1, 512, 7, 7]"):
                # No stacktrace found for following nodes
                conv1: "f32[1, 512, 7, 7]" = self.conv1(relu__default_14)
                bn1: "f32[1, 512, 7, 7]" = self.bn1(conv1);  conv1 = None
                relu: "f32[1, 512, 7, 7]" = self.relu(bn1);  bn1 = None
                conv2: "f32[1, 512, 7, 7]" = self.conv2(relu);  relu = None
                bn2: "f32[1, 512, 7, 7]" = self.bn2(conv2);  conv2 = None

                # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
                add__tensor_7: "f32[1, 512, 7, 7]" = torch.ops.aten.add_.Tensor(bn2, relu__default_14);  bn2 = relu__default_14 = None

                # No stacktrace found for following nodes
                relu_1: "f32[1, 512, 7, 7]" = getattr(self, "relu@1")(add__tensor_7);  add__tensor_7 = None
                return relu_1

            class conv1(torch.nn.Module):
                def forward(self, relu__default_14: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_18: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_14, weight, None, [1, 1], [1, 1]);  relu__default_14 = weight = None
                    return conv2d_default_18

            class bn1(torch.nn.Module):
                def forward(self, conv2d_default_18: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_18: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_18, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_18 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_18

            class relu(torch.nn.Module):
                def forward(self, batch_norm_default_18: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_15: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(batch_norm_default_18);  batch_norm_default_18 = None
                    return relu__default_15

            class conv2(torch.nn.Module):
                def forward(self, relu__default_15: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512, 512, 3, 3]" = self.weight

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/conv.py:553 in forward, code: return self._conv_forward(input, self.weight, self.bias)
                    conv2d_default_19: "f32[1, 512, 7, 7]" = torch.ops.aten.conv2d.default(relu__default_15, weight, None, [1, 1], [1, 1]);  relu__default_15 = weight = None
                    return conv2d_default_19

            class bn2(torch.nn.Module):
                def forward(self, conv2d_default_19: "f32[1, 512, 7, 7]"):
                    # No stacktrace found for following nodes
                    weight: "f32[512]" = self.weight
                    bias: "f32[512]" = self.bias
                    running_mean: "f32[512]" = self.running_mean
                    running_var: "f32[512]" = self.running_var

                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/batchnorm.py:194 in forward, code: return F.batch_norm(
                    batch_norm_default_19: "f32[1, 512, 7, 7]" = torch.ops.aten.batch_norm.default(conv2d_default_19, weight, bias, running_mean, running_var, False, 0.1, 1e-05, False);  conv2d_default_19 = weight = bias = running_mean = running_var = None
                    return batch_norm_default_19

            class relu@1(torch.nn.Module):
                def forward(self, add__tensor_7: "f32[1, 512, 7, 7]"):
                    # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
                    relu__default_16: "f32[1, 512, 7, 7]" = torch.ops.aten.relu_.default(add__tensor_7);  add__tensor_7 = None
                    return relu__default_16

    class avgpool(torch.nn.Module):
        def forward(self, relu__default_16: "f32[1, 512, 7, 7]"):
            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/pooling.py:1510 in forward, code: return F.adaptive_avg_pool2d(input, self.output_size)
            adaptive_avg_pool2d_default: "f32[1, 512, 1, 1]" = torch.ops.aten.adaptive_avg_pool2d.default(relu__default_16, [1, 1]);  relu__default_16 = None
            return adaptive_avg_pool2d_default

    class fc(torch.nn.Module):
        def forward(self, flatten_using_ints: "f32[1, 512]"):
            # No stacktrace found for following nodes
            weight: "f32[1000, 512]" = self.weight
            bias: "f32[1000]" = self.bias

            # File: /Users/animeshnd/miniconda3/envs/pytorch-env/lib/python3.11/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
            linear_default: "f32[1, 1000]" = torch.ops.aten.linear.default(flatten_using_ints, weight, bias);  flatten_using_ints = weight = bias = None
            return linear_default

