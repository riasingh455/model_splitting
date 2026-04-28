import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any


@dataclass
class SplitArtifact:
    split_id: int
    target_percentage: float
    actual_flops: float
    actual_percentage: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    module_names: List[str]
    exported_path: str
    cumulative_flops: float
    boundary_transfer_bytes: float


class ExportableSplit(nn.Module):
    def __init__(self, layers: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.names = [n for n, _ in layers]
        self.layers = nn.ModuleDict({n.replace(".", "_"): m for n, m in layers})
        self._name_map = {n: n.replace(".", "_") for n, _ in layers}

    def forward(self, x):
        for n in self.names:
            x = self.layers[self._name_map[n]](x)
        return x


class FlopAwarePipelineSplitterBase:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        self._dummy = torch.randn(input_shape, device=self.device)
        self.total_flops, self.layer_flops = self._count_flops()
        self.ordered_layers = self._get_ordered_layers()

    def architecture_name(self) -> str:
        return "base"

    def _is_leaf_module(self, module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _macs_to_flops(self, macs: float) -> float:
        return 2.0 * macs

    def _conv1d_flops(self, m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor) -> float:
        n, c_in, _ = x.shape
        _, c_out, l_out = y.shape
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        groups = m.groups
        macs = n * l_out * c_out * (c_in // groups) * k
        return self._macs_to_flops(macs)

    def _conv2d_flops(self, m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> float:
        n, c_in, _, _ = x.shape
        _, c_out, h_out, w_out = y.shape
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        groups = m.groups
        macs = n * h_out * w_out * c_out * (c_in // groups) * kh * kw
        return self._macs_to_flops(macs)

    def _linear_flops(self, m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
        if x.dim() == 2:
            macs = x.shape[0] * m.in_features * m.out_features
        elif x.dim() == 3:
            b, s, _ = x.shape
            macs = b * s * m.in_features * m.out_features
        else:
            macs = x.numel() * m.out_features / max(m.in_features, 1)
        return self._macs_to_flops(macs)

    def _elementwise_flops(self, x: torch.Tensor, ops_per_element: float = 1.0) -> float:
        return ops_per_element * x.numel()

    def _attention_flops(self, m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        if x.dim() != 3:
            return 0.0
        b, s, e = x.shape
        num_heads = getattr(m, "num_heads", 1)
        head_dim = e // max(num_heads, 1)
        qkv_macs = 3.0 * b * s * e * e
        qk_macs = b * num_heads * s * s * head_dim
        av_macs = b * num_heads * s * s * head_dim
        out_macs = b * s * e * e
        return self._macs_to_flops(qkv_macs + qk_macs + av_macs + out_macs)

    def _count_flops(self):
        layer_flops = {}
        handles = []

        def hook(name):
            def fn(module, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                y = out[0] if isinstance(out, (tuple, list)) else out
                flops = 0.0

                if isinstance(module, nn.Conv1d):
                    flops = self._conv1d_flops(module, x, y)
                elif isinstance(module, nn.Conv2d):
                    flops = self._conv2d_flops(module, x, y)
                elif isinstance(module, nn.Linear):
                    flops = self._linear_flops(module, x, y)
                elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU)):
                    flops = self._elementwise_flops(x, 1.0)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                         nn.LayerNorm, nn.GroupNorm,
                                         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    flops = self._elementwise_flops(x, 4.0)
                elif isinstance(module, nn.Dropout):
                    flops = 0.0
                elif module.__class__.__name__ == "MultiheadAttention":
                    flops = self._attention_flops(module, x, y)

                layer_flops[name] = float(flops)
            return fn

        for name, module in self.model.named_modules():
            if name and self._is_leaf_module(module):
                handles.append(module.register_forward_hook(hook(name)))

        with torch.no_grad():
            _ = self.model(self._dummy)

        for h in handles:
            h.remove()

        return float(sum(layer_flops.values())), layer_flops

    def _bytes_from_shape(self, shape: Optional[Tuple[int, ...]]) -> float:
        if shape is None:
            return 0.0
        total = 1
        for d in shape:
            total *= d
        return float(total * 4)

    def _find_best_split(self, start_idx: int, target_flops: float, lookahead: int) -> int:
        best_cut = start_idx
        best_cost = float("inf")
        for end in range(start_idx, len(self.ordered_layers)):
            candidate = sum(self.ordered_layers[i][2] for i in range(start_idx, end + 1))
            if candidate > 1.5 * target_flops and end > start_idx:
                break
            cost = abs(candidate - target_flops)
            if cost < best_cost:
                best_cost = cost
                best_cut = end
        return best_cut

    def split_by_flops_pipeline(
        self,
        flop_percentages: List[float],
        lookahead: int,
        out_dir: str,
        meta_name: str,
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)

        if abs(sum(flop_percentages) - 100.0) > 1e-6:
            raise ValueError("FLOP percentages must sum to 100")

        targets = [(p / 100.0) * self.total_flops for p in flop_percentages]
        x = self._dummy.clone()
        start_idx = 0
        cumulative = 0.0
        artifacts = []

        for split_id, target in enumerate(targets):
            if split_id == len(targets) - 1:
                group = self.ordered_layers[start_idx:]
            else:
                cut = self._find_best_split(start_idx, target, lookahead)
                group = self.ordered_layers[start_idx:cut + 1]

            module = ExportableSplit([(n, m) for n, m, _ in group]).eval().to(self.device)
            inp_shape = tuple(x.shape)
            exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")

            with torch.no_grad():
                ep = torch.export.export(module, (x,))
                torch.export.save(ep, exported_path)
                y = ep.module()(x)

            flops = sum(f for _, _, f in group)
            cumulative += flops
            artifacts.append(
                SplitArtifact(
                    split_id=split_id,
                    target_percentage=flop_percentages[split_id],
                    actual_flops=flops,
                    actual_percentage=(flops / self.total_flops) * 100 if self.total_flops > 0 else 0.0,
                    input_shape=inp_shape,
                    output_shape=tuple(y.shape),
                    module_names=[n for n, _, _ in group],
                    exported_path=exported_path,
                    cumulative_flops=cumulative,
                    boundary_transfer_bytes=self._bytes_from_shape(tuple(y.shape)),
                )
            )

            x = y.detach()
            start_idx += len(group)

        result = {
            "architecture": self.architecture_name(),
            "total_flops": self.total_flops,
            "splits": [a.__dict__ for a in artifacts],
        }

        with open(os.path.join(out_dir, meta_name), "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def _get_ordered_layers(self):
        raise NotImplementedError

class ViTStem(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        self.dropout = model.encoder.dropout

    def forward(self, x):
        x = self.conv_proj(x)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        b = x.shape[0]
        cls = self.class_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class ViTHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ln = model.encoder.ln
        self.heads = model.heads

    def forward(self, x):
        x = self.ln(x)
        x = x[:, 0]
        return self.heads(x)



class FlopAwareViTPipelineSplitter(FlopAwarePipelineSplitterBase):
    def architecture_name(self) -> str:
        return "vit"

    def _count_flops(self):
        layer_flops = {}
        handles = []

        def hook(name):
            def fn(module, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                y = out[0] if isinstance(out, (tuple, list)) else out
                flops = 0.0
                if isinstance(module, nn.Conv2d):
                    flops = self._conv2d_flops(module, x, y)
                elif isinstance(module, nn.Linear):
                    flops = self._linear_flops(module, x, y)
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.ReLU6, nn.LeakyReLU)):
                    flops = self._elementwise_flops(x, 1.0)
                elif isinstance(module, (nn.LayerNorm, nn.GroupNorm,
                                         nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                         nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    flops = self._elementwise_flops(x, 4.0)
                elif module.__class__.__name__ == "MultiheadAttention":
                    flops = self._attention_flops(module, x, y)
                layer_flops[name] = float(flops)
            return fn

        for name, module in self.model.named_modules():
            if name and self._is_leaf_module(module):
                handles.append(module.register_forward_hook(hook(name)))

        with torch.no_grad():
            _ = self.model(self._dummy)

        for h in handles:
            h.remove()

        return float(sum(layer_flops.values())), layer_flops

    def _get_ordered_layers(self):
        layers = [("stem", ViTStem(self.model), self.layer_flops.get("conv_proj", 0.0))]
        for i, blk in enumerate(self.model.encoder.layers):
            name = f"encoder_layers_{i}"
            flops = 0.0
            prefix = f"encoder.layers.encoder_layer_{i}"
            for k, v in self.layer_flops.items():
                if k.startswith(prefix):
                    flops += v
            if flops == 0.0:
                prefix2 = f"encoder.layers.{i}"
                for k, v in self.layer_flops.items():
                    if k.startswith(prefix2):
                        flops += v
            layers.append((name, blk, flops))
        layers.append(("head", ViTHead(self.model), self.layer_flops.get("heads", 0.0)))
        return layers

class ResNet18Stem(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ResNet18Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class FlopAwareResNet18PipelineSplitter(FlopAwarePipelineSplitterBase):
    def architecture_name(self) -> str:
        return "resnet18"

    def _get_ordered_layers(self):
        return [
            ("stem", ResNet18Stem(self.model), self.layer_flops.get("conv1", 0.0)
             + self.layer_flops.get("bn1", 0.0)
             + self.layer_flops.get("relu", 0.0)
             + self.layer_flops.get("maxpool", 0.0)),
            ("layer1", self.model.layer1, sum(v for k, v in self.layer_flops.items() if k.startswith("layer1."))),
            ("layer2", self.model.layer2, sum(v for k, v in self.layer_flops.items() if k.startswith("layer2."))),
            ("layer3", self.model.layer3, sum(v for k, v in self.layer_flops.items() if k.startswith("layer3."))),
            ("layer4", self.model.layer4, sum(v for k, v in self.layer_flops.items() if k.startswith("layer4."))),
            ("head", ResNet18Head(self.model), self.layer_flops.get("avgpool", 0.0) + self.layer_flops.get("fc", 0.0)),
        ]


class TCNStem(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class TCNBlockWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class TCNHead(nn.Module):
    def __init__(self, output_proj):
        super().__init__()
        self.output_proj = output_proj

    def forward(self, x):
        x = self.output_proj(x)
        return x.transpose(1, 2)


class FlopAwareTCNPipelineSplitter(FlopAwarePipelineSplitterBase):
    def architecture_name(self) -> str:
        return "tcn"

    def _count_flops(self):
        layer_flops = {}
        handles = []

        def hook(name):
            def fn(module, inp, out):
                x = inp[0] if isinstance(inp, (tuple, list)) else inp
                y = out[0] if isinstance(out, (tuple, list)) else out
                flops = 0.0
                if isinstance(module, nn.Conv1d):
                    flops = self._conv1d_flops(module, x, y)
                elif isinstance(module, (nn.ReLU, nn.Dropout, nn.Identity)):
                    flops = self._elementwise_flops(x, 1.0)
                layer_flops[name] = float(flops)
            return fn

        for name, module in self.model.named_modules():
            if name and self._is_leaf_module(module):
                handles.append(module.register_forward_hook(hook(name)))

        with torch.no_grad():
            _ = self.model(self._dummy)

        for h in handles:
            h.remove()

        return float(sum(layer_flops.values())), layer_flops

    def _get_ordered_layers(self):
        layers = [("stem", TCNStem(), 0.0)]
        for i, blk in enumerate(self.model.network):
            flops = 0.0
            for name, val in self.layer_flops.items():
                if name.startswith(f"network.{i}."):
                    flops += val
            layers.append((f"network_{i}", TCNBlockWrapper(blk), flops))
        layers.append(("head", TCNHead(self.model.output_proj), self.layer_flops.get("output_proj", 0.0)))
        return layers
    
if __name__ == "__main__":
    # ViT
    from torchvision.models import vision_transformer
    vit = vision_transformer.vit_b_16(weights=None).eval()
    vit_splitter = FlopAwareViTPipelineSplitter(vit, (1, 3, 224, 224))
    vit_meta = vit_splitter.split_by_flops_pipeline(
        [30, 33, 11, 18, 8],
        lookahead=5,
        out_dir="./vit_splits",
        meta_name="vit_meta.json",
    )

    # ResNet18
    # from torchvision.models import resnet18
    # resnet = resnet18(weights=None).eval()
    # resnet_splitter = FlopAwareResNet18PipelineSplitter(resnet, (1, 3, 224, 224))
    # resnet_meta = resnet_splitter.split_by_flops_pipeline(
    #     [20, 30, 30, 20],
    #     lookahead=5,
    #     out_dir="./resnet_splits",
    #     meta_name="resnet_meta.json",
    # )

    # TCN
    # tcn = SensorTCN(num_channels=8, hidden_channels=256, levels=8, kernel_size=5, output_channels=8).eval()
    # tcn_splitter = FlopAwareTCNPipelineSplitter(tcn, (1, 2048, 8))
    # tcn_meta = tcn_splitter.split_by_flops_pipeline(
    #     [30, 33, 11, 18, 8],
    #     lookahead=3,
    #     out_dir="./tcn_splits",
    #     meta_name="tcn_meta.json",
    # )