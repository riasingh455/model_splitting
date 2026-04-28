import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from torchvision.models import resnet18

@dataclass
class FlopAwarePipelineSplit:
    split_id: int
    target_percentage: float
    actual_flops: float
    actual_percentage: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    module_names: List[str]
    exported_path: str
    cumulative_flops: float
    output_tensor: Optional[torch.Tensor] = None

class ResNet18Stem(nn.Module):
    """ResNet18 stem: conv1 + bn1 + relu + maxpool"""
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
    """ResNet18 head: avgpool + fc"""
    def __init__(self, model):
        super().__init__()
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class FlopAwareResNet18PipelineSplitter:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        self._dummy = torch.randn(input_shape, device=self.device)
        # Replace with your real FLOP counter if desired
        self.total_flops = 1.8e9  # approximate for ResNet18
        self.layer_flops = self._estimate_resnet18_flops()
        self.ordered_layers = self._get_ordered_layers()

    def _estimate_resnet18_flops(self) -> Dict[str, float]:
        # Approximate FLOPs per stage (replace with your real counter)
        return {
            "stem": 1e7,
            "layer1": 2e8,
            "layer2": 4e8,
            "layer3": 6e8,
            "layer4": 6e8,
            "head": 1e6,
        }

    def _get_ordered_layers(self) -> List[Tuple[str, nn.Module, float]]:
        layers = []

        # Stem as one module
        stem = ResNet18Stem(self.model)
        layers.append(("stem", stem, self.layer_flops["stem"]))

        # ResNet stages with underscore names (no dots for torch.export)
        for i in range(1, 5):
            layer = getattr(self.model, f"layer{i}")
            name = f"layer{i}"  # no dots here already
            flops = self.layer_flops.get(name, 2e8)
            layers.append((name, layer, flops))

        # Head as one module
        head = ResNet18Head(self.model)
        layers.append(("head", head, self.layer_flops["head"]))

        return layers

    def split_by_flops_pipeline(
        self,
        flop_percentages: List[float],
        lookahead: int = 10,
        out_dir: str = "./resnet18_flop_pipeline_splits",
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)

        if abs(sum(flop_percentages) - 100.0) > 1e-6:
            raise ValueError("FLOP percentages must sum to 100")

        target_flops = [(p / 100.0) * self.total_flops for p in flop_percentages]
        splits = []
        split_flops = []
        start_idx = 0
        x = self._dummy.clone()

        for split_idx, target in enumerate(target_flops):
            if split_idx == len(target_flops) - 1:
                layer_group = self.ordered_layers[start_idx:]
            else:
                best_cut = self._find_best_split(start_idx, target, lookahead)
                layer_group = self.ordered_layers[start_idx : best_cut + 1]

            pipeline_module = self._create_pipeline_module(layer_group)
            split_meta = self._export_pipeline_split(
                pipeline_module, x.clone(), split_idx, layer_group, out_dir
            )

            splits.append(split_meta)
            split_flops.append(split_meta.actual_flops)
            x = split_meta.output_tensor
            start_idx = best_cut + 1

        result = self._create_metadata(splits, flop_percentages, split_flops)
        meta_path = os.path.join(out_dir, "resnet18_flop_pipeline_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def _find_best_split(self, start_idx: int, target_flops: float, lookahead: int) -> int:
        if start_idx >= len(self.ordered_layers) - 1:
            return len(self.ordered_layers) - 1

        best_cost = float("inf")
        best_cut = start_idx
        running_flops = 0.0

        for i in range(start_idx, len(self.ordered_layers)):
            running_flops += self.ordered_layers[i][2]
            if running_flops > 1.5 * target_flops:
                break

            end_limit = min(i + lookahead, len(self.ordered_layers))
            for j in range(i, end_limit):
                candidate_flops = sum(
                    l[2] for l in self.ordered_layers[start_idx : j + 1]
                )
                cost = abs(candidate_flops - target_flops)
                if cost < best_cost:
                    best_cost = cost
                    best_cut = j

        return best_cut

    def _create_pipeline_module(
        self, layer_group: List[Tuple[str, nn.Module, float]]
    ) -> nn.Module:
        """Build exportable sequential module from layer group"""
        layers = [(name, module) for name, module, _ in layer_group]
        return ExportableSplit(layers)

    def _export_pipeline_split(
        self,
        module: nn.Module,
        input_tensor: torch.Tensor,
        split_id: int,
        layer_group: List[Tuple[str, nn.Module, float]],
        out_dir: str,
    ) -> FlopAwarePipelineSplit:
        module = module.eval().to(self.device)

        with torch.no_grad():
            exported_model = torch.export.export(module, (input_tensor,))
            exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")
            torch.export.save(exported_model, exported_path)

            output_tensor = exported_model.module()(input_tensor)
            output_shape = tuple(output_tensor.shape)

        flops = sum(flops for _, _, flops in layer_group)

        return FlopAwarePipelineSplit(
            split_id=split_id,
            target_percentage=0.0,
            actual_flops=flops,
            actual_percentage=(flops / self.total_flops) * 100,
            input_shape=tuple(input_tensor.shape),
            output_shape=output_shape,
            module_names=[name for name, _, _ in layer_group],
            exported_path=exported_path,
            cumulative_flops=0.0,
            output_tensor=output_tensor.detach(),
        )

    def _create_metadata(
        self,
        splits: List[FlopAwarePipelineSplit],
        flop_percentages: List[float],
        split_flops: List[float],
    ) -> Dict:
        cumulative = 0.0
        result = {"total_flops": self.total_flops, "splits": []}

        for i, split in enumerate(splits):
            cumulative += split_flops[i]
            split.target_percentage = flop_percentages[i]
            split.cumulative_flops = cumulative

            result["splits"].append(
                {
                    "split_id": split.split_id,
                    "target_percentage": flop_percentages[i],
                    "actual_flops": split.actual_flops,
                    "actual_percentage": split.actual_percentage,
                    "input_shape": split.input_shape,
                    "output_shape": split.output_shape,
                    "module_names": split.module_names,
                    "exported_path": split.exported_path,
                    "cumulative_flops": cumulative,
                    "boundary_transfer_bytes": self._bytes_from_shape(
                        split.output_shape
                    )
                    / 1e6,
                }
            )

        return result

    def _bytes_from_shape(self, shape: Tuple[int, ...]) -> float:
        elements = 1
        for dim in shape:
            elements *= dim
        return elements * 4.0  # float32


class ExportableSplit(nn.Module):
    def __init__(self, layers: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.names = [n for n, _ in layers]
        self.layers = nn.ModuleDict({n: m for n, m in layers})

    def forward(self, x):
        for n in self.names:
            x = self.layers[n](x)
        return x


# Usage
if __name__ == "__main__":
    model = resnet18(weights=None).eval()
    splitter = FlopAwareResNet18PipelineSplitter(
        model, input_shape=(1, 3, 224, 224)
    )

    result = splitter.split_by_flops_pipeline(
        flop_percentages=[20, 30, 30, 20],
        lookahead=5,
        out_dir="./resnet18_flop_pipeline_splits",
    )

    print(f"\n✅ Saved {len(result['splits'])} ResNet18 FLOP-aware pipeline splits")
    print(f"📁 Export paths: split_*.pt2 (ONNX-ready)")
    print(f"📊 Metadata: resnet18_flop_pipeline_metadata.json")

    for split in result["splits"]:
        print(
            f"Split {split['split_id']}: {split['input_shape']} → {split['output_shape']} "
            f"({split['actual_percentage']:.1f}% FLOPs)"
        )