import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import tcn_library as tcn

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


class TCNBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class TCNStem(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class TCNHead(nn.Module):
    def __init__(self, output_proj: nn.Module):
        super().__init__()
        self.output_proj = output_proj

    def forward(self, x):
        x = self.output_proj(x)
        return x.transpose(1, 2)


class FlopAwareTCNPipelineSplitter:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        self._dummy = torch.randn(input_shape, device=self.device)
        self.total_flops, self.layer_flops = self._count_flops()
        self.ordered_layers = self._get_ordered_layers()

    def _is_leaf_module(self, module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _conv1d_flops(self, m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor) -> float:
        n, c_in, _ = x.shape
        _, c_out, l_out = y.shape
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        groups = m.groups
        macs = n * l_out * c_out * (c_in // groups) * k
        return 2.0 * macs

    def _relu_flops(self, x: torch.Tensor) -> float:
        return 1.0 * x.numel()

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
                    flops = self._relu_flops(x)
                layer_flops[name] = float(flops)
            return fn

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if self._is_leaf_module(module):
                handles.append(module.register_forward_hook(hook(name)))

        with torch.no_grad():
            _ = self.model(self._dummy)

        for h in handles:
            h.remove()

        total_flops = sum(layer_flops.values())
        return float(total_flops), layer_flops

    def _get_ordered_layers(self):
        layers = []
        layers.append(("stem", TCNStem(), 0.0))

        if hasattr(self.model, "network"):
            for i, blk in enumerate(self.model.network):
                name = f"network_{i}"
                flops = sum(
                    self.layer_flops.get(f"network.{i}.net.{j}", 0.0) for j in range(len(list(blk.net)))
                )
                layers.append((name, TCNBlockWrapper(blk), flops))

        if hasattr(self.model, "output_proj"):
            layers.append(("head", TCNHead(self.model.output_proj), self.layer_flops.get("output_proj", 0.0)))

        return layers

    def _collect_shapes_for_layers(self, layers: List[Tuple[str, nn.Module]], x: torch.Tensor):
        shapes = {}
        with torch.no_grad():
            for name, module in layers:
                shapes[name] = tuple(x.shape)
                x = module(x)
            final_shape = tuple(x.shape)
        return shapes, final_shape

    def _bytes_from_shape(self, shape: Optional[Tuple[int, ...]]) -> float:
        if shape is None:
            return 0.0
        elements = 1
        for d in shape:
            elements *= d
        return float(elements * 4)

    def _find_best_split(self, start_idx: int, target_flops: float, lookahead: int) -> int:
        best_cost = float("inf")
        best_cut = start_idx
        for end in range(start_idx, len(self.ordered_layers)):
            running = sum(self.ordered_layers[i][2] for i in range(start_idx, end + 1))
            if running > 1.5 * target_flops and end > start_idx:
                break
            for j in range(end, min(end + lookahead, len(self.ordered_layers))):
                cand = sum(self.ordered_layers[i][2] for i in range(start_idx, j + 1))
                cost = abs(cand - target_flops)
                if cost < best_cost:
                    best_cost = cost
                    best_cut = j
        return best_cut

    def split_by_flops_pipeline(
        self,
        flop_percentages: List[float],
        lookahead: int = 5,
        out_dir: str = "./tcn_flop_pipeline_splits",
    ) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)

        if abs(sum(flop_percentages) - 100.0) > 1e-6:
            raise ValueError("FLOP percentages must sum to 100")

        targets = [(p / 100.0) * self.total_flops for p in flop_percentages]
        splits = []
        split_flops = []
        start_idx = 0
        x = self._dummy.clone()
        cumulative = 0.0

        for split_id, target in enumerate(targets):
            if split_id == len(targets) - 1:
                layer_group = self.ordered_layers[start_idx:]
            else:
                best_cut = self._find_best_split(start_idx, target, lookahead)
                layer_group = self.ordered_layers[start_idx : best_cut + 1]

            split_module = ExportableSplit([(n, m) for n, m, _ in layer_group]).eval().to(self.device)

            with torch.no_grad():
                ep = torch.export.export(split_module, (x,))
                exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")
                torch.export.save(ep, exported_path)
                y = ep.module()(x)
                input_shape = tuple(x.shape)
                output_shape = tuple(y.shape)

            flops = sum(f for _, _, f in layer_group)
            cumulative += flops
            split_flops.append(flops)

            splits.append(
                FlopAwarePipelineSplit(
                    split_id=split_id,
                    target_percentage=flop_percentages[split_id],
                    actual_flops=flops,
                    actual_percentage=(flops / self.total_flops) * 100 if self.total_flops > 0 else 0.0,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    module_names=[n for n, _, _ in layer_group],
                    exported_path=exported_path,
                    cumulative_flops=cumulative,
                    boundary_transfer_bytes=self._bytes_from_shape(output_shape),
                )
            )

            x = y.detach()
            start_idx = len(self.ordered_layers) if split_id == len(targets) - 1 else start_idx + len(layer_group)

        result = {
            "architecture": "tcn",
            "total_flops": self.total_flops,
            "splits": [
                {
                    "split_id": s.split_id,
                    "target_percentage": s.target_percentage,
                    "actual_flops": s.actual_flops,
                    "actual_percentage": s.actual_percentage,
                    "input_shape": s.input_shape,
                    "output_shape": s.output_shape,
                    "module_names": s.module_names,
                    "exported_path": s.exported_path,
                    "cumulative_flops": s.cumulative_flops,
                    "boundary_transfer_bytes": s.boundary_transfer_bytes,
                }
                for s in splits
            ],
        }

        with open(os.path.join(out_dir, "tcn_flop_pipeline_metadata.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

if __name__ == "__main__":
    model = tcn.SensorTCN(
        num_channels=8,
        hidden_channels=256,
        levels=8,
        kernel_size=5,
        output_channels=8,
    ).eval()

    splitter = FlopAwareTCNPipelineSplitter(model, input_shape=(1, 2048, 8))
    result = splitter.split_by_flops_pipeline(
        flop_percentages=[30, 33, 11, 18, 8],
        lookahead=3,
        out_dir="./tcn_flop_pipeline_splits",
    )

    for split in result["splits"]:
        print(split["split_id"], split["input_shape"], "->", split["output_shape"])