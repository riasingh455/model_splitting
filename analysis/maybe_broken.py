import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


class ModelSplitter:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device

        self._dummy = self._dummy_input()
        self.total_flops, self.layer_flops = self._count_flops()
        self.ordered_layers = self._get_ordered_positive_flop_layers()

    def _dummy_input(self):
        return torch.randn(self.input_shape, device=self.device)

    def _is_leaf_module(self, module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _conv2d_flops(self, m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> float:
        n, _, h, w = x.shape
        out_n, out_c, out_h, out_w = y.shape
        kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        groups = m.groups
        cin = m.in_channels
        macs = out_n * out_h * out_w * out_c * (cin // groups) * kh * kw
        return 2.0 * macs

    def _conv1d_flops(self, m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor) -> float:
        n, c_in, l_in = x.shape
        _, c_out, l_out = y.shape
        k = m.kernel_size[0]
        groups = m.groups
        macs = n * l_out * c_out * (c_in // groups) * k
        return 2.0 * macs

    def _linear_flops(self, m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
        if x.dim() == 2:
            batch = x.shape[0]
            return 2.0 * batch * m.in_features * m.out_features
        if x.dim() == 3:
            b, s, _ = x.shape
            return 2.0 * b * s * m.in_features * m.out_features
        return 2.0 * x.numel() * m.out_features / max(m.in_features, 1)

    def _bn_flops(self, x: torch.Tensor) -> float:
        return 4.0 * x.numel()

    def _relu_flops(self, x: torch.Tensor) -> float:
        return 1.0 * x.numel()

    def _layernorm_flops(self, x: torch.Tensor) -> float:
        return 5.0 * x.numel()

    def _attention_flops(self, m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        if x.dim() != 3:
            return 0.0
        b, s, e = x.shape
        num_heads = getattr(m, "num_heads", 1)
        head_dim = e // num_heads if num_heads > 0 else e
        proj = 3.0 * 2.0 * b * s * e * e
        qk = 2.0 * b * num_heads * s * s * head_dim
        av = 2.0 * b * num_heads * s * s * head_dim
        out = 2.0 * b * s * e * e
        return proj + qk + av + out

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
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    flops = self._bn_flops(x)
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.ReLU6, nn.LeakyReLU)):
                    flops = self._relu_flops(x)
                elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    flops = self._layernorm_flops(x)
                elif module.__class__.__name__ == "MultiheadAttention":
                    flops = self._attention_flops(module, x, y)

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

    def _get_ordered_positive_flop_layers(self):
        ordered = []
        for name, module in self.model.named_modules():
            if name == "":
                continue
            flops = self.layer_flops.get(name, 0.0)
            if flops > 0 and self._is_leaf_module(module):
                ordered.append((name, module, flops))
        ordered.sort(key=lambda x: x[0])
        return ordered

    def _collect_output_shapes(self):
        shapes = {}
        handles = []

        def flatten_output(output):
            if isinstance(output, torch.Tensor):
                return output.shape
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        return item.shape
            return None

        def make_hook(name):
            def hook(module, inp, out):
                shapes[name] = flatten_output(out)
            return hook

        for name, module, _ in self.ordered_layers:
            handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = self.model(self._dummy)

        for h in handles:
            h.remove()

        return shapes

    def _boundary_transfer_bytes(self, shape: Optional[Tuple[int, ...]], dtype=torch.float32) -> float:
        if shape is None:
            return 0.0
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return float(total_elements * bytes_per_element)

    def _boundary_cost(
        self,
        flops: float,
        target_flops: float,
        transfer_bytes: float,
        alpha: float = 1.0,
        beta: float = 1e-6,
    ) -> float:
        flop_error = abs(flops - target_flops) / max(target_flops, 1e-9)
        return alpha * flop_error + beta * transfer_bytes

    def split(
        self,
        flop_percentages: List[float],
        lookahead: int = 10,
        alpha: float = 1.0,
        beta: float = 1e-6,
    ) -> Dict:
        if not flop_percentages:
            raise ValueError("flop_percentages cannot be empty")
        total_pct = sum(flop_percentages)
        if abs(total_pct - 100.0) > 1e-6:
            raise ValueError("FLOP percentages must sum to 100")

        layers = self.ordered_layers
        output_shapes = self._collect_output_shapes()

        target_flops = [(p / 100.0) * self.total_flops for p in flop_percentages]
        splits = []
        split_flops = []

        start_idx = 0

        for split_idx, target in enumerate(target_flops):
            if split_idx == len(target_flops) - 1:
                splits.append(layers[start_idx:])
                split_flops.append(sum(l[2] for l in layers[start_idx:]))
                break

            best_cost = float("inf")
            best_cut = start_idx

            running_flops = 0.0
            for i in range(start_idx, len(layers)):
                running_flops += layers[i][2]

                if running_flops < 0.5 * target:
                    continue
                if running_flops > 1.5 * target:
                    break

                end_limit = min(i + lookahead, len(layers))
                for j in range(i, end_limit):
                    candidate_flops = sum(l[2] for l in layers[start_idx : j + 1])
                    shape = output_shapes.get(layers[j][0])
                    transfer_bytes = self._boundary_transfer_bytes(shape)
                    cost = self._boundary_cost(candidate_flops, target, transfer_bytes, alpha=alpha, beta=beta)

                    if cost < best_cost:
                        best_cost = cost
                        best_cut = j

            splits.append(layers[start_idx : best_cut + 1])
            split_flops.append(sum(l[2] for l in layers[start_idx : best_cut + 1]))
            start_idx = best_cut + 1

        cumulative_sum = 0.0
        result = {"total_flops": self.total_flops, "splits": []}

        for i, (target_pct, layer_group) in enumerate(zip(flop_percentages, splits)):
            layers_info = []
            for layer_name, _, _ in layer_group:
                layers_info.append({
                    "layer": layer_name,
                    "flops": self.layer_flops.get(layer_name, 0.0),
                    "output_shape": output_shapes.get(layer_name),
                })

            cumulative_sum += split_flops[i]
            boundary_shape = layers_info[-1]["output_shape"] if layers_info else None

            result["splits"].append({
                "split_id": i,
                "target_percentage": target_pct,
                "actual_flops": split_flops[i],
                "actual_percentage": (split_flops[i] / self.total_flops) * 100 if self.total_flops > 0 else 0.0,
                "layers": layers_info,
                "boundary_output_shape": boundary_shape,
                "boundary_transfer_bytes": self._boundary_transfer_bytes(boundary_shape),
                "cumulative_flops": cumulative_sum,
            })

        return result


def print_split_summary(result: Dict):
    print("\n" + "=" * 100)
    print(f"Model Split Summary (Total FLOPs: {result['total_flops']:.2e})")
    print("=" * 100)

    for split in result["splits"]:
        boundary_mb = split.get("boundary_transfer_bytes", 0.0) / (1024 * 1024)
        print(
            f"\nSplit {split['split_id']}: Target {split['target_percentage']}% "
            f"(Actual: {split['actual_percentage']:.2f}%)"
        )
        print(f"  FLOPs: {split['actual_flops']:.2e}")
        print(f"  Cumulative FLOPs: {split['cumulative_flops']:.2e}")
        print(f"  Boundary transfer: {boundary_mb:.4f} MB")
        print(f"  Layers: {len(split['layers'])}")

        for layer_info in split["layers"]:
            output_shape_str = str(layer_info["output_shape"]) if layer_info["output_shape"] else "Unknown"
            print(f"    - {layer_info['layer']}")
            print(f"      FLOPs: {layer_info['flops']:.2e}, Output shape: {output_shape_str}")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("Network-Aware Model Splitting")
    print("=" * 100)

    import tcn_library

    model = tcn_library.SensorTCN(
        num_channels=8,
        hidden_channels=256,
        levels=8,
        kernel_size=5,
        output_channels=8,
    ).eval()

    example_input = torch.randn(1, 2048, 8)

    splitter = ModelSplitter(model, input_shape=(1, 2048, 8))
    #increase alpha for better compute, increase beta for better network
    result = splitter.split([30, 33, 11, 18, 8], lookahead=10, alpha=50, beta=1e-6)
    print_split_summary(result)