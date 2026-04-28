import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from torchvision.models import vision_transformer

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
    output_tensor: Optional[torch.Tensor] = None  # For chaining

class FlopAwareViTPipelineSplitter:
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
        self.model = model.eval()
        self.input_shape = input_shape
        self.device = next(model.parameters()).device
        self._dummy = torch.randn(input_shape, device=self.device)
        # Simplified FLOPs for demo - replace with your full _count_flops()
        self.total_flops = 1e9
        self.layer_flops = {f"encoder.layers.{i}": 8e7 for i in range(12)}
        self.ordered_layers = self._get_ordered_layers()
        
    def _get_ordered_layers(self):
        layers = []
        
        # Stem as ONE module (not separate parts)
        layers.append(("stem", VitStem(self.model), 2e7))  # Approximate FLOPs
        
        # Encoder layers individually
        for i in range(12):
            layer = self.model.encoder.layers[i]
            layers.append((f"encoder_layers_{i}", layer, self.layer_flops.get(f"encoder.layers.{i}", 8e7)))
        
        # Head as ONE module
        layers.append(("head", VitHead(self.model), 1e6))
        
        return layers
    
    def split_by_flops_pipeline(
        self,
        flop_percentages: List[float],
        lookahead: int = 10,
        out_dir: str = "./vit_flop_pipeline_splits"
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
            # Your FLOP-aware split logic
            best_cut = self._find_best_split(start_idx, target, lookahead)
            layer_group = self.ordered_layers[start_idx:best_cut+1]
            
            # Create and export pipeline module
            pipeline_module = self._create_pipeline_module(layer_group)
            split_meta = self._export_pipeline_split(pipeline_module, x.clone(), split_idx, layer_group, out_dir)
            
            splits.append(split_meta)
            split_flops.append(split_meta.actual_flops)
            x = split_meta.output_tensor  # Chain to next split
            start_idx = best_cut + 1
        
        # Create final metadata
        result = self._create_metadata(splits, flop_percentages, split_flops)
        meta_path = os.path.join(out_dir, "flop_pipeline_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def _find_best_split(self, start_idx: int, target_flops: float, lookahead: int) -> int:
        """Your FLOP-aware split finder"""
        if start_idx == len(self.ordered_layers) - 1:
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
                candidate_flops = sum(l[2] for l in self.ordered_layers[start_idx:j+1])
                cost = abs(candidate_flops - target_flops)
                
                if cost < best_cost:
                    best_cost = cost
                    best_cut = j
        
        return best_cut
    
    def _create_pipeline_module(self, layer_group: List[Tuple[str, nn.Module]]) -> nn.Module:
        """Build exportable sequential module"""
        layers = [(name, module) for name, module, _ in layer_group]
        return ExportableSplit(layers)
    
    def _export_pipeline_split(
        self, 
        module: nn.Module, 
        input_tensor: torch.Tensor, 
        split_id: int, 
        layer_group: List[Tuple[str, nn.Module, float]], 
        out_dir: str
    ) -> FlopAwarePipelineSplit:
        """Export with torch.export and capture shapes"""
        module = module.eval().to(self.device)
        
        # torch.export for ONNX compatibility
        with torch.no_grad():
            exported_model = torch.export.export(module, (input_tensor,))
            exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")
            torch.export.save(exported_model, exported_path)
            
            # Get output shape and tensor for chaining
            output_tensor = exported_model.module()(input_tensor)
            output_shape = tuple(output_tensor.shape)
        
        flops = sum(flops for _, _, flops in layer_group)
        
        meta = FlopAwarePipelineSplit(
            split_id=split_id,
            target_percentage=0.0,  # Filled later
            actual_flops=flops,
            actual_percentage=(flops / self.total_flops) * 100,
            input_shape=tuple(input_tensor.shape),
            output_shape=output_shape,
            module_names=[name for name, _, _ in layer_group],
            exported_path=exported_path,
            cumulative_flops=0.0,  # Filled later
            output_tensor=output_tensor.detach()
        )
        
        print(f"Split {split_id}: {meta.input_shape} → {meta.output_shape} "
              f"({meta.actual_percentage:.1f}%, {meta.actual_flops:.2e} FLOPs)")
        return meta
    
    def _create_metadata(self, splits: List[FlopAwarePipelineSplit], 
                        flop_percentages: List[float], split_flops: List[float]) -> Dict:
        cumulative = 0.0
        result = {"total_flops": self.total_flops, "splits": []}
        
        for i, split in enumerate(splits):
            cumulative += split_flops[i]
            split.target_percentage = flop_percentages[i]
            split.cumulative_flops = cumulative
            
            result["splits"].append({
                "split_id": split.split_id,
                "target_percentage": flop_percentages[i],
                "actual_flops": split.actual_flops,
                "actual_percentage": split.actual_percentage,
                "input_shape": split.input_shape,
                "output_shape": split.output_shape,
                "module_names": split.module_names,
                "exported_path": split.exported_path,
                "cumulative_flops": cumulative,
                "boundary_transfer_bytes": self._bytes_from_shape(split.output_shape) / 1e6
            })
        return result
    
    def _bytes_from_shape(self, shape: Tuple[int, ...]) -> float:
        elements = 1
        for dim in shape:
            elements *= dim
        return elements * 4.0  # float32 bytes

class ParameterWrapper(nn.Module):
    def __init__(self, param: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(param)
    
    def forward(self, x):
        return self.param

class ExportableSplit(nn.Module):
    def __init__(self, layers: List[Tuple[str, nn.Module]]):
        super().__init__()
        self.names = []
        self.layers = nn.ModuleDict()
        
        for name, module in layers:
            self.names.append(name)
            if isinstance(module, torch.Tensor):
                module = ParameterWrapper(module)
            self.layers[name] = module
    
    def forward(self, x):
        for n in self.names:
            x = self.layers[n](x)
        return x


class VitStem(nn.Module):
    """Combined ViT stem: patch projection + class token + pos embed + dropout"""
    def __init__(self, model):
        super().__init__()
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        self.dropout = model.encoder.dropout
    
    def forward(self, x):
        x = self.conv_proj(x)  # [B, C, H, W] -> [B, C, H*W]
        
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)  # [B, C, L] -> [B, L, C]
        
        B = x.shape[0]
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # Add CLS token
        
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class VitHead(nn.Module):
    """Combined ViT head: norm + cls token selection + classifier"""
    def __init__(self, model):
        super().__init__()
        self.ln = model.encoder.ln
        self.heads = model.heads
    
    def forward(self, x):
        x = self.ln(x)
        x = x[:, 0]  # CLS token
        return self.heads(x)


# Usage - now works!
if __name__ == "__main__":
    model = vision_transformer.vit_b_16(weights=None).eval()
    splitter = FlopAwareViTPipelineSplitter(model, input_shape=(1, 3, 224, 224))
    
    result = splitter.split_by_flops_pipeline(
        flop_percentages=[25, 25, 25, 25],
        lookahead=5,
        out_dir="./vit_flop_pipeline_splits_fixed"
    )
    
    print("\n✅ Fixed splits saved with full metadata!")