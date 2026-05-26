# import os
# import json
# import math
# import torch
# import torch.nn as nn
# from dataclasses import dataclass
# from typing import List, Tuple, Dict, Optional, Any
# from torch.profiler import profile, ProfilerActivity, record_function

# @dataclass
# class SplitArtifact:
#     split_id: int
#     target_percentage: float
#     actual_flops: float
#     actual_percentage: float
#     input_shape: Tuple[int, ...]
#     output_shape: Tuple[int, ...]
#     module_names: List[str]
#     exported_path: str
#     cumulative_flops: float
#     boundary_transfer_bytes: float
#     network_flops: Dict[str, float]
#     split_score: float
#     chosen_cut: int
#     # mem_usage: float


# class ExportableSplit(nn.Module):
#     def __init__(self, layers: List[Tuple[str, nn.Module]]):
#         super().__init__()
#         self.names = [n for n, _ in layers]
#         self.layers = nn.ModuleDict({n.replace(".", "_"): m for n, m in layers})
#         self._name_map = {n: n.replace(".", "_") for n, _ in layers}

#     def forward(self, x):
#         for n in self.names:
#             x = self.layers[self._name_map[n]](x)
#         return x


# class FlopAwarePipelineSplitterBase:
#     def __init__(self, model: nn.Module, input_shape: Tuple[int, ...]):
#         self.model = model.eval()
#         self.input_shape = input_shape
#         self.device = next(model.parameters()).device
#         self._dummy = torch.randn(input_shape, device=self.device)
#         self.total_flops, self.layer_flops = self._count_flops()
#         self.ordered_layers = self._get_ordered_layers()

#     def architecture_name(self) -> str:
#         return "base"

#     def _is_leaf_module(self, module: nn.Module) -> bool:
#         return len(list(module.children())) == 0

#     def _macs_to_flops(self, macs: float) -> float:
#         return 2.0 * macs

#     def _conv1d_flops(self, m: nn.Conv1d, x: torch.Tensor, y: torch.Tensor) -> float:
#         n, c_in, _ = x.shape
#         _, c_out, l_out = y.shape
#         k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
#         groups = m.groups
#         macs = n * l_out * c_out * (c_in // groups) * k
#         return self._macs_to_flops(macs)

#     def _conv2d_flops(self, m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> float:
#         n, c_in, _, _ = x.shape
#         _, c_out, h_out, w_out = y.shape
#         kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
#         groups = m.groups
#         macs = n * h_out * w_out * c_out * (c_in // groups) * kh * kw
#         return self._macs_to_flops(macs)

#     def _linear_flops(self, m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> float:
#         if x.dim() == 2:
#             macs = x.shape[0] * m.in_features * m.out_features
#         elif x.dim() == 3:
#             b, s, _ = x.shape
#             macs = b * s * m.in_features * m.out_features
#         else:
#             macs = x.numel() * m.out_features / max(m.in_features, 1)
#         return self._macs_to_flops(macs)

#     def _elementwise_flops(self, x: torch.Tensor, ops_per_element: float = 1.0) -> float:
#         return ops_per_element * x.numel()

#     def _attention_flops(self, m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
#         if x.dim() != 3:
#             return 0.0
#         b, s, e = x.shape
#         num_heads = getattr(m, "num_heads", 1)
#         head_dim = e // max(num_heads, 1)
#         qkv_macs = 3.0 * b * s * e * e
#         qk_macs = b * num_heads * s * s * head_dim
#         av_macs = b * num_heads * s * s * head_dim
#         out_macs = b * s * e * e
#         return self._macs_to_flops(qkv_macs + qk_macs + av_macs + out_macs)

#     def _count_flops(self):
#         layer_flops = {}
#         handles = []

#         def hook(name):
#             def fn(module, inp, out):
#                 x = inp[0] if isinstance(inp, (tuple, list)) else inp
#                 y = out[0] if isinstance(out, (tuple, list)) else out
#                 flops = 0.0
#                 if isinstance(module, nn.Conv1d):
#                     flops = self._conv1d_flops(module, x, y)
#                 elif isinstance(module, nn.Conv2d):
#                     flops = self._conv2d_flops(module, x, y)
#                 elif isinstance(module, nn.Linear):
#                     flops = self._linear_flops(module, x, y)
#                 elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU)):
#                     flops = self._elementwise_flops(x, 1.0)
#                 elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
#                                          nn.LayerNorm, nn.GroupNorm,
#                                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
#                     flops = self._elementwise_flops(x, 4.0)
#                 elif isinstance(module, nn.Dropout):
#                     flops = 0.0
#                 elif module.__class__.__name__ == "MultiheadAttention":
#                     flops = self._attention_flops(module, x, y)
#                 layer_flops[name] = float(flops)
#             return fn

#         for name, module in self.model.named_modules():
#             if name and self._is_leaf_module(module):
#                 handles.append(module.register_forward_hook(hook(name)))

#         with torch.no_grad():
#             _ = self.model(self._dummy)

#         for h in handles:
#             h.remove()

#         return float(sum(layer_flops.values())), layer_flops

#     def _bytes_from_shape(self, shape: Optional[Tuple[int, ...]]) -> float:
#         if shape is None:
#             return 0.0
#         total = 1
#         for d in shape:
#             total *= d
#         return float(total * 4)

#     def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[Tuple[str, nn.Module, float]]) -> float:
#         return 0.0

#     def _split_score(
#         self,
#         candidate_flops: float,
#         target_flops: float,
#         boundary_bytes: float,
#         network_penalty: float,
#         w_flop: float,
#         w_net: float,
#         w_comm: float,
#     ) -> float:
#         flop_error = abs(candidate_flops - target_flops) / max(target_flops, 1e-9)
#         comm_cost = boundary_bytes / max(1.0, 1024.0 * 1024.0)
#         return w_flop * flop_error + w_net * network_penalty + w_comm * comm_cost

#     def _find_best_split(
#         self,
#         start_idx: int,
#         target_flops: float,
#         lookahead: int,
#         w_flop: float,
#         w_net: float,
#         w_comm: float,
#     ) -> Tuple[int, float]:
#         best_cut = start_idx
#         best_score = float("inf")
#         end_limit_global = min(len(self.ordered_layers), start_idx + max(lookahead, 1) * 8)

#         for end in range(start_idx, end_limit_global):
#             candidate = sum(self.ordered_layers[i][2] for i in range(start_idx, end + 1))
#             if candidate > 1.5 * target_flops and end > start_idx:
#                 break
#             group = self.ordered_layers[start_idx:end + 1]
#             boundary_shape = None
#             network_pen = self._network_penalty(end, start_idx, group)
#             score = self._split_score(candidate, target_flops, self._bytes_from_shape(boundary_shape), network_pen, w_flop, w_net, w_comm)
#             if score < best_score:
#                 best_score = score
#                 best_cut = end

#         return best_cut, best_score

#     def _layer_group_flops(self, group: List[Tuple[str, nn.Module, float]]) -> Dict[str, float]:
#         return {name: float(flops) for name, _, flops in group}

#     def split_by_flops_pipeline(
#         self,
#         flop_percentages: List[float],
#         lookahead: int,
#         out_dir: str,
#         meta_name: str,
#         w_flop: float = 1.0,
#         w_net: float = 0.0,
#         w_comm: float = 0.0,
#         export: bool = False,
#         profiler: bool = False,
#     ) -> Dict[str, Any]:

#         if abs(sum(flop_percentages) - 100.0) > 1e-6:
#             raise ValueError("FLOP percentages must sum to 100")

#         targets = [(p / 100.0) * self.total_flops for p in flop_percentages]
#         x = self._dummy.clone()
#         start_idx = 0
#         cumulative = 0.0
#         artifacts = []

#         for split_id, target in enumerate(targets):
#             if split_id == len(targets) - 1:
#                 group = self.ordered_layers[start_idx:]
#                 cut = len(self.ordered_layers) - 1
#                 split_score = 0.0
#             else:
#                 cut, split_score = self._find_best_split(start_idx, target, lookahead, w_flop, w_net, w_comm)
#                 group = self.ordered_layers[start_idx:cut + 1]

#             module = ExportableSplit([(n, m) for n, m, _ in group]).eval().to(self.device)
#             inp_shape = tuple(x.shape)
#             exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")
#             # mem_usage = -1
#             with torch.no_grad():
#                 ep = torch.export.export(module, (x,), strict=True,
#                 dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},)
#                     # dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=2**32)}}#, "output": {0: torch.export.Dim("batch", min=1, max=1024)}}
#                 )
                
#                 # Path.unlink(f"{out_dir}/split_{split_id}.onnx")

#                 if export:
#                     os.makedirs(out_dir, exist_ok=True)
#                     # torch.export.save(ep, exported_path)

#                     # onnx_ep = torch.onnx.export(module, (x,), 
#                     onnx_ep = torch.onnx.export(ep, (x,), 
#                     input_names=["input"],
#                     output_names=["output"],
#                     # optimize=False if "vit" in out_dir else True,
#                     # dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},)#, "y": {0: torch.export.Dim("batch", min=1, max=1024)}}
#                     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} #if "vit" not in out_dir else {"input": {0: torch.export.Dim("batch", min=1, max=1024)}, "output": {0: torch.export.Dim("batch", min=1, max=1024)}}
#                     )

#                     from onnxsim import simplify
#                     import onnx
#                     from onnxruntime.quantization import quantize_dynamic, QuantType
#                     from pathlib import Path
#                     # onnx.save(onnx_ep,f"{out_dir}/split_{split_id}.onnx" )
#                     # onnx_model = onnx.load(f"{out_dir}/split_{split_id}.onnx")
#                     # if "vit" not in out_dir:
#                     # print(onnx_ep.model_proto)
#                     onnx_sim, check = simplify(onnx_ep.model_proto)#, perform_optimization=False if "vit" in out_dir else True, 
#                     # skip_shape_inference=True if "vit" in out_dir else False)
#                         # if check:
#                     onnx.save(onnx_sim, f"{out_dir}/split_{split_id}.simp.onnx")
#                     # else:
#                     #     onnx.save(onnx_ep.model_proto, f"{out_dir}/split_{split_id}.simp.onnx")
#                     quantize_dynamic(f"{out_dir}/split_{split_id}.simp.onnx", 
#                     f"{out_dir}/split_{split_id}_quant.onnx", weight_type=QuantType.QUInt8)
#                     Path.unlink(f"{out_dir}/split_{split_id}.simp.onnx")
#                         # quantize_dynamic(f"{out_dir}/split_{split_id}.onnx", 
#                         # f"{out_dir}/split_{split_id}_quant.onnx", weight_type=QuantType.QUInt8)
#                         # # Path.unlink(f"{out_dir}/split_{split_id}.simp.onnx")
#                 # if profiler:
#                 #     # onnx_mod = onnx.load(f"{out_dir}/split_{split_id}_quant.onnx")
#                 #     # import onnxruntime as ort
#                 #     # import numpy as np
#                 #     # onnx_mod = ort.InferenceSession(f"{out_dir}/split_{split_id}_quant.onnx")
#                 #     # input_map = {i.name: x.numpy() for ind, i in enumerate(onnx_mod.get_inputs())}
#                 #     # print(input_map)
#                 #     from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
#                 #     from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
#                 #     qparams = get_symmetric_quantization_config(is_per_channel=True)
#                 #     quantizer = XNNPACKQuantizer()
#                 #     quantizer.set_global(qparams)
#                 #     prepared_model = prepare_pt2e(ep.module(), quantizer)
#                 #     prepared_model(x)
#                 #     quantized_model = convert_pt2e(prepared_model)
#                 #     q_ep = torch.export.export(quantized_model, (x,))
#                 #     ep = torch.export.export(q_ep.module(), (x,))
#                 #     y = ep.module()(x)

#                 #     with profile(
#                 #         activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
#                 #     ) as prof:
#                 #         y = ep.module()(x)
#                         # y = onnx_mod.run(None, input_feed=input_map)
#                     # mem_usage = prof.key_averages()
#                     # print(mem_usage)
#                 # else:
#                 y = ep.module()(x)

#             flops = sum(f for _, _, f in group)
#             cumulative += flops
#             artifacts.append(
#                 SplitArtifact(
#                     split_id=split_id,
#                     target_percentage=flop_percentages[split_id],
#                     actual_flops=flops,
#                     actual_percentage=(flops / self.total_flops) * 100 if self.total_flops > 0 else 0.0,
#                     input_shape=inp_shape,
#                     output_shape=tuple(y.shape),
#                     module_names=[n for n, _, _ in group],
#                     exported_path=exported_path,
#                     cumulative_flops=cumulative,
#                     boundary_transfer_bytes=self._bytes_from_shape(tuple(y.shape)),
#                     network_flops=self._layer_group_flops(group),
#                     split_score=split_score,
#                     chosen_cut=cut,
#                     # mem_usage=mem_usage#(mem_usage/8 - self._bytes_from_shape(tuple(y.shape)))  if mem_usage!=-1 else mem_usage
#                 )
#             )

#             x = y.detach()
#             start_idx = cut + 1

#         result = {
#             "architecture": self.architecture_name(),
#             "total_flops": self.total_flops,
#             "weights": {"w_flop": w_flop, "w_net": w_net, "w_comm": w_comm},
#             "splits": [a.__dict__ for a in artifacts],
#         }
#         if export:
#             with open(os.path.join(out_dir, meta_name), "w") as f:
#                 json.dump(result, f, indent=2, default=str)

#         return result

#     def _get_ordered_layers(self):
#         raise NotImplementedError


# class ViTStem(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.conv_proj = model.conv_proj
#         self.class_token = model.class_token
#         self.pos_embedding = model.encoder.pos_embedding
#         self.dropout = model.encoder.dropout

#     def forward(self, x):
#         x = self.conv_proj(x)
#         if x.dim() == 4:
#             x = x.flatten(2).transpose(1, 2)
#         b = x.shape[0]
#         cls = self.class_token.expand(b, -1, -1)
#         x = torch.cat([cls, x], dim=1)
#         x = x + self.pos_embedding
#         x = self.dropout(x)
#         return x


# class ViTHead(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.ln = model.encoder.ln
#         self.heads = model.heads

#     def forward(self, x):
#         x = self.ln(x)
#         # x = x[:, 0]
#         if x.dim() == 3:
#             x = x[:, 0]
#         elif x.dim() != 2:
#             raise RuntimeError(f"Unexpected ViTHead input shape: {x.shape}")
#         return self.heads(x)

# # class ViTStem(nn.Module):
# #     def __init__(self, model):
# #         super().__init__()
# #         self.conv_proj = model.conv_proj
# #         self.class_token = model.class_token
# #         self.pos_embedding = model.encoder.pos_embedding
# #         self.dropout = model.encoder.dropout

# #     def forward(self, x):
# #         x = self.conv_proj(x)
# #         if x.dim() != 4:
# #             raise RuntimeError(f"Expected conv_proj output to be 4D, got {x.shape}")
# #         x = x.flatten(2).transpose(1, 2)  # [B, E, H, W] -> [B, N, E]
# #         b = x.shape[0]
# #         cls = self.class_token.expand(b, -1, -1)
# #         x = torch.cat([cls, x], dim=1)
# #         x = x + self.pos_embedding
# #         return self.dropout(x)


# # class ViTHead(nn.Module):
# #     def __init__(self, model):
# #         super().__init__()
# #         self.ln = model.encoder.ln
# #         self.heads = model.heads

# #     def forward(self, x):
# #         if x.dim() != 3:
# #             raise RuntimeError(f"Expected token tensor [B, S, E], got {x.shape}")
# #         x = self.ln(x)
# #         x = x[:, 0]
# #         return self.heads(x)


# class FlopAwareViTPipelineSplitter(FlopAwarePipelineSplitterBase):
#     def architecture_name(self) -> str:
#         return "vit"

#     def _count_flops(self):
#         layer_flops = {}
#         handles = []

#         def hook(name):
#             def fn(module, inp, out):
#                 x = inp[0] if isinstance(inp, (tuple, list)) else inp
#                 y = out[0] if isinstance(out, (tuple, list)) else out
#                 flops = 0.0
#                 if isinstance(module, nn.Conv2d):
#                     flops = self._conv2d_flops(module, x, y)
#                 elif isinstance(module, nn.Linear):
#                     flops = self._linear_flops(module, x, y)
#                 elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU)):
#                     flops = self._elementwise_flops(x, 1.0)
#                 elif isinstance(module, (nn.LayerNorm, nn.GroupNorm,
#                                          nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
#                                          nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
#                     flops = self._elementwise_flops(x, 4.0)
#                 elif module.__class__.__name__ == "MultiheadAttention":
#                     flops = self._attention_flops(module, x, y)
#                 layer_flops[name] = float(flops)
#             return fn

#         for name, module in self.model.named_modules():
#             if name and self._is_leaf_module(module):
#                 handles.append(module.register_forward_hook(hook(name)))

#         with torch.no_grad():
#             _ = self.model(self._dummy)

#         for h in handles:
#             h.remove()

#         return float(sum(layer_flops.values())), layer_flops

#     def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[Tuple[str, nn.Module, float]]) -> float:
#         name = group[-1][0]
#         return 0.0 if name == "head" or name.startswith("encoder_layers_") or name == "stem" else 1.0

#     # def _get_ordered_layers(self):
#     #     layers = [("stem", ViTStem(self.model), self.layer_flops.get("conv_proj", 0.0))]
#     #     for i, blk in enumerate(self.model.encoder.layers):
#     #         name = f"encoder_layer_{i}"
#     #         prefix = f"encoder.layers.encoder_layer_{i}"
#     #         flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix))
#     #         layers.append((name, blk, flops))
#     #     layers.append(("head", ViTHead(self.model), self.layer_flops.get("heads", 0.0)))
#     #     return layers
    
#     def _get_ordered_layers(self):
#         layers = [("stem", ViTStem(self.model), self.layer_flops.get("conv_proj", 0.0))]
#         for i, blk in enumerate(self.model.encoder.layers):
#             name = f"encoder_layers_{i}"
#             prefix = f"encoder.layers.encoder_layer_{i}"
#             flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix))
#             if flops == 0.0:
#                 prefix2 = f"encoder.layers.{i}"
#                 flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix2))
#             layers.append((name, blk, flops))
#         layers.append(("head", ViTHead(self.model), self.layer_flops.get("heads", 0.0)))
#         return layers

# class ResNet18Stem(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.conv1 = model.conv1
#         self.bn1 = model.bn1
#         self.relu = model.relu
#         self.maxpool = model.maxpool

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         return x


# class ResNet18Head(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.avgpool = model.avgpool
#         self.fc = model.fc

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)


# class FlopAwareResNet18PipelineSplitter(FlopAwarePipelineSplitterBase):
#     def architecture_name(self) -> str:
#         return "resnet18"

#     def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[Tuple[str, nn.Module, float]]) -> float:
#         name = group[-1][0]
#         return 0.0 if name in {"stem", "layer1", "layer2", "layer3", "layer4", "head"} else 1.0

#     def _get_ordered_layers(self):
#         return [
#             ("stem", ResNet18Stem(self.model),
#              self.layer_flops.get("conv1", 0.0) + self.layer_flops.get("bn1", 0.0) + self.layer_flops.get("relu", 0.0) + self.layer_flops.get("maxpool", 0.0)),
#             ("layer1", self.model.layer1, sum(v for k, v in self.layer_flops.items() if k.startswith("layer1."))),
#             ("layer2", self.model.layer2, sum(v for k, v in self.layer_flops.items() if k.startswith("layer2."))),
#             ("layer3", self.model.layer3, sum(v for k, v in self.layer_flops.items() if k.startswith("layer3."))),
#             ("layer4", self.model.layer4, sum(v for k, v in self.layer_flops.items() if k.startswith("layer4."))),
#             ("head", ResNet18Head(self.model), self.layer_flops.get("avgpool", 0.0) + self.layer_flops.get("fc", 0.0)),
#         ]


# # EDIT: Added EfficientNet-B0 head wrapper so the final split can run avgpool -> flatten -> classifier.
# class EfficientNetB0Head(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.avgpool = model.avgpool
#         self.classifier = model.classifier

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)


# # EDIT: Added EfficientNet-B0 splitter using torchvision EfficientNet's features blocks.
# class FlopAwareEfficientNetB0PipelineSplitter(FlopAwarePipelineSplitterBase):
#     def architecture_name(self) -> str:
#         return "efficientnet_b0"

#     def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[Tuple[str, nn.Module, float]]) -> float:
#         name = group[-1][0]
#         return 0.0 if name in {"stem", "final_conv", "head"} or name.startswith("features_") else 1.0

#     def _get_ordered_layers(self):
#         layers = [
#             ("stem", self.model.features[0], sum(v for k, v in self.layer_flops.items() if k.startswith("features.0.")))
#         ]

#         for i in range(1, len(self.model.features) - 1):
#             layers.append((
#                 f"features_{i}",
#                 self.model.features[i],
#                 sum(v for k, v in self.layer_flops.items() if k.startswith(f"features.{i}."))
#             ))

#         last_idx = len(self.model.features) - 1
#         layers.append((
#             "final_conv",
#             self.model.features[last_idx],
#             sum(v for k, v in self.layer_flops.items() if k.startswith(f"features.{last_idx}."))
#         ))

#         layers.append((
#             "head",
#             EfficientNetB0Head(self.model),
#             self.layer_flops.get("avgpool", 0.0) + sum(v for k, v in self.layer_flops.items() if k.startswith("classifier."))
#         ))

#         return layers

# class TCNStem(nn.Module):
#     def forward(self, x):
#         return x.transpose(1, 2)


# class TCNBlockWrapper(nn.Module):
#     def __init__(self, block):
#         super().__init__()
#         self.block = block

#     def forward(self, x):
#         return self.block(x)


# class TCNHead(nn.Module):
#     def __init__(self, output_proj):
#         super().__init__()
#         self.output_proj = output_proj

#     def forward(self, x):
#         x = self.output_proj(x)
#         return x.transpose(1, 2)


# class FlopAwareTCNPipelineSplitter(FlopAwarePipelineSplitterBase):
#     def architecture_name(self) -> str:
#         return "tcn"

#     def _count_flops(self):
#         layer_flops = {}
#         handles = []

#         def hook(name):
#             def fn(module, inp, out):
#                 x = inp[0] if isinstance(inp, (tuple, list)) else inp
#                 y = out[0] if isinstance(out, (tuple, list)) else out
#                 flops = 0.0
#                 if isinstance(module, nn.Conv1d):
#                     flops = self._conv1d_flops(module, x, y)
#                 elif isinstance(module, (nn.ReLU, nn.Dropout, nn.Identity)):
#                     flops = self._elementwise_flops(x, 1.0)
#                 layer_flops[name] = float(flops)
#             return fn

#         for name, module in self.model.named_modules():
#             if name and self._is_leaf_module(module):
#                 handles.append(module.register_forward_hook(hook(name)))

#         with torch.no_grad():
#             _ = self.model(self._dummy)

#         for h in handles:
#             h.remove()

#         return float(sum(layer_flops.values())), layer_flops

#     def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[Tuple[str, nn.Module, float]]) -> float:
#         name = group[-1][0]
#         return 0.0 if name in {"stem", "head"} or name.startswith("network_") else 1.0

#     def _get_ordered_layers(self):
#         layers = [("stem", TCNStem(), 0.0)]
#         for i, blk in enumerate(self.model.network):
#             flops = sum(v for k, v in self.layer_flops.items() if k.startswith(f"network.{i}."))
#             layers.append((f"network_{i}", TCNBlockWrapper(blk), flops))
#         layers.append(("head", TCNHead(self.model.output_proj), self.layer_flops.get("output_proj", 0.0)))
#         return layers

# if __name__=="__main__":
#     from torchvision.models import vision_transformer

#     # model = vision_transformer.vit_b_16(weights=None).eval()
#     # # splitter = FlopAwareViTPipelineSplitter(model, input_shape=(1, 3, 224, 224))
#     # splitter = FlopAwareViTPipelineSplitter(model, input_shape=(2, 3, 224, 224))

#     # # Tradeoff intuition
#     # #     If w_flop is high, the splitter prioritizes balanced compute across stages. -> keeps percentage close to provided percentage
#     # #     If w_net is high, it prefers clean architectural boundaries even if FLOPs are a little off. -> always ignored actually lmao
#     # #     If w_comm is high, it tries to minimize transfer size between splits, which matters when the boundary activation is large. -> keeps communication borders big
#     # result = splitter.split_by_flops_pipeline(
#     #     flop_percentages=[30, 33, 11, 18, 8],
#     #     lookahead=5,
#     #     out_dir="./vit_splits",
#     #     meta_name="meta.json",
#     #     w_flop=1.0,
#     #     w_net=0.5,
#     #     w_comm=0.1,
#     #     export=True
#     # )

# #reconstruction
# # meta = json.load(open("./vit_splits/vit_meta.json"))
# # modules = []
# # for part in meta["splits"]:
# #     ep = torch.export.load(part["exported_path"])
# #     modules.append(ep.module())

#     # import tcn_library as tcn
#     # model = tcn.SensorTCN(
#     # num_channels=8,
#     # hidden_channels=256,
#     # levels=8,
#     # kernel_size=5,
#     # output_channels=8,
#     # ).eval()

#     # splitter = FlopAwareTCNPipelineSplitter(model, input_shape=(1, 2048, 8))

#     # result = splitter.split_by_flops_pipeline(
#     #     flop_percentages=[30, 33, 11, 18, 8],
#     #     lookahead=3,
#     #     out_dir="./tcn_splits",
#     #     meta_name="meta.json",
#     #     w_flop=1.0,
#     #     w_net=0.5,
#     #     w_comm=0.1,
#     #     profiler=True,
#     #     export=True
#     # )

#     # from torchvision.models import resnet18
#     #
#     # model = resnet18(weights=None).eval()
#     # splitter = FlopAwareResNet18PipelineSplitter(model, input_shape=(2, 3, 224, 224))
#     #
#     # result = splitter.split_by_flops_pipeline(
#     #     flop_percentages=[20, 30, 30, 20],
#     #     lookahead=5,
#     #     out_dir="./resnet_splits",
#     #     meta_name="resnet_meta.json",
#     #     w_flop=1.0,
#     #     w_net=0.5,
#     #     w_comm=0.1,
#     #     export=True
#     # )

#     # EDIT: Added EfficientNet-B0 run block.
#     from torchvision.models import efficientnet_b0

#     model = efficientnet_b0(weights=None).eval()
#     splitter = FlopAwareEfficientNetB0PipelineSplitter(model, input_shape=(2, 3, 224, 224))

#     result = splitter.split_by_flops_pipeline(
#         flop_percentages=[50, 50],
#         lookahead=50,
#         out_dir="./efficientnet_b0_splits",
#         meta_name="efficientnet_b0_meta.json",
#         w_flop=0,
#         w_net=0,
#         w_comm=1,
#         export=True
#     )



import os
import json
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from torch.profiler import profile, ProfilerActivity, record_function

# Ordered layer tuple format used by all splitters:
# (name, module, flops, boundary_transfer_bytes_after_this_layer)
OrderedLayer = Tuple[str, nn.Module, float, float]

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
    network_flops: Dict[str, float]
    split_score: float
    chosen_cut: int
    # mem_usage: float


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

    def _add_boundary_bytes_to_layers(self, layers: List[Tuple[str, nn.Module, float]]) -> List[OrderedLayer]:
        """
        Given architecture-specific ordered layers of (name, module, flops),
        run one forward sweep and return (name, module, flops, boundary_bytes).

        boundary_bytes is the tensor transfer size immediately after that layer/stage,
        so split scoring can use it without recomputing candidate outputs.
        """
        annotated_layers: List[OrderedLayer] = []
        x = self._dummy.clone()

        with torch.no_grad():
            for name, module, flops in layers:
                module = module.eval().to(self.device)
                y = module(x)
                boundary_bytes = self._bytes_from_shape(tuple(y.shape))
                annotated_layers.append((name, module, flops, boundary_bytes))
                x = y.detach()

        return annotated_layers

    def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[OrderedLayer]) -> float:
        return 0.0

    def _split_score(
        self,
        candidate_flops: float,
        target_flops: float,
        boundary_bytes: float,
        network_penalty: float,
        w_flop: float,
        w_net: float,
        w_comm: float,
    ) -> float:
        flop_error = abs(candidate_flops - target_flops) / max(target_flops, 1e-9)
        comm_cost = boundary_bytes / max(1.0, 1024.0 * 1024.0)
        return w_flop * flop_error + w_net * network_penalty + w_comm * comm_cost

    def _find_best_split(
        self,
        start_idx: int,
        target_flops: float,
        lookahead: int,
        w_flop: float,
        w_net: float,
        w_comm: float,
    ) -> Tuple[int, float]:
        best_cut = start_idx
        best_score = float("inf")
        end_limit_global = min(len(self.ordered_layers), start_idx + max(lookahead, 1) * 8)

        for end in range(start_idx, end_limit_global):
            candidate = sum(self.ordered_layers[i][2] for i in range(start_idx, end + 1))
            if candidate > 1.5 * target_flops and end > start_idx:
                break
            group = self.ordered_layers[start_idx:end + 1]
            boundary_bytes = self.ordered_layers[end][3]
            network_pen = self._network_penalty(end, start_idx, group)
            score = self._split_score(candidate, target_flops, boundary_bytes, network_pen, w_flop, w_net, w_comm)
            if score < best_score:
                best_score = score
                best_cut = end

        return best_cut, best_score

    def _layer_group_flops(self, group: List[OrderedLayer]) -> Dict[str, float]:
        return {name: float(flops) for name, _, flops, _ in group}

    def split_by_flops_pipeline(
        self,
        flop_percentages: List[float],
        lookahead: int,
        out_dir: str,
        meta_name: str,
        w_flop: float = 1.0,
        w_net: float = 0.0,
        w_comm: float = 0.0,
        export: bool = False,
        profiler: bool = False,
    ) -> Dict[str, Any]:

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
                cut = len(self.ordered_layers) - 1
                split_score = 0.0
            else:
                cut, split_score = self._find_best_split(start_idx, target, lookahead, w_flop, w_net, w_comm)
                group = self.ordered_layers[start_idx:cut + 1]

            module = ExportableSplit([(n, m) for n, m, _, _ in group]).eval().to(self.device)
            inp_shape = tuple(x.shape)
            exported_path = os.path.join(out_dir, f"split_{split_id}.pt2")
            # mem_usage = -1
            with torch.no_grad():
                ep = torch.export.export(module, (x,), strict=True,
                dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},)
                    # dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=2**32)}}#, "output": {0: torch.export.Dim("batch", min=1, max=1024)}}
                )
                
                # Path.unlink(f"{out_dir}/split_{split_id}.onnx")

                if export:
                    os.makedirs(out_dir, exist_ok=True)
                    # torch.export.save(ep, exported_path)

                    # onnx_ep = torch.onnx.export(module, (x,), 
                    onnx_ep = torch.onnx.export(ep, (x,), 
                    input_names=["input"],
                    output_names=["output"],
                    # optimize=False if "vit" in out_dir else True,
                    # dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=1024)},)#, "y": {0: torch.export.Dim("batch", min=1, max=1024)}}
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} #if "vit" not in out_dir else {"input": {0: torch.export.Dim("batch", min=1, max=1024)}, "output": {0: torch.export.Dim("batch", min=1, max=1024)}}
                    )

                    from onnxsim import simplify
                    import onnx
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    from pathlib import Path
                    # onnx.save(onnx_ep,f"{out_dir}/split_{split_id}.onnx" )
                    # onnx_model = onnx.load(f"{out_dir}/split_{split_id}.onnx")
                    # if "vit" not in out_dir:
                    # print(onnx_ep.model_proto)
                    onnx_sim, check = simplify(onnx_ep.model_proto)#, perform_optimization=False if "vit" in out_dir else True, 
                    # skip_shape_inference=True if "vit" in out_dir else False)
                        # if check:
                    onnx.save(onnx_sim, f"{out_dir}/split_{split_id}.simp.onnx")
                    # else:
                    #     onnx.save(onnx_ep.model_proto, f"{out_dir}/split_{split_id}.simp.onnx")
                    quantize_dynamic(f"{out_dir}/split_{split_id}.simp.onnx", 
                    f"{out_dir}/split_{split_id}_quant.onnx", weight_type=QuantType.QUInt8)
                    Path.unlink(f"{out_dir}/split_{split_id}.simp.onnx")
                        # quantize_dynamic(f"{out_dir}/split_{split_id}.onnx", 
                        # f"{out_dir}/split_{split_id}_quant.onnx", weight_type=QuantType.QUInt8)
                        # # Path.unlink(f"{out_dir}/split_{split_id}.simp.onnx")
                # if profiler:
                #     # onnx_mod = onnx.load(f"{out_dir}/split_{split_id}_quant.onnx")
                #     # import onnxruntime as ort
                #     # import numpy as np
                #     # onnx_mod = ort.InferenceSession(f"{out_dir}/split_{split_id}_quant.onnx")
                #     # input_map = {i.name: x.numpy() for ind, i in enumerate(onnx_mod.get_inputs())}
                #     # print(input_map)
                #     from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
                #     from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
                #     qparams = get_symmetric_quantization_config(is_per_channel=True)
                #     quantizer = XNNPACKQuantizer()
                #     quantizer.set_global(qparams)
                #     prepared_model = prepare_pt2e(ep.module(), quantizer)
                #     prepared_model(x)
                #     quantized_model = convert_pt2e(prepared_model)
                #     q_ep = torch.export.export(quantized_model, (x,))
                #     ep = torch.export.export(q_ep.module(), (x,))
                #     y = ep.module()(x)

                #     with profile(
                #         activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
                #     ) as prof:
                #         y = ep.module()(x)
                        # y = onnx_mod.run(None, input_feed=input_map)
                    # mem_usage = prof.key_averages()
                    # print(mem_usage)
                # else:
                y = ep.module()(x)

            flops = sum(f for _, _, f, _ in group)
            cumulative += flops
            artifacts.append(
                SplitArtifact(
                    split_id=split_id,
                    target_percentage=flop_percentages[split_id],
                    actual_flops=flops,
                    actual_percentage=(flops / self.total_flops) * 100 if self.total_flops > 0 else 0.0,
                    input_shape=inp_shape,
                    output_shape=tuple(y.shape),
                    module_names=[n for n, _, _, _ in group],
                    exported_path=exported_path,
                    cumulative_flops=cumulative,
                    boundary_transfer_bytes=self._bytes_from_shape(tuple(y.shape)),
                    network_flops=self._layer_group_flops(group),
                    split_score=split_score,
                    chosen_cut=cut,
                    # mem_usage=mem_usage#(mem_usage/8 - self._bytes_from_shape(tuple(y.shape)))  if mem_usage!=-1 else mem_usage
                )
            )

            x = y.detach()
            start_idx = cut + 1

        result = {
            "architecture": self.architecture_name(),
            "total_flops": self.total_flops,
            "weights": {"w_flop": w_flop, "w_net": w_net, "w_comm": w_comm},
            "splits": [a.__dict__ for a in artifacts],
        }
        if export:
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
        # x = x[:, 0]
        if x.dim() == 3:
            x = x[:, 0]
        elif x.dim() != 2:
            raise RuntimeError(f"Unexpected ViTHead input shape: {x.shape}")
        return self.heads(x)

# class ViTStem(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.conv_proj = model.conv_proj
#         self.class_token = model.class_token
#         self.pos_embedding = model.encoder.pos_embedding
#         self.dropout = model.encoder.dropout

#     def forward(self, x):
#         x = self.conv_proj(x)
#         if x.dim() != 4:
#             raise RuntimeError(f"Expected conv_proj output to be 4D, got {x.shape}")
#         x = x.flatten(2).transpose(1, 2)  # [B, E, H, W] -> [B, N, E]
#         b = x.shape[0]
#         cls = self.class_token.expand(b, -1, -1)
#         x = torch.cat([cls, x], dim=1)
#         x = x + self.pos_embedding
#         return self.dropout(x)


# class ViTHead(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.ln = model.encoder.ln
#         self.heads = model.heads

#     def forward(self, x):
#         if x.dim() != 3:
#             raise RuntimeError(f"Expected token tensor [B, S, E], got {x.shape}")
#         x = self.ln(x)
#         x = x[:, 0]
#         return self.heads(x)


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
                elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.LeakyReLU)):
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

    def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[OrderedLayer]) -> float:
        name = group[-1][0]
        return 0.0 if name == "head" or name.startswith("encoder_layers_") or name == "stem" else 1.0

    # def _get_ordered_layers(self):
    #     layers = [("stem", ViTStem(self.model), self.layer_flops.get("conv_proj", 0.0))]
    #     for i, blk in enumerate(self.model.encoder.layers):
    #         name = f"encoder_layer_{i}"
    #         prefix = f"encoder.layers.encoder_layer_{i}"
    #         flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix))
    #         layers.append((name, blk, flops))
    #     layers.append(("head", ViTHead(self.model), self.layer_flops.get("heads", 0.0)))
    #     return layers
    
    def _get_ordered_layers(self):
        layers = [("stem", ViTStem(self.model), self.layer_flops.get("conv_proj", 0.0))]
        for i, blk in enumerate(self.model.encoder.layers):
            name = f"encoder_layers_{i}"
            prefix = f"encoder.layers.encoder_layer_{i}"
            flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix))
            if flops == 0.0:
                prefix2 = f"encoder.layers.{i}"
                flops = sum(v for k, v in self.layer_flops.items() if k.startswith(prefix2))
            layers.append((name, blk, flops))
        layers.append(("head", ViTHead(self.model), self.layer_flops.get("heads", 0.0)))
        return self._add_boundary_bytes_to_layers(layers)

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

    def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[OrderedLayer]) -> float:
        name = group[-1][0]
        return 0.0 if name in {"stem", "layer1", "layer2", "layer3", "layer4", "head"} else 1.0

    def _get_ordered_layers(self):
        layers = [
            ("stem", ResNet18Stem(self.model),
             self.layer_flops.get("conv1", 0.0) + self.layer_flops.get("bn1", 0.0) + self.layer_flops.get("relu", 0.0) + self.layer_flops.get("maxpool", 0.0)),
            ("layer1", self.model.layer1, sum(v for k, v in self.layer_flops.items() if k.startswith("layer1."))),
            ("layer2", self.model.layer2, sum(v for k, v in self.layer_flops.items() if k.startswith("layer2."))),
            ("layer3", self.model.layer3, sum(v for k, v in self.layer_flops.items() if k.startswith("layer3."))),
            ("layer4", self.model.layer4, sum(v for k, v in self.layer_flops.items() if k.startswith("layer4."))),
            ("head", ResNet18Head(self.model), self.layer_flops.get("avgpool", 0.0) + self.layer_flops.get("fc", 0.0)),
        ]
        return self._add_boundary_bytes_to_layers(layers)


# EDIT: Added EfficientNet-B0 head wrapper so the final split can run avgpool -> flatten -> classifier.
class EfficientNetB0Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# EDIT: Added EfficientNet-B0 splitter using torchvision EfficientNet's features blocks.
class FlopAwareEfficientNetB0PipelineSplitter(FlopAwarePipelineSplitterBase):
    def architecture_name(self) -> str:
        return "efficientnet_b0"

    def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[OrderedLayer]) -> float:
        name = group[-1][0]
        return 0.0 if name in {"stem", "final_conv", "head"} or name.startswith("features_") else 1.0

    def _get_ordered_layers(self):
        layers = [
            ("stem", self.model.features[0], sum(v for k, v in self.layer_flops.items() if k.startswith("features.0.")))
        ]

        for i in range(1, len(self.model.features) - 1):
            layers.append((
                f"features_{i}",
                self.model.features[i],
                sum(v for k, v in self.layer_flops.items() if k.startswith(f"features.{i}."))
            ))

        last_idx = len(self.model.features) - 1
        layers.append((
            "final_conv",
            self.model.features[last_idx],
            sum(v for k, v in self.layer_flops.items() if k.startswith(f"features.{last_idx}."))
        ))

        layers.append((
            "head",
            EfficientNetB0Head(self.model),
            self.layer_flops.get("avgpool", 0.0) + sum(v for k, v in self.layer_flops.items() if k.startswith("classifier."))
        ))

        return self._add_boundary_bytes_to_layers(layers)

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

    def _network_penalty(self, candidate_idx: int, start_idx: int, group: List[OrderedLayer]) -> float:
        name = group[-1][0]
        return 0.0 if name in {"stem", "head"} or name.startswith("network_") else 1.0

    def _get_ordered_layers(self):
        layers = [("stem", TCNStem(), 0.0)]
        for i, blk in enumerate(self.model.network):
            flops = sum(v for k, v in self.layer_flops.items() if k.startswith(f"network.{i}."))
            layers.append((f"network_{i}", TCNBlockWrapper(blk), flops))
        layers.append(("head", TCNHead(self.model.output_proj), self.layer_flops.get("output_proj", 0.0)))
        return self._add_boundary_bytes_to_layers(layers)

if __name__=="__main__":
    from torchvision.models import vision_transformer

    # model = vision_transformer.vit_b_16(weights=None).eval()
    # # splitter = FlopAwareViTPipelineSplitter(model, input_shape=(1, 3, 224, 224))
    # splitter = FlopAwareViTPipelineSplitter(model, input_shape=(2, 3, 224, 224))

    # # Tradeoff intuition
    # #     If w_flop is high, the splitter prioritizes balanced compute across stages. -> keeps percentage close to provided percentage
    # #     If w_net is high, it prefers clean architectural boundaries even if FLOPs are a little off. -> always ignored actually lmao
    # #     If w_comm is high, it tries to minimize transfer size between splits, which matters when the boundary activation is large. -> keeps communication borders big
    # result = splitter.split_by_flops_pipeline(
    #     flop_percentages=[30, 33, 11, 18, 8],
    #     lookahead=5,
    #     out_dir="./vit_splits",
    #     meta_name="meta.json",
    #     w_flop=1.0,
    #     w_net=0.5,
    #     w_comm=0.1,
    #     export=True
    # )

#reconstruction
# meta = json.load(open("./vit_splits/vit_meta.json"))
# modules = []
# for part in meta["splits"]:
#     ep = torch.export.load(part["exported_path"])
#     modules.append(ep.module())

    # import tcn as tcn
    # model = tcn.SensorTCN(
    # num_channels=8,
    # hidden_channels=256,
    # levels=8,
    # kernel_size=5,
    # output_channels=8,
    # ).eval()

    # splitter = FlopAwareTCNPipelineSplitter(model, input_shape=(1, 2048, 8))

    # result = splitter.split_by_flops_pipeline(
    #     flop_percentages=[30, 33, 11, 18, 8],
    #     lookahead=3,
    #     out_dir="./tcn_splits",
    #     meta_name="meta.json",
    #     w_flop=1.0,
    #     w_net=0.5,
    #     w_comm=0.1,
    #     profiler=True,
    #     export=True
    # )

    # from torchvision.models import resnet18
    
    # model = resnet18(weights=None).eval()
    # splitter = FlopAwareResNet18PipelineSplitter(model, input_shape=(2, 3, 224, 224))
    
    # result = splitter.split_by_flops_pipeline(
    #     flop_percentages=[20, 30, 30, 20],
    #     lookahead=5,
    #     out_dir="./resnet_splits",
    #     meta_name="resnet_meta.json",
    #     w_flop=1.0,
    #     w_net=0.5,
    #     w_comm=0.1,
    #     export=True
    # )

    # EDIT: Added EfficientNet-B0 run block.
    # from torchvision.models import efficientnet_b0

    # model = efficientnet_b0(weights=None).eval()
    # splitter = FlopAwareEfficientNetB0PipelineSplitter(model, input_shape=(2, 3, 224, 224))

    # result = splitter.split_by_flops_pipeline(
    #     flop_percentages=[20, 30, 30, 20],
    #     lookahead=5,
    #     out_dir="./efficientnet_b0_splits",
    #     meta_name="efficientnet_b0_meta.json",
    #     w_flop=0,
    #     w_net=0,
    #     w_comm=1,
    #     export=True
    # )