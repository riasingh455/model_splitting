from __future__ import annotations 

import tyro
from dataclasses import dataclass
import torch
import torch.nn as nn
# from typing import List
# from torchvision.models.quantization import ResNet18_QuantizedWeights
# from torchvision.models.quantization import resnet18
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights# mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights #vit_l_32, ViT_L_32_Weights
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.utils.flop_counter import FlopCounterMode
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower, EdgeCompileConfig
import torchao
# import onnxruntime
from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
# from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
# from torchao.quantization import PerGroup, Int8DynamicActivationIntxWeightConfig, Int8DynamicActivationInt8WeightConfig, quantize_ , Int4WeightOnlyConfig
# from torchao.quantization.qat import QATConfig

# from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
# from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import ArmInductorQuantizer, get_default_arm_inductor_quantization_config
# "XNNPACKQuantizer",
#     "get_symmetric_quantization_config",
# import torchao.quantization.pt2e.quantizer.arm_inductor_quantizer as aiq
# from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import ArmInductorQuantizer 
from pathlib import Path

@dataclass
class Args:
    world: int = 1
    batch_size: int = 4
    model_type: str = "resnet18"
    model_split_type:str = "children"
    image: str = "./bear.jpeg"
    custom: list[int] = None

def onnx_stuff(exp, ex_inp):
    import onnx
    from onnxsim import simplify
    # from onnx import helper, TensorProto
    model = onnx.load('test.onnx')
    # Replace all fixed batch dimensions with symbolic dimension
    # def make_dynamic(model):
    #     for input_tensor in model.graph.input:        
    #         # Set first dimension to be symbolic (variable)        
    #         dim = input_tensor.type.tensor_type.shape.dim[0]
    #         dim.ClearField("dim_value")
    #         dim.dim_param = "batch_size"
    #     for output_tensor in model.graph.output:
    #         if len(output_tensor.type.tensor_type.shape.dim) > 0:
    #             dim = output_tensor.type.tensor_type.shape.dim[0]
    #             dim.ClearField("dim_value")
    #             dim.dim_param = "batch_size" 
    #     return model
    # model = make_dynamic(model)
    # onnx.save(model, 'test_dynamic.onnx')
    model_simp, check = simplify(model)
    if check:
        onnx.save(model_simp, 'test_simp.onnx')
    # quantize_dynamic("test_dynamic.onnx", "test_quant.onnx", weight_type=QuantType.QInt8)
    quantize_dynamic("test_simp.onnx", "test_quant.onnx", weight_type=QuantType.QUInt8)

def executorch_stuff(fname, exp, cal_sample, ex_inp, star=False):

    #actually quantization lmao
    
    # sample_inputs = (torch.randn(1, 3, 224, 224), )
    qparams = get_symmetric_quantization_config(is_per_channel=True)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)
    # print("here?")
    # # print(exp.print_readable())
    # print(exp.named_modules())
    # print([mod._forward_hooks for mod in exp.pa() if hasattr(mod, "_forward_hooks" )])
    # print("here?")
    # exit()
    # training_ep = torch.export.export(exp, sample_inputs).module() # (2)
    # training_ep = exp # (2)
    prepared_model = prepare_pt2e(exp, quantizer) # (3)

    # for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
    if star==True:
        prepared_model(*cal_sample)
    else:
        prepared_model(cal_sample) # (4) Calibrate
    
    quantized_model = convert_pt2e(prepared_model) # (5)
    exported_program = torch.export.export(quantized_model, ex_inp, strict=True)
    re_exp = torch.export.export(exported_program.module(), ex_inp)
    torch.export.save(re_exp, f"{fname}.exp")#"resnet_exp_strict_rexp.model")
    onnx_stuff(exp, ex_inp)
    torch.onnx.export(exp, ex_inp, "test.onnx", input_names=['input'], output_names=['output'])
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    # quantize_dynamic("test.onnx", "test_quant.onnx", weight_type=QuantType.QInt8)
    # quantize_static("test.onnx", "test_quant.onnx", weight_type=QuantType.QInt8,per_channel=True)
    return re_exp.module()
    # torch.save(exported_program.module().state_dict(), "resnet_exp_strict.model")

    # edge = to_edge(exp, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    # et_program = edge.to_executorch()
    # # lower = edge.to_backend(XnnpackPartitioner())
    # # et_program = to_edge_transform_and_lower(
    # #                 exp,
    # #                 # compile_config = EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.aten._native_batch_norm_legit_functional.default]),
    # #                 compile_config = EdgeCompileConfig(_check_ir_validity=False),
    # #                 partitioner=[XnnpackPartitioner()]
    # #             ).to_executorch()
    # with open(fname, "wb") as f:
    #     f.write(et_program.buffer)

@torch.no_grad()
def stat_generator(pipe, world, model_type, model_split_type, example_input):
    stage_shapes = {}
    flop_stages = {}
    bs = list(example_input.shape)[0]
    app_dir = f"./{model_type}_{model_split_type}_{world}_{bs}" if args.custom==None else f"./{model_type}_{model_split_type}_{world}_{bs}_custom"
    output = torch.empty(size=tuple(example_input.shape), dtype=example_input.dtype)
    Path.mkdir(Path(app_dir), exist_ok=True)
    # quantizer = ArmInductorQuantizer()
    # # Specify you want INT8 for both activations and weights
    # quantizer.set_global(get_default_arm_inductor_quantization_config())
    # base_config = Int8DynamicActivationIntxWeightConfig(
    #     weight_dtype=torch.int4,
    #     weight_granularity=PerGroup(32),
    # )
    # qparams = get_symmetric_quantization_config(is_per_channel=True) # (1)
    # quantizer = XNNPACKQuantizer()
    # quantizer.set_global(qparams)
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights).eval()
    # stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    # tail = nn.Sequential(model.avgpool, nn.Flatten(1), model.fc)



    # forced_staged = {}
    # if world == 1:
    #     forced_staged = {0:model}
    # if world == 2:
    #     # Stage0: stem + layer1 + layer2
    #     # Stage1: layer3 + layer4 + tail
    #     s0 = nn.Sequential(stem, model.layer1, model.layer2)
    #     s1 = nn.Sequential(model.layer3, model.layer4, tail)
    #     forced_staged = {0:s0, 1:s1}

    # if world == 3:
    #     # Stage0: stem + layer1
    #     # Stage1: layer2 + layer3
    #     # Stage2: layer4 + tail
    #     s0 = nn.Sequential(stem, model.layer1)
    #     s1 = nn.Sequential(model.layer2, model.layer3)
    #     s2 = nn.Sequential(model.layer4, tail)
    #     forced_staged = {0:s0, 1:s1, 2:s2}
    
    #manually checking here:
    # output = forced_staged[0](example_input)
    # executorch_stuff("", forced_staged[0])
    # print("passed first executorch?") 
    # qparams = get_symmetric_quantization_config(is_per_channel=True)
    # quantizer = XNNPACKQuantizer()
    # quantizer.set_global(qparams)

    # training_ep = torch.export.export(forced_staged[0], (example_input,)).module() # (2)
    # output = forced_staged[0](example_input)
    # prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

    # exit()

    for rank in range(world):
        # temp = forced_staged[rank]#pipe.get_stage_module(rank)
        temp = pipe.get_stage_module(rank)
        # print(temp)
        # exit()
        # print(temp)
        # quantize_(temp, QATConfig(base_config, step="prepare"))
        # #fake train here lmao xd
        # quantize_(temp, QATConfig(base_config, step="convert"))
        # print(temp)
        # exit()
        # print([v.dtype for k, v in temp.named_parameters()])
        # quantize_(temp, Int8DynamicActivationInt8WeightConfig())
        # temp.compile()
        # print([ type(v.weights) for k, v in temp.named_childrens()])
        # if rank==0:
        #     temp = torch.export.export(temp, (example_input,)).module()
        # else:
        #     if type(output) == type(tuple()):
        #         temp = torch.export.export(temp, output).module()
        #     else:
        #         temp = torch.export.export(temp, (output,)).module()

        
        # flop_counter = FlopCounterMode(display=False, depth=None)
        # with flop_counter:
        if rank==0:
            # print(example_input.shape)
            # n_output = temp(example_input)
#             from torchao.quantization.quant_api import int4_weight_only

# m = nn.Sequential(nn.Linear(32, 1024), nn.Linear(1024, 32))
# quantize_(m, Int4WeightOnlyConfig(group_size=32, version=1))
            # print(temp)
            # quantize_(temp, Int4WeightOnlyConfig(group_size=32, version=1) )
            # print(temp)
            # with torch.no_grad():
            exp =  torch.export.export(temp, (example_input,), strict=True)
            temp = executorch_stuff(f"{app_dir}/exe_split_{rank}.pte", exp.module(), example_input, (example_input,))
            n_output = temp.forward(example_input)
            # exp=exp.run_decompositions(decomp_table=torch.export.default_decompositions())
            # exp = exp.module()
            # pre_model = prepare_pt2e(exp, quantizer=)#, quantizer)
            # pre_model.forward(example_input)
            # con_model = convert_pt2e(pre_model)
            # exp = torch.export.export(con_model, (example_input,))
            

                # pt2_path = torch._inductor.aoti_compile_and_package(exp, package_path=f"{app_dir}/ind_split_{rank}.pt2")
                # print(pt2_path)
            # exp = torch.compile(exp.module(), backend="inductor")x
            # pre_model = prepare_pt2e(exp, quantizer)
            # pre_model.forward(example_input)
            # con_model = convert_pt2e(pre_model)
            # # con_model.compile()
            # exp = torch.export.export(con_model, (example_input,))
            # quantize_(, Int8DynamicActivationInt8WeightConfig())
            # torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
            # torch.save(exp, f"{app_dir}/split_{rank}.pt2")
            # torch.save(exp.module().state_dict(), f"{app_dir}/split_{rank}_state.pt2")
        else:
            if type(output)!=type(example_input):
                exp = torch.export.export(temp, output)
                #TODO make first output actually *output!
                temp = executorch_stuff(f"{app_dir}/exe_split_{rank}.pte", exp.module(), output, output, star=True)
                # torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
                n_output = temp.forward(*output)
                
            else:
                exp = torch.export.export(temp, (output,))
                temp = executorch_stuff(f"{app_dir}/exe_split_{rank}.pte", exp.module(), output, (output,))
                # torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
                n_output = temp.forward(output)
        
        if rank not in stage_shapes:
            if type(output)!=type(example_input):
                stage_info = [len(output), [list(o.shape) for o in output]]
                new_output = torch.cat([o.flatten() for o in output]).flatten()
                stage_info = [tuple(new_output.shape), f"{new_output.dtype}"] + stage_info
                stage_shapes[rank] = [i for i in stage_info]
            else:
                stage_shapes[rank] = [tuple(output.shape), f"{output.dtype}"]
        
        #flop counter here instead
        if rank not in flop_stages:
            flop_counter = FlopCounterMode(display=False, depth=None)
            with flop_counter:
                if type(output) != type(example_input):
                    temp.forward(*output)
                else:
                    temp.forward(output)
                flop_stages[rank] = flop_counter.get_total_flops()
        
        output = n_output

        # if rank not in flop_stages:
            # flop_stages[rank] = flop_counter.get_total_flops()
    #additional rank for final output
    if world not in stage_shapes:
        if type(output)!=type(example_input):
            stage_info = [len(output), [list(o.shape) for o in output]]
            new_output = torch.cat([o.flatten() for o in output]).flatten()
            stage_info = [tuple(new_output.shape), f"{new_output.dtype}"] + stage_info
            stage_shapes[world] = [i for i in stage_info]
        else:
            stage_shapes[world] = [tuple(output.shape), f"{output.dtype}"]
    flop_file=f"{app_dir}/flop.dict"
    f=open(flop_file, "w")
    f.write(f"{flop_stages}\n")
    stage_file=f"{app_dir}/stages.dict"
    f=open(stage_file, "w")
    f.write(f"{stage_shapes}\n")

def model_splitter(model, model_type, split_type, world, batch_size, specific_chunks=[]):
    split_model = {}
    if split_type=="modules":
        split_model = {k:v for k,v in model.named_modules() if k!='' and k not in model.named_children()}
    elif split_type=="children":
        split_model = {k:v for k,v in model.named_children() if k!='_guards_fn'}
    # split_model = {k:0 for k in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']}
    if world > len(split_model):
            world = len(split_model)
    
    sizes=[]
    if len(specific_chunks)==0:
        sizes = [len(split_model) // world + (1 if i < len(split_model) % world else 0) for i in range(world)] 
    else:
        if world!=len(specific_chunks):
            world=len(specific_chunks)
        raw_sizes = {k: int(round(specific_chunks[k]/100*len(split_model))) for k in range(len(specific_chunks))}
        diff = sum(raw_sizes.values()) - len(split_model)
        if diff>0:
            #rounding error causes overestimation
            #remove difference from heaviest layer
            descending_sort = {k:v for k,v in sorted(raw_sizes.items(), key=lambda x: x[1], reverse=True)}
            descending_keys = list(descending_sort.keys())
            d_counter=0
            while diff > 0:
                new_diff = descending_sort[descending_keys[d_counter]] - diff
                if new_diff <= 0:
                    diff = -1*new_diff + 1
                    descending_sort[descending_keys[d_counter]] = 1
                else:
                    diff = 0
                    descending_sort[descending_keys[d_counter]] = new_diff
                d_counter+=1
            sizes = [descending_sort[r] for r in range(len(descending_sort))]
        elif diff < 0:
            #rounding error causes underestimation
            #add diff to lightest layers
            diff=-1*diff
            ascending_sort = {k:v for k,v in sorted(raw_sizes.items(), key=lambda x: x[1])}
            ascending_keys = list(ascending_sort.keys())
            ascending_sort[ascending_keys[0]]+=diff
            sizes = [ascending_sort[r] for r in range(len(ascending_sort))]
        


        






    split_spec = {}
    split_names = list(split_model.keys())
    r=0
    counter=0
    print(split_model.keys())
    # exit()
    # print(len(split_model))
    # print(sizes)
    while counter < len(sizes) and r+sizes[counter] < len(split_model):
        temp_key = split_names[r+sizes[counter]-1]
        split_spec[temp_key] = SplitPoint.END
        r+=sizes[counter]
        counter+=1
    if split_names[-1] not in split_spec:
        split_spec[split_names[-1]] = SplitPoint.END

    example_input = torch.randn(1*batch_size, 3, 224, 224)
    print(split_spec)
    # exit()
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec )
    # print(pipe)

    # print(pipe.num_stages)
    print(split_spec, pipe.num_stages)
    print(split_model.keys())

    # exit()



    stat_generator(pipe, world, model_type, split_type, example_input)


if __name__ == "__main__":
    torch.backends.mkldnn.enabled = False
    #based on world size
    #split model 
    #get communication information
    #get flop count
    #export model as .pt2
    args = tyro.cli(Args)
    # backend="qnnpack"
    # qconfig = torch.quantization.get_default_qconfig(backend)  
    # torch.backends.quantized.engine = backend
    weights = ResNet18_Weights.DEFAULT
    # weights = ResNet18_QuantizedWeights.DEFAULT
    # print(weights.meta["backend"])
    pretrained_model = resnet18(weights=weights).eval()
    # pretrained_model = torch.export.load("resnet_exp_strict_rexp.model").module()
    # ex = torch.empty(1*args.batch_size, 3, 224, 224)
    # pretrained_model = torch.export.export(pretrained_model, (ex, ), strict=True).module()
    # c=0
    # for n, v in enumerate(pretrained_model.layers):
    #     if n=='':
    #         pretrained_model.add_module(f"dummy_name_{c}", v)
    # pretrained_model = torch.export.export(pretrained_model, (ex, ), strict=True).module()

    # exit()
    # pretrained_model = torch.export.load("resnet_exp_strict_rexp.model").module()

    # quantize_(pretrained_model, Int8DynamicActivationInt8WeightConfig())
    # example_input = torch.empty(1*args.batch_size, 3, 224, 224)

    # pretrained_model = torch.export.export(pretrained_model, (example_input, )).module()

    # quantizer = aiq.ArmInductorQuantizer()
    # quantizer.set_global(aiq.get_default_arm_inductor_quantization_config())
    # example_input = torch.randn(1*args.batch_size, 3, 224, 224)
    # exp_pretrained_model = torch.export.export(pretrained_model, (example_input,) ).module()
    # pre_model = prepare_pt2e(exp_pretrained_model, quantizer)
    # pre_model(example_input)
    # pretrained_model = convert_pt2e(pre_model)
    # quantize_(pretrained_model, Int8DynamicActivationInt8WeightConfig())
    
    if args.model_type=="mbv3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        pretrained_model = mobilenet_v3_small(weights=weights).eval()
        # import torchvision.models as models
        # from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
        # pretrained_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
        # # sample_inputs = (torch.randn(1, 3, 224, 224), )
        # # et_program = to_edge_transform_and_lower(
        # #         torch.export.export(pretrained_model, sample_inputs),
        # #         partitioner=[XnnpackPartitioner()]
        # #     ).to_executorch()
        # # exit()
    elif args.model_type=="eb0":
        weights = EfficientNet_B0_Weights.DEFAULT
        pretrained_model = efficientnet_b0(weights=weights).eval()
    
    elif args.model_type=="vit":
        # weights=ViT_L_32_Weights
        # pretrained_model = vit_l_32(weights=weights).eval()
        weights=ViT_B_16_Weights
        pretrained_model = vit_b_16(weights=weights).eval()
    
    model_splitter(pretrained_model, args.model_type, 
    args.model_split_type, args.world, args.batch_size, args.custom if args.custom!=None else [])

    
