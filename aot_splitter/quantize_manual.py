#quantize manual
import torch
from torchao.quantization import Int8WeightOnlyConfig, PerGroup, Int8DynamicActivationIntxWeightConfig, Int8DynamicActivationInt8WeightConfig, quantize_
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
torch.save(model.state_dict(), "resnet_full.model")

# quantize_(model, Int8DynamicActivationInt8WeightConfig())
quantize_(model, Int8WeightOnlyConfig())
torch.save(model.state_dict(), "resnet_quant.model")
base_config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype=torch.int4,
        weight_granularity=PerGroup(128),
    )
model = resnet18(weights=weights).eval()
quantize_(model, base_config)
torch.save(model.state_dict(), "resnet_quant_4.model")
print([k for k,_ in model.named_children()])

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).eval()
torch.save(model.state_dict(), "resnet_50.model")
quantize_(model, Int8WeightOnlyConfig())
torch.save(model.state_dict(), "resnet_50_quant.model")

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import  EdgeProgramManager, to_edge_transform_and_lower, to_edge, EdgeCompileConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )
qparams = get_symmetric_quantization_config(is_per_channel=True)
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

training_ep = torch.export.export(model, sample_inputs).module() # (2)
prepared_model = prepare_pt2e(training_ep, quantizer) # (3)

for cal_sample in [torch.randn(1, 3, 224, 224)]: # Replace with representative model inputs
	prepared_model(cal_sample) # (4) Calibrate

quantized_model = convert_pt2e(prepared_model) # (5)

# print({k:v for k,v in quantized_model.named_children()})
# exit()
# quantized_model = torch.compile(quantized_model)
exported_program = torch.export.export(quantized_model, sample_inputs, strict=True)
# exp = exported_program.run_decompositions(None)
re_exp = torch.export.export(exported_program.module(), sample_inputs)
torch.save(exported_program.module().state_dict(), "resnet_exp_strict.model")
torch.export.save(re_exp, "resnet_exp_strict_rexp.model")

#test load module
from PIL import Image
new_load_model = torch.export.load("resnet_exp_strict_rexp.model").module()
print([k for k,_ in new_load_model.named_modules()])

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
# old_res = model(*sample_inputs)
# print(old_res-res)

def data_loader(model_type, batch_num, batch_size, image_path):
    weights, transforms= None, None

    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        transforms=weights.transforms()
    
    img_batch=[transforms(Image.open(image_path).convert("RGB"))]*batch_size
    batches = [torch.stack(img_batch)]*batch_num
    return batches

import numpy as np
def top1_label(model_type, output):
    categories=None
    if model_type=="resnet18":
        from torchvision.models import ResNet18_Weights
        weights=ResNet18_Weights.DEFAULT
        categories = weights.meta["categories"]

    #weights = ResNet18_Weights.DEFAULT
    #categories = weights.meta["categories"]
    idx=0
    try:
        idx = int(output.argmax(dim=1).item())
        # print(np.argsort(output)[:-5])
        print([categories[idx] for idx in [ int(output.argsort(dim=1)[0][i]) for i in [-1, -2, -3, -4, -5] ]])
        # print(output.argsort()[0][-1], output.argsort()[0][-2], output.argsort()[0][-3], output.argsort()[0][-4], output.argsort()[0][-5])
    except Exception as e:
        idx = output.argmax(dim=1).tolist()
    return f"{idx}: {categories[idx]}" if type(idx)!=list else ", ".join([str(categories[i]) for i in idx])

inputs = data_loader("resnet18", 1, 1, "../penguin.jpeg")
old_res = model(*inputs)
print(top1_label("resnet18", old_res))
new_res = new_load_model(*inputs)
print(top1_label("resnet18", new_res))



# print(new_load_model.print_readable())


exit()
# torch.save(exp.module().state_dict(), "resnet_exec_quant_state.model")

from executorch.exir.passes.external_constants_pass import delegate_external_constants_pass_unlifted

# Tag the unlifted ep.module().
tagged_module = exported_program.module()
delegate_external_constants_pass_unlifted(
    module=tagged_module,
    gen_tag_fn=lambda x: "resnet_alt.model", # This is the filename the weights will be saved to. In this case, weights will be saved as "model.ptd"
)
# Re-export to get the EP.
exported_program = torch.export.export(tagged_module, sample_inputs)
et_program = to_edge_transform_and_lower(
    exported_program
    ,partitioner = [XnnpackPartitioner()]
).to_executorch()

# torch.save(et_program.exported_program().state_dict, "test_save.model")

# et = to_edge(
#     torch.export.export(quantized_model, sample_inputs), 
#     compile_config=EdgeCompileConfig(_check_ir_validity=False))

# torch.save(et.module().state_dict(), "resnet_to_e.model")
# print(et)

# # to_edge_transform_and_lower( # (6)
# #     torch.export.export(quantized_model, sample_inputs),
# #     partitioner=[XnnpackPartitioner()],
# # ).to_executorch()
et_program.write_tensor_data_to_file("./")
with open("resnet_exec_quant.model", "wb") as f:
    f.write(et_program.buffer)

#read executorch and then split?????
import torch
from executorch.extension.pybindings import portable_lib 

input_tensor = torch.randn(1, 3, 32, 32)
module = portable_lib._load_for_executorch("resnet_exec_quant.model", "resnet_alt.model.ptd")
print(module)
# outputs = module.forward([input_tensor])