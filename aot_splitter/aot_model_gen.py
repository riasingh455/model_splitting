import tyro
from dataclasses import dataclass
import torch
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights# mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights # mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.utils.flop_counter import FlopCounterMode

from pathlib import Path

@dataclass
class Args:
    world: int = 1
    batch_size: int = 4
    model_type: str = "resnet18"
    model_split_type:str = "children"
    image: str = "./bear.jpeg"


def stat_generator(pipe, world, model_type, model_split_type, example_input):
    stage_shapes = {}
    flop_stages = {}
    bs = list(example_input.shape)[0]
    app_dir = f"./{model_type}_{model_split_type}_{world}_{bs}"
    output = torch.empty(size=tuple(example_input.shape), dtype=example_input.dtype)
    Path.mkdir(Path(app_dir), exist_ok=True)
    for rank in range(world):
        temp = pipe.get_stage_module(rank)
        if rank not in stage_shapes:
            if type(output)!=type(example_input):
                stage_info = [len(output), [list(o.shape) for o in output]]
                new_output = torch.cat([o.flatten() for o in output]).flatten()
                stage_info = [tuple(new_output.shape), new_output.dtype] + stage_info
                stage_shapes[rank] = [i for i in stage_info]
            else:
                stage_shapes[rank] = [tuple(output.shape), output.dtype]
        flop_counter = FlopCounterMode(display=False, depth=None)
        with flop_counter:
            if rank==0:
                output = temp.forward(example_input)
                exp = torch.export.export(temp, (example_input,))
                torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
            else:
                if type(output)!=type(example_input):
                    exp = torch.export.export(temp, output)
                    torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
                    output = temp.forward(*output)
                    
                else:
                    exp = torch.export.export(temp, (output,))
                    torch.export.save(exp, f"{app_dir}/split_{rank}.pt2")
                    output = temp.forward(output)
            if rank not in flop_stages:
                flop_stages[rank] = flop_counter.get_total_flops()
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
        split_model = {k:v for k,v in model.named_children()}
    
    if world > len(split_model):
            world = len(split_model)
    
    sizes = [len(split_model) // world + (1 if i < len(split_model) % world else 0) for i in range(world)] if len(specific_chunks)==0 else specific_chunks
    split_spec = {}
    split_names = list(split_model.keys())
    r=0
    counter=0
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
    pipe = pipeline(model, mb_args=(example_input,), split_spec=split_spec )
    # print(pipe.num_stages)
    print(split_spec, pipe.num_stages)


    stat_generator(pipe, world, model_type, split_type, example_input)


if __name__ == "__main__":
    #based on world size
    #split model 
    #get communication information
    #get flop count
    #export model as .pt2
    args = tyro.cli(Args)
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights).eval()
    
    if args.model_type=="mbv3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        pretrained_model = mobilenet_v3_small(weights=weights).eval()
    elif args.model_type=="eb0":
        weights = EfficientNet_B0_Weights.DEFAULT
        pretrained_model = efficientnet_b0(weights=weights).eval()
    
    model_splitter(pretrained_model, args.model_type, 
    args.model_split_type, args.world, args.batch_size, [])

    
