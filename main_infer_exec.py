import tyro
from dataclasses import dataclass
import pipeline_library 
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch.fx.passes.split_module import split_module

@dataclass
class Args:
    # Path to instruction profile results
    stages: int
    assigned_rank:int
    device: str = "cpu"
    image: str = "./bear.jpeg"
    images: tuple[str,...] = () #if multiple images provided, each image run an iters+warmups number of times 
    iters:int = 10 #number of times to inference a single image and measures time
    warmup: int = 0 #number of times to not measure time to warmup the inference
    # local_pipeline: bool = False

    @property
    def backend(self):
        return "gloo" if self.device=="cpu" else "nccl"

if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device(args.device)
    #import model and do splits before hand
    pipeline_library.set_one_core_behavior()
    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights).eval()
    pipe_mod = pipeline_library.FBModel(pretrained_model, {}, device)
    pipe_mod.split(args.assigned_rank, args.stages)
    pre_proc = weights.transforms()
    data_labels = weights.meta['categories'] #subject to change depending on fine-tuning and training
    #early gc potential
    weights=None
    pretrained_model=None 
    x0=None
    l = 1 if len(args.images)==0 else len(args.images)
    for _ in range(l):
        if args.assigned_rank==0:
            x0 = pipe_mod.load_image_tensor(args.image, pre_proc) 
        output = pipe_mod.pipeline_inference(args.backend, args.stages, args.assigned_rank, args.warmup, args.iters, x0)
        if output!=None:
            res = pipe_mod.top1_label(data_labels, output)
            print(res)





