import tyro
from dataclasses import dataclass
from torch.export import load
from torch.distributed import init_process_group

@dataclass
class Args:
    world: int = 1
    rank: int = 0
    batch_size: int = 4
    batch_num: int = 10
    model_type: str = "resnet18"
    model_split_type:str = "children"
    image: str = "./bear.jpeg"
    ip: str = "127.0.0.1"
    port: int = 9123
    cores: int = 4
    iters: int = 15
    warmup: int = 1
    backend: str = "gloo"

def custom_pipeline(aot_dir, batch_num, world, rank, input=[]):
    split_pt = f"/{aot_dir}/split_{rank}.pt2"
    stage_dict = f"/{aot_dir}/stages.dict"
    
    #can actually just use the communication from the stage dict?







if __name__ == "__main__":
    args = tyro.cli(Args)
    print(f"Model {args.model_type} and split {args.model_split_type}")
    
    init_method = f"tcp://{args.ip}:{args.port}"
    init_process_group(backend=args.backend, init_method=init_method, 
    world_size=args.world, rank=args.rank)




