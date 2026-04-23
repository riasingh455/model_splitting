# to run: python tcn.py --model-type sensor_tcn --model-split-type modules --world 3 --batch-size 4 

from __future__ import annotations

import tyro
from dataclasses import dataclass
import torch
import torch.nn as nn
from pathlib import Path

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

from torch.distributed.pipelining import pipeline, SplitPoint
from torch.utils.flop_counter import FlopCounterMode

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


@dataclass
class Args:
    world: int = 1
    batch_size: int = 4
    model_type: str = "resnet18"
    model_split_type: str = "children"
    image: str = "./bear.jpeg"
    custom: list[int] = None


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class SensorTCN(nn.Module):
    """
    Temporal Convolutional Network for multichannel IoT/sensor time-series data.

    Input shape:
        [batch, time_steps, sensor_channels]

    Output shape:
        [batch, time_steps, sensor_channels]
    """

    def __init__(
        self,
        num_channels=8,
        hidden_channels=256,
        levels=8,
        kernel_size=5,
        output_channels=8,
    ):
        super().__init__()

        layers = []
        in_channels = num_channels

        for i in range(levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=0.0,
                )
            )
            in_channels = hidden_channels

        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(
            hidden_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, x):
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        x = self.network(x)
        x = self.output_proj(x)

        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        return x


def executorch_stuff(fname, exp, cal_sample, ex_inp, star=False):
    qparams = get_symmetric_quantization_config(is_per_channel=True)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(qparams)

    prepared_model = prepare_pt2e(exp, quantizer)

    if star:
        prepared_model(*cal_sample)
    else:
        prepared_model(cal_sample)

    quantized_model = convert_pt2e(prepared_model)
    exported_program = torch.export.export(quantized_model, ex_inp, strict=True)
    re_exp = torch.export.export(exported_program.module(), ex_inp)
    torch.export.save(re_exp, f"{fname}.exp")
    return re_exp.module()


@torch.no_grad()
def stat_generator(pipe, world, model_type, model_split_type, example_input):
    stage_shapes = {}
    flop_stages = {}

    bs = list(example_input.shape)[0]
    app_dir = f"./{model_type}_{model_split_type}_{world}_{bs}"
    output = torch.empty(size=tuple(example_input.shape), dtype=example_input.dtype)
    Path.mkdir(Path(app_dir), exist_ok=True)

    for rank in range(world):
        temp = pipe.get_stage_module(rank)

        if rank == 0:
            exp = torch.export.export(temp, (example_input,))
            temp = executorch_stuff(
                f"{app_dir}/exe_split_{rank}.pte",
                exp.module(),
                example_input,
                (example_input,),
            )
            n_output = temp.forward(example_input)

        else:
            if type(output) != type(example_input):
                exp = torch.export.export(temp, output)
                temp = executorch_stuff(
                    f"{app_dir}/exe_split_{rank}.pte",
                    exp.module(),
                    output,
                    output,
                    star=True,
                )
                n_output = temp.forward(*output)

            else:
                exp = torch.export.export(temp, (output,))
                temp = executorch_stuff(
                    f"{app_dir}/exe_split_{rank}.pte",
                    exp.module(),
                    output,
                    (output,),
                )
                n_output = temp.forward(output)

        if rank not in stage_shapes:
            if type(output) != type(example_input):
                stage_info = [len(output), [list(o.shape) for o in output]]
                new_output = torch.cat([o.flatten() for o in output]).flatten()
                stage_info = [tuple(new_output.shape), f"{new_output.dtype}"] + stage_info
                stage_shapes[rank] = [i for i in stage_info]
            else:
                stage_shapes[rank] = [tuple(output.shape), f"{output.dtype}"]

        if rank not in flop_stages:
            flop_counter = FlopCounterMode(display=False, depth=None)
            with flop_counter:
                if type(output) != type(example_input):
                    temp.forward(*output)
                else:
                    temp.forward(output)
                flop_stages[rank] = flop_counter.get_total_flops()

        output = n_output

    if world not in stage_shapes:
        if type(output) != type(example_input):
            stage_info = [len(output), [list(o.shape) for o in output]]
            new_output = torch.cat([o.flatten() for o in output]).flatten()
            stage_info = [tuple(new_output.shape), f"{new_output.dtype}"] + stage_info
            stage_shapes[world] = [i for i in stage_info]
        else:
            stage_shapes[world] = [tuple(output.shape), f"{output.dtype}"]

    flop_file = f"{app_dir}/flop.dict"
    with open(flop_file, "w") as f:
        f.write(f"{flop_stages}\n")

    stage_file = f"{app_dir}/stages.dict"
    with open(stage_file, "w") as f:
        f.write(f"{stage_shapes}\n")


def model_splitter(
    model,
    model_type,
    split_type,
    world,
    batch_size,
    example_input,
    specific_chunks=[],
):
    split_model = {}

    if split_type == "modules":
        split_model = {
            k: v
            for k, v in model.named_modules()
            if k != "" and k not in model.named_children()
        }

    elif split_type == "children":
        split_model = {
            k: v
            for k, v in model.named_children()
            if k != "_guards_fn"
        }

    if world > len(split_model):
        world = len(split_model)

    sizes = []

    if len(specific_chunks) == 0:
        sizes = [
            len(split_model) // world + (1 if i < len(split_model) % world else 0)
            for i in range(world)
        ]

    else:
        if world != len(specific_chunks):
            world = len(specific_chunks)

        raw_sizes = {
            k: int(round(specific_chunks[k] / 100 * len(split_model)))
            for k in range(len(specific_chunks))
        }

        diff = sum(raw_sizes.values()) - len(split_model)

        if diff > 0:
            descending_sort = {
                k: v
                for k, v in sorted(
                    raw_sizes.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

            descending_keys = list(descending_sort.keys())
            d_counter = 0

            while diff > 0:
                new_diff = descending_sort[descending_keys[d_counter]] - diff

                if new_diff <= 0:
                    diff = -1 * new_diff + 1
                    descending_sort[descending_keys[d_counter]] = 1
                else:
                    diff = 0
                    descending_sort[descending_keys[d_counter]] = new_diff

                d_counter += 1

            sizes = [descending_sort[r] for r in range(len(descending_sort))]

        elif diff < 0:
            diff = -1 * diff

            ascending_sort = {
                k: v
                for k, v in sorted(
                    raw_sizes.items(),
                    key=lambda x: x[1],
                )
            }

            ascending_keys = list(ascending_sort.keys())
            ascending_sort[ascending_keys[0]] += diff
            sizes = [ascending_sort[r] for r in range(len(ascending_sort))]

    split_spec = {}
    split_names = list(split_model.keys())

    r = 0
    counter = 0

    print(split_model.keys())

    while counter < len(sizes) and r + sizes[counter] < len(split_model):
        temp_key = split_names[r + sizes[counter] - 1]
        split_spec[temp_key] = SplitPoint.END
        r += sizes[counter]
        counter += 1

    if split_names[-1] not in split_spec:
        split_spec[split_names[-1]] = SplitPoint.END

    print(split_spec)

    pipe = pipeline(
        model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )

    print(split_spec, pipe.num_stages)
    print(split_model.keys())

    stat_generator(
        pipe,
        world,
        model_type,
        split_type,
        example_input,
    )


if __name__ == "__main__":
    torch.backends.mkldnn.enabled = False

    args = tyro.cli(Args)

    weights = ResNet18_Weights.DEFAULT
    pretrained_model = resnet18(weights=weights).eval()
    example_input = torch.randn(1 * args.batch_size, 3, 224, 224)

    if args.model_type == "mbv3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        pretrained_model = mobilenet_v3_small(weights=weights).eval()
        example_input = torch.randn(1 * args.batch_size, 3, 224, 224)

    elif args.model_type == "eb0":
        weights = EfficientNet_B0_Weights.DEFAULT
        pretrained_model = efficientnet_b0(weights=weights).eval()
        example_input = torch.randn(1 * args.batch_size, 3, 224, 224)

    elif args.model_type == "vit":
        weights = ViT_B_16_Weights
        pretrained_model = vit_b_16(weights=weights).eval()
        example_input = torch.randn(1 * args.batch_size, 3, 224, 224)

    elif args.model_type == "sensor_tcn":
        pretrained_model = SensorTCN(
            num_channels=8,
            hidden_channels=256,
            levels=8,
            kernel_size=5,
            output_channels=8,
        ).eval()

        example_input = torch.randn(
            args.batch_size,
            2048,
            8,
        )

    model_splitter(
        pretrained_model,
        args.model_type,
        args.model_split_type,
        args.world,
        args.batch_size,
        example_input,
        args.custom if args.custom is not None else [],
    )
