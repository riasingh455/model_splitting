import torch
import torch.nn as nn

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
