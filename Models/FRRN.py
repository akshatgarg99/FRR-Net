import torch
import torch.nn as nn
from collections import OrderedDict
from .FRRN_utils import *


class FRRNet(nn.Module):
    """
    implementation table A of Full-Resolution Residual Networks
    """

    def __init__(self, in_channels=3, out_channels=21, layer_blocks=(3, 4, 2, 2)):
        super(FRRNet, self).__init__()

        # 5×5
        self.first = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=5, padding=2)),
                ('bn', nn.BatchNorm2d(48)),
                ('relu', nn.ReLU()),
            ]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()

        # 3×48 Residual Unit
        self.reslayers_in = nn.Sequential(*[BasicBlock(48, 48, efficient=False) for _ in range(3)])

        # divide
        self.divide = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1)

        # frrlayer 1
        self.frrnlayer1 = FRRLayer(48, 96, factor=2, num_blocks=layer_blocks[0])

        # frrlayer2
        self.frrnlayer2 = FRRLayer(96, 192, factor=4, num_blocks=layer_blocks[1])

        # frrnlayer3
        self.frrnlayer3 = FRRLayer(192, 384, factor=8, num_blocks=layer_blocks[2])

        # frrnlayer4
        self.frrnlayer4 = FRRLayer(384, 384, factor=16, num_blocks=layer_blocks[3])

        # defrrnlayer1
        self.defrrnlayer1 = FRRLayer(384, 192, factor=8, num_blocks=2)

        # defrrnlayer2
        self.defrrnlayer2 = FRRLayer(192, 192, factor=4, num_blocks=2)

        # defrrnlayer3
        self.defrrnlayer3 = FRRLayer(192, 96, factor=2, num_blocks=2)

        # join
        self.compress = nn.Conv2d(96 + 32, 48, kernel_size=1)

        # 3×48 reslayer

        self.reslayers_out = nn.Sequential(*[BasicBlock(48, 48, efficient=True) for _ in range(3)])

        self.out_conv = nn.Conv2d(48, out_channels, 1)

    def forward(self, x):
        x = self.first(x)
        y = self.reslayers_in(x)

        z = self.divide(y)
        y = self.pool(y)

        y, z = self.frrnlayer1(y, z)

        y = self.pool(y)
        y, z = self.frrnlayer2(y, z)

        y = self.pool(y)
        y, z = self.frrnlayer3(y, z)

        y = self.pool(y)
        y, z = self.frrnlayer4(y, z)

        y = self.up(y)
        y, z = self.defrrnlayer1(y, z)

        y = self.up(y)
        y, z = self.defrrnlayer2(y, z)

        y = self.up(y)
        y, z = self.defrrnlayer3(y, z)

        y = self.up(y)
        refine = self.compress(torch.cat((y, z), 1))

        out = self.reslayers_out(refine)
        out = self.out_conv(out)
        return out
