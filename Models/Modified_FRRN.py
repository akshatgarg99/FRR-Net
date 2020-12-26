from collections import OrderedDict
from Models.Utils.Modified_FRRN_utils import *


class FRRNet(nn.Module):
    """
    implementation table A of Full-Resolution Residual Networks
    """

    def __init__(self, in_channels=3, out_channels=21, layer_blocks=(4, 4, 3, 2)):
        super(FRRNet, self).__init__()

        # 5×5
        self.first = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=5, padding=2)),
                ('bn', nn.BatchNorm2d(48)),
                ('relu', nn.ReLU()),
            ]))

        self.expand = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1)),
                ('bn', nn.BatchNorm2d(64)),
            ]))

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.relu = nn.ReLU()

        # 3×48 Residual Unit
        self.reslayers_in = nn.Sequential(*[BasicBlock(48, 48, efficient=False) for _ in range(3)])

        # divide
        self.divide = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=1)

        # pool
        self.pool1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)

        # frrlayer 1
        self.frrnlayer1 = FRRLayer(48, 96, factor=2, num_blocks=layer_blocks[0])

        # pool
        self.pool2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)

        # frrlayer2
        self.frrnlayer2 = FRRLayer(96, 192, factor=4, num_blocks=layer_blocks[1])

        # pool
        self.pool3 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)

        # frrnlayer3
        self.frrnlayer3 = FRRLayer(192, 384, factor=8, num_blocks=layer_blocks[2])

        # pool
        self.pool4 = nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1)

        # frrnlayer4
        self.frrnlayer4 = FRRLayer(384, 512, factor=16, num_blocks=layer_blocks[3])

        # intervalayer
        self.interval = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(128 + 64, 128, kernel_size=1)),
                ('bn', nn.BatchNorm2d(128))
            ]))

        # upsample
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)

        # defrrnlayer1
        self.defrrnlayer1 = FRRLayer(512, 384, factor=8, num_blocks=3)

        # intervalayer
        self.interval2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(128 + 64, 128, kernel_size=1)),
                ('bn', nn.BatchNorm2d(128))
            ]))

        # upsample
        self.up2 = nn.ConvTranspose2d(384, 384, 2, stride=2)

        # defrrnlayer2
        self.defrrnlayer2 = FRRLayer(384, 192, factor=4, num_blocks=2)

        # intervalayer
        self.interval3 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(128 + 64, 128, kernel_size=1)),
                ('bn', nn.BatchNorm2d(128))
            ]))

        # upsample
        self.up3 = nn.ConvTranspose2d(192, 192, 2, stride=2)

        # defrrnlayer3
        self.defrrnlayer3 = FRRLayer(192, 96, factor=2, num_blocks=2)

        # upsample
        self.up4 = nn.ConvTranspose2d(96, 96, 2, stride=2)

        # join
        self.compress = nn.Conv2d(96 + 128, 48, kernel_size=1)

        # 3×48 reslayer

        self.reslayers_out = nn.Sequential(*[BasicBlock(48, 48, efficient=False) for _ in range(3)])

        self.out_conv1 = nn.Conv2d(48 + 3, 24, 1)

        self.out_conv2 = nn.Conv2d(24, out_channels, 1)

    def forward(self, x):
        exp = self.expand(x)
        f = self.first(x)
        y = self.reslayers_in(f)

        z = self.divide(y)
        y = self.pool1(y)

        y, z = self.frrnlayer1(y, z)

        y = self.pool2(y)
        y, z = self.frrnlayer2(y, z)

        y = self.pool3(y)
        y, z = self.frrnlayer3(y, z)

        y = self.pool4(y)
        y, z = self.frrnlayer4(y, z)

        z = torch.cat((z, exp), 1)
        z = self.interval(z)

        y = self.up1(y)
        y, z = self.defrrnlayer1(y, z)

        z = torch.cat((z, exp), 1)
        z = self.interval2(z)

        y = self.up2(y)
        y, z = self.defrrnlayer2(y, z)

        z = torch.cat((z, exp), 1)
        z = self.interval3(z)

        y = self.up3(y)
        y, z = self.defrrnlayer3(y, z)

        y = self.up4(y)
        refine = self.compress(torch.cat((y, z), 1))

        out = self.reslayers_out(refine)
        out = torch.cat((out, x), 1)

        out = self.out_conv1(out)
        out = self.out_conv2(out)
        return out
