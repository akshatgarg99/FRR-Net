import torch
import torch.nn as nn
from torch.utils import checkpoint as cp


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def residual_func(relu, norm1, conv1, norm2, conv2):
    def rbc_function(*inputs):
        output_ = relu(norm1(conv1(*inputs)))
        output = relu(torch.add(norm2(conv2(output_)), *input))
        return output

    return rbc_function


def frr_func(relu, norm1, conv1, norm2, conv2):
    def rbc_func(*inputs):
        cat = torch.cat(inputs, 1)
        output_ = relu(norm1(conv1(cat)))
        output = relu(norm2(conv2(output_)))
        return output

    return rbc_func


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, efficient=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        if self.efficient:
            resfunc = residual_func(self.relu, self.norm1, self.conv1, self.norm2, self.conv2)
            ret = cp.checkpoint(resfunc, x)
            return ret

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)

        return out


class FRRU(nn.Module):
    def __init__(self, y_in_c, y_out_c, factor=2, z_c=128, efficient=False):
        super(FRRU, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=factor, padding=1)
        self.conv1 = conv3x3(y_in_c + z_c, y_out_c)
        self.bn1 = nn.BatchNorm2d(y_out_c)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(y_out_c, y_out_c)
        self.bn2 = nn.BatchNorm2d(y_out_c)
        self.convz = nn.Conv2d(in_channels=y_out_c, out_channels=z_c, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=factor)
        self.efficient = efficient

    def forward(self, y, z):
        z_ = self.pool(z)

        if self.efficient:
            frr = frr_func(self.relu, self.bn1, self.conv1, self.bn2, self.conv2)
            cat = cp.checkpoint(frr, y, z_)

            z_out = z + self.up(self.convz(cat))
            return cat, z_out

        cat = torch.cat((y, z_), 1)
        cat = self.relu(self.bn1(self.conv1(cat)))
        y = self.relu(self.bn2(self.conv2(cat)))

        z_out = z + self.up(self.convz(y))

        return y, z_out


class FRRLayer(nn.Module):
    def __init__(self, in_channels, out_channels, factor, num_blocks, z_c=128):
        super(FRRLayer, self).__init__()
        self.frr1 = FRRU(in_channels, out_channels, z_c=z_c, factor=factor)
        self.nexts = torch.nn.ModuleList(
            [FRRU(out_channels, out_channels, factor=factor, efficient=False) for _ in range(1, num_blocks)])

    def forward(self, y, z):
        y, z = self.frr1(y, z)
        for m in self.nexts:
            y, z = m(y, z)
        return y, z
