import torch
import torch.nn

from .resnet import ResNet
from .resnet import ResFullConvNet

import thelper.modules


class SqueezeExcitationLayer(torch.nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SqueezeExcitationBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes, 1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.se = SqueezeExcitationLayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def conv3x3(in_planes, out_planes, stride=1):
        return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SqueezeExcitationNet(thelper.modules.Module):

    def __init__(self, task, name, layers):
        super().__init__(task, name)
        self.logger.info("initializing senet with layers={} and {} classes".format(layers, task.get_nb_classes()))
        self.model = ResNet(SqueezeExcitationBlock, layers, task.get_nb_classes())

    def forward(self, x):
        return self.model.forward(x)


class SqueezeExcitationFullConvNet(thelper.modules.Module):

    def __init__(self, task, name, layers):
        super().__init__(task, name)
        self.logger.info("initializing senet with layers={} and {} classes".format(layers, task.get_nb_classes()))
        self.model = ResFullConvNet(SqueezeExcitationBlock, layers, task.get_nb_classes())

    def forward(self, x):
        return self.model.forward(x)
