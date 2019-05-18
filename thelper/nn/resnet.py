import torch
import torch.nn
import torch.utils.model_zoo

import thelper.nn
import thelper.nn.coordconv


class Module(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, coordconv=False, radius_channel=True):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.coordconv = coordconv
        self.radius_channel = radius_channel

    def _make_conv2d(self, *args, **kwargs):
        if self.coordconv:
            return thelper.nn.coordconv.CoordConv2d(*args, radius_channel=self.radius_channel, **kwargs)
        else:
            return torch.nn.Conv2d(*args, **kwargs)


class BasicBlock(Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, coordconv=False, radius_channel=True):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, coordconv=False, radius_channel=True):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = self._make_conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


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


class SqueezeExcitationBlock(Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, coordconv=False, radius_channel=True):
        super().__init__(inplanes, planes, stride, downsample, coordconv, radius_channel)
        self.conv1 = self._make_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = self._make_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.se = SqueezeExcitationLayer(planes, reduction)

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


class ResNet(thelper.nn.Module):

    def __init__(self, task, block=BasicBlock, layers=[3, 4, 6, 3], strides=[1, 2, 2, 2], input_channels=3,
                 flexible_input_res=False, pool_size=7, coordconv=False, radius_channel=True):
        # TODO: add pretrained param to toggle loading weights from imagenet before applying task?
        super().__init__(task)
        if isinstance(block, str):
            block = thelper.utils.import_class(block)
        if not issubclass(block, Module):
            raise AssertionError("block type must be subclass of thelper.nn.resnet.Module")
        if not isinstance(layers, list) or not isinstance(strides, list):
            raise AssertionError("expected layers/strides to be provided as list of ints")
        if len(layers) != len(strides):
            raise AssertionError("layer/strides length mismatch")
        self.inplanes = 64
        self.coordconv = coordconv
        self.radius_channel = radius_channel
        self.conv1 = self._make_conv2d(in_channels=input_channels, out_channels=self.inplanes,
                                       kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3])
        self.out_features = 512
        self.layer5 = None
        if len(layers) > 4:
            self.layer5 = self._make_layer(block, 1024, layers[4], stride=strides[4])
            self.out_features = 1024
        if flexible_input_res:
            self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            if pool_size < 1:
                raise AssertionError("invalid avg pool size for non-flex resolution")
            self.avgpool = torch.nn.AvgPool2d(pool_size, stride=1)
        self.out_features *= block.expansion
        self.fc = torch.nn.Linear(self.out_features, 1000)  # output type/count will be specialized by task after init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, thelper.nn.coordconv.CoordConv2d):
                torch.nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        self.set_task(task)

    def _make_conv2d(self, *args, **kwargs):
        if self.coordconv:
            return thelper.nn.coordconv.CoordConv2d(*args, radius_channel=self.radius_channel, **kwargs)
        else:
            return torch.nn.Conv2d(*args, **kwargs)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                self._make_conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Classification), "missing impl for non-classif task type"
        num_classes = len(task.class_names)
        if self.fc.out_features != num_classes:
            self.fc = torch.nn.Linear(self.out_features, num_classes)
        self.task = task


class FCResNet(ResNet):

    def __init__(self, task, ckptdata, map_location="cpu", avgpool_size=0):
        if isinstance(ckptdata, str):
            ckptdata = thelper.utils.load_checkpoint(ckptdata, map_location=map_location)
        model_type = ckptdata["model_type"]
        if model_type != "thelper.nn.resnet.ResNet":
            raise AssertionError("cannot convert non-resnet model to fully conv with this impl")
        model_params = ckptdata["model_params"]
        if isinstance(ckptdata["task"], str):
            old_model_task = thelper.tasks.create_task(ckptdata["task"])
        else:
            old_model_task = ckptdata["task"]
        self.task = None
        self.avgpool_size = avgpool_size
        super().__init__(old_model_task, **model_params)
        self.load_state_dict(ckptdata["model"], strict=False)  # assumes model always stored as weight dict
        self.finallayer = torch.nn.Conv2d(self.out_features, self.fc.out_features, kernel_size=1)
        self.finallayer.weight = torch.nn.Parameter(self.fc.weight.view(self.fc.out_features, self.out_features, 1, 1))
        self.finallayer.bias = torch.nn.Parameter(self.fc.bias)
        self.set_task(task)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.layer5 is not None:
            x = self.layer5(x)
        if self.avgpool_size > 0:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.avgpool_size, stride=1)
        x = self.finallayer(x)
        return x

    def set_task(self, task):
        assert isinstance(task, (thelper.tasks.Segmentation, thelper.tasks.Classification)), \
            "missing impl for non-segm/classif task type"
        num_classes = len(task.class_names)
        if self.fc.out_features != num_classes:
            self.fc = torch.nn.Linear(self.out_features, num_classes)
            self.finallayer = torch.nn.Conv2d(self.out_features, num_classes, kernel_size=1)
            self.finallayer.weight = torch.nn.Parameter(self.fc.weight.view(self.fc.out_features, self.out_features, 1, 1))
            self.finallayer.bias = torch.nn.Parameter(self.fc.bias)
        self.task = task
