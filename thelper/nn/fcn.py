# inspired from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py

import numpy as np
import torch
import torch.nn
import torch.nn.functional

import thelper.nn


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(thelper.nn.Module):
    def __init__(self, task, init_vgg16=True):
        super().__init__(task)
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # res = 1/2
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # res = 1/4
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # res = 1/8
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # res = 1/16
        self.conv_block5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # res = 1/32
        self.classifier = None
        self.upscaler = None
        self.set_task(task)
        if init_vgg16:
            import torchvision
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.init_vgg16_params(vgg16)

    def forward(self, x):
        conv = self.conv_block1(x)
        conv = self.conv_block2(conv)
        conv = self.conv_block3(conv)
        conv = self.conv_block4(conv)
        conv = self.conv_block5(conv)
        score = self.classifier(conv)
        out = self.upscaler(score)
        out = out[:, :, 19:(19 + x.size()[2]), 19:(19 + x.size()[3])].contiguous()
        #out = torch.nn.functional.upsample(score, x.size()[2:], mode="bilinear")
        return out

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), "missing impl for non-segm task type"
        num_classes = len(task.class_names)
        if self.classifier is None or list(self.classifier.modules())[-1].out_channels != num_classes:
            self.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(),
                torch.nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout2d(),
                torch.nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0),
            )
            self.upscaler = torch.nn.ConvTranspose2d(num_classes, num_classes,
                                                     kernel_size=64, stride=32, bias=False)
            self.upscaler.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))
        self.task = task

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]: ranges[idx][1]], conv_block):
                if isinstance(l1, torch.nn.Conv2d) and isinstance(l2, torch.nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
