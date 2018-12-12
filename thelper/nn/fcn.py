# inspired from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py

import torch
import torch.nn
import torch.nn.functional

import thelper.nn


class FCN32s(thelper.nn.Module):
    def __init__(self, task, learned_billinear=False, init_vgg16=True):
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
        self.set_task(task)
        self.learned_billinear = learned_billinear
        if self.learned_billinear:
            raise NotImplementedError
        if init_vgg16:
            import torchvision
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.init_vgg16_params(vgg16)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        out = torch.nn.functional.upsample(score, x.size()[2:], mode="bilinear")
        return out

    def set_task(self, task):
        if isinstance(task, thelper.tasks.Segmentation):
            num_classes = len(task.get_class_names())
            if self.classifier is None or self.classifier[-1].out_features != num_classes:
                self.classifier = torch.nn.Sequential(
                    torch.nn.Conv2d(512, 4096, 7),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout2d(),
                    torch.nn.Conv2d(4096, 4096, 1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout2d(),
                    torch.nn.Conv2d(4096, num_classes, 1),
                )
        else:
            raise AssertionError("missing impl for non-segm task type")
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
