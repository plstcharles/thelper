import logging
logger = logging.getLogger(__file__)
import torch
import torch.nn as nn
import numpy as np
from thelper.nn.netutils import  *

__all__ = ['srcnn', 'SRCNN']


class SRCNN(nn.Module):

    def __init__(self, num_channels=3, base_filter=64,  groups=1, **kwargs):

        super(SRCNN, self).__init__()
        self.conv1 = ConvBlock(num_channels, base_filter*groups,
                               kernel_size=9,
                               stride=1,
                               padding=0,
                               norm=None,
                               groups=groups,
                               activation='relu')
        self.conv2 = ConvBlock(base_filter*groups, base_filter // 2*groups, kernel_size=5, stride=1, padding=0, norm=None, groups=groups,activation='relu')
        self.conv3 = ConvBlock((base_filter // 2)*groups, num_channels, kernel_size=5, stride=1, padding=0, activation=None, norm=None,groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            weights_init_xavier(m)


def srcnn(pretrained=False, **kwargs):
    """Constructs a srcnn.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SRCNN(**kwargs)
    if pretrained:
        logger.debug("No pre-trained net available")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
