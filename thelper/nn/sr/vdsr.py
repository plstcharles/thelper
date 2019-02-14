import logging
logger = logging.getLogger(__file__)
import torch
import torch.nn as nn
import numpy as np
from thelper.nn.netutils import  *
import thelper

__all__ = ['vdsr', 'VDSR']


class VDSR(torch.nn.Module):

    def __init__(self, num_channels=1, base_filter=64, kernel_size0=3, num_residuals=18, groups=1, activation='relu', norm='batch', **kwargs):
        super(VDSR, self).__init__()
        self.kernel_size0 = kernel_size0
        self.num_channels = num_channels
        self.input_conv = ConvBlock(input_size=num_channels,
                                    output_size=base_filter,
                                    kernel_size=self.kernel_size0,
                                    stride=1,
                                    padding=self.kernel_size0 // 2,
                                    norm=norm,
                                    bias=False,
                                    groups=groups,
                                    activation=activation)

        self.num_residuals = num_residuals
        conv_blocks = []
        if self.num_residuals:
            for _ in range(self.num_residuals):
                conv_blocks.append(ConvBlock(input_size = base_filter,
                                                 output_size = base_filter,
                                                 kernel_size= 3,
                                                 stride=1,
                                                 padding=1,
                                                 norm=norm,
                                                 bias=False,
                                                 groups=groups,
                                                 activation=activation,
                                                 ))

            self.residual_layers = nn.Sequential(*conv_blocks)

        self.output_conv = ConvBlock(input_size=base_filter,
                                     output_size=num_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     activation=None,
                                     norm=None,
                                     bias=False,
                                     groups=groups)

        self.weight_init()


    def forward(self, x):
        x0 = x.view(x.shape[0]*x.shape[1],1, x.shape[2], x.shape[3])
        residual = x0
        x0 = self.input_conv(x0)
        if self.num_residuals > 0:
            x0 = self.residual_layers(x0)
        x0 = self.output_conv(x0)
        x0 = torch.add(x0, residual)
        x0 = x0.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return x0

    def weight_init(self):
        for m in self.modules():
            weights_init_kaming(m)


def vdsr(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VDSR(**kwargs)
    if pretrained:
        logger.debug("No pre-trained net available")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
