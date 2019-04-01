import torch

import thelper.nn


class VDSR(thelper.nn.Module):
    """Implements the VDSR architecture.

    See Kim et al., "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" (2015) for more
    information (https://arxiv.org/abs/1511.04587).
    """

    def __init__(self, task, num_channels=1, base_filter=64, kernel_size0=3,
                 num_residuals=18, groups=1, activation='relu', norm='batch'):
        super(VDSR, self).__init__(task)
        self.kernel_size0 = kernel_size0
        self.num_channels = num_channels
        self.input_conv = thelper.nn.common.ConvBlock(input_size=num_channels, output_size=base_filter, kernel_size=self.kernel_size0,
                                                      stride=1, padding=self.kernel_size0 // 2, norm=norm, bias=False, groups=groups,
                                                      activation=activation)
        self.num_residuals = num_residuals
        conv_blocks = []
        if self.num_residuals:
            for _ in range(self.num_residuals):
                conv_blocks.append(thelper.nn.common.ConvBlock(input_size=base_filter, output_size=base_filter, kernel_size=3,
                                                               stride=1, padding=1, norm=norm, bias=False, groups=groups,
                                                               activation=activation))
            self.residual_layers = torch.nn.Sequential(*conv_blocks)
        self.output_conv = thelper.nn.common.ConvBlock(input_size=base_filter, output_size=num_channels, kernel_size=3,
                                                       stride=1, padding=1, activation=None, norm=None, bias=False, groups=groups)
        self.weight_init()
        self.set_task(task)

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
            thelper.nn.common.weights_init_kaiming(m)

    def set_task(self, task):
        if not isinstance(task, thelper.tasks.Regression):
            raise AssertionError("VDSR architecture only available for super res regression tasks")
        self.task = task
