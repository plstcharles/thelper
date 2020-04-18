import math

import torch
import torch.nn

import thelper.nn.coordconv
import thelper.nn.srm

warned_bad_input_size_power2 = False


class BasicBlock(torch.nn.Module):
    """Default (double-conv) block used in U-Net layers."""

    def __init__(self, in_channels, out_channels, coordconv=False, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coordconv = coordconv
        self.kernel_size = kernel_size
        self.padding = padding
        self.layer = torch.nn.Sequential(
            thelper.nn.coordconv.make_conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            thelper.nn.coordconv.make_conv2d(in_channels=out_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, coordconv=coordconv),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UNet(thelper.nn.Module):
    """U-Net implementation. Not identical to the original.

    This version includes batchnorm and transposed conv2d layers for upsampling. Coordinate Convolutions
    (CoordConv) can also be toggled on if requested (see :mod:`thelper.nn.coordconv` for more information).
    """

    def __init__(self, task, in_channels=3, mid_channels=512, coordconv=False, srm=False):
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.coordconv = coordconv
        self.srm = srm
        self.pool = torch.nn.MaxPool2d(2)
        self.srm_conv = thelper.nn.srm.setup_srm_layer(in_channels) if srm else None
        self.encoder_block1 = BasicBlock(in_channels=in_channels + 3 if srm else in_channels,
                                         out_channels=mid_channels // 16,
                                         coordconv=coordconv)
        self.encoder_block2 = BasicBlock(in_channels=mid_channels // 16,
                                         out_channels=mid_channels // 8,
                                         coordconv=coordconv)
        self.encoder_block3 = BasicBlock(in_channels=mid_channels // 8,
                                         out_channels=mid_channels // 4,
                                         coordconv=coordconv)
        self.encoder_block4 = BasicBlock(in_channels=mid_channels // 4,
                                         out_channels=mid_channels // 2,
                                         coordconv=coordconv)
        self.mid_block = BasicBlock(in_channels=mid_channels // 2,
                                    out_channels=mid_channels,
                                    coordconv=coordconv)
        self.upsampling_block1 = torch.nn.ConvTranspose2d(in_channels=mid_channels,
                                                          out_channels=mid_channels // 2,
                                                          kernel_size=2, stride=2)
        self.decoder_block1 = BasicBlock(in_channels=mid_channels,
                                         out_channels=mid_channels // 2,
                                         coordconv=coordconv)
        self.upsampling_block2 = torch.nn.ConvTranspose2d(in_channels=mid_channels // 2,
                                                          out_channels=mid_channels // 4,
                                                          kernel_size=2, stride=2)
        self.decoder_block2 = BasicBlock(in_channels=mid_channels // 2,
                                         out_channels=mid_channels // 4,
                                         coordconv=coordconv)
        self.upsampling_block3 = torch.nn.ConvTranspose2d(in_channels=mid_channels // 4,
                                                          out_channels=mid_channels // 8,
                                                          kernel_size=2, stride=2)
        self.decoder_block3 = BasicBlock(in_channels=mid_channels // 4,
                                         out_channels=mid_channels // 8,
                                         coordconv=coordconv)
        self.upsampling_block4 = torch.nn.ConvTranspose2d(in_channels=mid_channels // 8,
                                                          out_channels=mid_channels // 16,
                                                          kernel_size=2, stride=2)
        self.final_block = None
        self.num_classes = None
        self.set_task(task)

    def forward(self, x):
        global warned_bad_input_size_power2
        if not warned_bad_input_size_power2 and len(x.shape) == 4:
            if not math.log(x.shape[-1], 2).is_integer() or not math.log(x.shape[-2], 2).is_integer():
                warned_bad_input_size_power2 = True
                thelper.nn.logger.warning("unet input size should be power of 2 (e.g. 256x256, 512x512, ...)")
        if self.srm_conv is not None:
            noise = self.srm_conv(x)
            x = torch.cat([x, noise], dim=1)
        encoded1 = self.encoder_block1(x)  # 512x512
        encoded2 = self.encoder_block2(self.pool(encoded1))  # 256x256
        encoded3 = self.encoder_block3(self.pool(encoded2))  # 128x128
        encoded4 = self.encoder_block4(self.pool(encoded3))  # 64x64
        embedding = self.mid_block(self.pool(encoded4))  # 32x32
        decoded1 = self.decoder_block1(torch.cat([encoded4, self.upsampling_block1(embedding)], dim=1))
        decoded2 = self.decoder_block2(torch.cat([encoded3, self.upsampling_block2(decoded1)], dim=1))
        decoded3 = self.decoder_block3(torch.cat([encoded2, self.upsampling_block3(decoded2)], dim=1))
        out = self.final_block(torch.cat([encoded1, self.upsampling_block4(decoded3)], dim=1))
        return out

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), "missing impl for non-segm task type"
        if self.final_block is None or self.num_classes != len(task.class_names):
            self.num_classes = len(task.class_names)
            self.final_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=self.mid_channels // 8,
                                out_channels=self.mid_channels // 16,
                                kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(in_channels=self.mid_channels // 16,
                                out_channels=self.num_classes,
                                kernel_size=1),
            )
        self.task = task
