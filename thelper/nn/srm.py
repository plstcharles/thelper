import numpy as np
import torch
import torch.nn


class SRMWrapper(torch.nn.Module):
    """Wraps a base model for Steganalysis Rich Model (SRM)-based noise analysis."""

    def __init__(self, base_model: torch.nn.Module, input_channels: int = 3):
        """Creates a SRM analysis layer and prepares internal params."""
        # note: the base model should expect to process 3 extra channels in its inputs!
        super().__init__()
        self.input_channels = input_channels
        self.base_model = base_model
        self.srm_conv = setup_srm_layer(input_channels)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Adds a stack of noise channels to the input tensor, and processes it using the base model."""
        # simply put, this is an early fusion of noise features...
        noise = self.srm_conv(img)
        img = torch.cat([img, noise], dim=1)
        return self.base_model(img)


def setup_srm_weights(input_channels: int = 3) -> torch.Tensor:
    """Creates the SRM kernels for noise analysis."""
    # note: values taken from Zhou et al., "Learning Rich Features for Image Manipulation Detection", CVPR2018
    srm_kernel = torch.from_numpy(np.array([
        [  # srm 1/2 horiz
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  1., -2.,  1.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
        ], [  # srm 1/4
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
            [ 0., -1.,  2., -1.,  0.],  # noqa: E241,E201
            [ 0.,  2., -4.,  2.,  0.],  # noqa: E241,E201
            [ 0., -1.,  2., -1.,  0.],  # noqa: E241,E201
            [ 0.,  0.,  0.,  0.,  0.],  # noqa: E241,E201
        ], [  # srm 1/12
            [-1.,  2., -2.,  2., -1.],  # noqa: E241,E201
            [ 2., -6.,  8., -6.,  2.],  # noqa: E241,E201
            [-2.,  8., -12., 8., -2.],  # noqa: E241,E201
            [ 2., -6.,  8., -6.,  2.],  # noqa: E241,E201
            [-1.,  2., -2.,  2., -1.],  # noqa: E241,E201
        ]
    ])).float()
    srm_kernel[0] /= 2
    srm_kernel[1] /= 4
    srm_kernel[2] /= 12
    return srm_kernel.view(3, 1, 5, 5).repeat(1, input_channels, 1, 1)


def setup_srm_layer(input_channels: int = 3) -> torch.nn.Module:
    """Creates a SRM convolution layer for noise analysis."""
    weights = setup_srm_weights(input_channels)
    conv = torch.nn.Conv2d(input_channels, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(weights)
    return conv
