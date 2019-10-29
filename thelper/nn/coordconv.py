import torch
import torch.nn


def get_coords_map(height, width, centered=True, normalized=True, noise=None, dtype=torch.float32):
    """Returns a HxW intrinsic coordinates map tensor (shape=2xHxW)."""
    x = torch.arange(width, dtype=dtype).unsqueeze(0)
    y = torch.arange(height, dtype=dtype).unsqueeze(0)
    if centered:
        x -= (width - 1) // 2
        y -= (height - 1) // 2
    if normalized:
        x /= width - 1
        y /= height - 1
    x = x.repeat(height, 1)
    y = y.t().repeat(1, width)
    if noise is not None:
        assert isinstance(noise, float) and noise >= 0, "invalid noise stddev value"
        x = torch.normal(mean=x, std=noise)
        y = torch.normal(mean=y, std=noise)
    return torch.stack([x, y])


class AddCoords(torch.nn.Module):
    """Creates a torch-compatible layer that adds intrinsic coordinate layers to input tensors."""
    def __init__(self, centered=True, normalized=True, noise=None, radius_channel=False):
        super().__init__()
        self.centered = centered
        self.normalized = normalized
        self.noise = noise
        self.radius_channel = radius_channel

    def forward(self, in_tensor):
        batch_size, channels, height, width = in_tensor.shape
        coords_map = get_coords_map(height, width, self.centered, self.normalized, self.noise)
        if self.radius_channel:
            middle_slice = coords_map[:, (height - 1) // 2, (width - 1) // 2]
            radius = torch.sqrt(torch.pow(coords_map[0, :, :] - middle_slice[0], 2) +
                                torch.pow(coords_map[1, :, :] - middle_slice[1], 2))
            coords_map = torch.cat([coords_map, radius.unsqueeze(0)], dim=0)
        coords_map = coords_map.repeat(batch_size, 1, 1, 1)
        dev = in_tensor.device
        out = torch.cat([in_tensor, coords_map.to(dev)], dim=1)
        return out


class CoordConv2d(torch.nn.Module):
    """CoordConv-equivalent of torch's default Conv2d model layer.

    See Liu et al. (2018), "An intriguing failing of convolutional neural networks..."
    for more information (https://arxiv.org/abs/1807.03247).
    """

    def __init__(self, in_channels, *args, centered=True, normalized=True,
                 noise=None, radius_channel=False, **kwargs):
        super().__init__()
        self.addcoord = AddCoords(centered=centered, normalized=normalized,
                                  noise=noise, radius_channel=radius_channel)
        extra_ch = 3 if radius_channel else 2
        self.conv = torch.nn.Conv2d(in_channels + extra_ch, *args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose2d(torch.nn.Module):
    """CoordConv-equivalent of torch's default ConvTranspose2d model layer.

    See Liu et al. (2018), "An intriguing failing of convolutional neural networks..."
    for more information (https://arxiv.org/abs/1807.03247).
    """

    def __init__(self, in_channels, *args, centered=True, normalized=True,
                 noise=None, radius_channel=False, **kwargs):
        super().__init__()
        self.addcoord = AddCoords(centered=centered, normalized=normalized,
                                  noise=noise, radius_channel=radius_channel)
        extra_ch = 3 if radius_channel else 2
        self.convT = torch.nn.ConvTranspose2d(in_channels + extra_ch, *args, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out
