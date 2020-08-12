import torch.nn


class PixelCNN(torch.nn.Module):

    def __init__(
            self,
            n_iters: int = 5,
            input_ch: int = 1024,
            bottleneck_ch: int = 256,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=input_ch,
                    out_channels=bottleneck_ch,
                    kernel_size=(1, 1),
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    in_channels=bottleneck_ch,
                    out_channels=bottleneck_ch,
                    kernel_size=(1, 3),
                    padding=(0, 1),
                ),
                torch.nn.ZeroPad2d((0, 0, 1, 0)),
                torch.nn.Conv2d(
                    in_channels=bottleneck_ch,
                    out_channels=bottleneck_ch,
                    kernel_size=(2, 1),
                    padding=0,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    in_channels=bottleneck_ch,
                    out_channels=input_ch,
                    kernel_size=(1, 1),
                ),
                # add activation function here? only if not last block?
            ) for _ in range(n_iters)])

    def forward(self, in_patches):
        cres = in_patches
        for layer in self.layers:
            c = layer(cres)
            cres = cres + c
        cres = torch.nn.functional.relu(cres, inplace=True)
        return cres
