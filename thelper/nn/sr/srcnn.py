
import thelper.nn


class SRCNN(thelper.nn.Module):
    """Implements the SRCNN architecture.

    See Dong et al., "Image Super-Resolution Using Deep Convolutional Networks" (2014) for more
    information (https://arxiv.org/abs/1501.00092).
    """

    def __init__(self, task, num_channels=1, base_filter=64, groups=1):
        super(SRCNN, self).__init__(task)
        self.conv1 = thelper.nn.common.ConvBlock(num_channels, base_filter * groups, kernel_size=9,
                                                 stride=1, padding=0, activation="relu", norm=None, groups=groups)
        self.conv2 = thelper.nn.common.ConvBlock(base_filter * groups, base_filter // 2 * groups, kernel_size=5,
                                                 stride=1, padding=0, activation="relu", norm=None, groups=groups)
        self.conv3 = thelper.nn.common.ConvBlock((base_filter // 2) * groups, num_channels, kernel_size=5,
                                                 stride=1, padding=0, activation=None, norm=None, groups=groups)
        self.set_task(task)

    def forward(self, x):
        x0 = x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = x0.view(x.shape[0], x.shape[1], x0.shape[2], x0.shape[3])
        return x0

    def weight_init(self):
        for m in self.modules():
            thelper.nn.common.weights_init_xavier(m)

    def set_task(self, task):
        if not isinstance(task, thelper.tasks.Regression):
            raise AssertionError("SRCNN architecture only available for super res regression tasks")
        self.task = task
