import torchvision

import thelper.nn.segmentation.base


class FCNResNet50(thelper.nn.segmentation.base.SegmModelBase):
    """
    This class is a thin wrapper for :func:`torchvision.models.segmentation.fcn_resnet50` (``torchvision > 0.6``).

    .. note::
        Contributed by Mario Beaulieu <mario.beaulieu@crim.ca>.

    .. seealso::
        | Liang-Chieh et al., `Rethinking Atrous Convolution for Semantic Image Segmentation
          <https://arxiv.org/abs/1706.05587>`_ [arXiv], 2017.
    """
    def __init__(self, task, pretrained=False):
        self.model_cls = torchvision.models.segmentation.fcn_resnet50
        self.in_channels = 512
        super().__init__(task, pretrained=pretrained)


class FCNResNet101(thelper.nn.segmentation.base.SegmModelBase):
    """
    This class is a thin wrapper for :func:`torchvision.models.segmentation.fcn_resnet50` (``torchvision > 0.6``).

    .. note::
        Contributed by Mario Beaulieu <mario.beaulieu@crim.ca>.

    .. seealso::
        | Liang-Chieh et al., `Rethinking Atrous Convolution for Semantic Image Segmentation
          <https://arxiv.org/abs/1706.05587>`_ [arXiv], 2017.
    """
    def __init__(self, task, pretrained=False):
        self.model_cls = torchvision.models.segmentation.fcn_resnet101
        self.in_channels = 512
        super().__init__(task, pretrained=pretrained)
