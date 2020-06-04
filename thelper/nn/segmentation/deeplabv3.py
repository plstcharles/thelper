import torchvision

import thelper.nn.segmentation.base


class DeepLabV3ResNet50(thelper.nn.segmentation.base.SegmModelBase):
    """
    This class is a thin wrapper for :func:`torchvision.models.segmentation.deeplabv3_resnet101`
    (``torchvision > 0.6``).

    .. note::
        Contributed by Mario Beaulieu <mario.beaulieu@crim.ca>.

    .. seealso::
        | Liang-Chieh et al., `Rethinking Atrous Convolution for Semantic Image Segmentation
          <https://arxiv.org/abs/1706.05587>`_ [arXiv], 2017.
    """
    def __init__(self, task, pretrained=False):
        self.model_cls = torchvision.models.segmentation.deeplabv3_resnet50
        self.in_channels = 256
        super().__init__(task, pretrained=pretrained)


class DeepLabV3ResNet101(thelper.nn.segmentation.base.SegmModelBase):
    """
    This class is a thin wrapper for :func:`torchvision.models.segmentation.deeplabv3_resnet101`
    (``torchvision > 0.6``).

    .. note::
        Contributed by Mario Beaulieu <mario.beaulieu@crim.ca>.

    .. seealso::
        | Liang-Chieh et al., `Rethinking Atrous Convolution for Semantic Image Segmentation
          <https://arxiv.org/abs/1706.05587>`_ [arXiv], 2017.
    """
    def __init__(self, task, pretrained=False):
        self.model_cls = torchvision.models.segmentation.deeplabv3_resnet101
        self.in_channels = 256
        super().__init__(task, pretrained=pretrained)
