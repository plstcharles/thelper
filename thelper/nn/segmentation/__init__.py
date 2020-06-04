"""Neural network and model package for segemnation.

This package contains classes that define blocks and modules used in various neural network for segmentation
"""

import logging

import thelper.nn  # noqa: F401
from thelper.nn.segmentation.deeplabv3 import DeepLabV3ResNet50  # noqa: F401
from thelper.nn.segmentation.deeplabv3 import DeepLabV3ResNet101  # noqa: F401
from thelper.nn.segmentation.fcn import FCNResNet50  # noqa: F401
from thelper.nn.segmentation.fcn import FCNResNet101  # noqa: F401

# dirty redirection for backward compat with Mario's stuff
deeplabv3_resnet50 = DeepLabV3ResNet50  # noqa: C0103
deeplabv3_resnet101 = DeepLabV3ResNet101  # noqa: C0103
fcn_resnet50 = FCNResNet50  # noqa: C0103
fcn_resnet101 = FCNResNet101  # noqa: C0103

logger = logging.getLogger("thelper.nn.segm")
