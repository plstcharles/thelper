"""Neural network and model package for segemnation.

This package contains classes that define blocks and modules used in various neural network for segmentation
"""

import logging

import thelper.nn  # noqa: F401
from thelper.nn.segmentation.deeplabv3 import deeplabv3_resnet50
from thelper.nn.segmentation.deeplabv3 import deeplabv3_resnet101  # noqa: F401

logger = logging.getLogger("thelper.nn.sr")
