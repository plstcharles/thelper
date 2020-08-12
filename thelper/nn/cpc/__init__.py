"""Model package for Contrastive Predictive Coding-based learning.

See the following articles for more information:
  - https://arxiv.org/abs/1807.03748
  - https://arxiv.org/abs/1905.09272
"""

import logging

import thelper.nn.cpc.base  # noqa: F401
import thelper.nn.cpc.pixelcnn  # noqa: F401

logger = logging.getLogger("thelper.nn.cpc")
