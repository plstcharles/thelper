"""Neural network and model package for supperresolution.

This package contains classes that define blocks and modules used in various neural network for supperesoltion
architectures. Most of these classes have been adapted from external sources; see their individual
headers for more information.
"""

import logging

import thelper.nn  # noqa: F401
from thelper.nn.sr.srcnn import SRCNN  # noqa: F401
from thelper.nn.sr.vdsr import VDSR  # noqa: F401

logger = logging.getLogger("thelper.nn.sr")
