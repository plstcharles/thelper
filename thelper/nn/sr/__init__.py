"""Neural network and model package for supperresolution.

This package contains classes that define blocks and modules used in various neural network for supperesoltion
architectures. Most of these classes have been adapted from external sources; see their individual
headers for more information.
"""

import logging

import logging

import thelper.nn.utils  # noqa: F401
import thelper.nn.netutils  # noqa: F401
from thelper.nn.sr.srcnn import srcnn  # noqa: F401
from thelper.nn.sr.vdsr import vdsr  # noqa: F401
from thelper.nn.utils import create_model  # noqa: F401

logger = logging.getLogger("thelper.nn.sr")
