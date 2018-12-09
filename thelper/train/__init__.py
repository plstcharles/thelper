"""Trainer package.

This package contains classes specialized for training models on various tasks.
"""

import logging

import thelper.train.base  # noqa: F401
import thelper.train.classif  # noqa: F401
import thelper.train.segm  # noqa: F401
import thelper.train.utils  # noqa: F401
from thelper.train.base import Trainer  # noqa: F401
from thelper.train.classif import ImageClassifTrainer  # noqa: F401
from thelper.train.segm import ImageSegmTrainer  # noqa: F401
from thelper.train.utils import create_trainer  # noqa: F401

logger = logging.getLogger("thelper.train")
