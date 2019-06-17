"""Trainer package.

This package contains classes specialized for training models on various tasks.
"""

import logging

from thelper.train.base import Trainer  # noqa: F401
from thelper.train.classif import ImageClassifTrainer  # noqa: F401
from thelper.train.detect import ObjDetectTrainer  # noqa: F401
from thelper.train.regr import RegressionTrainer  # noqa: F401
from thelper.train.segm import ImageSegmTrainer  # noqa: F401
from thelper.train.utils import create_trainer  # noqa: F401

logger = logging.getLogger("thelper.train")
