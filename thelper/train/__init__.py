"""Trainer package.

This package contains classes specialized for training models on various tasks.
"""

import logging

import thelper.train.trainers  # noqa: F401
import thelper.train.utils  # noqa: F401
from thelper.train.trainers import Trainer  # noqa: F401
from thelper.train.trainers import ImageClassifTrainer  # noqa: F401
from thelper.train.utils import create_trainer  # noqa: F401

logger = logging.getLogger("thelper.train")
