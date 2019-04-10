"""Optimization/metrics package.

This package contains metrics implementations used to monitor training sessions and evaluate models,
and optimization methods used to control the learning behavior of these models.
"""

import logging

import thelper.optim.metrics  # noqa: F401
import thelper.optim.schedulers  # noqa: F401
import thelper.optim.utils  # noqa: F401
from thelper.optim.metrics import PSNR  # noqa: F401
from thelper.optim.metrics import BinaryAccuracy  # noqa: F401
from thelper.optim.metrics import CategoryAccuracy  # noqa: F401
from thelper.optim.metrics import ClassifLogger  # noqa: F401
from thelper.optim.metrics import ClassifReport  # noqa: F401
from thelper.optim.metrics import ConfusionMatrix  # noqa: F401
from thelper.optim.metrics import ExternalMetric  # noqa: F401
from thelper.optim.metrics import MeanAbsoluteError  # noqa: F401
from thelper.optim.metrics import MeanSquaredError  # noqa: F401
from thelper.optim.metrics import Metric  # noqa: F401
from thelper.optim.metrics import RawPredictions  # noqa: F401
from thelper.optim.metrics import ROCCurve  # noqa: F401
from thelper.optim.schedulers import CustomStepLR  # noqa: F401
from thelper.optim.utils import create_loss_fn  # noqa: F401
from thelper.optim.utils import create_metrics  # noqa: F401
from thelper.optim.utils import create_optimizer  # noqa: F401
from thelper.optim.utils import create_scheduler  # noqa: F401
from thelper.optim.utils import get_lr  # noqa: F401

logger = logging.getLogger("thelper.optim")
