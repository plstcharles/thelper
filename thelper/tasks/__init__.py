"""Task definition package.

This package contains classes and functions whose role is to define the input/output formats and operations
expected from a trained model. This essentially defines the 'goal' of the model, and is used to specialize
and automate its training.
"""

import logging

from thelper.tasks.classif import Classification  # noqa: F401
from thelper.tasks.detect import Detection  # noqa: F401
from thelper.tasks.regr import Regression  # noqa: F401
from thelper.tasks.regr import SuperResolution  # noqa: F401
from thelper.tasks.segm import Segmentation  # noqa: F401
from thelper.tasks.utils import Task  # noqa: F401
from thelper.tasks.utils import create_global_task  # noqa: F401
from thelper.tasks.utils import create_task  # noqa: F401

logger = logging.getLogger("thelper.tasks")
