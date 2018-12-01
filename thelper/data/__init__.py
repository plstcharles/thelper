"""Dataset parsing/loading package.

This package contains classes and functions whose role is to fetch the data required to train, validate,
and test a model. The :func:`thelper.data.utils.create_loaders` function contained herein is responsible for
preparing the task and data loaders for a training session. This package also contains the base interfaces
for dataset parsers.
"""

import logging

import thelper.data.utils  # noqa: F401
from thelper.data.parsers import ClassificationDataset  # noqa: F401
from thelper.data.parsers import Dataset  # noqa: F401
from thelper.data.parsers import ExternalDataset  # noqa: F401
from thelper.data.parsers import ImageDataset  # noqa: F401
from thelper.data.parsers import ImageFolderDataset  # noqa: F401
from thelper.data.utils import create_loaders  # noqa: F401
from thelper.data.utils import create_parsers  # noqa: F401

logger = logging.getLogger("thelper.data")
