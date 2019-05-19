"""Dataset parsing/loading package.

This package contains classes and functions whose role is to fetch the data required to train, validate,
and test a model. The :func:`thelper.data.utils.create_loaders` function contained herein is responsible for
preparing the task and data loaders for a training session. This package also contains the base interfaces
for dataset parsers.
"""

import logging

import thelper.data.loaders  # noqa: F401
import thelper.data.parsers  # noqa: F401
import thelper.data.pascalvoc  # noqa: F401
import thelper.data.samplers  # noqa: F401
import thelper.data.utils  # noqa: F401
from thelper.data.loaders import DataLoader  # noqa: F401
from thelper.data.loaders import default_collate  # noqa: F401
from thelper.data.parsers import ClassificationDataset  # noqa: F401
from thelper.data.parsers import Dataset  # noqa: F401
from thelper.data.parsers import ExternalDataset  # noqa: F401
from thelper.data.parsers import HDF5Dataset  # noqa: F401
from thelper.data.parsers import ImageDataset  # noqa: F401
from thelper.data.parsers import ImageFolderDataset  # noqa: F401
from thelper.data.parsers import SegmentationDataset  # noqa: F401
from thelper.data.parsers import SuperResFolderDataset  # noqa: F401
from thelper.data.pascalvoc import PASCALVOC  # noqa: F401
from thelper.data.samplers import SubsetRandomSampler  # noqa: F401
from thelper.data.samplers import SubsetSequentialSampler  # noqa: F401
from thelper.data.samplers import WeightedSubsetRandomSampler  # noqa: F401
from thelper.data.utils import create_hdf5  # noqa: F401
from thelper.data.utils import create_loaders  # noqa: F401
from thelper.data.utils import create_parsers  # noqa: F401
from thelper.data.utils import get_class_weights  # noqa: F401
from thelper.tasks.detect import BoundingBox  # noqa: F401

logger = logging.getLogger("thelper.data")
