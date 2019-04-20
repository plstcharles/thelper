"""Graphical User Interface (GUI) package.

This package contains various tools and annotators used to simplify user interactions with data
or models. Since the training framework is CLI-based, these tools are not used to train models,
but can be helpful when debugging them. They can also be used to annotate and explore datasets.
"""

import logging

import thelper.gui.annotators  # noqa: F401
import thelper.gui.utils  # noqa: F401
from thelper.gui.annotators import Annotator  # noqa: F401
from thelper.gui.annotators import ImageSegmentAnnotator  # noqa: F401
from thelper.gui.utils import create_annotator  # noqa: F401
from thelper.gui.utils import create_key_listener  # noqa: F401

logger = logging.getLogger("thelper.gui")
