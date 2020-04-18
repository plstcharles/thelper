"""Infer package.

This package contains classes specialized for inference of models on various tasks.
"""

import logging

from thelper.infer.base import Tester  # noqa: F401
from thelper.infer.impl import ImageClassifTester, ImageSegmTester, ObjDetectTester, RegressionTester  # noqa: F401
from thelper.infer.utils import create_tester  # noqa: F401

logger = logging.getLogger(__name__)
