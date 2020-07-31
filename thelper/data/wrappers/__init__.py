"""Dataset wrappers package.

This package contains classes and functions that can wrap other datasets to transform
them in very specific ways. This includes e.g. the patch splitter used for CPC training.
"""

import logging

import thelper.data.wrappers.patch  # noqa: F401

logger = logging.getLogger(__name__)
