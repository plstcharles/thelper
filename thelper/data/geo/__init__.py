"""Geospatial dataset parsing/loading package.

This package contains classes and functions whose role is to fetch the data required to train, validate,
and test a model on geospatial data. Importing the modules inside this package requires GDAL.
"""

import logging

import thelper.data.geo.gdl  # noqa: F401
import thelper.data.geo.ogc  # noqa: F401
import thelper.data.geo.parsers  # noqa: F401
import thelper.data.geo.utils  # noqa: F401

logger = logging.getLogger(__name__)
