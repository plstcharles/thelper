"""Top-level package for the 'thelper' framework.

Running ``import thelper`` will recursively import all important subpackages and modules.
"""

import logging

import thelper.cli  # noqa: F401
import thelper.data  # noqa: F401
import thelper.nn  # noqa: F401
import thelper.optim  # noqa: F401
import thelper.tasks  # noqa: F401
import thelper.train  # noqa: F401
import thelper.transforms  # noqa: F401
import thelper.typedefs  # noqa: F401
import thelper.utils  # noqa: F401

logger = logging.getLogger("thelper")

__url__ = "https://github.com/plstcharles/thelper"
__version__ = "0.3.2"
