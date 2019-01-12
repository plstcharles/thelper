"""
Entrypoint module, in case you use `python -m thelper`.
"""

import sys

import thelper.cli


def init():
    if __name__ == "__main__":
        sys.exit(thelper.cli.main())


init()
