#!/usr/bin/env python

import glob
import io
import os
import re

import setuptools
from setuptools.command.build_ext import build_ext


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ).read()


# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if "TOXENV" in os.environ and "SETUPPY_CFLAGS" in os.environ:
    os.environ["CFLAGS"] = os.environ["SETUPPY_CFLAGS"]


class OptionalBuildExt(build_ext):
    """Allow the building of C extensions to fail."""

    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            self._unavailable(e)
            self.extensions = []  # avoid copying missing files (it would fail).

    def _unavailable(self, e):
        print("*" * 80)
        print("""WARNING:

    An optional code optimization (C extension) could not be compiled.

    Optimizations for this package will not be available!
        """)
        print("CAUSE:")
        print("")
        print("    " + repr(e))
        print("*" * 80)


on_rtd = os.environ.get('READTHEDOCS') == 'True'

with open("requirements.txt") as reqfd:
    install_requires = reqfd.read()

setuptools.setup(
    name="thelper",
    version="0.3.2",
    license="Apache Software License 2.0",
    description="Training framework & tools for PyTorch-based machine learning projects.",
    long_description="%s\n%s" % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub("", read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst"))
    ),
    long_description_content_type="text/x-rst",
    author="Pierre-Luc St-Charles",
    author_email="stcharpl@crim.ca",
    url="https://github.com/plstcharles/thelper",
    packages=setuptools.find_packages(exclude=("test",)),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",  # we assume dict insert order will be kept intact... (>=3.6)
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=["pytorch", "trainer", "loader"],
    install_requires=install_requires if not on_rtd else [],  # bypass deps install on rtd
    python_requires="~=3.6",
    extras_require={
        "rst": ["docutils>=0.11"],
    },
    entry_points={
        "console_scripts": [
            "thelper = thelper.cli:main",
        ]
    },
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=[
        setuptools.Extension(
            os.path.splitext(os.path.relpath(path, "src").replace(os.sep, "."))[0],
            sources=[path],
            include_dirs=[os.path.dirname(path)]
        )
        for root, _, _ in os.walk("src")
        for path in glob.glob(os.path.join(root, "*.c"))
    ],
)
