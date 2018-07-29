#!/usr/bin/env python

import glob
import io
import os
import re

import setuptools
from setuptools.command.build_ext import build_ext


def read(*names,**kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__),*names),
        encoding=kwargs.get("encoding","utf8")
    ).read()


# Enable code coverage for C code: we can't use CFLAGS=-coverage in tox.ini, since that may mess with compiling
# dependencies (e.g. numpy). Therefore we set SETUPPY_CFLAGS=-coverage in tox.ini and copy it to CFLAGS here (after
# deps have been safely installed).
if "TOXENV" in os.environ and "SETUPPY_CFLAGS" in os.environ:
    os.environ["CFLAGS"] = os.environ["SETUPPY_CFLAGS"]


class optional_build_ext(build_ext):
    """Allow the building of C extensions to fail."""

    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            self._unavailable(e)
            self.extensions = []  # avoid copying missing files (it would fail).

    def _unavailable(self,e):
        print("*"*80)
        print("""WARNING:

    An optional code optimization (C extension) could not be compiled.

    Optimizations for this package will not be available!
        """)

        print("CAUSE:")
        print("")
        print("    "+repr(e))
        print("*"*80)


setuptools.setup(
    name="thelper",
    version="0.0.0",
    license="Apache Software License 2.0",
    description="Provides training help & tools for PyTorch-based machine learning projects.",
    long_description="%s\n%s"%(
        re.compile("^.. start-badges.*^.. end-badges",re.M|re.S).sub("",read("README.rst")),
        re.sub(":[a-z]+:`~?(.*?)`",r"``\1``",read("CHANGELOG.rst"))
    ),
    author="Pierre-Luc St-Charles",
    author_email="pierreluc.stcharles@gmail.com",
    url="https://github.com/plstcharles/thelper",
    packages=setuptools.find_packages("src"),
    package_dir={"":"src"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py")],
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=["pytorch","trainer","loader"],
    install_requires=[
        "augmentor>=0.2.2",
        "matplotlib>=2.2.2",
        "numpy>=1.14.0",
        "opencv-python>=3.3.0",
        "torch>=0.4.1",
        # "scikit-learn>=0.19.1", check ver and readd later
        # "scipy>=1.1.0", check ver and readd later
        "torchvision>=0.2.1",
    ],
    python_requires="~=3.5",
    extras_require={
        "rst":["docutils>=0.11"],
    },
    entry_points={
        "console_scripts":[
            "thelper = thelper.cli:main",
        ]
    },
    cmdclass={"build_ext":optional_build_ext},
    ext_modules=[
        setuptools.Extension(
            os.path.splitext(os.path.relpath(path,"src").replace(os.sep,"."))[0],
            sources=[path],
            include_dirs=[os.path.dirname(path)]
        )
        for root,_,_ in os.walk("src")
        for path in glob.glob(os.path.join(root,"*.c"))
    ],
)
