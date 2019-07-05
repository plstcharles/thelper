.. _install-guide:

============
Installation
============

Anaconda
========

Starting with v0.2.5, the latest stable version of the framework can be installed directly (with its
dependencies) via `Anaconda <https://docs.anaconda.com/anaconda/install/>`_. In a conda environment,
simply enter::

  $ conda config --env --add channels plstcharles
  $ conda config --env --add channels conda-forge
  $ conda config --env --add channels pytorch
  $ conda config --env --add channels albumentations
  $ conda install thelper

This should install the latest stable version of the framework on Windows and Linux, for Python
3.6 or 3.7. You can check the release notes `on GitHub`__.

.. __: https://github.com/plstcharles/thelper/blob/master/CHANGELOG.rst


Docker
======

Starting with v0.3.2, the latest stable version of the framework is pre-built and available from the
docker hub. To get a copy, simply pull it via::

  $ docker pull plstcharles/thelper

You should then be able to launch sessions in containers as such::

  $ docker run -it plstcharles/thelper thelper <CLI_ARGS_HERE>


Installing from source
======================

If you wish to modify the framework's source code, follow the installation instructions below.

Linux
-----

You can use the provided Makefile to automatically create a conda environment on your system that will contain
the thelper framework and all its dependencies. In your terminal, simply enter::

  $ cd <THELPER_ROOT>
  $ make install

If you already have conda installed somewhere, you can force the Makefile to use it for the installation of the
new environment by setting the ``CONDA_HOME`` variable before calling make::

  $ export CONDA_HOME=/some/path/to/miniconda3
  $ cd <THELPER_ROOT>
  $ make install

The newly created conda environment will be called 'thelper', and can then be activated using::

  $ conda activate thelper

Or, assuming conda is not already in your path::

  $ source /some/path/to/miniconda3/bin/activate thelper


Other systems
-------------

If you cannot use the Makefile, you will have to install the dependencies yourself. These dependencies are
listed in the `requirements file <https://github.com/plstcharles/thelper/blob/master/requirements.txt>`_,
and can also be installed using the conda environment configuration file provided `here`__. For the latter
case, call the following from your terminal::

  $ conda env create --file <THELPER_ROOT>/conda-env.yml -n thelper

.. __: https://github.com/plstcharles/thelper/blob/master/conda-env.yml

Then, simply activate your environment and install the thelper package within it::

  $ conda activate thelper
  $ pip install -e <THELPER_ROOT> --no-deps

On the other hand, although it is *not* recommended since it tends to break PyTorch, you can install the dependencies
directly through pip::

  $ pip install -r <THELPER_ROOT>/requirements.txt
  $ pip install -e <THELPER_ROOT> --no-deps


Testing the installation
========================

You should now be able to print the thelper package version number to see if the package is properly installed and
that all dependencies can be loaded at runtime::

  (conda-env:thelper) username@hostname:~/devel/thelper$ python
    Python X.Y.Z |Anaconda, Inc.| (default, YYY XX ZZZ, AA:BB:CC)
    [GCC X.Y.Z] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import thelper
    >>> print(thelper.__version__)
    x.y.z

You can now refer to the `[user guide]`__ for more information on how to use the framework.

.. __: https://thelper.readthedocs.io/en/latest/user-guide.html


Documentation
=============

The sphinx documentation is generated automatically via `readthedocs.io <https://readthedocs.org/projects/thelper/>`_,
but it might still be incomplete due to buggy apidoc usage/platform limitations. To build it yourself, use the makefile::

  $ cd <THELPER_ROOT>
  $ make docs

The HTML documentation should then be generated inside ``<THELPER_ROOT>/docs/build/html``. To browse it, simply open the
``index.html`` file there.
