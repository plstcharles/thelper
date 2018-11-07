============
Installation
============

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

If you cannot use the Makefile, you will have to install the dependencies yourself. These dependencies are listed in
the `requirements file <https://github.com/plstcharles/thelper/blob/master/requirements.txt>`_, and can also be installed
using the conda environment configuration file provided `here <https://github.com/plstcharles/thelper/blob/master/conda-env.yml>`_.
For the latter case, call the following from your terminal::

  $ conda env create --file <THELPER_ROOT>/conda-env.yml -n thelper

Then, simply activate your environment and install the thelper package within it::

  $ conda activate thelper
  $ pip install -e <THELPER_ROOT> --no-deps

On the other hand, although it is *not* recommended since it tends to break PyTorch, you can install the dependencies
directly through pip::

  $ pip install -r <THELPER_ROOT>/requirements.txt
  $ pip install -e <THELPER_ROOT> --no-deps


Testing the installation
------------------------

You should now be able to print the thelper package version number to see if the package is properly installed and
that all dependencies can be loaded at runtime::

  (conda-env:thelper) username@hostname:~/devel/thelper$ python
    Python X.Y.Z |Anaconda, Inc.| (default, YYY XX ZZZ, AA:BB:CC)
    [GCC X.Y.Z] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import thelper
    >>> print(thelper.__version__)
    x.y.z
