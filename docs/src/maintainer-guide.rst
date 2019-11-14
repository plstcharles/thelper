.. _maintainer-guide:

================
Maintainer Guide
================

This guide provides an overview of the basic steps required for testing, creating new releases, and deploying
new builds of the framework. For installation instructions, refer to the installation guide :ref:`[here]
<install-guide>`.

As of November 2019 (v0.4.3), the framework's Continuous Integration (CI) pipeline is based primarily on
`Travis CI <travis_>`_ and `Docker Hub <dockerhub_>`_. Automated conda builds have slowly been getting
harder and harder to fix on Travis, and have gradually been phased out in favor of installation from
source.

Maintainers are expected to be using an installation from source and running on a platform that supports
Makefiles. All commands below are also provided under the assumption that they will be executed from the
root directory of the framework.

  .. _travis: https://travis-ci.org
  .. _dockerhub: https://hub.docker.com


Testing
=======

The simplest way to test whether local changes have broken something in the framework is to use make::

    $ make test-all

This command will start by running linters (flake8, isort, twine, check-manifest), and then execute all
tests and produce a coverage report. To only run the linters, you can use::

    $ make check

The framework tests are based on pytest, and can be executed indenpendently via::

    $ make test

Since Travis CI runs on GPU-less platforms, unit tests and integration tests with mocked device-aware
components are preferred. Future regression tests should also keep this limitation in consideration.

Tests can leave some logs and artifacts in the working directory, especially if they are cancelled in
the middle of a run. To get rid of these, you can use::

    $ make clean-test


Releases
========

To tag a commit for a release, you should use ``bumpversion``. It is pre-configured to update all
framework version references everywhere and automatically create a new commit with the required tag.
Bumping the version works by incrementing the patch (v0.0.X), minor (v0.X.0) or major (vX.0.0) integer::

    $ bumpversion patch
        or
    $ bumpversion minor
        or
    $ bumpversion major

Creating and pushing any tag on GitHub will trigger the deployment phase on Travis CI.


Documentation
=============

Building the documentation can be accomplished by simply calling::

    $ make docs

This will create the documentation pages in HTML format and display them in your browser. The same
documentation will be built and deployed on `readthedocs.io <https://readthedocs.org/projects/thelper/>`_.


Deployment
==========

Travis CI will automatically attempt to deploy the framework after successfully testing a tagged version.
The deployment will target PyPI, Docker Hub, and Anaconda. If any of these steps fail, the deployment
can be completed manually as specified below.

Python Package Index (PyPI)
---------------------------

A source distribution (sdist) and be prepared and uploaded using the following commands::

    $ python setup.py sdist
    $ twine upload dist/* --skip-existing

This will allow you to upload to your own project page, or to the `origin <https://pypi.org/project/thelper/>`_
(if you have collaborator access).

Docker Hub
----------

A docker image can be prepared and uploaded using the following commands::

    $ docker build -t ${DOCKER_REPO}:${TAG} -t ${DOCKER_REPO}:latest .
    $ docker push ${DOCKER_REPO}

Again, uploading to the `original project page <https://hub.docker.com/r/plstcharles/thelper>`_ will require
collaborator access, but you can also upload the image to your own private repository.


Anaconda
--------

The Anaconda package build process is very long, and requires a lot of disk space. It tends to fail on
Travis CI, and must often be completed manually. To do so, you must first configure your conda environment
to use custom channels to find proper project dependencies::

    $ conda config --prepend channels conda-forge
    $ conda config --prepend channels albumentations
    $ conda config --prepend channels pytorch

Then, updating conda itself is never a bad idea::

    $ conda update -q conda

To build and upload a package, you must also install the correct CLI tools::

    $ conda install conda-build conda-verify anaconda-client

Finally, using the meta-config already available inside the project, you can build the package via::

    $ conda build ci/

Instructions will be printed in the terminal regarding how to upload the built packages.
