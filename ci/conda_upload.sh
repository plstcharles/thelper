#!/bin/bash

export CONDA_BUILD_PATH=~/conda-build

mkdir $CONDA_BUILD_PATH
conda config --set anaconda_upload no
conda build .
anaconda -t $CONDA_TOKEN upload -u $CONDA_USERNAME \
    $CONDA_BUILD_PATH/$TRAVIS_OS_NAME-64/thelper-0.2.4.tar.bz2 --force