#!/bin/bash

conda config --set anaconda_upload no
conda build .
anaconda -t $CONDA_TOKEN upload -u $CONDA_USERNAME \
    $CONDA_BUILD_PATH/thelper-0.2.5$CONDA_BUILD_SUFFIX.tar.bz2 --force
