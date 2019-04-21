FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ARG PYTHON_VERSION=3.6
ARG PYTORCH_VERSION=1.0.1
ARG TORCHVISION_VERSION=0.2.2
ARG MKL_VERSION=2019.1

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION \
        numpy pyyaml scipy ipython mkl=${MKL_VERSION} mkl-include=${MKL_VERSION} cython typing matplotlib lz4 scikit-learn \
        pytest pytest-cov pytest-mock tqdm shapely gitpython mock && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
     /opt/conda/bin/conda install -y -c albumentations albumentations && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN pip install ninja h5py imgaug==0.2.5 augmentor tensorboardX pynput opencv-python-headless

RUN git clone -b v${PYTORCH_VERSION} --progress --verbose --single-branch https://github.com/pytorch/pytorch.git /opt/pytorch && \
    git clone -b v${TORCHVISION_VERSION} --progress --verbose --single-branch https://github.com/pytorch/vision.git /opt/torchvision

WORKDIR /opt/pytorch
RUN git submodule update --init
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

WORKDIR /opt/torchvision
RUN pip install -v .

WORKDIR /opt/thelper
COPY . .
RUN pip install -e . --no-deps

WORKDIR /workspace
RUN chmod -R a+w /workspace
