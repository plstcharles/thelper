FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

ARG PYTHON_VERSION=3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential git curl vim ca-certificates less && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION

ENV CONDA_HOME /opt/conda
ENV PATH /opt/conda/bin:$PATH

WORKDIR /opt/thelper
COPY . .
RUN sed -i 's/thelper/base/g' conda-env.yml
RUN conda env update --file conda-env.yml && pip install opencv-python-headless
RUN pip install -q -e . --no-deps

WORKDIR /workspace
RUN chmod -R a+w /workspace
