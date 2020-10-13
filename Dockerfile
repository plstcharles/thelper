FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
LABEL name="thelper"
LABEL description="Training framework and CLI for PyTorch-based machine learning projects"
LABEL vendor="Centre de Recherche Informatique de Montréal / Computer Research Institute of Montreal (CRIM)"
LABEL version="0.6.2"

ARG PYTHON_VERSION=3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential git curl vim ca-certificates less && rm -rf /var/lib/apt/lists/*

ENV CONDA_HOME /opt/conda
ENV PATH ${CONDA_HOME}/bin:$PATH
RUN curl -o ~/miniconda.sh -LO  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${CONDA_HOME} && \
    rm ~/miniconda.sh && \
    ${CONDA_HOME}/bin/conda install -y python=$PYTHON_VERSION

ENV PROJ_LIB ${CONDA_HOME}/share/proj
ENV THELPER_HOME /opt/thelper
WORKDIR ${THELPER_HOME}

# NOTE:
#  force full reinstall with *possibly* updated even if just changing source
#  this way we make sure that it works with any recent dependency update
COPY . .
RUN sed -i 's/thelper/base/g' conda-env.yml
RUN conda env update --file conda-env.yml \
    && pip install opencv-python-headless \
    && conda clean --all -f -y
RUN pip install -q -e . --no-deps

WORKDIR /workspace
RUN chmod -R a+w /workspace

# set default command
# NOTE:
#   avoid using 'entrypoint' as it requires explicit override which not all services do automatically
#   command is easier to override as it is the default docker run CLI input after option flags
#       ie:
#           command:        docker run [options] <your-cmd-override>
#       vs:
#           entrypoint:     docker run [options] --entrypoint="" <your-cmd-override>
#           # without "" override, Dockerfile entrypoint is executed and override command is completly ignored
CMD ["python", "-m", "thelper"]

