ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-base

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6

SHELL ["/bin/bash", "-c"]

# update and clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* && \
    # get miniconda
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    opt/conda/bin/conda config --set always_yes yes && \
    opt/conda/bin/conda update -q conda

# set conda path
ENV PATH /opt/conda/bin:$PATH

# create stable environment
RUN conda create -n icevision-stable python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION} -c pytorch && \
    source activate icevision-stable && \
    pip install -U pip wheel setuptools && \
    pip install icevision[all] && \
    conda clean -ya && \
    conda info && \
    conda list

# create dev environment
RUN conda create -n icevision-dev python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION} -c pytorch && \
    source activate icevision-dev && \
    pip install -U pip wheel setuptools && \
    pip install git+https://github.com/airctic/icevision.git@master --upgrade && \
    conda clean -ya && \
    conda info && \
    conda list

# default conda env
ENV CONDA_DEFAULT_ENV icevision-stable

# make sure we have 2 envs and icevision-stable is activated
RUN conda init bash && conda env list

CMD ["/bin/bash"]
