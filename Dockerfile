ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-base

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6
ARG ENV_NAME="icevision-stable"

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
    rm -rf /var/lib/apt/lists/*

# get miniconda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    opt/conda/bin/conda config --set always_yes yes && \
    opt/conda/bin/conda update -q conda

# set conda path
ENV PATH /opt/conda/bin:$PATH

# copy everything
COPY . .

# create stable or dev environment
RUN conda create -n ${ENV_NAME} python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION} -c pytorch && \
    source activate ${ENV_NAME} && \
    pip install -U pip wheel setuptools && \
    if [ ${ENV_NAME} = "icevision-dev" ]; then \
        pip install ".[all,dev]" ; \
    else \
        pip install icevision ; \
    fi && \
    conda clean -ya && \
    conda info && \
    conda list

# make sure we have correct env
RUN conda env list

CMD ["/bin/bash"]
