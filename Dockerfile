ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-base

ARG PYTHON_VERSION=3.6
ARG PYTORCH_VERSION=1.5

SHELL ["/bin/bash", "-c"]

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

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    opt/conda/bin/conda config --set always_yes yes --set changeps1 no && \
    opt/conda/bin/conda update -q conda

ENV PATH /opt/conda/bin:$PATH

RUN conda create -y -n icevision-stable python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} torchvision cudatoolkit=${CUDA_VERSION} -c pytorch && \
    source activate icevision-stable && \
    pip install -U pip wheel setuptools && \
    pip install icevision && \
    conda clean -ya && \
    conda info && \
    conda list

COPY . .

RUN conda create -y -n icevision-dev python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} torchvision cudatoolkit={CUDA_VERSION} -c pytorch && \
    source activate icevision-dev && \
    pip install -U pip wheel setuptools && \
    pip install ".[all,dev]" && \
    conda clean -ya && \
    conda info && \
    conda list

CMD ["/bin/bash"]
