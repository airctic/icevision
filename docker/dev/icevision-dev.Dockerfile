
# Base image
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime


# Metadata
# LABEL maintainer=""
LABEL environment_type="Icevision Dev"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# Run commands on initialising the container
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get clean

# Build Env
RUN apt-get install -y \
    apt-utils \
    build-essential \
    byobu \
    bzip2 \
    ca-certificates \
    curl \
    git-core \
    htop \
    libcurl4-openssl-dev \
    libssl-dev wget \
    pkg-config \
    sudo \
    tree \
    wget \
    unzip

RUN pip install --upgrade pip
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Expose port for Jupyter
EXPOSE 8889

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get update
RUN add-apt-repository ppa:kelleyk/emacs
RUN apt update
RUN apt install emacs -y
RUN apt-get update


# Install python packages
COPY requirements.txt ./
COPY no-dep-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r no-dep-requirements.txt --no-dependencies

# === MMDetection === #
# MM-CV needs to be torch and cuda version specific
#  Torch 1.7 w/ CUDA 11.0
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
#  Torch 1.8 w/ CUDA 11.1
#  RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
RUN pip install mmdet

# Jupyter extensions
RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter contrib nbextension install --user

# Black for Jupyer
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
RUN jupyter nbextension enable jupyter-black-master/jupyter-black


# Copy Jupyter Settings
COPY run-jupyter.sh /run-jupyter.sh
RUN chmod +x /run-jupyter.sh

# There can only be one CMD instruction in a Dockerfile. If you list more 
#     than one CMD then only the last CMD will take effect.
# The main purpose of a CMD is to provide defaults for an executing container.

# You could start a bash shell and be inside it
CMD ["/bin/bash"]
# OR
# You could start a notebook server (in the background?)
# CMD ["/run-jupyter.sh"]