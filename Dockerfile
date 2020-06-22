# Choose the base image from to take.
# Using slim images is best practice

ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-base

# install versions
ARG PYTHON_VERSION=3.6
ARG PYTORCH_VERSION=1.5
ARG MANTISSHRIMP_VERSION=master

# This is one of the best practice. 
# This technique is known as “cache busting”.
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates

# add non-root user
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser

# install miniconda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya && \
    /home/containeruser/conda/bin/conda install -y python=$PYTHON_VERSION 

# add conda to path
ENV PATH /home/containeruser/conda/bin:$PATH

# install Pytorch with cuda dependencies. This pytorch is GPU accelerated.
RUN pip install torch==$PYTORCH_VERSION

# Now install mantisshrimp
# We need only the master branch not all branches
RUN git clone https://github.com/lgvaz/mantisshrimp.git --single-branch --branch $MANTISSHRIMP_VERSION

# WORKDIR "/mantisshrimp"

# This is another good practice to cache the files before installing.
COPY requirements.txt  requirements.txt
COPY requirements-extra.txt requirements-extra.txt
RUN pip install -r requirements.txt && \
    pip install -r requirements-extra.txt && \
    pip install ./mantisshrimp/ && \
    pip install git+git://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
# CMD ls    
# COPY Important files

COPY mantisshrimp mantisshrimp
COPY examples examples
COPY samples samples
COPY tutorials tutorials
COPY tests tests

CMD ["/bin/bash"]

