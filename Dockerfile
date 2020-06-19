# Choose the base image from to take.
# Using slim images is best practice
FROM ubuntu:18.04

# Install dependencies
# Do this first for caching
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get install -y git
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
# RUN pip3 install git+git://github.com/lgvaz/mantisshrimp.git
RUN git clone https://github.com/lgvaz/mantisshrimp.git
WORKDIR "/mantisshrimp"

COPY requirements.txt  requirements.txt
COPY requirements-extra.txt requirements-extra.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r requirements-extra.txt
RUN pip3 install git+git://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip3 install .

# COPY Important files

COPY mantisshrimp mantisshrimp
COPY examples examples
COPY samples samples
COPY tutorials tutorials
COPY tests tests

# We need to expose port an run a dummy output with wsgi server
