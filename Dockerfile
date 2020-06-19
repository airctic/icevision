# Choose the base image from to take.
# Using slim images is best practice
FROM python:3.6-slim

# Install dependencies
# Do this first for caching
COPY requirements.txt  requirements.txt
RUN pip install git+git://github.com/lgvaz/mantisshrimp.git
RUN pip install -r requirements.txt
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# COPY Important files
COPY mantisshrimp mantisshrimp
COPY examples examples
COPY samples samples
COPY tutorials tutorials

# We need to expose port an run a dummy output with wsgi server
