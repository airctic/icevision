FROM pytorch/pytorch:latest

ARG BUILD="dev"

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
    # config miniconda
    opt/conda/bin/conda config --set always_yes yes && \
    opt/conda/bin/conda update -q conda && \
    pip install -U pip wheel setuptools

# set conda path
ENV PATH /opt/conda/bin:$PATH

RUN if [ $BUILD == "prod" ] ; then \
        echo "Production Build" && pip install icevision[all] icedata \
    fi

RUN if [ $BUILD == "dev" ] ; then \
        git clone https://github.com/airctic/icevision && \
        git clone https://github.com/airctic/icedata \
        echo "Development Build" && \
        cd icevision && pip install ".[all]" && \
        cd .. && cd icedata && pip install . \
    fi

RUN conda install -c conda-forge notebook && \
    conda clean -ya && \
    echo '#!/bin/bash\njupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser' >> run_jupyter.sh

RUN chmod u+x run_jupyter.sh && \
    conda init bash && conda env list

CMD ["/bin/bash"]
