# syntax=docker/dockerfile:experimental
FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler \
    libjson-c-dev \
    intltool \
    libx11-dev \
    libxext-dev

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/spiralpp

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n spiralpp python=3.7 \
    numpy \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing

# Activate environment in .bashrc.
RUN echo "conda activate spiralpp" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

# Clone spiralpp.
WORKDIR /src/spiralpp

COPY .git /src/spiralpp/.git

RUN git reset --hard

# install spiral env
RUN git submodule update --init --recursive \
    && wget -c https://github.com/mypaint/mypaint-brushes/archive/v1.3.0.tar.gz -O - | tar -xz -C third_party \
    && git clone https://github.com/dli/paint third_party/paint \
    && patch third_party/paint/shaders/setbristles.frag third_party/paint-setbristles.patch

WORKDIR /src/spiralpp/spiral-envs

RUN pip install --no-cache-dir six scipy

RUN patch setup.py setup.patch
RUN patch CMakeLists.txt cmakelists.patch

RUN pip install -e .

# Install PyTorch.

# Would like to install PyTorch via pip. Unfortunately, there's binary
# incompatability issues (https://github.com/pytorch/pytorch/issues/18128).
# Otherwise, this would work:
# # # Install PyTorch. This needs increased Docker memory.
# # # (https://github.com/pytorch/pytorch/issues/1022)
# # RUN pip download torch
# # RUN pip install torch*.whl
#
WORKDIR /src

RUN git clone --single-branch --branch v1.5.0 --recursive https://github.com/pytorch/pytorch

WORKDIR /src/pytorch

ENV CMAKE_PREFIX_PATH ${CONDA_PREFIX}

RUN python setup.py install

WORKDIR /src

RUN git clone --single-branch --branch v0.5.1 https://github.com/pytorch/vision.git

WORKDIR /src/vision 

RUN python setup.py install

WORKDIR /src/spiralpp

# Collect and install grpc.
RUN conda install protobuf
RUN ./scripts/install_grpc.sh

# Install nest.
RUN pip install nest/

# Install PolyBeast's requirements.
RUN pip install -r requirements.txt

# Compile libtorchbeast.
ENV LD_LIBRARY_PATH ${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN python setup.py install

ENV OMP_NUM_THREADS 1

# Run.
CMD ["bash", "-c", "python -m torchbeast.polybeast \
	--dataset omniglot \
	--total_steps 100000000 \
	--episode_length 40 \
	--use_pressure \
	--batch_size 32"]



# Docker commands:
#   docker rm spiralpp -v
#   docker build -t spiralpp .
#   docker run --name spiralpp spiralpp
# or
#   docker run --name spiralpp -it spiralpp /bin/bash
