FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8

ARG APT_INSTALL="apt-get install -y --no-install-recommends" 
ARG PIP_INSTALL="python -m pip --no-cache-dir install --user"
ARG GIT_CLONE="git clone --depth 10"

RUN    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update 

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive ${APT_INSTALL} \
        build-essential \
        apt-utils \
        libsm6 libxext6 libxrender-dev \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        ffmpeg \
        && \
    ${GIT_CLONE} https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install 

# ==================================================================
# python
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive ${APT_INSTALL} \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive ${APT_INSTALL} \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python 

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*



# ==================================================================
# Add non-priviledged user
# ------------------------------------------------------------------

ARG UID=1000
RUN useradd -m -l -r -u $UID deep-dance -g sudo 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $UID
WORKDIR /home/deep-dance
ENV PATH="/home/deep-dance/.local/bin:${PATH}"

RUN wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    rm get-pip.py 

# ==================================================================
# copy requirements and install python packages
# ------------------------------------------------------------------


COPY requirements.txt /home/deep-dance 

RUN ${PIP_INSTALL} --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html

# requierements from file and detectron
RUN ${PIP_INSTALL} -r requirements.txt \
        && \
# we install precompiled version because cuda has to be available during compilation which it isn't while building the docker container
# which makes things more complicated otherwise. -> see Dockerfiles in official 
# Detectron2 repository
    ${PIP_INSTALL} detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html 

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
