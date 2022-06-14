FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer "Maksym Andriushchenko <maksym.andriushchenko@epfl.ch>"

ARG DEBIAN_FRONTEND=noninteractive  # needed to prevent some questions during the Docker building phase

# install some necessary tools
RUN apt-get update 
RUN apt-get install -y \
    cmake \
    curl \
    htop \
    locales \
    python3 \
    python3-pip \
    sudo \
    unzip \
    vim \
    git \
    wget \
    zsh \
    libssl-dev \
    libffi-dev \
    libmagickwand-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    openssh-server
RUN rm -rf /var/lib/apt/lists/*
RUN mkdir /var/run/sshd


# configure environments
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


RUN pip3 install --upgrade pip  # needed for opencv
RUN pip3 install -U setuptools  # may be needed for opencv
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html  # needed for A100 pods


# python packages
RUN pip3 install --upgrade \
    scipy \
    numpy \
    jupyter notebook \
    ipdb \
    pyyaml \
    easydict \
    requests \
    matplotlib \
    seaborn
RUN export LC_ALL=en_US.UTF-8

# Configure user and group
ENV SHELL=/bin/bash

ENV HOME=/home/$NB_USER

RUN ln -s /usr/bin/python3 /usr/bin/python

# expose the port to ssh
EXPOSE 22  
CMD ["/usr/sbin/sshd", "-D"]
