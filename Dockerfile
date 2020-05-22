#!/bin/sh

FROM ubuntu:18.04
FROM nvidia/cuda:latest

LABEL maintainer="Nam Nguyen Duc <namduc2308@gmail.com>"

COPY ["requirements.txt", "/root/requirements.txt"]

RUN apt-get update
RUN apt-get install -y \
  sudo vim man less git wget bzip2 build-essential \
  ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
  ffmpeg

# Install miniconda to /miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda update -y conda
RUN conda --version

RUN pip install -r /root/requirements.txt

COPY ["torch", "/usr/src/app/torch"]

WORKDIR /usr/src/app/torch



