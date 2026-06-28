FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-setuptools \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m ensurepip --upgrade && \
    python3.11 -m pip install --upgrade pip setuptools && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
RUN python3.11 -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python3.11 -m pip install dominate>=2.8.0 Pillow>=10.0.0 wandb>=0.16.0 visdom>=0.1.8.8

COPY . .
