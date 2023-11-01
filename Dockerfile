ARG CUDA_TAG=12.2.2-base-ubuntu22.04

FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        curl \ 
        git \
        python3-pip \
        python3-dev \
        python3-opencv \ 
        libfontconfig-dev 

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

ARG UID=1000
ARG GID=1000

RUN useradd -m -u $UID -U app \
    && chown -R $UID:$GID /app

USER app

EXPOSE 8888
ENTRYPOINT [ "jupyter-lab", "--no-browser", "--allow-root", "--ip=0.0.0.0" ]