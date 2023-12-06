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
        libfontconfig-dev \
        ninja-build \
        unzip

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ENV POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    # POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 
    # paths
    # this is where our requirements + virtual environment will live
    # PYSETUP_PATH="/opt/pysetup" \
    # VENV_PATH="/opt/pysetup/.venv"


# RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install "poetry==${POETRY_VERSION}"

# prepend poetry and venv to path
# ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN poetry config installer.max-workers 10

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .
COPY README.md .
COPY gettention/ ./gettention/

RUN poetry install

ARG UID=1000
ARG GID=1000

RUN useradd -m -u $UID -U app \
    && chown -R $UID:$GID /app

USER app

EXPOSE 8888
ENTRYPOINT [ "jupyter-lab", "--no-browser", "--allow-root", "--ip=0.0.0.0" ]