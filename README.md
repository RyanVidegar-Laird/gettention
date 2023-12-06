# Intro
Course project for UNC COMP755, 2023.

# Install

## venv
Requirements:
 - Python >=3.10, < 3.13
 - Poetry >= 1.5.1

```
git clone https://github.com/RyanVidegar-Laird/gettention.git
cd gettention
python -m venv .venv
source .venv/bin/activate

# default includes dev deps and extras (snakemake)
poetry install
```

## Docker

Requirements: 
 - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```
docker build -t gettention .
docker run --rm -it --gpus all -v $PWD:/app gettention
```
