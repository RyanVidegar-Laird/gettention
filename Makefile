ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: snakemake

snakemake: 
	docker run --rm --gpus all -v ${ROOT_DIR}:/app -w /app --entrypoint "" pytorch_gpu:latest snakemake --cores all
