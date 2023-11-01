ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: snakemake
.PHONY: render

snakemake: 
	docker run --rm --gpus all -v ${ROOT_DIR}:/app -w /app --entrypoint "" pytorch_gpu:latest snakemake --cores all

render: 
	cd ${ROOT_DIR}/report/mid; \
	pdflatex  midway_report.tex; \
	bibtex midway_report; \
	pdflatex  midway_report.tex; \
	pdflatex  midway_report.tex;
