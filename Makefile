ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: snakemake
.PHONY: render

snakemake: 
	docker run --rm --gpus all -v ${ROOT_DIR}:/app -w /app --entrypoint "" pytorch_gpu:latest snakemake --cores all

render: 
	cd ${ROOT_DIR}/report/; \
	pdflatex  final_report.tex; \
	bibtex final_report; \
	pdflatex  final_report.tex; \
	pdflatex  final_report.tex;
