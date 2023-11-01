ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: snakemake
.PHONY: render

snakemake: 
	docker run --rm --gpus all -v ${ROOT_DIR}:/app -w /app --entrypoint "" pytorch_gpu:latest snakemake --cores all

render: 
	cd ${ROOT_DIR}/report; \
	pdflatex -output-directory=.midway_report/ midway_report.tex; \
	mv ${ROOT_DIR}/report/.midway_report/midway_report.pdf ${ROOT_DIR}/report/