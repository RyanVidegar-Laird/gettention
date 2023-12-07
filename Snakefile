rule all: 
	input:
		performer_weights = 'data/pfalciparum/performer_model_weights.pth',
		transformer_weights = 'data/pfalciparum/transformer_model_weights.pth',
		liver = 'data/liver_atlas/info.txt',
		anndata = 'data/liver_atlas/GSE151530.gz.h5ad'

# TODO! add basic arg parsing to train scripts to dedup

rule train_transformer_classifier:
	input: 
		anndata = 'data/pfalciparum/pf10xIDC.gz.h5ad',
		train_script = 'src/python/02-transformer_pfalci.py',
		train_idx = 'data/pfalciparum/train_indices.pkl',
		test_idx = 'data/pfalciparum/test_indices.pkl',
	output:
		model_weights = 'data/pfalciparum/transformer_model_weights.pth',
		train_losses = 'data/pfalciparum/transformer_train_losses.pkl',
		test_losses = 'data/pfalciparum/transformer_test_losses.pkl'
	shell:
		"""
		python3 {input.train_script}
		"""		
		
rule train_performer_classifier:
	input: 
		anndata = 'data/pfalciparum/pf10xIDC.gz.h5ad',
		train_script = 'src/python/01-performer_pfalci.py',
		train_idx = 'data/pfalciparum/train_indices.pkl',
		test_idx = 'data/pfalciparum/test_indices.pkl',
	output:
		model_weights = 'data/pfalciparum/performer_model_weights.pth',
		train_losses = 'data/pfalciparum/performer_train_losses.pkl',
		test_losses = 'data/pfalciparum/performer_test_losses.pkl'
	shell:
		"""
		python3 {input.train_script}
		"""		

rule split_pfalciparum:
	input:
		script = 'src/python/split_pfalci.py',
		anndata = 'data/pfalciparum/pf10xIDC.gz.h5ad'	
	output:		
		train_idx = 'data/pfalciparum/train_indices.pkl',
		test_idx = 'data/pfalciparum/test_indices.pkl',
	shell:
		"""
		python3 {input.script}	
		"""
rule process_liver:
	input:
		script = 'src/python/preprocess_liver.py',
		info = 'data/liver_atlas/info.txt',
		barcodes = 'data/liver_atlas/barcodes.tsv',
		genes = 'data/liver_atlas/genes.tsv',
		mtx = 'data/liver_atlas/matrix.mtx'
	output:
		anndata = 'data/liver_atlas/GSE151530.gz.h5ad'
	shell:
		"""
		python3 {input.script}
		"""
rule fetch_liver:
	params:
		url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151530/suppl'
	output:
		info = 'data/liver_atlas/info.txt',
		barcodes = 'data/liver_atlas/barcodes.tsv',
		genes = 'data/liver_atlas/genes.tsv',
		mtx = 'data/liver_atlas/matrix.mtx'

	shell:
		"""
		mkdir -p data/liver_atlas
		curl -o - {params.url}/GSE151530_Info.txt.gz | gunzip > {output.info}

		curl -o - {params.url}/GSE151530_barcodes.tsv.gz | gunzip > {output.barcodes}

		curl -o - {params.url}/GSE151530_genes.tsv.gz | gunzip > {output.genes}
		
		curl -o - {params.url}/GSE151530_matrix.mtx.gz | gunzip > {output.mtx}
		""" 
		

rule fetch_process_pfalciparum:
	localrule: True
	input:
		preprocess_script = 'src/python/00-preprocess_falciparum.py'
	output:
		raw_counts = 'data/pfalciparum/pf10xIDC_counts.arrow',
		pheno = 'data/pfalciparum/pf10xIDC_pheno.arrow',
		anndata = 'data/pfalciparum/pf10xIDC.gz.h5ad'	
	params:
		MCA_Commit = "4e19a713d0681b118cc7e229133489f039b8766b"
	shell:
		"""
		mkdir -p data/pfalciparum
        curl -SsL https://raw.github.com/vhowick/MalariaCellAtlas/{params.MCA_Commit}/Expression_Matrices/10X/pf10xIDC/pf10xIDC_counts.csv.zip -o data/pfalciparum/pf10xIDC_counts.csv.zip

		curl -SsL https://raw.githubusercontent.com/vhowick/MalariaCellAtlas/{params.MCA_Commit}/Expression_Matrices/10X/pf10xIDC/pf10xIDC_pheno.csv -o data/pfalciparum/pf10xIDC_pheno.csv

		unzip data/pfalciparum/pf10xIDC_counts.csv.zip -x "__MACOSX/*" -d data/pfalciparum/

		rm data/pfalciparum/pf10xIDC_counts.csv.zip
		
		python3 {input.preprocess_script}

		rm data/pfalciparum/pf10xIDC_counts.csv data/pfalciparum/pf10xIDC_pheno.csv
        """
