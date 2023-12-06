rule all:
	input:
		'data/pfalciparum/performer_model_weights.pth'
		
rule train_performer_classifier:
	input: 
		anndata = 'data/pfalciparum/pf10xIDC.gz.h5ad',
		train_script = 'src/python/01-performer_pfalci.py'
	
	output:
		train_idx = 'data/pfalciparum/train_indices.pkl',
		ttest_idx = 'data/pfalciparum/test_indices.pkl',
		model_weights = 'data/pfalciparum/performer_model_weights.pth',
		train_losses = 'data/pfalciparum/performer_train_losses.pkl',
		test_losses = 'data/pfalciparum/performer_test_losses.pkl'
	shell:
		"""
		python {input.train_script}
		"""		

rule fetch_process_pfaciparum:
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
        curl -SsL https://raw.github.com/vhowick/MalariaCellAtlas/{params.MCA_Commit}/Expression_Matrices/10X/pf10xIDC/pf10xIDC_counts.csv.zip -o data/pfalciparum/pf10xIDC_counts.csv.zip

		curl -SsL https://raw.githubusercontent.com/vhowick/MalariaCellAtlas/{params.MCA_Commit}/Expression_Matrices/10X/pf10xIDC/pf10xIDC_pheno.csv -o data/pfalciparum/pf10xIDC_pheno.csv

		unzip data/pfalciparum/pf10xIDC_counts.csv.zip -x "__MACOSX/*" -d data/pfalciparum/

		rm data/pfalciparum/pf10xIDC_counts.csv.zip
		
		python3 {input.preprocess_script}

		rm data/pfalciparum/pf10xIDC_counts.csv data/pfalciparum/pf10xIDC_pheno.csv
        """
