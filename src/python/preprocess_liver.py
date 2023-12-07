from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 0

# Setup relative paths
PROJ_PATH = Path()
DATA_PATH = PROJ_PATH / "data" / "liver_atlas"

adata = sc.read_10x_mtx(DATA_PATH, var_names="gene_symbols")

cell_types = pd.read_csv(DATA_PATH / 'info.txt', sep='\t', index_col=2)["Type"]
adata.obs["Celltype"] = cell_types

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

sc.pp.normalize_total(adata, target_sum=None, exclude_highly_expressed=True, inplace=True)
sc.pp.log1p(adata)

adata.write_h5ad(DATA_PATH / "GSE151530.gz.h5ad", compression="gzip")