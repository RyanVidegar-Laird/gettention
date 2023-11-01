from pathlib import Path
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

# Setup relative paths
PROJ_PATH = Path()
DATA_PATH = PROJ_PATH / 'data' / 'pfalciparum'

path_raw_counts = DATA_PATH /'pf10xIDC_counts.csv'
path_pheno = DATA_PATH / 'pf10xIDC_pheno.csv'

# Read CSVs downloaded from Github, transform into format expected by `scanpy.AnnData`
df_raw_counts = pd.read_csv(path_raw_counts, index_col=0).transpose()
df_raw_counts.index.names = ['Cells']
df_raw_counts.columns.names = ['Genes']

df_pheno = pd.read_csv(path_pheno, index_col = 0)
df_pheno.index.names = ['Cells']

# Create AnnData object. Add cell phenotype info for later convenience
adata = sc.AnnData(df_raw_counts)
adata.obs = adata.obs.merge(df_pheno, left_index=True, right_index=True)

# Make sparse
adata.X = csr_matrix(adata.X)

# Normalize gene counts per million per cell
# New layers are created within the object in case prior data is needed
adata.layers['counts_per_million'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=10e6, layer="counts_per_million")

# Log transform: ln(X + 1)
adata.layers['log_norm'] = adata.layers['counts_per_million'].copy()
sc.pp.log1p(adata, layer='counts_per_million')

# Save raw to disk. Keep intermediate dfs in case needed later (csvs are deleted)
df_raw_counts.to_feather(DATA_PATH / 'pf10xIDC_counts.arrow')
df_pheno.to_feather(DATA_PATH /'pf10xIDC_pheno.arrow')

# Save AnnData object as .h5ad (basically a hdf5 with some addtl. metadata for Scanpy & other libs
adata.write_h5ad(DATA_PATH / "pf10xIDC.gz.h5ad", compression='gzip')