from pathlib import Path

import scanpy as sc
from sklearn.model_selection import train_test_split
import pickle
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJ_ROOT = Path()
DATA_PATH = PROJ_ROOT / "data" / "liver_atlas"
H5AD_PATH = DATA_PATH / 'GSE151530.gz.h5ad'

logging.info(f"Loading: {H5AD_PATH}")
adata = sc.read_h5ad(filename=H5AD_PATH)


TEST_SIZE = 0.3
SEED = 42

# https://stackoverflow.com/a/68338670
#   generate indices: instead of the actual data, pass in integers instead
train_indices, test_indices, _, _ = train_test_split(
    range(len(adata)),
    adata.obs.Celltype.values,
    stratify=adata.obs.Celltype.values,
    test_size=TEST_SIZE,
    random_state=SEED,
)

# save train/test indices for comparing same train/test sets between methods
idx_paths = [DATA_PATH / "train_indices.pkl", DATA_PATH / "test_indices.pkl"]

for p, loader in dict(zip(idx_paths, [train_indices, test_indices])).items():
    with p.open("wb") as f:
        pickle.dump(loader, f)