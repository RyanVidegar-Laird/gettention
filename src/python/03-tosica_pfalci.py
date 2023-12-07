from pathlib import Path

import scanpy as sc
import TOSICA
import pickle

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJ_ROOT = Path()
DATA_PATH = PROJ_ROOT / "data" / "pfalciparum"

logging.info(f"Loading: {DATA_PATH / 'pf10xIDC.gz.h5ad'}")
adata = sc.read_h5ad(filename=DATA_PATH / "pf10xIDC.gz.h5ad")

# Load train/test indices
with (DATA_PATH / "train_indices.pkl").open("rb") as f:
    train_indices = pickle.load(f)

with (DATA_PATH / "test_indices.pkl").open("rb") as f:
    test_indices = pickle.load(f)

TOSICA.train(adata[train_indices], gmt_path = None, label_name='bulk',project=DATA_PATH)