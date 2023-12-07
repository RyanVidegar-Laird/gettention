from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import scanpy as sc
from gettention.data import SCDataset
from sklearn.model_selection import train_test_split
import pickle
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJ_ROOT = Path()
DATA_PATH = PROJ_ROOT / "data" / "pfalciparum"

logging.info(f"Loading: {DATA_PATH / 'pf10xIDC.gz.h5ad'}")
adata = sc.read_h5ad(filename=DATA_PATH / "pf10xIDC.gz.h5ad")

sc_data = SCDataset(adata, "bulk", "log_trans", device)

TEST_SIZE = 0.3
SEED = 42

# https://stackoverflow.com/a/68338670
#   generate indices: instead of the actual data, pass in integers instead
train_indices, test_indices, _, _ = train_test_split(
    range(len(sc_data)),
    sc_data.labels.cpu(),
    stratify=sc_data.labels.cpu(),
    test_size=TEST_SIZE,
    random_state=SEED,
)

# save train/test indices for comparing same train/test sets between methods
idx_paths = [DATA_PATH / "train_indices.pkl", DATA_PATH / "test_indices.pkl"]

for p, loader in dict(zip(idx_paths, [train_indices, test_indices])).items():
    with p.open("wb") as f:
        pickle.dump(loader, f)