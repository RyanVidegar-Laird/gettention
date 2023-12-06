from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset

import scanpy as sc
from gettention.data import SCDataset
from gettention.model import PerformerClassifier

import numpy as np
from sklearn.model_selection import train_test_split

import pickle
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Data
# ----------------------------------------------------

PROJ_ROOT = Path()
DATA_PATH = PROJ_ROOT / "data" / "pfalciparum"

logging.info(f"Loading: {DATA_PATH / 'pf10xIDC.gz.h5ad'}")
adata = sc.read_h5ad(filename=DATA_PATH / "pf10xIDC.gz.h5ad")

sc_data = SCDataset(adata, "bulk", "log_trans", device)

# Training
# -------------------------------------

N_CELLS, M_GENES = sc_data.x.shape
K_CLASSES = sc_data.labels.shape[1]

model = PerformerClassifier(N_CELLS, M_GENES, K_CLASSES).to(device)

N_EPOCHS = 50
BATCH_SIZE = 8

TEST_SIZE = 0.3
SEED = 424242

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

# generate subset based on indices
train_split = Subset(sc_data, train_indices)
test_split = Subset(sc_data, test_indices)

# create batched loaders
train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)

BATCHES_PER_EPOCH = len(train_indices) // BATCH_SIZE
LOSS_CUTOFF = 1e-8
LEARNING_RATE = 1e-3

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

logging.info(f"Training PerformerClassifier for max {N_EPOCHS} epochs")

epoch_mean_train_loss = []
epoch_mean_test_loss = []
best_test_loss = 1e6

logging.info(f"Batch size; Batches/epoch: {BATCH_SIZE}; {BATCHES_PER_EPOCH}")

for epoch in range(N_EPOCHS):
    # if epoch > 1:
    #     loss_diff = epoch_mean_test_loss[-2] - epoch_mean_test_loss[-1]
    #     if loss_diff < LOSS_CUTOFF:
    #         logging.info(f"Met minimum loss diff cutoff at epoch: {epoch - 1}")
    #         break

    train_loss = []
    test_loss = []

    logging.info(f"Epoch: {epoch}")

    start = time.time()

    model.train()
    for i in range(BATCHES_PER_EPOCH):
        # take a batch
        X_batch, y_batch = next(iter(train_loader))

        # forward pass
        with autocast(device_type=device.type):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.float())

        # backward pass
        scaler.scale(loss).backward()
        train_loss.append(loss.detach().clone().cpu().numpy())

        # update weights
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    end = time.time()
    batches_per_sec = BATCHES_PER_EPOCH / (end - start)
    logging.info(f"\tMean batches/s: {batches_per_sec:.3f}")

    # evail test loss
    with torch.no_grad():
        for test_vals, test_labs in iter(test_loader):
            test_pred = model(test_vals)
            tloss = loss_fn(test_pred, test_labs.float())
            test_loss.append(tloss.cpu().numpy())

    mean_test_loss = np.mean(test_loss)
    epoch_mean_test_loss.append(mean_test_loss)

    # save mean train loss within epoch
    mean_train_loss = np.mean(train_loss)
    epoch_mean_train_loss.append(mean_train_loss)

    logging.info(
        f"\tMean train--test loss: {mean_train_loss:.4f}--{mean_test_loss:.4f}"
    )

    if mean_test_loss < best_test_loss:
        best_test_loss = mean_test_loss
        MODEL_PATH = DATA_PATH / "performer_model_weights.pth"
        logging.info("\t\tSaving new best model")
        torch.save(model.state_dict(), MODEL_PATH)

logging.info(f"Saving train losses to {DATA_PATH / 'performer_train_losses.pkl'}")
with (DATA_PATH / "performer_train_losses.pkl").open("wb") as f:
    pickle.dump(epoch_mean_train_loss, f)

logging.info(f"Saving test losses to {DATA_PATH / 'performer_test_losses.pkl'}")
with (DATA_PATH / "performer_test_losses.pkl").open("wb") as f:
    pickle.dump(epoch_mean_test_loss, f)
