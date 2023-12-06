from scanpy import AnnData

import torch
from torch.utils.data import Dataset


class SCDataset(Dataset):
    """
    andata: object from `scanpy`
    label_col: andata cell obs column of annotated cell labels
    layer: which transformed expression layer to use in andata
    device: which gpu/cpu to use, from `torch.device`
    """

    def __init__(
        self, adata: AnnData, label_col: str, layer: str, device: torch.device
    ):
        super().__init__()

        # get desired labels for all cells
        cell_types = set(adata.obs[label_col])

        # assign each cell label to an int
        self.class_map = {
            key: value for key, value in zip(cell_types, range(0, len(cell_types)))
        }

        # transform int cell labels to one-hot tensors
        self.intlabels = torch.LongTensor(
            [self.class_map[i] for i in adata.obs.bulk.values]
        )
        self.labels = torch.nn.functional.one_hot(self.intlabels).to(device)

        # messy class mapping of one_hot tensor --> class name
        class_map = {}
        for key, value in self.class_map.items():
            ohc = torch.nn.functional.one_hot(torch.tensor([value]), len(cell_types))
            ohc = torch.squeeze(ohc)
            ohc = tuple(ohc.tolist())
            class_map[ohc] = key

        # store class_map for later use
        self.class_map = class_map

        # extract gene expression matrix (N cells x M genes)
        self.x = torch.tensor(adata.layers[layer]).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        x = self.x[idx]

        # returns a M_genes x 1 cell tensor for easier transform with expression embedding
        return x[None, :].T, label

    def label_name(self, label: torch.Tensor) -> str:
        """
        label: one-hot encoded cell label

        Returns:
            Simple helper to convert one-hot label tensor to cell label name
        """

        return self.class_map[tuple(label.tolist())]
