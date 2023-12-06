import torch
import torch.nn as nn
from torch.amp import autocast

from performer_pytorch import Performer


# Performer
class PerformerClassifier(nn.Module):
    """
    [TODO] Document: Cell classifier using Performer...
    """

    # being very explicit in dims for sake of learning/consistency
    def __init__(
        self,
        n_cells,
        m_genes,
        k_classes,
        d_emb=64,
        enc_depth=4,
        enc_heads=2,
        enc_dim_head=32,
        ff_dropout=0.1,
        attn_dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # NOTE! (d_emb - 1) is used as 1d gene counts will be appended during forward pass
        self.emb = torch.nn.Embedding(m_genes, d_emb - 1)

        self.device_type = device.type

        self.performer_encoder = Performer(
            dim=d_emb,
            depth=enc_depth,
            heads=enc_heads,
            dim_head=enc_dim_head,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            causal=False,
            reversible=False,
            shift_tokens=False,
        )

        self.classifier = nn.Linear(d_emb, k_classes)

    def forward(self, x):
        # x is a batch_size * m_genes * 1 count tensor from the data loadter
        # concat each gene count values to the end of the embedding ---
        # --> N_cells x M_genes x D_embedding (+1) tensor
        batch_size = x.shape[0]
        emb = self.emb.weight.expand(batch_size, -1, -1)
        x = torch.concat((emb, x), 2)

        with autocast(enabled=False, device_type=self.device_type):
            # known issue with performer encoder and autocast that I haven't debugged yet
            # https://github.com/openai/triton/issues/217
            # probably need to update performer dep fast-transformers
            # it would speed up performance by ~
            x = self.performer_encoder(x)

        # take the mean of the encoding --> N x D tensor for classification
        # [TODO] find a more interesting approach
        # (consider multiple hidden layers with non-linear activations)
        x = x.mean(dim=1)

        # use linear classifier to predict N x D --> N x K_classes
        x = self.classifier(torch.squeeze(x, 0))
        return x
