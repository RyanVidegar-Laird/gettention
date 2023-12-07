import torch
from torch import nn, Tensor
from torch.amp import autocast
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from performer_pytorch import Performer


# Performer
class PerformerClassifier(nn.Module):
    """
    Performer Classifier
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


class TransformerClassifier(nn.Module):
    """
    Transformer Classifier
    """

    def __init__(
        self,
        # number of classes in the model
        num_classes: int,
        # ntoken: number of expected genes
        ntoken: int,
        # d_model: dimension of encoder (choose)
        d_model: int,
        # number of heads in ``nn.MultiheadAttention``
        nhead: int,
        # d_hid: dimension of the feedforward network model, default is 200
        d_hid: int,
        # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
        nlayers: int,
        learning_rate: int,
        # dropout: dropout value, default is 0.1, but set to 0.5
        dropout: float = 0.5,
    ):
        super().__init__()
        self.emb = torch.nn.Embedding(ntoken, d_model - 1)
        self.d_emb = torch.tensor([d_model])

        # unlike TOSICA, do not use positional encoding
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # TransformerEncoder is a stack of N encoder layers (6 in TOSICA)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # applies a linear transofmration to the data
        self.linear = nn.Linear(d_model, num_classes)
        self.d_model = d_model
        self.lr = learning_rate

        # Saving attention layer
        # Adapted from: https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91?permalink_comment_id=4407423#gistcomment-4407423
        # very memory intensive...
        # Modify the self-attention block of the last layer to capture attention weight
        # self.transformer_encoder.eval()

        # Initialize a list to store attention from the last layer
        self.last_layer_attention = None

        # Register a hook to the last layer's self attention module
        def save_last_layer_attention(module, input, output):
            # Save only the attention weights, which are typically the second output from MultiheadAttention
            self.last_layer_attention = output[1]

        # save_output = SaveOutput()
        patch_attention(self.transformer_encoder.layers[-1].self_attn)
        hook_handle = self.transformer_encoder.layers[
            -1
        ].self_attn.register_forward_hook(save_last_layer_attention)

    def get_last_layer_attention(self):
        # Retrieve the stored attention weights
        return self.last_layer_attention

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_cells, N_genes, 1_count]``

        Returns:
            output Tensor of shape 1xK class predictions
        """
        # x is a batch_size * m_genes * 1 count tensor from the data loadter
        # concat each gene count values to the end of the embedding ---
        # --> N_cells x M_genes x D_embedding (+1) tensor
        batch_size = x.shape[0]
        emb = self.emb.weight.expand(batch_size, -1, -1)
        x = torch.concat((emb, x), 2)

        # the transformer encoder layer
        output = self.transformer_encoder(x)

        output = output.mean(dim=1)

        # the linear layer
        output = self.linear(torch.squeeze(output, 0))
        return output


def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap
