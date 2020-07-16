import math

from rosetta.modules.positional_embedding import SinusoidalPositionalEncoder
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder as _TransformerDecoder
from torch.nn import TransformerDecoderLayer as _TransformerDecoderLayer
import torch.nn.functional as F


class TransformerDecoder(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 **kwargs):
        super().__init__()

        decoder_layer = _TransformerDecoderLayer(d_model, nhead,
                                                 dim_feedforward, dropout,
                                                 activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = _TransformerDecoder(decoder_layer,
                                                       num_layers,
                                                       decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
