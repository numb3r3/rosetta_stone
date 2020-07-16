import math

from rosetta.modules.positional_embedding import SinusoidalPositionalEncoder
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder as _TransformerEncoder
from torch.nn import TransformerEncoderLayer as _TransformerEncoderLayer
import torch.nn.functional as F


class TransformerEncoder(nn.Module):

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 **kwargs):
        super().__init__()

        self.position_encoder = SinusoidalPositionalEncoder(d_model)
        self.pos_dropout = nn.Dropout(p=0.1)

        encoder_layer = _TransformerEncoderLayer(d_model, nhead,
                                                 dim_feedforward, dropout,
                                                 activation)
        encoder_norm = nn.LayerNorm(d_model)

        self.transformer_encoder = _TransformerEncoder(encoder_layer,
                                                       num_layers,
                                                       encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        memory = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory
