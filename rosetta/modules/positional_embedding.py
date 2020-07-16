"""Positonal Encoding Module."""

import math

import torch
import torch.nn as nn


class LearnedPositionalEncoder(nn.Module):
    """This module produces LearnedPositionalEmbedding."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, input, offset=0):
        """Input is expected to be of size [bsz x seq_len]."""
        input_shape = input.size()
        bsz, seq_len = input_shape[0], input_shape[1]
        positions = offset + torch.arange(seq_len)
        res = self.embedding(positions).unsqueeze(1).expand(-1, bsz, -1).transpose(0, 1)
        return res


class SinusoidalPositionalEncoder(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.embedding_dim = d_model
        pe = SinusoidalPositionalEncoder.get_embedding(max_len, d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input, offset=0):
        """Input is expected to be of size [bsz x seq_len]."""
        input_shape = input.size()
        bsz, seq_len = input_shape[0], input_shape[1]
        mx_position = seq_len + offset
        if self.pe is None or mx_position > self.pe.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEncoder.get_embedding(
                mx_position, self.embedding_dim
            )

        positions = offset + torch.arange(seq_len, device=input.device)
        res = self.pe.index_select(0, positions)
        res = res.unsqueeze(1)
        res = res.expand(-1, bsz, -1).transpose(0, 1)
        return res
