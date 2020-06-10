import os

import pytest
from rosetta.modules.position_embedding import SinusoidalPositionalEncoder
import torch


def test_sinusoidal_embedding():
    encoder = SinusoidalPositionalEncoder(12)

    x = torch.randn(6, 4)

    # position_ids = torch.arange(input_shape[1], dtype=torch.long)
    # position_ids = position_ids.unsqueeze(0).expand(input_shape)

    embeds = encoder(x)
    
