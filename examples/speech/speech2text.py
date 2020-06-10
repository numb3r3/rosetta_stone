import math
from typing import Dict

from rosetta.modules.nn import LayerNorm
from rosetta.modules.positional_embedding import SinusoidalPositionalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.decoder import TransformerDecoder
from .models.embedding import CNNExtractor, VGGExtractor
from .models.encoder import TransformerEncoder


class Speech2TextModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_conf: Dict,
        decoder_conf: Dict,
        attention_conf: Dict,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # self.prenet = VGGExtractor(input_dim)
        self.prenet = CNNExtractor(input_dim, encoder_conf["d_model"])

        self.position_encoder = SinusoidalPositionalEncoder(encoder_conf["d_model"])
        self.pos_dropout = nn.Dropout(p=dropout_rate)

        self.encoder = TransformerEncoder(**encoder_conf)
        self.decoder = TransformerDecoder(**decoder_conf)

    def forward(
        self,
        audio_frame_feats,
        audio_frame_length,
        text_input_ids,
        text_input_length,
        text_input_masks,
        **kwargs
    ):
        print(audio_frame_feats.shape)
        print(audio_frame_length)

        # down-sampling step
        audio_frame_feats, _ = self.prenet(audio_frame_feats, audio_frame_feats.size(1))

        # add positional embedding
        audio_frame_feats += self.position_encoder(audio_frame_feats)
        audio_frame_feats = self.pos_dropout(audio_frame_feats)

        # TODO: add frame masking after downsampling

        # apply transformer encoder
        audio_frame_encoding = self.encoder(audio_frame_feats)

        loss = None
        predicts = {}
        metrics = {}
        return predicts, loss, metrics
