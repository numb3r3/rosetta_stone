import math
from typing import Dict

# from rosetta.modules.nn import LayerNorm
from rosetta.modules.positional_embedding import SinusoidalPositionalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.decoder import TransformerDecoder
from .models.embedding import CNNExtractor, VGGExtractor
from .models.encoder import TransformerEncoder


class Speech2TextModel(nn.Module):

    def __init__(self,
                 feature_dim: int,
                 vocab_size: int,
                 encoder_conf: Dict,
                 decoder_conf: Dict,
                 attention_conf: Dict,
                 dropout_rate: float = 0.1,
                 prenet_name: str = None,
                 **kwargs):
        super().__init__()

        self.encoder_dim = encoder_conf['d_model']
        self.decoder_dim = decoder_conf['d_model']

        self.prenet_name = prenet_name

        self.prenet = None
        # 4x reduction on time feature extraction
        if prenet_name == 'vgg':
            # TODO: fix the output dim issue
            self.prenet = VGGExtractor(feature_dim)
        elif prenet_name == 'cnn':
            self.prenet = CNNExtractor(feature_dim, self.encoder_dim)

        self.position_encoder = SinusoidalPositionalEncoder(self.encoder_dim)
        self.pos_dropout = nn.Dropout(p=dropout_rate)

        self.transformer_encoder = TransformerEncoder(**encoder_conf)
        self.transformer_decoder = TransformerDecoder(**decoder_conf)

        self.output_layer = nn.Linear(self.decoder_dim, vocab_size)

    def forward(self, audio_frame_feats, audio_frame_length, text_input_ids,
                text_input_length, text_input_masks, **kwargs):

        # down-sampling step
        if self.prenet:
            audio_frame_feats, _ = self.prenet(audio_frame_feats,
                                               audio_frame_feats.size(1))

        # add positional embedding

        # inspired from "Attention is All You Need"
        # (https://arxiv.org/abs/1706.03762)
        embedding_scale = torch.sqrt(torch.FloatTensor([self.encoder_dim])).to(
            audio_frame_feats.device)

        audio_frame_feats = audio_frame_feats * embedding_scale

        audio_frame_feats += self.position_encoder(audio_frame_feats)
        audio_frame_feats = self.pos_dropout(audio_frame_feats)

        # TODO: add frame masking after downsampling

        # apply transformer encoder
        audio_frame_encoding = self.transformer_encoder(audio_frame_feats)

        loss = None
        predicts = {}
        metrics = {}
        return predicts, loss, metrics
