import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VGGExtractor(nn.Module):
    """VGG extractor for ASR described in
    https://arxiv.org/pdf/1706.02737.pdf."""

    def __init__(self, input_dim):
        super(VGGExtractor, self).__init__()
        self.init_dim = 64
        self.hide_dim = 128
        in_channel, freq_dim, out_dim = self.check_dim(input_dim)
        self.in_channel = in_channel
        self.freq_dim = freq_dim
        self.out_dim = out_dim

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channel, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.init_dim, self.init_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
            nn.Conv2d(self.init_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hide_dim, self.hide_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Half-time dimension
        )

    def check_dim(self, input_dim):
        # Check input dimension, delta feature should be stack over channel.
        if input_dim % 13 == 0:
            # MFCC feature
            return int(input_dim / 13), 13, (13 // 4) * self.hide_dim
        elif input_dim % 40 == 0:
            # Fbank feature
            return int(input_dim / 40), 40, (40 // 4) * self.hide_dim
        else:
            raise ValueError(
                'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '
                + input_dim)

    def view_input(self, feature, feat_len):
        # downsample time
        feat_len = feat_len // 4
        # crop sequence s.t. t%4==0
        if feature.shape[1] % 4 != 0:
            feature = feature[:, :-(feature.shape[1] % 4), :].contiguous()
        bs, ts, ds = feature.shape
        # stack feature according to result of check_dim
        feature = feature.view(bs, ts, self.in_channel, self.freq_dim)
        feature = feature.transpose(1, 2)

        return feature, feat_len

    def forward(self, feature, feat_len):
        # Feature shape BSxTxD -> BS x CH(num of delta) x T x D(acoustic feature dim)
        feature, feat_len = self.view_input(feature, feat_len)
        # Foward
        feature = self.extractor(feature)
        # BSx128xT/4xD/4 -> BSxT/4x128xD/4
        feature = feature.transpose(1, 2)
        #  BS x T/4 x 128 x D/4 -> BS x T/4 x 32D
        feature = feature.contiguous().view(feature.shape[0], feature.shape[1],
                                            self.out_dim)
        return feature, feat_len


class CNNExtractor(nn.Module):
    """A simple 2-layer CNN extractor for acoustic feature down-sampling."""

    def __init__(self, input_dim, out_dim):
        super(CNNExtractor, self).__init__()

        self.out_dim = out_dim
        self.extractor = nn.Sequential(
            nn.Conv1d(input_dim, out_dim, 4, stride=2, padding=1),
            nn.Conv1d(out_dim, out_dim, 4, stride=2, padding=1),
        )

    def forward(self, feature, feat_len):
        # Fixed down-sample ratio
        feat_len = feat_len // 4
        # Channel first
        feature = feature.transpose(1, 2)
        # Foward
        feature = self.extractor(feature)
        # Channel last
        feature = feature.transpose(1, 2)

        return feature, feat_len
