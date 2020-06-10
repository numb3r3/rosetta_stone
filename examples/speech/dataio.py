import json
import os
import sys
from typing import Dict

from rosetta.core.dataio import BaseDataIO
from rosetta.data.tokenization import Tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchaudio


def load_audio(path, sample_rate: int = 16000):
    waveform, _ = torchaudio.load(path, normalization=True)
    waveform = torchaudio.transforms.Resample(orig_freq=_, new_freq=sample_rate)(
        waveform
    )
    # waveform = waveform.numpy().T
    # if len(waveform.shape) > 1:
    #     if waveform.shape[1] == 1:
    #         waveform = waveform.squeeze()
    #     else:
    #         waveform = waveform.mean(axis=1)  # multiple channels, average
    return waveform, sample_rate


def calc_mean_invstddev(feature):
    if len(feature.shape) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = torch.mean(feature, dim=0)
    var = torch.var(feature, dim=0)
    # avoid division by ~zero
    if (var < sys.float_info.epsilon).any():
        return mean, 1.0 / (torch.sqrt(var) + sys.float_info.epsilon)
    return mean, 1.0 / torch.sqrt(var)


def calcMN(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


class SpeechDataset(Dataset):
    """ Loads and processes the audio dataset for easier training. """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 64,
        audio_conf: Dict = None,
        # feat_type: str = "fbank",
        # sample_rate: int = 16000,
        # frame_length: float = 25.0,
        # frame_shift: float = 10.0,
        # num_mel_bins: int = 40,
        # delta_order: int = 0,
        # delta_window_size: int = 2,
        # apply_cmvn: bool = True,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self._manifests = []
        with open(manifest_path, "r", encoding="utf-8") as fin:
            for line in fin:
                manifest = json.loads(line)
                self._manifests.append(manifest)

        # audio config setting
        self.feat_type = audio_conf["feat_type"]
        self.sample_rate = audio_conf["sample_rate"]
        self.frame_length = audio_conf["frame_length"]
        self.frame_shift = audio_conf["frame_shift"]
        self.num_mel_bins = audio_conf["num_mel_bins"]
        self.delta_order = audio_conf["delta_order"]
        self.delta_window_size = audio_conf["delta_window_size"]
        self.apply_cmvn = audio_conf["apply_cmvn"]

        self.audio_feature_fn = (
            torchaudio.compliance.kaldi.fbank
            if self.feat_type == "fbank"
            else torchaudio.compliance.kaldi.mfcc
        )
        self.max_seq_len = max_seq_len

        self.kwargs = kwargs

    def __getitem__(self, index):
        manifest = self._manifests[index]

        audio_path = manifest["audio_filepath"]
        if not os.path.exists(audio_path):
            raise FileNotFoundError("Audio file not found: {}".format(audio_path))

        # load and resample audio signals
        waveform, sample_rate = load_audio(audio_path, self.sample_rate)

        # The following steps are adapted from:
        # https://zhuanlan.zhihu.com/p/126201460

        # 1. extract fbank or mfcc
        audio_frame_feats = self.audio_feature_fn(
            waveform,
            channel=-1,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            num_mel_bins=self.num_mel_bins,
        )

        # 2. add delta
        _delta = audio_frame_feats
        deltas = []
        if self.delta_order > 0:
            for _ in range(self.delta_order):
                _delta = torchaudio.functional.compute_deltas(
                    _delta, win_length=self.delta_window_size
                )
                deltas.append(_delta)

            audio_frame_feats = torch.cat([audio_frame_feats] + deltas, dim=-1)

        # 3. apply cmvn
        if self.apply_cmvn:
            audio_frame_feats = calcMN(audio_frame_feats.cpu().detach())

        audio_frame_length = torch.LongTensor([audio_frame_feats.size(0)])
        # audio_frame_length = audio_frame_feats.size(0)

        text = manifest["text"]

        # TODO: preprocessing text, to remove punctuation

        text_tokenized_dict = self.tokenizer.encode_plus(
            text,
            text_pair=None,
            add_special_tokens=False,
            max_length=self.max_seq_len,
            pad_to_max_length=False,
        )

        text_input_ids, text_input_masks = (
            text_tokenized_dict["input_ids"],
            text_tokenized_dict["attention_mask"],
        )

        text_input_ids += [self.tokenizer.eos_token_idx]
        text_input_length = len(text_input_ids)
        text_input_masks += [1]

        text_input_ids = torch.LongTensor(text_input_ids)
        text_input_length = torch.LongTensor([text_input_length])
        text_input_masks = torch.LongTensor(text_input_masks)
        return (
            audio_frame_feats,
            audio_frame_length,
            text_input_ids,
            text_input_length,
            text_input_masks,
        )

        # sample = {"net_input": {"src_tokens": source, "src_lengths": frames_lengths}}

    def __len__(self):
        return len(self._manifests)


class SpeechDataIO(BaseDataIO):
    def __init__(self, tokenizer_name_or_path: str = "bert-base-chinese", **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = Tokenizer.load(tokenizer_name_or_path, do_lower_case=True)
        self.tokenizer.eos_token = "<EOS>"
        self.tokenizer.eos_token_idx = 1

    def create_dataset(self, data_path: str, mode: str = "train", **kwargs) -> Dataset:
        return SpeechDataset(data_path, self.tokenizer, **kwargs)

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Dict[str, torch.Tensor]:

        frame_feats, frame_length = [], []
        text_input_ids, text_input_length, text_input_masks = [], [], []

        with torch.no_grad():
            for example in batch:
                frame_feats.append(example[0])
                frame_length.append(example[1])

                text_input_ids.append(example[2])
                text_input_length.append(example[3])
                text_input_masks.append(example[4])

        # Zero-padding
        audio_frame_feats = pad_sequence(frame_feats, batch_first=True)
        audio_frame_length = torch.LongTensor(frame_length)
        text_input_ids = pad_sequence(text_input_ids, batch_first=True)
        text_input_length = torch.LongTensor(text_input_length)
        text_input_masks = pad_sequence(text_input_masks, batch_first=True)

        return (
            audio_frame_feats,
            audio_frame_length,
            text_input_ids,
            text_input_length,
            text_input_masks,
        )
