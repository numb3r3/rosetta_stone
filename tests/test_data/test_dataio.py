import os

import pytest
from rosetta.core.dataio import BaseDataIO
import torch
from torch.utils.data.dataset import Dataset


class DumyDataset(Dataset):
    def __init__(self, size=100):
        self.features = [{"x": torch.tensor([1] * 5)}] * size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class DumyDataIO(BaseDataIO):
    def __init__(self):
        super().__init__()

    def create_dataset(self, data_path: str, mode: str = "train", **kwargs):
        dt = DumyDataset()
        return dt


dirname = os.path.dirname(__file__)


def test_dataio():

    dataio = DumyDataIO()
    loader = dataio.create_data_loader(data_path=None, batch_size=5)
    sampled_data = list(iter(loader))
    assert len(sampled_data) == 20


def test_masked_text_dataio():
    from rosetta.datasets.text.masked_text import MaskedTextDataIO

    dataio = MaskedTextDataIO(tokenizer_name="bert-base-cased")
    loader = dataio.create_data_loader(
        data_path=os.path.join(dirname, "sonnets.txt"), batch_size=5
    )


def test_aishell_dataio():
    from rosetta.datasets.speech.aishell import AiShellDataIO

    config = {
        "feat_type": "fbank",
        "feat_dim": 40,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0,
        "apply_cmvn": True,
        "delta_order": 2,
        "delta_window_size": 2,
    }
    kwargs = {}
    kwargs["audio_config"] = config
    
    # kwargs[
    #     "data_path"
    # ] = "/workspace/project-nas-10251-sz/open_speech_data/Aishell/data_aishell"

    dataio = AiShellDataIO(tokenizer_name_or_path="bert-base-chinese", **kwargs)
    # loader = dataio.create_data_loader(data_path=None, batch_size=5)
