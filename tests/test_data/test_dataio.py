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

    def create_dataset(self, file_paths, **kwargs):
        dt = DumyDataset()
        return dt


dirname = os.path.dirname(__file__)


def test_dataio():

    dataio = DumyDataIO()
    loader = dataio.create_data_loader(file_paths=None, batch_size=5)
    sampled_data = list(iter(loader))
    assert len(sampled_data) == 20


def test_masked_text_dataio():
    from rosetta.datasets.text.masked_text import MaskedTextDataIO

    dataio = MaskedTextDataIO(tokenizer_name="bert-base-cased")
    loader = dataio.create_data_loader(
        file_paths=os.path.join(dirname, "sonnets.txt"), batch_size=5
    )
