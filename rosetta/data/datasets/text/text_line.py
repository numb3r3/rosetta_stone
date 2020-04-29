import os

import torch
from torch.utils.data.dataset import Dataset

from ... import helper
from ..tokenization import Tokenizer


from transformers.tokenization_utils import PreTrainedTokenizer


class TextLineDataset(Dataset):
    """

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        max_length: int,
        local_rank=-1,
        logger=None,
    ):
        if logger is None:
            logger = helper.get_logger(__name__)
        self.logger = logger

        assert os.path.isfile(file_path)

        self.logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path) as f:
            lines = [
                line.strip() for line in f.read().splitlines() if len(line.strip()) > 0
            ]

        batch_encoding = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=max_length
        )
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
