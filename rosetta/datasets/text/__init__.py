from typing import Dict, List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class SentDataset(Dataset):
    """"""

    def __init__(
        self,
        file_paths: Union[List[str], str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        **kwargs
    ):

        if not isinstance(file_paths, list):
            file_paths = [file_paths]

        lines = []
        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    lines.append(line)

        batch_encoding = tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, max_length=max_length
        )
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
