import logging
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.tokenization_utils import PreTrainedTokenizer

# from transformers.data.processors.utils import InputExample, InputFeatures
from ...processor.glue import InputExample, InputFeatures


logger = logging.getLogger(__name__)


class SentenceDataset(Dataset):
    """

    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = -1,
        set_type: str = "train",
        local_rank: int = -1,
        hparams: Dict = dict(),
    ):
        logger.info(f"Creating features from dataset file at {file_path}")

        examples = self._create_examples(file_path, set_type, limit_length)

        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=hparams["hparams"],
            label_list=self.labels,
            output_mode="classification",
        )

    @property
    def labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(
        self, file_path: str, set_type: str, limit_length: Optional[int] = -1
    ):
        """Creates examples for the training and dev sets."""
        examples = []
        count = 0
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                count += 1
                if limit_length > 0 and count > limit_length:
                    break
                guid = "%s-%s" % (set_type, i)
                items = line.split("\t")
                text_a = items[-1]
                if len(items) == 2:
                    label = items[0] if len(items) == 2 else "0"
                examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
