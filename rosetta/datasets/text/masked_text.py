from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from . import SentDataset
from ...core.dataio import BaseDataIO
from ...data.tokenization import Tokenizer


class MaskedTextDataIO(BaseDataIO):
    def __init__(
        self,
        tokenizer_name: str = "bert",
        max_length: int = 256,
        mlm_prob: float = 0.15,
        **kwargs,
    ):
        self.tokenizer = Tokenizer.load(tokenizer_name, do_lower_case=True)
        self.max_length = max_length
        self.mlm_prob = mlm_prob

    def create_dataset(self, file_paths: List[str], mode: str = "train", **kwargs):
        return SentDataset(
            file_paths, tokenizer=self.tokenizer, max_length=self.max_length
        )

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(batch)
        inputs, labels = self.mask_tokens(batch)
        return (inputs, labels)
        # return {"input_ids": inputs, "masked_lm_labels": labels}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(
                examples, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability mlm_prob defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
