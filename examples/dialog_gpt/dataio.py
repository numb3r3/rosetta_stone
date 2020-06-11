from functools import reduce
import json
from typing import Tuple

from rosetta.core.dataio import BaseDataIO
from rosetta.data.tokenization import Tokenizer
import torch
from torch.utils.data import Dataset


class ContextTransform(object):
    def __init__(self, tokenizer, max_len: int = 512, max_history: int = 5):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history

        self.cls_id, self.sep_id, self.pad_id = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", "[PAD]"]
        )

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list = [], [], []

        # the expected sequence is "[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
        for text in texts[::-1][
            : self.max_history
        ]:  # make sure to keep the last utterence
            tokenized_dict = self.tokenizer.encode_plus(
                text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=False,
            )
            input_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["attention_mask"],
            )
            segment_ids = [0] * len(input_ids)
            if len(input_ids_list) > 0:  # ignore CLS token
                input_ids = input_ids[1:]
                segment_ids = segment_ids[1:]
                input_masks = input_masks[1:]

            input_ids_list = input_ids + input_ids_list
            segment_ids_list = segment_ids + segment_ids_list
            input_masks_list = input_masks + input_masks_list

            if len(input_ids_list) >= self.max_len:
                input_ids_list = input_ids_list[-self.max_len :]
                segment_ids_list = segment_ids_list[-self.max_len :]
                input_masks_list = input_masks_list[-self.max_len :]
                break

        input_ids_list += [self.pad_id] * (self.max_len - len(input_ids_list))
        segment_ids_list += [0] * (self.max_len - len(segment_ids_list))
        input_masks_list += [0] * (self.max_len - len(input_masks_list))

        assert len(input_ids_list) == self.max_len
        assert len(segment_ids_list) == self.max_len
        assert len(input_masks_list) == self.max_len

        return input_ids_list, segment_ids_list, input_masks_list


class DialogDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        max_contexts_length: int = 256,
        max_history: int = 5,
        sample_cnt: int = -1,
        **kwargs,
    ):
        self.data_source = json.load(open(data_path, "r", encoding="utf-8"))
        if sample_cnt > 0:
            self.data_source = self.data_source[:sample_cnt]
        self.transformed_data = {}

        self.context_transform = ContextTransform(
            tokenizer=tokenizer, max_len=max_contexts_length, max_history=max_history
        )

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        if index in self.transformed_data:
            return self.transformed_data[index]
        else:
            utternces = self.data_source[index]

            ctx_input_ids, ctx_segment_ids, ctx_masks = self.context_transform(
                utternces
            )  # [token_ids],[seg_ids],[masks]
            self.transformed_data[index] = (ctx_input_ids, ctx_segment_ids, ctx_masks)
            return self.transformed_data[index]


class DialogDataIO(BaseDataIO):
    def __init__(
        self,
        tokenizer_name_or_path: str = "bert-base-cased",
        max_contexts_length: int = 256,
        max_history: int = 10,
        **kwargs,
    ):
        self.tokenizer = Tokenizer.load(tokenizer_name_or_path, do_lower_case=True)
        self.max_contexts_length = max_contexts_length
        self.max_history = max_history

    def create_dataset(self, data_path: str, mode: str = "train", **kwargs):
        sample_cnt = -1
        if mode != "train":
            sample_cnt = 1000
        return DialogDataset(
            self.tokenizer,
            data_path,
            max_contexts_length=self.max_contexts_length,
            max_history=self.max_history,
            sample_cnt=sample_cnt,
        )

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Tuple[torch.Tensor]:

        context_input_ids, context_segment_ids, context_masks, lm_labels = (
            [],
            [],
            [],
            [],
        )

        for sample in batch:
            input_ids_list, segment_ids_list, input_masks_list = sample

            # NOTE: right-truncated input for easy loss calculating
            context_input_ids.append(input_ids_list[:-1])
            context_segment_ids.append(segment_ids_list[:-1])
            context_masks.append(input_masks_list[:-1])

            # NOTE: left-shifted labels for easy loss calculating
            lm_labels.append(
                [
                    x if x != self.tokenizer.pad_token_id else -100
                    for x in input_ids_list[1:]
                ]
            )

        long_tensors = [
            context_input_ids,
            context_segment_ids,
            context_masks,
            lm_labels,
        ]

        context_input_ids, context_segment_ids, context_masks, lm_labels = (
            torch.tensor(t, dtype=torch.long) for t in long_tensors
        )

        return (context_input_ids, context_segment_ids, context_masks, lm_labels)
