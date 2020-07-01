from functools import reduce
import glob
from itertools import chain
import json
from typing import Tuple

from rosetta.core.dataio import BaseDataIO
from rosetta.data.tokenization import Tokenizer
import torch
from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer


class ContextTransform(object):
    def __init__(self, tokenizer, max_seq_len=512, max_history=10):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_history = max_history

        self.cls_id, self.sep_id, self.pad_id = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", "[PAD]"]
        )

    def __call__(self, texts):
        input_ids_list, segment_ids_list, input_masks_list = [], [], []

        total_seq_size = 0

        for text in texts[::-1][: self.max_history]:  # 优先保证最后一个context的信息量
            tokenized_dict = self.tokenizer.encode_plus(
                text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                pad_to_max_length=False,
            )
            input_ids, input_masks = (
                tokenized_dict["input_ids"],
                tokenized_dict["attention_mask"],
            )
            segment_ids = [0] * len(input_ids)

            seq_len = self.max_seq_len - total_seq_size
            if seq_len < len(input_ids) and total_seq_size > 0:
                break

            if len(input_ids_list) > 0:  # ignore CLS token
                input_ids = input_ids[1:]
                segment_ids = segment_ids[1:]
                input_masks = input_masks[1:]

            input_ids_list.append(input_ids)
            segment_ids_list.append(segment_ids)
            input_masks_list.append(input_masks)

            total_seq_size += len(input_ids)

            if total_seq_size >= self.max_seq_len:
                break

        # reverse the context order
        input_ids = [self.tokenizer.cls_token_id] + list(chain(*input_ids_list[::-1]))
        segment_ids = [0] + list(chain(*segment_ids_list[::-1]))
        input_masks = [1] + list(chain(*input_masks_list[::-1]))

        # padding sequence
        input_ids += [self.pad_id] * (self.max_seq_len - len(input_ids))
        segment_ids += [0] * (self.max_seq_len - len(segment_ids))
        input_masks += [0] * (self.max_seq_len - len(input_masks))

        assert len(input_ids) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(input_masks) == self.max_seq_len

        return input_ids, segment_ids, input_masks


class SpeechDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        max_contexts_length: int = 256,
        max_history: int = 10,
        **kwargs,
    ):
        self._data = []
        for fn in glob.glob(data_path):
            with open(fn, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    utterences = json.loads(line)
                    self._data.append(utterences)

        self.transformed_data = {}

        self.context_transform = ContextTransform(
            tokenizer=tokenizer,
            max_seq_len=max_contexts_length,
            max_history=max_history,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if index in self.transformed_data:
            return self.transformed_data[index]
        else:
            utternces = self._data[index]

            ctx_input_ids, ctx_segment_ids, ctx_masks = self.context_transform(
                utternces
            )  # [token_ids],[seg_ids],[masks]
            self.transformed_data[index] = (ctx_input_ids, ctx_segment_ids, ctx_masks)
            return self.transformed_data[index]


class SpeechDataIO(BaseDataIO):
    def __init__(
        self,
        tokenizer_name_or_path: str = "bert-base-cased",
        max_contexts_length: int = 256,
        max_history: int = 10,
        **kwargs,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name_or_path, do_lower_case=True
        )
        self.max_contexts_length = max_contexts_length
        self.max_history = max_history

    def create_dataset(self, data_path: str, mode: str = "train", **kwargs):
        sample_cnt = -1
        # if mode != "train":
        #     sample_cnt = 1000
        return SpeechDataset(
            self.tokenizer,
            data_path,
            max_contexts_length=self.max_contexts_length,
            max_history=self.max_history,
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

            # # NOTE: right-truncated input for easy loss calculating
            # context_input_ids.append(input_ids_list[:-1])
            # context_segment_ids.append(segment_ids_list[:-1])
            # context_masks.append(input_masks_list[:-1])

            # # NOTE: left-shifted labels for easy loss calculating
            # lm_labels.append(
            #     [
            #         x if x != self.tokenizer.pad_token_id else -100
            #         for x in input_ids_list[1:]
            #     ]
            # )

            context_input_ids.append(input_ids_list)
            context_segment_ids.append(segment_ids_list)
            context_masks.append(input_masks_list)

            lm_labels.append(
                [
                    x if x != self.tokenizer.pad_token_id else -100
                    for x in input_ids_list
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
