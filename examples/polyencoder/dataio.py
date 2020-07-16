from typing import Dict, Tuple

from rosetta.core.dataio import BaseDataIO
from rosetta.data.tokenization import Tokenizer
from rosetta.utils.distribute import get_global_rank
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .transform import SelectionJoinTransform, SelectionSequentialTransform


class SelectionDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        context_transform,
        response_transform,
        sample_cnt: int = -1,
    ):
        self.context_transform = context_transform
        self.response_transform = response_transform

        self.data_source = []
        self.transformed_data = {}

        with open(file_path, encoding='utf-8') as f:
            group = {'context': None, 'responses': [], 'labels': []}
            for line in f:
                split = line.strip().split('\t')
                lbl, context, response = int(split[0]), split[1:-1], split[-1]

                if lbl == 1 and len(group['responses']) > 0:
                    self.data_source.append(group)
                    group = {'context': None, 'responses': [], 'labels': []}
                    if sample_cnt > 0 and len(self.data_source) >= sample_cnt:
                        break

                group['responses'].append(response)
                group['labels'].append(lbl)
                group['context'] = context

            if len(group['responses']) > 0:
                self.data_source.append(group)

        # if get_global_rank() <= 0:
        #     self.prepare()

    def prepare(self):
        print('prepare datasets ...')
        for idx in tqdm(range(len(self.data_source))):
            self.__get_single_item__(idx)

        self.data_source = [0] * len(self.transformed_data)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        if index in self.transformed_data:
            key_data = self.transformed_data[index]
            return key_data
        else:
            group = self.data_source[index]
            context, responses, labels = (
                group['context'],
                group['responses'],
                group['labels'],
            )
            transformed_context = self.context_transform(
                context)  # [token_ids],[seg_ids],[masks]
            transformed_responses = self.response_transform(
                responses)  # [token_ids],[seg_ids],[masks]
            _tranformed_data = transformed_context, transformed_responses, labels
            self.transformed_data[index] = _tranformed_data

            return _tranformed_data


class ConversationDataIO(BaseDataIO):

    def __init__(
        self,
        tokenizer_name_or_path: str = 'bert-base-cased',
        max_contexts_length: int = 256,
        max_response_length: int = 64,
        max_history: int = 10,
        **kwargs,
    ):
        tokenizer = Tokenizer.load(tokenizer_name_or_path, do_lower_case=True)
        self.pad_id = 0

        self.context_transform = SelectionJoinTransform(
            tokenizer=tokenizer,
            max_seq_len=max_contexts_length,
            max_history=max_history,
        )

        self.response_transform = SelectionSequentialTransform(
            tokenizer=tokenizer,
            max_seq_len=max_response_length,
            max_history=None,
            pair_last=False,
        )

    def create_dataset(self,
                       data_path: str,
                       mode: str = 'train',
                       download: bool = True,
                       **kwargs):
        return SelectionDataset(data_path, self.context_transform,
                                self.response_transform)

    def collate_fn(self,
                   batch,
                   tensor_names=None,
                   mode: str = 'train',
                   **kwargs) -> Tuple[torch.Tensor]:

        contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        batch_size = len(batch)
        labels_batch = []
        max_ctx_seq_len = 0
        max_resp_seq_len = 0
        for sample in batch:
            (contexts_token_ids, contexts_segment_ids, contexts_input_masks), (
                responses_token_ids_list,
                responses_segment_ids_list,
                responses_input_masks_list,
                _,
            ) = sample[:2]

            if len(contexts_token_ids) > max_ctx_seq_len:
                max_ctx_seq_len = len(contexts_token_ids)

            contexts_token_ids_list_batch.append(contexts_token_ids)
            contexts_segment_ids_list_batch.append(contexts_segment_ids)
            contexts_input_masks_list_batch.append(contexts_input_masks)

            _max_resp_seq_len = max([len(_) for _ in responses_token_ids_list])
            if _max_resp_seq_len > max_resp_seq_len:
                max_resp_seq_len = _max_resp_seq_len

            responses_token_ids_list_batch.append(responses_token_ids_list)
            responses_segment_ids_list_batch.append(responses_segment_ids_list)
            responses_input_masks_list_batch.append(responses_input_masks_list)

            labels_batch.append(sample[-1])

        for i in range(batch_size):
            input_ids = contexts_token_ids_list_batch[i]
            segment_ids = contexts_segment_ids_list_batch[i]
            input_mask_ids = contexts_input_masks_list_batch[i]

            contexts_token_ids_list_batch[i] += [self.pad_id] * (
                max_ctx_seq_len - len(input_ids))
            contexts_segment_ids_list_batch[i] += [0] * (
                max_ctx_seq_len - len(segment_ids))
            contexts_input_masks_list_batch[i] += [0] * (
                max_ctx_seq_len - len(input_mask_ids))

            responses_token_ids_list = responses_token_ids_list_batch[i]
            responses_segment_ids_list = responses_segment_ids_list_batch[i]
            responses_input_masks_list = responses_input_masks_list_batch[i]

            for _ in range(len(responses_token_ids_list)):
                input_ids = responses_token_ids_list[_]
                segment_ids = responses_segment_ids_list[_]
                input_mask_ids = responses_input_masks_list[_]

                responses_token_ids_list[_] += [self.pad_id] * (
                    max_resp_seq_len - len(input_ids))
                responses_segment_ids_list[_] += [0] * (
                    max_resp_seq_len - len(segment_ids))
                responses_input_masks_list[_] += [0] * (
                    max_resp_seq_len - len(input_mask_ids))

        long_tensors = [
            contexts_token_ids_list_batch,
            contexts_segment_ids_list_batch,
            contexts_input_masks_list_batch,
            responses_token_ids_list_batch,
            responses_segment_ids_list_batch,
            responses_input_masks_list_batch,
        ]

        contexts_token_ids_list_batch, contexts_segment_ids_list_batch, contexts_input_masks_list_batch, responses_token_ids_list_batch, responses_segment_ids_list_batch, responses_input_masks_list_batch = (
            torch.tensor(t, dtype=torch.long) for t in long_tensors)

        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        return (
            contexts_token_ids_list_batch,
            contexts_segment_ids_list_batch,
            contexts_input_masks_list_batch,
            responses_token_ids_list_batch,
            responses_segment_ids_list_batch,
            responses_input_masks_list_batch,
            labels_batch,
        )
