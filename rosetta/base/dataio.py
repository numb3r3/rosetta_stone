import functools
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .. import helper


class BaseDataIO:
    """ Generates and stores PyTorch DataLoader objects for the train, dev and test datasets.

    """

    def __init__(self, **kwargs):
        self.logger = helper.get_logger(__name__)

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        A custom collate function that formats the batch as a dictionary where the key is
        the name of the tensor and the value is the tensor itself
        """
        assert isinstance(batch, list)

        _tensor_names = tensor_names if tensor_names else list(batch[0].keys())

        ret = {}
        for key in _tensor_names:
            data = [x[key] for x in batch]
            ret[key] = torch.stack(data)

        return ret

    def create_dataset(self, file_paths: List[str], **kwargs):
        raise NotImplementedError

    def create_data_loader(
        self,
        file_paths: List[str],
        batch_size: int,
        mode: str = "train",
        pin_memory: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        """
        Wraps a PyTorch Dataset with a DataLoader.
        :param dataset: Dataset to be wrapped.
        :type dataset: Dataset
        :param sampler: PyTorch sampler used to pick samples in a batch.
        :type sampler: Sampler
        :param batch_size: Number of samples in the batch.
        :param num_workers: number of workers to use for the DataLoader
        :type num_workers: int
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        :type pin_memory: bool
        :return: A DataLoader that wraps the input Dataset.
        """
        dataset = self.create_dataset(file_paths, **kwargs)
        shuffle = True if mode == "train" else False
        tensor_names = None
        if type(dataset).__name__ == "_StreamingDataSet":
            tensor_names = dataset.tensor_names
        sampler = None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            sampler=sampler(dataset) if sampler else None,
            batch_size=batch_size,
            collate_fn=functools.partial(
                self.collate_fn,
                batch_size=batch_size,
                tensor_names=tensor_names,
                mode=mode,
                **kwargs
            ),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
