import functools
from typing import Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .. import helper
from ..utils.distribute import get_global_rank, get_world_size, is_distributed


class BaseDataIO:
    """Generates and stores PyTorch DataLoader objects for the train, dev and
    test datasets."""

    def __init__(self, **kwargs):
        self.logger = helper.get_logger(__name__)

    def collate_fn(self,
                   batch,
                   tensor_names=None,
                   mode: str = 'train',
                   **kwargs) -> Tuple[torch.Tensor]:
        """A custom collate function for data loading that formats the batch as
        a tuple tensors."""
        assert isinstance(batch, list)

        _tensor_names = tensor_names if tensor_names else list(batch[0].keys())

        ret = {}
        for key in _tensor_names:
            data = [x[key] for x in batch]
            ret[key] = torch.stack(data)

        return ret

    def create_dataset(self, data_path: str, mode: str = 'train', **kwargs):
        raise NotImplementedError

    def create_data_loader(self,
                           data_path: str,
                           batch_size: int,
                           mode: str = 'train',
                           pin_memory: bool = True,
                           num_workers: int = 0,
                           start_epoch: bool = 0,
                           **kwargs):
        """Wraps a PyTorch Dataset with a DataLoader.

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
        dataset = self.create_dataset(data_path, mode, **kwargs)
        is_train = True if mode == 'train' else False
        tensor_names = None
        if type(dataset).__name__ == '_StreamingDataSet':
            tensor_names = dataset.tensor_names

        sampler = None
        if is_distributed():
            sampler_kwargs = dict(
                num_replicas=get_world_size(), rank=get_global_rank())
            sampler = DistributedSampler(dataset, **sampler_kwargs)

            # In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method
            # at the beginning of each epoch **before** creating the `DataLoader` iterator is necessary
            # to make shuffling work properly across multiple epochs.
            #  Otherwise, the same ordering will be always used.
            sampler.set_epoch(start_epoch)
        elif is_train:
            sampler = RandomSampler(dataset, True)

        loader_kwargs = dict()
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (num_workers > 0 and hasattr(mp, '_supports_context')
                and mp._supports_context
                and 'forkserver' in mp.get_all_start_methods()):
            loader_kwargs['multiprocessing_context'] = 'forkserver'

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=functools.partial(
                self.collate_fn,
                batch_size=batch_size,
                tensor_names=tensor_names,
                mode=mode,
                **kwargs),
            pin_memory=pin_memory,
            num_workers=num_workers,
            **loader_kwargs)
