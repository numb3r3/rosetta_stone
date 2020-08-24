import torch
from torch.utils.data import DataLoader

from torch.utils.data import DistributedSampler


class DataPrefetcher(object):
    """prefetch data."""

    def __init__(self, loader: DataLoader, start_epoch: int = 0):
        if not torch.cuda.is_available():
            raise RuntimeError('Prefetcher needs CUDA, but not available!')
        self.loader = iter(loader)
        self.epoch = start_epoch - 1

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.loader.sampler is not None and isinstance(
                self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(self.epoch)
        self.epoch += 1

        stream = torch.cuda.Stream()
        is_first = True

        for next_data in self.loader:
            with torch.cuda.stream(stream):
                next_data = [
                    x.to(device='cuda', non_blocking=True)
                    if torch.is_tensor(x) else x for x in self.next_data
                ]

            if not is_first:
                yield data
            else:
                is_first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data

    def __next__(self):
        if self._cuda_available:
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
