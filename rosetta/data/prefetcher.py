import torch
from torch.utils.data import DataLoader

from torch.utils.data import DistributedSampler


class DataPrefetcher(object):
    """prefetch data."""

    def __init__(self, loader: DataLoader):
        if not torch.cuda.is_available():
            raise RuntimeError('Prefetcher needs CUDA, but not available!')
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream()
        is_first = True

        for next_data in iter(self.loader):
            with torch.cuda.stream(stream):
                next_data = [
                    x.to(device='cuda', non_blocking=True)
                    if torch.is_tensor(x) else x for x in next_data
                ]

            if not is_first:
                yield data
            else:
                is_first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data
