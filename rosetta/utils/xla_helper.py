import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.xla_multiprocessing as xmp


# At global scope.
SERIAL_EXEC = xmp.MpSerialExecutor()


def get_tpu_sampler(dataset: Dataset, shuffle: bool = True):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )
