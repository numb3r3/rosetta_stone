"""utils for distributed training."""
import builtins
import importlib.util
import os
import sys
from typing import Optional, Callable

from torch import distributed
from torch.cuda import device_count

from .. import helper

logger = helper.get_logger(__name__)
args = ' '.join(sys.argv)

# IS_DISTRIBUTED is used to handle horovod
IS_DISTRIBUTED_HOROVOD = False

# decorators


def if_is_master(func: Callable) -> Callable:
    """Wraps function that is active only if it is the master process.

    :param func: Any function
    :return:
    """

    @wraps(func)
    def inner(*args, **kwargs) -> Optional:
        if is_master():
            return func(*args, **kwargs)

    return inner


# distributed


def init_distributed(
    use_horovod: bool = False,
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    warning: bool = True,
):
    """Simple initializer for distributed training.

    :param use_horovod: If use horovod as distributed backend
    :param backend: backend of torch.distributed.init_process_group
    :param init_method: init_method of torch.distributed.init_process_group
    :param warning: Warn if this method is called multiple times
    :return:
    """

    if not is_distributed_available():
        raise RuntimeError(
            'Distributed training is not available on this machine')

    if use_horovod:
        global IS_DISTRIBUTED_HOROVOD
        IS_DISTRIBUTED_HOROVOD = True
        if backend is not None or init_method is not None:
            raise RuntimeError(
                'Try to use horovod, but `backend` and `init_method` are not None'
            )

        if is_horovod_available():
            import horovod.torch as hvd

            hvd.init()
            logger.debug('init horovod')
        else:
            raise RuntimeError('horovod is not available!')

    else:
        # default values
        backend = backend or 'nccl'
        init_method = init_method or 'env://'

        if not is_distributed():
            raise RuntimeError(
                f'For distributed training, use `python -m torch.distributed.launch '
                f'--nproc_per_node={device_count()} {args}` ...')

        if distributed.is_initialized():
            if warning:
                logger.warn('`distributed` is already initialized. Skipped.')
        else:
            distributed.init_process_group(
                backend=backend, init_method=init_method)
        logger.info('init distributed')

    if not is_master():

        def no_print(*values, **kwargs):
            pass

        builtins.print = no_print


def is_horovod_available() -> bool:
    disable_horovod = int(os.environ.get('DISABLE_HOROVOD', 0))
    return (importlib.util.find_spec('horovod')
            is not None) and (disable_horovod == 0)


def is_distributed_available() -> bool:
    return distributed.is_available() or is_horovod_available()


def is_distributed() -> bool:
    # to handle horovod
    return get_world_size() > 1


def get_local_rank() -> int:
    # returns -1 if not distributed, else returns local rank
    # it works before dist.init_process_group
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.local_rank()
    else:
        return int(os.environ.get('LOCAL_RANK', 0))


def get_global_rank() -> int:
    # returns 0 if not distributed, else returns global rank
    # it works before dist.init_process_group
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.rank()
    else:
        return int(os.environ.get('RANK', 0))


def is_master() -> bool:
    return get_global_rank() == 0


def get_num_nodes() -> int:
    # assume all nodes have the same number of gpus
    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def get_num_nodes() -> int:
    # assume all nodes have the same number of gpus
    if not is_distributed():
        return 1
    else:
        return get_world_size() // device_count()


def get_world_size() -> int:
    if IS_DISTRIBUTED_HOROVOD:
        import horovod.torch as hvd

        return hvd.size()
    else:
        return int(os.environ.get('WORLD_SIZE', 1))
