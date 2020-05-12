import contextlib
from functools import partial as Partial
from typing import Dict, Iterable, Mapping, Optional
import warnings

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader, DistributedSampler

# from ..utils.containers import TensorMap, TensorTuple
from .. import helper
from ..utils.distribute import (
    get_global_rank,
    get_local_rank,
    is_distributed,
    is_horovod_available,
)


# try:
#     from apex import amp

#     AMP_AVAILABLE = True
# except ImportError:
#     AMP_AVAILABLE = False


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Scheduler = None,
        evaluate_every=100,
        device: Optional[torch.device or str] = None,
        use_horovod: bool = False,
        use_amp: bool = False,
        use_sync_bn: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        self.logger = helper.get_logger(__name__)

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        if use_horovod and not is_horovod_available():
            raise RuntimeError("horovod is not available!")

        if is_distributed():
            if use_sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            rank = get_local_rank()
            torch.cuda.set_device(rank)
            if get_global_rank() > 0:
                # to avoid overwriting
                verbose = False

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError(
                f"Unknown type for `model`. Expected nn.Module but got {type(model)}"
            )

        # if data_parallel and not isinstance(self.model, nn.DataParallel) and torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        if "cuda" in str(self.device):
            self.model.to(self.device)
        else:
            # usually, this is not expected
            self.logger.info(
                f"cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})"
            )

        if not use_horovod and is_distributed():
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[rank]
            )

        if isinstance(self.model, nn.parallel.DistributedDataParallel) or isinstance(
            self.model, nn.DataParallel
        ):
            self.accessible_model = self.model.module
        else:
            self.accessible_model = self.model

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.set_optimizer()
        self.set_scheduler()

        if use_horovod:
            import horovod.torch as hvd

            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.optimizer = hvd.DistributedOptimizer(
                self.optimizer, named_parameters=self.model.named_parameters()
            )

        self._verbose = verbose

        self._global_step = -1
        self._epoch = -1
        self._is_train = True

        # self._use_amp = use_amp
        # if use_amp and not AMP_AVAILABLE:
        #     raise ImportError(
        #         f"Got use_amp = {use_amp}, but cannot find apex. "
        #         "Please install Apex if you want to make use of automatic mixed precision. "
        #         "https://github.com/NVIDIA/apex"
        #     )

        self._use_amp = use_amp
        if use_amp:
            if not hasattr(torch.cuda, "amp"):
                warnings.warn("amp is not available")
                self._use_amp = False
            else:
                self.scaler = torch.cuda.amp.GradScaler()

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    @property
    def is_train(self):
        return self._is_train

    def __enter__(self):
        """
        >>> with Trainer(...) as trainer:
        >>>     trainer.train(...)
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def _iteration(
        self, feed_dict: Dict[str, torch.Tensor], mode: str = "eval"
    ) -> Mapping[str, torch.Tensor]:
        """ Iteration part, user can override via duck typing or override_iteration ::
            def iteration(self, feed_dict: Dict[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
                input, labels = data
                output = self.model(input)
                loss = self.loss_f(output, labels)
                if self.is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                return TensorMap(loss=loss, output=output)
        :param data: data used during a iteration
        :return: TensorMap or Dict
        """

        # input, labels = data
        compat_nullcontext = (
            contextlib.nullcontext
            if hasattr(contextlib, "nullcontext")
            else contextlib.suppress
        )
        context = torch.cuda.amp.autocast if self._use_amp else compat_nullcontext
        with context():
            try:
                output, loss, metrics = self.model(**feed_dict)
            except Exception:
                raise ValueError(
                    f"The implemented module = {type(self.model)} should return 3-tuples, i.e, output, loss, metrics. "
                )

        if mode == "train":
            # increment step here
            self._global_step += 1
            self.optimizer.zero_grad()
            if self._use_amp:
                self.scaler(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # TODO: enable custome grad norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

            # update step for lr_scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()
        return output, loss, metrics

    # def _iteration(self, data: Dict[str, torch.Tensor], mode: str):
    #     """ Iteration level training loop
    #     :param data: should be TensorTuple
    #     :param mode: train, test or val
    #     :return:
    #     """
    #     results = self.iteration(data)
    #     if self.is_train and self.scheduler is not None:
    #         self.scheduler.step()

    def _loop(self, data_loader: Iterable or DataLoader, mode: str):

        for data in data_loader:
            # if self.is_train:
            #     # increment step here
            #     self.step += 1
            self._iteration(data, mode)

        self.logger.debug(f"epoch {self.epoch} finished")

    def train(self, data_loader: Iterable or DataLoader):
        """ Perform the training procedure for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. 
        """

        self._is_train = True
        self._epoch += 1

        # Turn on the train mode
        self.model.train()

        with torch.enable_grad():
            self._loop(data_loader, mode="train")
            # for batch in data_loader:
            #     self.iteration(batch, mode)
            self.logger.info(f"epoch {self.epoch} finished")

        if isinstance(data_loader, DataLoader) and isinstance(
            data_loader.sampler, DistributedSampler
        ):
            data_loader.sampler.set_epoch(self.epoch)

    def eval(self, data_loader: Iterable or DataLoader, set_name: str = None):
        """ Evaluate the model.

        :param data_loader:
        :param mode: Name of this loop. Default is `dev`. 
        :return:
        """

        self._is_train = False

        # Turn on the evaluation mode
        self.model.eval()

        with torch.no_grad():
            self._loop(data_loader, mode="eval")

    def run(
        self,
        train_loader: Iterable or DataLoader,
        eval_loaders: Iterable or DataLoader or Dict[str, Iterable or DataLoader],
        total_steps: int,
        eval_intervals: int,
    ):
        """ Train the model for a given iterations. This module is almost equal to ::
            for ep in range(total_steps):
                trainer.train(train_loader)
                for k, v in eval_loaders.items():
                    trainer.eval(v, k)
        :param train_loader:
        :param eval_loaders:
        :param total_steps:
        :param eval_intervals:
        :return:
        """

        class ProxyLoader(object):
            def __init__(self, loader):
                self.loader = loader

            def __len__(self):
                return eval_intervals

            def __iter__(self):
                counter = 0
                while True:
                    for data in self.loader:
                        if counter == eval_intervals:
                            return  # from python 3.7, this is valid
                        yield data
                        counter += 1

        train_loader = ProxyLoader(train_loader)
        if not isinstance(eval_loaders, Dict) and (
            isinstance(eval_loaders, Iterable) or isinstance(eval_loaders, DataLoader)
        ):
            eval_loaders = {"eval": eval_loaders}

        for ep in range(total_steps // eval_intervals):
            self.train(train_loader)
            if isinstance(train_loader.loader, DataLoader) and isinstance(
                train_loader.loader.sampler, DistributedSampler
            ):
                train_loader.loader.sampler.set_epoch(self.epoch)
            for name, loader in eval_loaders.items():
                self.eval(loader, name)

    def set_optimizer(self):
        """ Set optimizer(s) for model(s). You can override as ::
            class YourTrainer(TrainerBase):
                def set_optimizer(self):
                    self.optimizer = torch.optim.SGD(self.model.parameters())
        """

        optimizer = self.optimizer
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.optimizer = optimizer
        elif isinstance(optimizer, Partial):
            if not issubclass(optimizer.func, Optimizer):
                raise TypeError(
                    f"`optimizer.func` is expected to be subclass of `Optimizer`"
                    f" but got {type(optimizer.func)}"
                )
            self.optimizer = optimizer(self.model.parameters())
        else:
            raise TypeError(f"Unexpected type {type(optimizer)} for `optimizer`")

    def set_scheduler(self):
        """ Set scheduler(s) for optimizer(s). You can override as ::
            class YourTrainer(TrainerBase):
                def set_scheduler(self):
                    self.scheduler = torch.optim.lr_scheduler.Foo(self.optimizer)
        """

        scheduler = self.lr_scheduler
        if scheduler is not None and self.optimizer is None:
            raise TypeError("Optimizer is not set, so scheduler cannot be set")

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.lr_scheduler = scheduler
        elif isinstance(scheduler, Partial):
            if not issubclass(scheduler.func, Scheduler):
                raise TypeError(
                    f"`scheduler.func` is expected to be subclass of `_LRScheduler`"
                    f" but got {type(scheduler.func)}"
                )
            self.lr_scheduler = scheduler(self.optimizer)
        else:
            raise TypeError(f"Unexpected type {type(scheduler)} for `scheduler`")
