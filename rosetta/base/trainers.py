import contextlib
from functools import partial as Partial
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple
import warnings

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader

from .. import helper
from ..utils.containers import TensorMap, TensorTuple


class Trainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_f: Optional[Callable or Dict[str, Callable]],
        scheduler: Scheduler = None,
        device: Optional[torch.device or str] = None,
        use_amp: bool = False,
        logger=None,
        **kwargs,
    ):

        if logger is None:
            logger = helper.get_logger(__name__)
        self.logger = logger

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device

        if isinstance(model, nn.Module):
            self.model = model
        else:
            raise TypeError(
                f"Unknown type for `model`. Expected nn.Module but got {type(model)}"
            )

        if "cuda" in str(self.device):
            self.model.to(self.device)
        else:
            # usually, this is not expected
            self.logger.info(
                f"cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})"
            )

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.set_optimizer()
        self.set_scheduler()

        self.loss_f = loss_f

        self.step = -1
        self.epoch = -1
        self.is_train = True

        self._use_amp = use_amp
        if use_amp:
            if not hasattr(torch.cuda, "amp"):
                warnings.warn("amp is not available")
                self._use_amp = False
            else:
                self.scaler = torch.cuda.amp.GradScaler()

    def __enter__(self):
        """
        >>> with Trainer(...) as trainer:
        >>>     trainer.train(...)
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """ Iteration part, user can override via duck typing or override_iteration ::
            def iteration(self, data: Tuple[torch.Tensor]) -> Mapping[str, torch.Tensor]:
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

        input, labels = data
        compat_nullcontext = (
            contextlib.nullcontext
            if hasattr(contextlib, "nullcontext")
            else contextlib.suppress
        )
        context = torch.cuda.amp.autocast if self._use_amp else compat_nullcontext
        with context():
            output = self.model(input)
            loss = self.loss_f(output, labels)
        if self.is_train:
            self.optimizer.zero_grad()
            if self._use_amp:
                self.scaler(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        return TensorMap(loss=loss, output=output)

    def _iteration(self, data: Tuple[torch.Tensor], mode: str):
        """ Iteration level training loop
        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """
        results = self.iteration(data)
        if self.is_train and self.scheduler is not None:
            self.scheduler.step()

    def _loop(self, data_loader: Iterable or DataLoader, mode: str):

        for data in data_loader:
            if self.is_train:
                # increment step here
                self.step += 1
            self._iteration(data, mode)

        self.logger.debug(f"epoch {self.epoch} finished")

    def train(self, data_loader: Iterable or DataLoader, mode: str = "train"):
        """ Training the model for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. 
        """

        self.is_train = True
        self.epoch += 1

        # Turn on the train mode
        self.model.train()

        if hasattr(self.loss_f, "train"):
            self.loss_f.train()

        with torch.enable_grad():
            self._loop(data_loader, mode=mode)

    def eval(self, data_loader: Iterable or DataLoader, mode: str = "eval"):
        """ Evaluate the model.
        
        :param data_loader:
        :param mode: Name of this loop. Default is `dev`. 
        :return:
        """

        self.is_train = False

        # Turn on the evaluation mode
        self.model.eval()

        if hasattr(self.loss_f, "eval"):
            self.loss_f.eval()

        with torch.no_grad():
            self._loop(data_loader, mode=mode)

    def run(
        self,
        train_loader: Iterable or DataLoader,
        val_loaders: Iterable or DataLoader or Dict[str, Iterable or DataLoader],
        total_iterations: int,
        val_intervals: int,
    ):

        """ Train the model for a given iterations. This module is almost equal to ::
            for ep in range(total_iterations):
                trainer.train(train_loader)
                for k, v in eval_loaders.items():
                    trainer.eval(v, k)
        :param train_loader:
        :param val_loaders:
        :param total_iterations:
        :param val_intervals:
        :return:
        """

        class ProxyLoader(object):
            def __init__(self, loader):
                self.loader = loader

            def __len__(self):
                return val_intervals

            def __iter__(self):
                counter = 0
                while True:
                    for data in self.loader:
                        if counter == val_intervals:
                            return  # from python 3.7, this is valid
                        yield data
                        counter += 1

        train_loader = ProxyLoader(train_loader)
        if not isinstance(val_loaders, Dict) and (
            isinstance(val_loaders, Iterable) or isinstance(val_loaders, DataLoader)
        ):
            val_loaders = {"eval": val_loaders}

        for ep in range(total_iterations // val_intervals):
            self.train(train_loader)
            if isinstance(train_loader.loader, DataLoader) and isinstance(
                train_loader.loader.sampler, DistributedSampler
            ):
                train_loader.loader.sampler.set_epoch(self.epoch)
            for name, loader in val_loaders.items():
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

        scheduler = self.scheduler
        if scheduler is not None and self.optimizer is None:
            raise TypeError("Optimizer is not set, so scheduler cannot be set")

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.scheduler = scheduler
        elif isinstance(scheduler, Partial):
            if not issubclass(scheduler.func, Scheduler):
                raise TypeError(
                    f"`scheduler.func` is expected to be subclass of `_LRScheduler`"
                    f" but got {type(scheduler.func)}"
                )
            self.scheduler = scheduler(self.optimizer)
        else:
            raise TypeError(f"Unexpected type {type(scheduler)} for `scheduler`")
