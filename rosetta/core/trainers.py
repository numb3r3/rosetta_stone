from functools import partial as Partial
import os
import time
from typing import Dict, Iterable, Optional, Tuple, Mapping, Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader, DistributedSampler

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


from .. import helper
from ..utils.containers import AverageDictMeter
from ..utils.distribute import (
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_distributed,
    is_horovod_available,
)
from ..utils.logx import logx, to_np


class Trainer(object):

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Scheduler = None,
        log_interval: int = 10,
        evaluate_every: int = 100,
        gradient_accumulation_steps: int = 1,
        update_scheduler_by_epoch: bool = False,
        device: Optional[torch.device or str] = None,
        use_cudnn_benchmark: bool = True,
        use_cuda_nonblocking: bool = False,
        use_sync_bn: bool = False,
        use_horovod: bool = False,
        use_amp: bool = False,
        use_prefetcher=False,
        verbose: bool = True,
        **kwargs,
    ):
        self.logger = helper.get_logger(__name__)

        self._eval_metric = kwargs['checkpoint_selector']['eval_metric']
        self._higher_better = kwargs['checkpoint_selector']['higher_better']
        self._best_metric = -float('inf') if self._higher_better else float(
            'inf')

        self._log_interval = log_interval
        self._verbose = verbose
        self._step = -1
        self._epoch = -1
        self._is_train = None
        self._loss = None

        self._use_prefetcher = use_prefetcher

        self.gradient_accumulation_steps = gradient_accumulation_steps

        if device is None:
            self.device = device or (torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu'))
        else:
            self.device = device

        for k, v in kwargs.items():
            if hasattr(self, k):
                raise AttributeError(f'{self} already has {k}')
            if torch.is_tensor(v):
                v = v.to(self.device)
            if isinstance(v, nn.Module):
                v.to(self.device)
            kwargs[k] = v

        self.kwargs = kwargs

        # setup for distributed
        self._use_sync_bn = use_sync_bn
        if is_distributed():
            if self._use_sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.logger.info(
                    'BNs of model are converted to nn.SyncBatchNorm')

            rank = get_local_rank()
            torch.cuda.set_device(rank)
            if get_global_rank() > 0:
                # to avoid overwriting
                self._verbose = False

        # setup model
        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, dict):
            self.model = nn.ModuleDict(model)
        else:
            raise TypeError(
                f'Unknown type for `model`. Expected nn.Module or Dict[str, Module], but got {type(model)}'
            )

        if 'cuda' in str(self.device):
            self.model.to(self.device)
            torch.backends.cudnn.benchmark = use_cudnn_benchmark
            self._cuda_nonblocking = use_cuda_nonblocking
            self.logger.debug(
                f'cuda: True, cudnn.benchmark: {use_cudnn_benchmark}, '
                f'cuda.nonblocking: {use_cuda_nonblocking}')
        else:
            self._cuda_nonblocking = False
            # usually, this is not expected
            self.logger.info(
                f'cuda: False (torch.cuda.is_available()={torch.cuda.is_available()})'
            )

        if not use_horovod and is_distributed():
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True)

        # self.accessible_model is useful for e.g., checkpointing
        if isinstance(self.model,
                      nn.parallel.DistributedDataParallel) or isinstance(
                          self.model, nn.DataParallel):
            self.accessible_model = self.model.module
        else:
            self.accessible_model = self.model

        # setup optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._update_scheduler_by_epoch = update_scheduler_by_epoch
        self.set_optimizer()
        self.set_scheduler()

        if use_horovod:
            if not is_horovod_available():
                raise RuntimeError('horovod is not available!')
            import horovod.torch as hvd

            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            self.optimizer = hvd.DistributedOptimizer(
                self.optimizer, named_parameters=self.model.named_parameters())

        self._use_amp = use_amp
        self.scaler = None
        if self._use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            self.logger.info('AMP is activated')

    @property
    def log_interval(self):
        return self._log_interval

    @property
    def step(self):
        return self._step

    @property
    def epoch(self):
        return self._epoch

    @property
    def best_metric(self):
        return self._best_metric

    @property
    def is_train(self):
        return self._is_train

    def _iteration(self,
                   data: Tuple[torch.Tensor],
                   mode: str = 'train') -> Tuple[torch.Tensor]:
        """Iteration level training loop.

        :param data: should be TensorTuple
        :param mode: train, test or val
        :return:
        """

        output, loss, metrics = self.iteration(data, mode)

        return output, loss, metrics

    def iteration(self,
                  data: Tuple[torch.Tensor],
                  mode: str = 'train') -> Tuple[torch.Tensor]:

        with torch.cuda.amp.autocast(
                self._use_amp) if self._use_amp else nullcontext():
            try:
                output, loss, metrics = self.model(*data)
                # collapse all losses if they are scattered on multiple gpus
                loss = loss.mean()
            except Exception:
                raise ValueError(
                    f'The implemented module = {type(self.model)} should return 3-tuples, i.e, output, loss, metrics. '
                )

        self._loss = loss / self.gradient_accumulation_steps

        is_update_step = ((self.step + 1) %
                          self.gradient_accumulation_steps == 0)

        if self.is_train:
            # backprop computation
            if self._use_amp:
                self.scaler.scale(self._loss).backward()
            else:
                self._loss.backward()

            # log the layers and layers gradient histogram and distributions
            # NOTE: the visualization step must be called before `zero_grad()`
            if (self.step + 1) % (self.log_interval *
                                  self.gradient_accumulation_steps) == 0:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    if value is not None and value.grad is not None:
                        logx.add_histogram('model/' + tag, to_np(value),
                                           self.step)

                        logx.add_histogram('model/' + tag + '/grad',
                                           to_np(value.grad), self.step)

            # update the parameters
            if is_update_step:
                if self._use_amp:
                    if self.kwargs.get('gradient_clip', None):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.kwargs['gradient_max_norm'])

                    # optimizer.zero_grad()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.model.zero_grad()
                    # self.optimizer.zero_grad()
                else:
                    if self.kwargs.get('gradient_clip', None):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.kwargs['gradient_max_norm'])

                    self.optimizer.step()
                    self.model.zero_grad()
                    # self.optimizer.zero_grad()

                if self.scheduler is not None and not self._update_scheduler_by_epoch:
                    self.scheduler.step()

        return output, loss, metrics

    def _loop(self,
              data_loader: Iterable or DataLoader,
              mode: str = 'train',
              **kwargs):
        batch_size, total_size = ((data_loader.batch_size,
                                   len(data_loader.dataset)) if isinstance(
                                       data_loader, DataLoader) else
                                  (len(data_loader), len(data_loader)))
        total_batchs = total_size // batch_size

        # keep tracking the model's metric
        avg_metrics = AverageDictMeter()
        start_time = time.time()

        prefetcher = None
        if self._use_prefetcher:
            if is_distributed():
                from ..data.prefetcher import DataPrefetcher
                prefetcher = DataPrefetcher(data_loader)
                self.logger.info(
                    'use `rosetta_stone.data.prefetcher.DataPrefetcher` as prefetcher'
                )
            else:
                # NOTE: async dataloader is better than DataPrefetcher,
                #         However, it does not work with DDP
                from ..data.async_data import AsyncDataLoader
                prefetcher = AsyncDataLoader(data_loader)
                self.logger.info(
                    'use `rosetta_stone.data.async_data.AsyncDataLoader` as prefetcher'
                )

            # from ..data.prefetcher import DataPrefetcher
            # prefetcher = DataPrefetcher(data_loader)
            # self.logger.info(
            #     'use `rosetta_stone.data.prefetcher.DataPrefetcher` as prefetcher'
            # )

        for batch_idx, batch_data in enumerate(prefetcher or data_loader):
            # move batch of samples to device
            batch_data = [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch_data
            ]

            if self.is_train:
                self._step += 1

            output, loss, metrics = self._iteration(batch_data, mode)
            if loss is not None:
                # capture metrics
                metrics.update({'loss': loss})

            avg_metrics.update(metrics)

            if (batch_idx + 1) % (self.log_interval *
                                  self.gradient_accumulation_steps) == 0:
                if mode == 'train':
                    elapsed = time.time() - start_time

                    logx.msg('| epoch {:3d} | {:5d}/{:5d} batches | '
                             'lr {:02.6f} | ms/batch {:5.2f} | '
                             'loss {:5.3f}'.format(
                                 self.epoch,
                                 (batch_idx + 1) * get_world_size(),
                                 total_batchs,
                                 self.scheduler.get_lr()[0],
                                 elapsed * 1000 /
                                 (self.log_interval *
                                  self.gradient_accumulation_steps *
                                  get_world_size()),
                                 loss.item(),
                             ))

                    start_time = time.time()

                    logx.add_scalar(
                        '%s/learning_rate' % mode,
                        self.scheduler.get_lr()[0],
                        self.step,
                    )
                    logx.metric(mode, metrics, self.step)

        return avg_metrics

    def train(self, data_loader: Iterable or DataLoader, **kwargs):
        """Training the model for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`. Passed to callbacks.
        """

        self._is_train = True
        self._epoch += 1
        self._loss = None

        # Turn on the train mode
        self.model.train()

        avg_metrics = None
        with torch.enable_grad():
            avg_metrics = self._loop(data_loader, mode='train', **kwargs)

        if self.scheduler is not None and self._update_scheduler_by_epoch:
            self.scheduler.step()

        # For distributed training, to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will be always used.
        if isinstance(data_loader, DataLoader) and isinstance(
                data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(self.epoch)

        return avg_metrics

    def eval(self, data_loader: Iterable or DataLoader, **kwargs):
        """Evaluate the model.

        :param data_loader:
        :param mode: Name of this loop. Default is `test`. Passed to callbacks.
        :return:
        """

        self._is_train = False
        self._loss = None

        # Turn on the evaluation mode
        self.model.eval()

        eval_metrics = None
        with torch.no_grad():
            eval_metrics = self._loop(data_loader, mode='eval', **kwargs)

        logx.metric('validate', eval_metrics, self.epoch)

        return eval_metrics

    def run(
        self,
        train_loader: Iterable or DataLoader,
        eval_loaders: Iterable or DataLoader
        or Dict[str, Iterable or DataLoader],
        total_steps: int,
        eval_intervals: int,
    ):
        """Train the model for a given iterations. This module is almost equal
        to :: for ep in range(total_iterations): trainer.train(train_loader)
        for k, v in val_loaders.items(): trainer.test(v, k)

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
        if not isinstance(eval_loaders,
                          Dict) and (isinstance(eval_loaders, Iterable)
                                     or isinstance(eval_loaders, DataLoader)):
            eval_loaders = {'eval': eval_loaders}

        for ep in range(total_steps // eval_intervals):
            self.train(train_loader)
            if isinstance(train_loader.loader, DataLoader) and isinstance(
                    train_loader.loader.sampler, DistributedSampler):
                train_loader.loader.sampler.set_epoch(self.epoch)
            for name, loader in eval_loaders.items():
                self.eval(loader, name)

    def state_dict(self) -> Mapping[str, Any]:

        return {
            'model': self.accessible_model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'scheduler':
            self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'step': self.step,
            'update_scheduler_by_epoch': self._update_scheduler_by_epoch,
            'use_sync_bn': self._use_sync_bn,
            'use_amp': self._use_amp
        }

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        resume_optimizer: bool = False) -> None:
        self.accessible_model.load_state_dict(state_dict['model'])
        if resume_optimizer:
            self.optimizer.load_state_dict(state_dict['optim'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.scheduler.last_epoch = state_dict['epoch']
            self._epoch = state_dict['epoch']
            self._update_scheduler_by_epoch = state_dict[
                'update_scheduler_by_epoch']
            self._use_sync_bn = state_dict['use_sync_bn']
            self._use_amp = state_dict['use_amp']

    def save_checkpoint(self, eval_metrics, **kwargs):
        """checkpoint saving."""

        if self._eval_metric not in eval_metrics:
            raise ValueError(
                f"The model's metric {self._eval_metric} is not available!")

        metric = eval_metrics[self._eval_metric]

        self._best_metric = (
            max(self.best_metric, metric) if self._higher_better else min(
                self.best_metric, metric))

        state_dict = self.state_dict()

        logx.save_model(
            state_dict,
            metric=metric,
            epoch=self.epoch,
            higher_better=self._higher_better,
        )

    def load_checkpoint(self,
                        resume_file: str,
                        resume_optimizer: bool = False,
                        **kwargs):
        """Restore a model and return a dict with any meta data included in the
        snapshot."""
        if os.path.isfile(resume_file):
            checkpoint = torch.load(
                resume_file, map_location=torch.device('cpu'))

            logx.msg("=> loaded checkpoint '{}' (epoch {})".format(
                resume_file, checkpoint['epoch']))
            self.load_state_dict(checkpoint, resume_optimizer=resume_optimizer)
        else:
            logx.msg("=> no checkpoint found at '{}'".format(resume_file))
            raise FileNotFoundError(
                f'checkpoint file {resume_file} not found!')

    def set_optimizer(self) -> None:
        """Set optimizer(s) for model(s).

        You can override as::
            class YourTrainer(TrainerBase):
                def set_optimizer(self):
                    self.optimizer = torch.optim.SGD(self.model.parameters())
        :return:
        """

        optimizer = self.optimizer
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.optimizer = optimizer

        elif isinstance(optimizer, Partial):
            if not issubclass(optimizer.func, Optimizer):
                raise TypeError(
                    f'`optimizer.func` is expected to be subclass of `Optimizer`'
                    f' but got {type(optimizer.func)}')

            grouped_parameters = self.model.parameters()
            if hasattr(self.model, 'optimizer_grouped_parameters'):
                grouped_parameters = self.model.optimizer_grouped_parameters

            self.optimizer = optimizer(grouped_parameters)

        # elif isinstance(optimizer, dict):
        #     if not isinstance(self.model, nn.ModuleDict):
        #         raise TypeError(
        #             'When `optimizer` is `dict`, `model` also needs to be `dict` or `nn.ModuleDict`'
        #         )

        #     if isinstance(list(optimizer.values())[0], Partial):
        #         optimizer = {
        #             k: v(self.model[k].parameters())
        #             for k, v in optimizer.items() if v is not None
        #         }
        #     self.optimizer = StepDict(Optimizer, **optimizer)

        else:
            raise TypeError(
                f'Unexpected type {type(optimizer)} for `optimizer`')

    def set_scheduler(self) -> None:
        """Set scheduler(s) for optimizer(s).

        You can override as ::
            class YourTrainer(TrainerBase):
                def set_scheduler(self):
                    self.scheduler = torch.optim.lr_scheduler.Foo(self.optimizer)
        :return:
        """

        scheduler = self.scheduler
        if scheduler is not None and self.optimizer is None:
            raise TypeError('Optimizer is not set, so scheduler cannot be set')

        if isinstance(scheduler, Scheduler) or scheduler is None:
            self.scheduler = scheduler

        elif isinstance(scheduler, Partial):
            self.scheduler = scheduler(self.optimizer)

        elif isinstance(scheduler, dict):
            if not isinstance(self.optimizer, StepDict):
                raise TypeError(
                    'When `scheduler` is `dict`, `optimizer` is also needs to be `dict`'
                )

            _scheduler = {}
            for k, v in scheduler.items():
                if isinstance(v, Partial):
                    v = v(self.optimizer[k])
                _scheduler[k] = v
            self.scheduler = StepDict(Scheduler, **_scheduler)

        else:
            raise TypeError(
                f'Unexpected type {type(scheduler)} for `scheduler`')
