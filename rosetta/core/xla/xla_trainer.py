from functools import partial as Partial
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader, DistributedSampler
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

from .. import helper
from ..utils.containers import AverageDictMeter
from ..utils.logx import logx


class XLATrainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Scheduler = None,
        log_interval: int = 10,
        evaluate_every: int = 100,
        device: Optional[torch.device or str] = None,
        verbose: bool = True,
        resume: str = None,
        reset_lr_scheduler: bool = False,
        **kwargs,
    ):
        self.logger = helper.get_logger(__name__)

        self._eval_metric = kwargs["checkpoint_selector"]["eval_metric"]
        self._higher_better = kwargs["checkpoint_selector"]["higher_better"]
        self._best_metric = -float("inf") if self._higher_better else float("inf")

        self._log_interval = log_interval
        self._verbose = verbose
        self._global_step = -1
        self._epoch = -1
        self._is_train = None

        self.kwargs = kwargs

        if isinstance(model, nn.Module):
            # Only instantiate model weights once in memory.
            wrapped_model = xmp.MpModelWrapper(model)
            self.model = wrapped_model
        else:
            raise TypeError(
                f"Unknown type for `model`. Expected nn.Module but got {type(model)}"
            )

        if device is None:
            self.device = xm.xla_device()
        else:
            self.device = device

        # optionally resume from a checkpoint
        if resume:
            state_dict, checkpoint_metas = self.load_checkpoint(resume)

            # resume model
            self.model.load_state_dict(state_dict)

        if "xla" in str(self.device):
            self.model.to(self.device)
            self._cuda_nonblocking = use_cuda_nonblocking
        else:
            self._cuda_nonblocking = False

        self.optimizer = optimizer
        self.set_optimizer()

        # scale the learning rate by the number of workers to account for
        # increased total batch size
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= max(1.0, xm.xrt_world_size() // 2)

        self.lr_scheduler = lr_scheduler
        self.set_scheduler()

    @property
    def log_interval(self):
        return self._log_interval

    @property
    def global_step(self):
        return self._global_step

    @property
    def epoch(self):
        return self._epoch

    @property
    def best_metric(self):
        return self._best_metric

    @property
    def is_train(self):
        return self._is_train

    def _iteration(
        self, batch_data: Tuple[torch.Tensor], mode: str = "eval"
    ) -> Tuple[torch.Tensor]:
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

        try:
            output, loss, metrics = self.model(*batch_data)
        except Exception:
            raise ValueError(
                f"The implemented module = {type(self.model)} should return 3-tuples, i.e, output, loss, metrics. "
            )

        if mode == "train":
            # increment step here
            self._global_step += 1

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            if self.kwargs.get("gradient_clip", None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.kwargs["gradient_max_norm"]
                )
            xm.optimizer_step(self.optimizer)

            # update step for lr_scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

        return output, loss, metrics

    def _loop(self, data_loader: Iterable or DataLoader, mode: str, **kwargs):
        batch_size, total_size = (
            (data_loader.batch_size, len(data_loader.dataset))
            if isinstance(data_loader, DataLoader)
            else (len(data_loader), len(data_loader))
        )
        total_batchs = total_size // batch_size
        # keep tracking the model's metric
        avg_metrics = AverageDictMeter()
        start_time = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            # move batch of samples to device
            batch_data = [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch_data
            ]

            output, loss, metrics = self._iteration(batch_data, mode)

            if loss is not None:
                # capture metrics
                metrics.update({"loss": loss})

            avg_metrics.update(metrics)

            if (batch_idx + 1) % self.log_interval == 0:
                if mode == "train":
                    elapsed = time.time() - start_time

                    logx.msg(
                        "| epoch {:3d} | {:5d}/{:5d} batches | "
                        "lr {:02.6f} | ms/batch {:5.2f} | "
                        "loss {:5.3f}".format(
                            self.epoch,
                            (batch_idx + 1) * get_world_size(),
                            total_batchs,
                            self.lr_scheduler.get_lr()[0],
                            elapsed * 1000 / (self.log_interval * get_world_size()),
                            loss.item(),
                        )
                    )

                    start_time = time.time()

                    logx.add_scalar(
                        "%s/learning_rate" % mode,
                        self.lr_scheduler.get_lr()[0],
                        self.global_step,
                    )
                    logx.metric(mode, metrics, self.global_step)

        return avg_metrics

    def train(self, data_loader: Iterable or DataLoader, **kwargs):
        """Perform the training procedure for an epoch.

        :param data_loader:
        :param mode: Name of this loop. Default is `train`.
        """

        self._is_train = True
        self._epoch += 1

        # Turn on the train mode
        self.model.train()

        avg_metrics = None
        with torch.enable_grad():
            avg_metrics = self._loop(data_loader, mode="train", **kwargs)

        if isinstance(data_loader, DataLoader) and isinstance(
            data_loader.sampler, DistributedSampler
        ):
            data_loader.sampler.set_epoch(self.epoch)
        return avg_metrics

    def eval(self, data_loader: Iterable or DataLoader, **kwargs):
        """Evaluate the model.

        :param data_loader:
        :param mode: Name of this loop. Default is `dev`.
        :return:
        """

        self._is_train = False

        # Turn on the evaluation mode
        self.model.eval()

        eval_metrics = None
        with torch.no_grad():
            eval_metrics = self._loop(data_loader, mode="eval", **kwargs)

        logx.metric("validate", eval_metrics, self.epoch)

        return eval_metrics

    def run(
        self,
        train_loader: Iterable or DataLoader,
        eval_loaders: Iterable or DataLoader or Dict[str, Iterable or DataLoader],
        total_steps: int,
        eval_intervals: int,
    ):
        """Train the model for a given iterations. This module is almost equal
        to :: for ep in range(total_steps): trainer.train(train_loader) for k,
        v in eval_loaders.items(): trainer.eval(v, k)

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

    def save_checkpoint(self, eval_metrics, **kwargs):
        """checkpoint saving."""
        metric = eval_metrics[self._eval_metric]

        self._best_metric = (
            max(self.best_metric, metric)
            if self._higher_better
            else min(self.best_metric, metric)
        )

        # TODO: save amp states when using amp
        save_dict = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "state_dict": self.model.state_dict(),
            "metrics": eval_metrics,
            "best_metric": self.best_metric,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
        }

        self.save_model(
            save_dict,
            metric=metric,
            epoch=self.epoch,
            higher_better=self._higher_better,
        )

    def save_model(
        self,
        save_dict,
        metric,
        epoch,
        higher_better=True,
        delete_old=True,
        use_xla=False,
    ):
        """Saves a model to disk. Keeps a separate copy of latest and best
        models.

        Arguments:
            save_dict: dictionary to save to checkpoint
            epoch: epoch number, used to name checkpoint
            metric: metric value to be used to evaluate whether this is the
                    best result
            higher_better: True if higher valued metric is better, False
                    otherwise
            delete_old: Delete prior 'lastest' checkpoints. By setting to
                    false, you'll get a checkpoint saved every time this
                    function is called.
        """
        if not xm.is_master_ordinal(local=False):
            return

        save_dict["__metric"] = metric

        if os.path.exists(logx.save_ckpt_fn) and delete_old:
            os.remove(logx.save_ckpt_fn)

        # Save out current model
        logx.save_ckpt_fn = os.path.join(
            logx.logdir, "last_checkpoint_ep{}.pth".format(epoch)
        )

        # save_func = xm.save if use_xla else torch.save
        xm.save(save_dict, logx.save_ckpt_fn)

        logx.save_metric = metric

        is_better = logx.is_better(logx.save_metric, logx.best_metric, higher_better)
        if is_better:
            from shutil import copyfile

            if os.path.exists(logx.best_ckpt_fn):
                os.remove(logx.best_ckpt_fn)
            logx.best_ckpt_fn = os.path.join(
                logx.logdir, "best_checkpoint_ep{}.pth".format(epoch)
            )
            logx.best_metric = logx.save_metric
            copyfile(logx.save_ckpt_fn, logx.best_ckpt_fn)
        return is_better

    def load_checkpoint(self, resume_file: str, **kwargs):
        """Restore a model and return a dict with any meta data included in the
        snapshot."""
        if os.path.isfile(resume_file):
            # logx.msg("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file, map_location=torch.device("cpu"))

            # self._epoch = checkpoint["epoch"]
            # self._global_step = checkpoint["global_step"]
            # self._best_metric = checkpoint["best_metric"]

            # self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            logx.msg(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume_file, checkpoint["epoch"]
                )
            )
            state_dict = checkpoint["state_dict"]
            meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
            return (state_dict, meta)
        else:
            logx.msg("=> no checkpoint found at '{}'".format(resume_file))
            return None

    def set_optimizer(self):
        """Set optimizer(s) for model(s).

        You can override as ::
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
        """Set scheduler(s) for optimizer(s).

        You can override as ::
        class YourTrainer(TrainerBase):
            def set_scheduler(self):
                self.scheduler = torch.optim.lr_scheduler.Foo(self.optimizer)
        """

        lr_scheduler = self.lr_scheduler
        if lr_scheduler is not None and self.optimizer is None:
            raise TypeError("Optimizer is not set, so scheduler cannot be set")

        if isinstance(lr_scheduler, Scheduler) or lr_scheduler is None:
            self.lr_scheduler = lr_scheduler

        elif isinstance(lr_scheduler, Partial):
            if not issubclass(lr_scheduler.func, Scheduler):
                raise TypeError(
                    f"`scheduler.func` is expected to be subclass of `_LRScheduler`"
                    f" but got {type(lr_scheduler.func)}"
                )
            self.lr_scheduler = lr_scheduler(self.optimizer)

        else:
            raise TypeError(f"Unexpected type {type(lr_scheduler)} for `lr_scheduler`")
