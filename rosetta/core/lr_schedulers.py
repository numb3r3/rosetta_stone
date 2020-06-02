from functools import partial
import math

from torch.optim import lr_scheduler as _lr_scheduler


def ConstantLR(last_epoch=-1):
    """Create a schedule with a constant learning rate.
    """
    return partial(_lr_scheduler.LambdaLR, lambda _: 1, **locals())


def StepLR(step_size, gamma=0.1, last_epoch=-1):
    """ Decays the learning rate of each parameter group by gamma every step_size epochs. 
    """
    return partial(_lr_scheduler.StepLR, **locals())


def MultiStepLR(milestones, gamma=0.1, last_epoch=-1):
    """ Decays the learning rate of each parameter group by gamma 
    once the number of epoch reaches one of the milestones.
    """
    return partial(_lr_scheduler.MultiStepLR, **locals())


def LambdaLR(lr_lambda, last_epoch=-1):
    return partial(_lr_scheduler.LambdaLR, **locals())


def ExponentialLR(T_max, eta_min=0, last_epoch=-1):
    """ Decays the learning rate of each parameter group by gamma every epoch.
    """
    return partial(_lr_scheduler.ExponentialLR, **locals())


def ReduceLROnPlateau(
    mode="min",
    factor=0.1,
    patience=10,
    verbose=False,
    threshold=1e-4,
    threshold_mode="rel",
    cooldown=0,
    min_lr=0,
    eps=1e-8,
):
    return partial(_lr_scheduler.ReduceLROnPlateau, **locals())


def ConstantWithWarmup(warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    strategy during which the learning rate increases linearly.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1.0, warmup_steps)

        return 1.0

    return partial(_lr_scheduler.LambdaLR, **locals())


def LinearAnnealingWithWarmup(
    warmup_steps, constant_steps, total_training_steps, last_epoch=-1
):
    """ Set the learning rate of each parameter group using a linear annealing schedule
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1.0, warmup_steps)
        elif current_step < warmup_steps + constant_steps:
            return 1.0
        else:
            return max(0.0, total_training_steps - current_step - constant_steps) / max(
                1.0, total_training_steps - warmup_steps - constant_steps
            )

    return partial(_lr_scheduler.LambdaLR, **locals())


def CosineAnealingWithWarmup(
    warmup_steps, total_training_steps, num_cycles=0.5, last_epoch=-1
):
    """ Set the learning rate of each parameter group using a cosine annealing schedule
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1.0, warmup_steps)
        elif current_step < warmup_steps + constant_steps:
            return 1.0
        else:
            progress = (current_step - warmup_steps) / max(
                1, total_training_steps - num_warmup_steps
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

    return partial(_lr_scheduler.LambdaLR, **locals())


def _exponential_decay(factor_rate, step, decay_steps=20000, decay_rate=0.5):
    """Exponential decay.

    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies an exponential decay function
    to a provided initial learning rate.

    The function returns the decayed learning rate.  It is computed as:

    ```python
    factor_rate = factor_rate *
                            decay_rate ^ (global_step / decay_steps)
    ```
    """

    factor_rate *= factor_rate ** (step // decay_steps)
    return factor_rate


def _cyclic_decay(factor_rate, step, decay_steps=1000, decay_rate=0.1):
    """Cyclic decay."""
    min_factor_rate = factor_rate * decay_rate
    cycle = math.cos(math.mod(step * math.pi / decay_steps, math.pi)) * 0.5 + 0.5
    factor_rate = (factor_rate - min_factor_rate) * cycle + min_factor_rate
    return factor_rate


def DecayedLRWithWarmup(
    warmup_steps,
    constant_steps,
    decay_method="exponential",
    decay_steps=20000,
    decay_rate=0.5,
    last_epoch=-1,
):
    return partial(_DecayedLRWithWarmup, **locals())


class _DecayedLRWithWarmup(_lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        constant_steps,
        decay_method="exponential",
        decay_steps=20000,
        decay_rate=0.5,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.decay_method = decay_method
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        def _lr_lambda(current_step):
            factor_rate = 1.0

            if current_step < self.warmup_steps:
                warmup_percent_done = current_step / max(1.0, self.warmup_steps)
                factor_rate = factor_rate * warmup_percent_done

            current_step = max(
                0.0, current_step - self.warmup_steps - self.constant_steps
            )

            if self.decay_method == "exponential":
                factor_rate = _exponential_decay(
                    factor_rate=factor_rate,
                    step=current_step,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate,
                )
            elif self.decay_method == "cyclic":
                factor_rate = _cyclic_decay(
                    factor_rate=factor_rate,
                    step=current_step,
                    decay_steps=self.decay_steps,
                    decay_rate=self.decay_rate,
                )
            else:
                raise NotImplementedError(
                    f"The specified decay method {self.decay_method} has not been supported!"
                )
            return factor_rate

        return [base_lr * _lr_lambda(self.last_epoch) for base_lr in self.base_lrs]
