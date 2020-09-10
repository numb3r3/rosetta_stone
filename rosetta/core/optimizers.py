"""Pytorch optimization definitions and functions."""
from typing import Any, Dict, List, Tuple, Union, Iterable, Callable
from functools import partial
import torch
import math
import re


# https://github.com/allenai/allennlp/blob/master/allennlp/training/optimizers.py
def make_parameter_groups(
    model_parameters: List[Tuple[str, torch.nn.Parameter]],
    groups: List[Tuple[List[str], Dict[str, Any]]] = None,
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.parameters`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.
    `groups` contains something like:
    ```
    [
        (["regex1", "regex2"], {"lr": 1e-3}),
        (["regex3"], {"lr": 1e-4})
    ]
    ```
    All of key-value pairs specified in each of these dictionaries will passed passed as-is
    to the optimizer, with the exception of a dictionaries that specify `requires_grad` to be `False`:
    ```
    [
        ...
        (["regex"], {"requires_grad": False})
    ]
    ```
    When a parameter group has `{"requires_grad": False}`, the gradient on all matching parameters
    will be disabled and that group will be dropped so that it's not actually passed to the optimizer.
    Ultimately, the return value of this function is in the right format to be passed directly
    as the `params` argument to a pytorch `Optimizer`.
    If there are multiple groups specified, this is list of dictionaries, where each
    dict contains a "parameter group" and groups specific options, e.g., {'params': [list of
    parameters], 'lr': 1e-3, ...}.  Any config option not specified in the additional options (e.g.
    for the default group) is inherited from the top level arguments given in the constructor.  See:
    <https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options>.  See also our
    `test_optimizer_parameter_groups` test for an example of how this works in this code.
    The dictionary's return type is labeled as `Any`, because it can be a `List[torch.nn.Parameter]`
    (for the "params" key), or anything else (typically a float) for the other keys.
    """
    if groups:
        # In addition to any parameters that match group specific regex,
        # we also need a group for the remaining "default" group.
        # Those will be included in the last entry of parameter_groups.
        parameter_groups: Union[List[Dict[str, Any]],
                                List[torch.nn.Parameter]] = [{
                                    'params': []
                                } for _ in range(len(groups) + 1)]
        # add the group specific kwargs
        for k in range(len(groups)):
            parameter_groups[k].update(groups[k][1])

        regex_use_counts: Dict[str, int] = {}
        parameter_group_names: List[set] = [
            set() for _ in range(len(groups) + 1)
        ]
        for name, param in model_parameters:
            # Determine the group for this parameter.
            group_index = None
            for k, group_regexes in enumerate(groups):
                for regex in group_regexes[0]:
                    if regex not in regex_use_counts:
                        regex_use_counts[regex] = 0
                    if re.search(regex, name):
                        if group_index is not None and group_index != k:
                            raise ValueError(
                                '{} was specified in two separate parameter groups'
                                .format(name))
                        group_index = k
                        regex_use_counts[regex] += 1

            if group_index is not None:
                parameter_groups[group_index]['params'].append(param)
                parameter_group_names[group_index].add(name)
            else:
                # the default group
                parameter_groups[-1]['params'].append(param)
                parameter_group_names[-1].add(name)

        # find and remove any groups with 'requires_grad = False'
        no_grad_group_indices: List[int] = []
        for k, (names, group) in enumerate(
                zip(parameter_group_names, parameter_groups)):
            if group.get('requires_grad') is False:
                no_grad_group_indices.append(k)
                logging.info(
                    'Disabling gradient for the following parameters: %s',
                    names)
                for param in group['params']:
                    param.requires_grad_(False)

                # warn about any other unused options in that group.
                unused_options = {
                    key: val
                    for key, val in group.items()
                    if key not in ('params', 'requires_grad')
                }
                if unused_options:
                    logger.warning('Ignoring unused options %s for %s',
                                   unused_options, names)
        parameter_group_names = [
            names for (k, names) in enumerate(parameter_group_names)
            if k not in no_grad_group_indices
        ]
        parameter_groups = [
            group for (k, group) in enumerate(parameter_groups)
            if k not in no_grad_group_indices
        ]

        # log the remaining parameter groups
        logger.info('Done constructing parameter groups.')
        for k in range(len(parameter_groups)):
            group_options = {
                key: val
                for key, val in parameter_groups[k].items() if key != 'params'
            }
            logger.info('Group %s: %s, %s', k, list(parameter_group_names[k]),
                        group_options)

        # check for unused regex
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning(
                    'When constructing parameter groups, %s does not match any parameter name',
                    regex,
                )

    else:
        parameter_groups = [param for name, param in model_parameters]

    # Log the number of parameters to optimize
    num_parameters = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum(parameter.numel()
                                  for parameter in parameter_group['params'])
        else:
            num_parameters += parameter_group.numel()  # type: ignore
    logger.info('Number of trainable parameters: %s', num_parameters)
    return parameter_groups


def Adam(lr=1e-3,
         betas=(0.9, 0.999),
         eps=1e-8,
         weight_decay=0.0,
         amsgrad=False):
    return partial(torch.optim.Adam, **locals())


def Adadelta(lr=1.0, rho=0.9, eps=1e-06, weight_decay=0):
    """

    Arguments:
        rho (float, optional) – coefficient used for computing a running average of squared gradients (default: 0.9)
        eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-6)
        lr (float, optional) – coefficient that scale delta before it is applied to the parameters (default: 1.0)
        weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
    """

    return partial(torch.optim.Adadelta, **locals())


def SGD(lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return partial(torch.optim.SGD, **locals())


def Adafactor(lr=None,
              beta1=None,
              eps=(1e-30, 1e-3),
              clip_threshold=1.0,
              weight_decay=0.0,
              scale_parameter=True,
              relative_step=True,
              warmup_init=False):
    """Note that this optimizer internally adjusts the learning rate depending
    on the *scale_parameter*, *relative_step* and.

    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    # NOTE: the parameter eps is frezzen here
    eps = (1e-30, 1e-3)

    # Cannot combine manual lr and relative_step options
    if lr is not None:
        relative_step = False

    return partial(_Adafactor, **locals())


# https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py
class _Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(self,
                 params,
                 lr=None,
                 eps=(1e-30, 1e-3),
                 clip_threshold=1.0,
                 decay_rate=-0.8,
                 beta1=None,
                 weight_decay=0.0,
                 scale_parameter=True,
                 relative_step=True,
                 warmup_init=False):
        if lr is not None and relative_step:
            raise ValueError(
                'Cannot combine manual lr and relative_step options')
        if warmup_init and not relative_step:
            raise ValueError('warmup_init requires relative_step=True')

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group['lr']
        if param_group['relative_step']:
            min_step = 1e-6 * param_state['step'] if param_group[
                'warmup_init'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'][1], param_state['RMS'])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel()**0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row /
                    exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_()
        c_factor = exp_avg_sq_col.rsqrt()
        return torch.mm(r_factor.unsqueeze(-1), c_factor.unsqueeze(0))

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adafactor does not support sparse gradients.')

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(
                    group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(
                            grad_shape[:-1]).to(grad)
                        state['exp_avg_sq_col'] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)

                    state['RMS'] = 0
                else:
                    if use_first_moment:
                        state['exp_avg'] = state['exp_avg'].to(grad)
                    if factored:
                        state['exp_avg_sq_row'] = state['exp_avg_sq_row'].to(
                            grad)
                        state['exp_avg_sq_col'] = state['exp_avg_sq_col'].to(
                            grad)
                    else:
                        state['exp_avg_sq'] = state['exp_avg_sq'].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                group['lr'] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = (grad**2) + group['eps'][0]
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row,
                                                  exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) /
                             group['clip_threshold']).clamp_(min=1.0))
                update.mul_(group['lr'])

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(
                        update, alpha=1 - group['beta1'])
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(
                        p_data_fp32,
                        alpha=-group['weight_decay'] * group['lr'])

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss


def AdamW(lr=1e-3,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=0.0,
          correct_bias=True):
    return partial(AdamWeightDecayOptimizer, **locals())


class AdamWeightDecayOptimizer(torch.optim.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.

    https://github.com/google-research/bert/blob/master/optimization.py
    https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        """Implements Adam algorithm with weight decay fix.

        Parameters:
            lr (float): learning rate. Default 1e-3.
            betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
            eps (float): Adams epsilon. Default: 1e-6
            weight_decay (float): Weight decay. Default: 0.0
            correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        """
        if lr < 0.0:
            raise ValueError(
                'Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter: {} - should be in [0.0, 1.0['.format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter: {} - should be in [0.0, 1.0['.format(
                    betas[1]))
        if not 0.0 <= eps:
            raise ValueError(
                'Invalid epsilon value: {} - should be >= 0.0'.format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1**state['step']
                    bias_correction2 = 1.0 - beta2**state['step']
                    step_size = step_size * math.sqrt(
                        bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(
                        p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss


def LAMB(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, adam=False):
    return partial(_Lamb, **locals())


# https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
class _Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self,
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0,
                 adam: bool = False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.'
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    'lr']  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss
