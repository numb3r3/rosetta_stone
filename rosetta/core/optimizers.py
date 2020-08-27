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


def SGD(lr=1e-1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return partial(torch.optim.SGD, **locals())


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
