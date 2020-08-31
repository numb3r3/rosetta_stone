from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import math
import numpy
import torch


def move_to_device(obj, cuda_device: Union[torch.device, int]):
    """Given a structure (possibly) containing Tensors on the CPU, move all the
    Tensors to the specified GPU (or do nothing, if they should be on the
    CPU)."""
    from allennlp.common.util import int_to_device

    cuda_device = int_to_device(cuda_device)

    if cuda_device == torch.device('cpu') or not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {
            key: move_to_device(value, cuda_device)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [move_to_device(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, cuda_device)
                               for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, cuda_device) for item in obj)
    else:
        return obj


def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor,
                                   max_length: int) -> torch.BoolTensor:
    """Given a variable of shape `(batch_size,)` that represents the sequence
    lengths of each batch element, this function returns a `(batch_size,
    max_length)` mask variable.

    For example, if our input was `[2, 2, 3]`, with a `max_length` of 4, we'd
    return `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`. We require
    `max_length` here instead of just computing it from the input
    `sequence_lengths` because it lets us avoid finding the max, then copying
    that value from the GPU to the CPU so that we can use it to construct a new
    tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """Sort a batch first tensor by some specified lengths.

    # Parameters
    tensor : `torch.FloatTensor`, required.
        A batch first Pytorch tensor.
    sequence_lengths : `torch.LongTensor`, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    # Returns
    sorted_tensor : `torch.FloatTensor`
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : `torch.LongTensor`
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : `torch.LongTensor`
        Indices into the sorted_tensor such that
        `sorted_tensor.index_select(0, restoration_indices) == original_tensor`
    permutation_index : `torch.LongTensor`
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(
            sequence_lengths, torch.Tensor):
        raise ConfigurationError(
            'Both the tensor and sequence lengths must be torch.Tensors.')

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(
        0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(
        0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_dropout_mask(dropout_rate: float, tensor_for_masking: torch.Tensor):
    """Computes and returns an element-wise dropout mask for a given tensor,
    where each element in the mask is dropped out with probability
    dropout_probability.

    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    # Parameters
    dropout_rate : `float`, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : `torch.Tensor`, required.
    # Returns
    `torch.FloatTensor`
        A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_rate).
        This scaling ensures expected values and variances of the output of applying this mask
        and the original tensor are the same.
    """
    binary_mask = (torch.rand(tensor_for_masking.size()) > dropout_rate).to(
        tensor_for_masking.device)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_rate)
    return dropout_mask
