import json
import logging
from typing import List, Optional, Union

import dataclasses
from dataclasses import dataclass
from transformers.tokenization_utils import PreTrainedTokenizer

from . import DataProcessor


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """A single set of features of data.

    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


class SentenceClassificationProcessor(DataProcessor):
    """Generic processor for a single sentence classification data set."""

    def __init__(
        self, examples=None, labels=None, mode="classification", verbose=False
    ):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(
                labels=self.labels, examples=self.examples[idx]
            )
        return self.examples[idx]

    def add_examples_from_csv(
        self,
        file_name: str,
        split_name: str = "train",
        column_label: int = 0,
        column_text: int = 1,
        ignore_header: bool = False,
    ):
        lines = self._read_csv(file_name)
        if ignore_header:
            lines = lines[1:]
        examples = []
        for (i, line) in enumerate(lines):
            text = line[column_text]
            label = line[column_label]
            guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label)
            )

        return examples

    def get_features(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = None,
        pad_on_left: bool = False,
        pad_token: int = 0,
        mask_padding_with_zero: bool = True,
        return_tensors: bool = None,
    ):
        """
        Convert examples in a list of ``InputFeatures``
        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}

        all_input_ids = []
        for (ex_index, example) in enumerate(self.examples):
            input_ids = tokenizer.encode(
                example.text_a,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )
            all_input_ids.append(input_ids)

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        features = []
        for (ex_index, (input_ids, example)) in enumerate(
            zip(all_input_ids, self.examples)
        ):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )

            assert (
                len(input_ids) == batch_length
            ), "Error with input length {} vs {}".format(len(input_ids), batch_length)
            assert (
                len(attention_mask) == batch_length
            ), "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if self.mode == "classification":
                label = label_map[example.label]
            elif self.mode == "regression":
                label = float(example.label)
            else:
                raise ValueError(self.mode)

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
                )
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, label=label
                )
            )

        return features
