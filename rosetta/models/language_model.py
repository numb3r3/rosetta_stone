"""Many of the modeling parts here come from the great transformers repository: 
https://github.com/huggingface/transformers.
Thanks for the great work! """

import os
from typing import Dict

import torch
from torch import nn
from transformers import modeling_bert

from .. import helper


class BertLM(nn.Module):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.logger = helper.get_logger(__name__)

        if "pretrained_model_path" in kwargs:
            pretrained_model_path = kwargs["pretrained_model_path"]
            self._restore_from_pretrained(pretrained_model_path)
        elif "bert_config_file" in kwargs:
            config_path = kwargs["bert_config_file"]
            config = modeling_bert.BertConfig.from_pretrained(config_path)
            self.model = modeling_bert.BertModel(config)
        else:
            raise ValueError("Pleas provide pretrained model or config path!")
        self.extraction_layer = kwargs.get("extraction_layer", -1)

    def _restore_from_pretrained(self, pretrained_model_path: str, **kwargs):
        # We need to differentiate between loading model using custom format and Pytorch-Transformers format
        config_path = os.path.join(pretrained_model_path, "bert_config.json")
        if os.path.exists(config_path):
            bert_config = modeling_bert.BertConfig.from_pretrained(config_path)

            model_path = os.path.join(pretrained_model_path, "bert_model.bin")
            self.model = modeling_bert.BertModel.from_pretrained(
                model_path, config=bert_config, **kwargs
            )
        else:
            # Pytorch-transformer Style
            self.model = modeling_bert.BertModel.from_pretrained(
                pretrained_model_path, **kwargs
            )

    # def cross_entropy_one_hot(input, target):
    #     _, labels = target.max(dim=0)
    #     return nn.CrossEntropyLoss()(input, labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        **kwargs
    ):
        """
        Perform the forward pass of the BERT model.
        :param input_ids: The ids of each token in the input sequence. Is a tensor of shape [batch_size, max_seq_len]
        :type input_ids: torch.Tensor
        :param segment_ids: The id of the segment. For example, in next sentence prediction, the tokens in the
           first sentence are marked with 0 and those in the second are marked with 1.
           It is a tensor of shape [batch_size, max_seq_len]
        :type segment_ids: torch.Tensor
        :param padding_mask: A mask that assigns a 1 to valid input tokens and 0 to padding tokens
           of shape [batch_size, max_seq_len]
        :return: Embeddings for each token in the input sequence.
        """

        output_tuple = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        if self.model.encoder.output_hidden_states == True:
            sequence_output, pooled_output, all_hidden_states = (
                output_tuple[0],
                output_tuple[1],
                output_tuple[2],
            )
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

        # sequence_output = all_hidden_states[self.extraction_layer]

        # sequence_output = self.dropout(sequence_output)

        # # not available in earlier layers
        # if self.extraction_layer != -1:
        #     pooled_output = None
        # return (sequence_output, pooled_output)

        # # Forward pass through model
        # logits = self.model.forward(**batch)
        # per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)

        # if self.model.encoder.output_hidden_states:
        #     sequence_output, pooled_output, all_hidden_states = (
        #         output_tuple[0],
        #         output_tuple[1],
        #         output_tuple[2],
        #     )
        #     return ((sequence_output, pooled_output, all_hidden_states), loss, metrics)
        # else:
        #     sequence_output, pooled_output = output_tuple[0], output_tuple[1]
        #     return ((sequence_output, pooled_output), loss, metrics)

        # loss = None
        # metrics = {}
        # predicts = {"sequence_output": sequence_output, "pooled_output": pooled_output}

        # return predicts, loss, metrics
