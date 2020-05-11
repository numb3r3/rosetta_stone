"""Many of the modeling parts here come from the great transformers repository: 
https://github.com/huggingface/transformers.
Thanks for the great work! """

import os
from typing import Dict

import torch
from torch import nn
from transformers import modeling_bert

from .. import helper


class Bert(nn.Module):
    """
    A BERT model that wraps HuggingFace's implementation
    (https://github.com/huggingface/transformers) to fit the LanguageModel class.
    Paper: https://arxiv.org/abs/1810.04805
    """

    def __init__(self, pretrained_model_path: str = None, **kwargs):
        super().__init__()
        self.logger = helper.get_logger(__name__)

        if pretrained_model_path:
            self.restore_from_pretrained(pretrained_model_path)
        elif "bert_config_path" in kwargs:
            config_path = kwargs["bert_config_path"]
            config = modeling_bert.BertConfig.from_pretrained(config_path)
            self.model = modeling_bert.BertModel(config)
        else:
            raise ValueError("Pleas provide pretrained model or config path!")

    def restore_from_pretrained(self, pretrained_model_path: str):
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

    def forward(self, input_ids, segment_ids, padding_mask, **kwargs):
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
            input_ids, token_type_ids=segment_ids, attention_mask=padding_mask
        )
        if self.model.encoder.output_hidden_states:
            sequence_output, pooled_output, all_hidden_states = (
                output_tuple[0],
                output_tuple[1],
                output_tuple[2],
            )
            return sequence_output, pooled_output, all_hidden_states
        else:
            sequence_output, pooled_output = output_tuple[0], output_tuple[1]
            return sequence_output, pooled_output

    def get_output_dims(self):
        # These are the names of the attributes in various model configs which refer to the number of dimensions
        # in the output vectors
        OUTPUT_DIM_NAMES = ["dim", "hidden_size", "d_model"]

        config = self.model.config
        for odn in OUTPUT_DIM_NAMES:
            if odn in dir(config):
                return getattr(config, odn)
        else:
            raise Exception(
                "Could not infer the output dimensions of the language model"
            )

    def enable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = True

    def disable_hidden_states_output(self):
        self.model.encoder.output_hidden_states = False

    def save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "bert_config.json")
        with open(save_filename, "w") as file:
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # Save Weights
        save_filename = os.path.join(save_dir, "bert_model.bin")
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_filename)
        self.save_config(save_dir)
