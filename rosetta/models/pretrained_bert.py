import torch

from ..modules.prediction_head import BertLMHead
from .language_model import BertLM


class PretrainedBert(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bert_lm = BertLM(**kwargs)

        self.predict_head = BertLMHead(**kwargs)
        # set shared weights for LM finetuning
        self.predict_head.set_shared_weights(
            self.bert_lm.model.embeddings.word_embeddings.weight
        )

        self.dropout = torch.nn.Dropout(kwargs.get("embeds_dropout_prob", 0.1))

    def forward(
        self,
        input_ids,
        masked_lm_labels,
        **kwargs
    ):
        """
        Push data through the whole model and returns logits. The data will propagate through the language
        model and each of the attached prediction heads.

        :param kwargs: Holds all arguments that need to be passed to the language model and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """

        # Run forward pass of language model
        sequence_output, pooled_output = self.bert_lm(input_ids, **kwargs)
        
        output = self.dropout(sequence_output)
        logits = self.predict_head(output)

        loss_per_sample = self.predict_head.logits_to_loss(
            logits=logits, labels=masked_lm_labels
        )

        loss = loss_per_sample.mean()
        metrics = {}
        predicts = {}

        return predicts, loss, metrics
