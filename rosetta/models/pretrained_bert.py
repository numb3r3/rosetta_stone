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
            self.bert_lm.model.embeddings.word_embeddings.weight)

        self.dropout = torch.nn.Dropout(kwargs.get('embeds_dropout_prob', 0.1))

        weight_decay_rate = kwargs.get('weight_decay_rate', 0)
        no_decay = kwargs.get('no_decay', ['bias', 'LayerNorm.weight'])

        self.optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(
            weight_decay_rate=weight_decay_rate, no_decay=no_decay)

    def get_optimizer_grouped_parameters(self,
                                         weight_decay_rate: float = 0.,
                                         no_decay=[],
                                         **kwargs):
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                weight_decay_rate,
            },
            {
                'params': [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay':
                0.0
            },
        ]
        return optimizer_grouped_parameters

    def forward(self, input_ids, masked_lm_labels, **kwargs):
        """Push data through the whole model and returns logits. The data will
        propagate through the language model and each of the attached
        prediction heads.

        :param kwargs: Holds all arguments that need to be passed to the language model and prediction head(s).
        :return: all logits as torch.tensor or multiple tensors.
        """

        # Run forward pass of language model
        sequence_output, pooled_output = self.bert_lm(input_ids, **kwargs)

        output = self.dropout(sequence_output)
        logits = self.predict_head(output)

        loss_per_sample = self.predict_head.logits_to_loss(
            logits=logits, labels=masked_lm_labels)

        loss = loss_per_sample.mean()
        metrics = {'loss': loss}
        predicts = {}

        return predicts, loss, metrics
