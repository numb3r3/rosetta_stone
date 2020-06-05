from rosetta.models.language_model import BertLM
import torch
from torch import nn
import torch.nn.functional as F


class DSSMBert(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.bert_lm = BertLM(**kwargs)

        self.dropout = nn.Dropout(kwargs["dropout_rate"])
        self.context_fc = nn.Linear(kwargs["bert_hidden_size"], kwargs["fc_hidden_size"])
        self.response_fc = nn.Linear(kwargs["bert_hidden_size"], kwargs["fc_hidden_size"])

    def forward(
        self,
        context_input_ids,
        context_segment_ids,
        context_input_masks,
        responses_input_ids,
        responses_segment_ids,
        responses_input_masks,
        labels=None,
    ):
        # only select the first response (whose lbl==1)
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        # if isinstance(self.bert, DistilBertModel):
        #     context_vec = self.bert(context_input_ids, context_input_masks)[
        #         -1
        #     ]  # [bs,dim]
        #     context_vec = context_vec[:, 0]
        # else:
        #     context_vec = self.bert_lm(
        #         context_input_ids, context_input_masks, context_segment_ids
        #     )[
        #         -1
        #     ]  # [bs,dim]

        context_sequence_output, context_pooled_output = self.bert_lm(
            context_input_ids, context_input_masks, context_segment_ids
        )

        context_vec = self.context_fc(self.dropout(context_pooled_output))
        context_vec = F.normalize(context_vec, 2, -1)

        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_segment_ids = responses_segment_ids.view(-1, seq_length)

        # if isinstance(self.bert, DistilBertModel):
        #     responses_vec = self.bert(responses_input_ids, responses_input_masks)[
        #         -1
        #     ]  # [bs,dim]
        #     responses_vec = responses_vec[:, 0]
        # else:
        #     responses_vec = self.bert(
        #         responses_input_ids, responses_input_masks, responses_segment_ids
        #     )[
        #         -1
        #     ]  # [bs,dim]

        responses_sequence_output, responses_pooled_output = self.bert_lm(
            responses_input_ids, responses_input_masks, responses_segment_ids
        )

        responses_vec = responses_pooled_output.view(batch_size, res_cnt, -1)

        responses_vec = self.response_fc(self.dropout(responses_vec))
        responses_vec = F.normalize(responses_vec, 2, -1)

        predicts = {}
        metrics = {}
        loss = None

        if labels is not None:
            responses_vec = responses_vec.squeeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product * 5, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()

            metrics["loss"] = loss
        else:
            context_vec = context_vec.unsqueeze(1)
            dot_product = torch.matmul(
                context_vec, responses_vec.permute(0, 2, 1)
            )  # take this as logits
            dot_product.squeeze_(1)
            cos_similarity = (dot_product + 1) / 2
            predicts["cosine_sim"] = cos_similarity
            return cos_similarity

        return predicts, loss, metrics
