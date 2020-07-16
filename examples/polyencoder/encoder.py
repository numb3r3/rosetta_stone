from rosetta.models.language_model import BertLM
import torch
from torch import nn
import torch.nn.functional as F


def dot_attention(q, k, v, v_mask=None, dropout=None):
    attention_weights = torch.matmul(q, k.transpose(-1, -2))
    if v_mask is not None:
        attention_weights *= v_mask.unsqueeze(1)
    attention_weights = F.softmax(attention_weights, -1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, v)
    return output


class PolyDSSMBert(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        bert_pretrained_model_path = kwargs.pop("bert_pretrained_model_path")

        self.bert_lm = BertLM(
            pretrained_model_path=bert_pretrained_model_path, from_tf=True
        )

        self.poly_m = kwargs.pop("poly_m")
        self.poly_code_embeddings = nn.Embedding(
            self.poly_m + 1, kwargs["bert_hidden_size"]
        )

        self.dropout = nn.Dropout(kwargs["dropout_rate"])
        self.context_fc = nn.Linear(
            kwargs["bert_hidden_size"], kwargs["fc_hidden_size"]
        )
        self.response_fc = nn.Linear(
            kwargs["bert_hidden_size"], kwargs["fc_hidden_size"]
        )

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

        batch_size, res_cnt, seq_length = responses_input_ids.shape

        ## poly context encoder
        context_sequence_output, context_pooled_output = self.bert_lm(
            context_input_ids, context_input_masks, context_segment_ids
        )

        poly_code_ids = torch.arange(
            self.poly_m, dtype=torch.long, device=context_input_ids.device
        )
        poly_code_ids += 1
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        context_vecs = dot_attention(
            poly_codes,
            context_sequence_output,
            context_sequence_output,
            context_input_masks,
            self.dropout,
        )

        context_vec = self.context_fc(self.dropout(context_vecs))
        context_vec = F.normalize(context_vec, 2, -1)

        ## response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_segment_ids = responses_segment_ids.view(-1, seq_length)

        responses_sequence_output, responses_pooled_output = self.bert_lm(
            responses_input_ids, responses_input_masks, responses_segment_ids
        )

        poly_code_ids = torch.zeros(
            batch_size * res_cnt, 1, dtype=torch.long, device=responses_input_ids.device
        )
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        responses_vec = dot_attention(
            poly_codes,
            responses_sequence_output,
            responses_sequence_output,
            responses_input_masks,
            self.dropout,
        )
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        responses_vec = self.response_fc(self.dropout(responses_vec))
        responses_vec = F.normalize(responses_vec, 2, -1)

        ## poly final context vector aggregation
        if labels is not None:
            responses_vec = responses_vec.view(1, batch_size, -1).expand(
                batch_size, batch_size, self.vec_dim
            )
        final_context_vec = dot_attention(
            responses_vec, context_vecs, context_vecs, None, self.dropout
        )
        final_context_vec = F.normalize(
            final_context_vec, 2, -1
        )  # [bs, res_cnt, dim], res_cnt==bs when training

        dot_product = torch.sum(
            final_context_vec * responses_vec, -1
        )  # [bs, res_cnt], res_cnt==bs when training

        predicts = {}
        metrics = {}
        loss = None

        if labels is not None:
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product * 5, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
        else:
            cos_similarity = (dot_product + 1) / 2
            predicts["cosine_sim"] = cos_similarity
            return cos_similarity

        return predicts, loss, metrics


class DSSMBert(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        bert_pretrained_model_path = kwargs.pop("bert_pretrained_model_path")

        self.bert_lm = BertLM(
            pretrained_model_path=bert_pretrained_model_path, from_tf=True
        )

        self.dropout = nn.Dropout(kwargs["dropout_rate"])
        self.context_fc = nn.Linear(
            kwargs["bert_hidden_size"], kwargs["fc_hidden_size"]
        )
        self.response_fc = nn.Linear(
            kwargs["bert_hidden_size"], kwargs["fc_hidden_size"]
        )

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

        context_sequence_output, context_pooled_output = self.bert_lm(
            context_input_ids, context_input_masks, context_segment_ids
        )

        context_vec = self.context_fc(self.dropout(context_pooled_output))
        context_vec = F.normalize(context_vec, 2, -1)

        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_segment_ids = responses_segment_ids.view(-1, seq_length)

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
