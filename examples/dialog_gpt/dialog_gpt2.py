import torch
from torch import nn
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel


def compute_accuracy(logits, labels, device, pad_id=-100):
    """计算非pad_id的平均loss和准确率.

    :param outputs:
    :param labels:
    :param device:
    :return:
    """

    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    _, preds = shift_logits.max(
        dim=-1
    )  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(
        pad_id
    )  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
    correct = correct.float().sum()

    accuracy = correct / num_targets
    return accuracy


def accuracy(logits, target, topk=(1,), pad_id=0):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        _, preds = logits.max(
            dim=-1
        )  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        not_ignore = target.ne(
            pad_id
        )  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (target == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets

        return accuracy


class DialogGPT2(nn.Module):
    def __init__(self, vocab_size: int, pretrained_model: str = None, **kwargs):
        super().__init__()
        if pretrained_model:
            # resume from a pretrained model
            print("resume from a pretrained model: %s" % pretrained_model)
            self.gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        else:
            # start from scratch
            model_config = GPT2Config(**kwargs["gpt2_model_conf"])
            self.gpt2_model = GPT2LMHeadModel(config=model_config)

            # resize vocab size
            self.gpt2_model.resize_token_embeddings(vocab_size)

        # self.smoothing = LabelSmoothing(local_rank, self.vocab.size, self.vocab.padding_idx, smoothing_factor)

    def forward(
        self,
        context_input_ids,
        context_segment_ids=None,
        context_masks=None,
        lm_labels=None,
        **kwargs
    ):

        # logits, *_ = self.gpt2_model(
        #     context_input_ids,
        #     token_type_ids=context_segment_ids,
        #     attention_mask=context_masks,
        # )

        # predicts = {"logits": logits}
        loss = None
        metrics = {}

        # if lm_labels is not None:
        #     loss, accuracy = calculate_loss_and_accuracy(logits, lm_labels, logits.device, pad_id=0)
        #     metrics["accuracy"] = accuracy

        if lm_labels is not None:
            # train mode
            loss, logits, *_ = self.gpt2_model(
                context_input_ids,
                labels=lm_labels,
                token_type_ids=context_segment_ids,
                attention_mask=context_masks,
            )
            metrics["accuracy"] = compute_accuracy(
                logits, lm_labels, logits.device, pad_id=-100
            )

        else:
            logits, *_ = self.gpt2_model(
                context_input_ids,
                token_type_ids=context_segment_ids,
                attention_mask=context_masks,
            )

        predicts = {"logits": logits}

        return predicts, loss, metrics

    # def label_smotthing_loss(self, y_pred, y, y_mask, avg=True):
    #     seq_len, bsz = y.size()

    #     y_pred = torch.log(y_pred.clamp(min=1e-8))
    #     loss = self.smoothing(y_pred.view(seq_len * bsz, -1), y.view(seq_len * bsz, -1))
    #     if avg:
    #         return loss / torch.sum(y_mask)
    #     else:
    #         return loss / bsz
