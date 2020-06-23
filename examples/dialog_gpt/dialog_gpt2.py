import torch
from torch import nn
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class DialogGPT2(nn.Module):
    def __init__(self, vocab_size: int, pretrained_model: str = None, **kwargs):
        super().__init__()
        if pretrained_model:
            # resume from a pretrained model
            self.gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        else:
            # start from scratch
            model_config = GPT2Config(**kwargs["gpt2_model_conf"])
            self.gpt2_model = GPT2LMHeadModel(config=model_config)

            # resize vocab size
            self.gpt2_model.resize_token_embeddings(vocab_size)

    def forward(
        self,
        context_input_ids,
        context_segment_ids=None,
        context_masks=None,
        lm_labels=None,
        **kwargs
    ):
        loss = None
        if lm_labels:
            # train mode
            loss, logits, *_ = self.gpt2_model(
                context_input_ids,
                labels=lm_labels,
                token_type_ids=context_segment_ids,
                attention_mask=context_masks,
            )
        else:
            logits, *_ = self.gpt2_model(
                context_input_ids,
                token_type_ids=context_segment_ids,
                attention_mask=context_masks,
            )

        predicts = {"logits": logits}
        metrics = {}

        if lm_labels:
            # measure accuracy and record loss
            acc1, acc5 = accuracy(
                logits.view(-1, logits.shape[-1]), lm_labels.view(-1), topk=(1, 5)
            )
            metrics["accuracy"] = acc1
            metrics["acc1"] = acc1
            metrics["acc5"] = acc5

        return predicts, loss, metrics
