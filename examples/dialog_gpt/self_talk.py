import argparse
import importlib
from itertools import chain
import os
import sys

from rosetta import __version__, helper
from termcolor import colored
import torch
import torch.nn.functional as F
from transformers import BertTokenizer


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True
        )  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def build_input_from_segments(
    history, reply, tokenizer, with_eos=True, no_speaker_aware=False
):
    """Build a sequence of input from 3 segments: persona, history and last
    reply."""
    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [
        [speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])
    ]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    if not no_speaker_aware:
        instance["token_type_ids"] = [bos] + [
            speaker2 if i % 2 else speaker1
            for i, s in enumerate(sequence[1:])
            for _ in s
        ]
    return instance, sequence


def sample_sequence(
    history, tokenizer, model, hparams, generated=None, no_speaker_aware=False
):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if generated is None:
        generated = []

    for i in range(hparams["max_response_length"]):
        instance, sequence = build_input_from_segments(
            history,
            generated,
            tokenizer,
            with_eos=False,
            no_speaker_aware=no_speaker_aware,
        )
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long).unsqueeze(0)

        token_type_ids = None
        if "token_type_ids" in instance:
            token_type_ids = torch.tensor(
                instance["token_type_ids"], dtype=torch.long
            ).unsqueeze(0)

        outputs, _, _ = model(
            context_input_ids=input_ids, context_segment_ids=token_type_ids
        )
        next_token_logits = outputs["logits"][0, -1, :]

        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for id in set(generated):
            next_token_logits[id] /= hparams["repetition_penalty"]

        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")
        filtered_logits = top_k_top_p_filtering(
            next_token_logits,
            top_k=hparams["decode_topk"],
            top_p=hparams["decode_topp"],
        )

        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), num_samples=1
        )
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        generated.append(next_token.item())

    return generated


def main(args):
    logger = helper.set_logger("dialog_gpt2", verbose=True)

    first_hparams = helper.parse_args("app.yaml", args.first_model_name, "default")

    model_pkg = importlib.import_module(first_hparams["model_package"])
    model_cls_ = getattr(model_pkg, first_hparams.get("model_class", "Model"))
    first_model = model_cls_(**first_hparams)

    first_tokenizer = BertTokenizer(vocab_file=first_hparams["tokenizer_name_or_path"])

    second_hparams = helper.parse_args("app.yaml", args.second_model_name, "default")

    model_pkg = importlib.import_module(second_hparams["model_package"])
    model_cls_ = getattr(model_pkg, second_hparams.get("model_class", "Model"))
    second_model = model_cls_(**second_hparams)

    second_tokenizer = BertTokenizer(
        vocab_file=second_hparams["tokenizer_name_or_path"]
    )

    history = [first_tokenizer.encode("你好呀")]
    print("开始和chatbot聊天，输入CTRL + Z以退出")

    while True:
        try:
            # query = input(colored("user:", "cyan"))

            # history.append(tokenizer.encode(query))

            with torch.no_grad():
                response_ids = sample_sequence(
                    history,
                    first_tokenizer,
                    first_model,
                    first_hparams,
                    no_speaker_aware=True,
                )
                response_text = first_tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                print(
                    colored("chatbot #1: " + "".join(response_text.split(" ")), "green")
                )
                history.append(response_ids)

                response_ids = sample_sequence(
                    history,
                    second_tokenizer,
                    second_model,
                    second_hparams,
                    no_speaker_aware=False,
                )
                response_text = second_tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )
                print(
                    colored("chatbot #2: " + "".join(response_text.split(" ")), "cyan")
                )
                history.append(response_ids)

            history = history[-(2 * second_hparams["max_history"] + 1) :]
            input("Press Enter to continue...")
            # response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
            # print(colored("chatbot: " + "".join(response_text.split(" ")), "green"))

        except KeyboardInterrupt:
            break


def parse_args():
    # create the argument parser
    parser = argparse.ArgumentParser(
        description="%s, a toolkit based on pytorch. "
        "Visit %s for tutorials and documents."
        % (
            colored("DialoGPT2 v%s" % __version__, "green"),
            colored(
                "https://git.huya.com/wangfeng2/rosetta_stone",
                "cyan",
                attrs=["underline"],
            ),
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("first_model_name", type=str, help="the second model name")
    parser.add_argument("second_model_name", type=str, help="the second model name")

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    parser.add_argument(
        "--no_speaker_aware",
        action="store_true",
        default=False,
        help="disable speaker aware mode",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
