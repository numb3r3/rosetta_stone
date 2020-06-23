import argparse
import importlib
import os
import sys

from rosetta import __version__, helper
from termcolor import colored
import torch
import torch.nn.functional as F
from transformers import BertTokenizer


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


def main(args):
    logger = helper.set_logger("dialog_gpt2", verbose=True)
    hparams = helper.parse_args("app.yaml", args.model_name, "default")

    model_pkg = importlib.import_module(hparams["model_package"])
    model_cls_ = getattr(model_pkg, hparams.get("model_class", "Model"))
    model = model_cls_(**hparams)

    tokenizer = BertTokenizer(vocab_file=hparams["tokenizer_name_or_path"])

    # restore model
    if args.resume:
        resume_file = args.resume
        if os.path.isfile(resume_file):
            logger.info("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.error("=> no checkpoint found at '{}'".format(resume_file))
            sys.exit(1)

    model.eval()

    history = []
    print("开始和chatbot聊天，输入CTRL + Z以退出")

    while True:
        try:
            text = input("user:")

            history.append(tokenizer.encode(text))

            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(
                history[-hparams["max_history"] :]
            ):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)

            curr_input_tensor = torch.tensor(input_ids).long()

            generated = []
            # 最多生成max_len个token
            for _ in range(hparams["max_response_length"]):
                outputs, _, _ = model(context_input_ids=curr_input_tensor)
                next_token_logits = outputs["logits"][-1, :]

                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= hparams["repetition_penalty"]

                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float(
                    "Inf"
                )

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
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

            history.append(generated)
            text = tokenizer.convert_ids_to_tokens(generated)
            print("chatbot:" + "".join(text))

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

    parser.add_argument("model_name", type=str, help="the model name")

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
