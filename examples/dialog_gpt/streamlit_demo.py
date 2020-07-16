import sys  # isort:skip
import streamlit as st  # isort:skip


@st.cache
def _set_sys_path():
    print('set sysm path ...')
    sys.path.append('.')


_set_sys_path()

import argparse
import importlib
from itertools import chain
import os

from rosetta import __version__, helper
import torch
import torch.nn.functional as F


MODEL_NAMES = ['gpt2_chitchat']

MODELS = [
    {
        'model_label': '基础预训练模型',
        'model_name': 'gpt2_chitchat',
        'resume_from': None
    },
    {
        'model_label':
        '正恒YJ-画心',
        'model_name':
        'gpt2_chitchat',
        'resume_from':
        '/workspace/project-nas-10251-sh/models/gpt2_chitchat_20200630',
    },
]

SPECIAL_TOKENS = ['[CLS]', '[SEP]', '[PAD]', '[speaker1]', '[speaker2]']


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
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
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def build_input_from_segments(history,
                              reply,
                              tokenizer,
                              with_eos: bool = True,
                              speaker_aware: bool = False):
    """Build a sequence of input from 3 segments: persona, history and last
    reply."""
    # bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = ([[tokenizer.cls_token_id]] + history +
                [reply + ([tokenizer.sep_token_id] if with_eos else [])])

    instance = {}
    if speaker_aware:
        sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s
                                    for i, s in enumerate(sequence[1:])]
        instance['token_type_ids'] = [bos] + [
            speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
            for _ in s
        ]

    instance['input_ids'] = list(chain(*sequence))

    return instance, sequence


def sample_sequence(history,
                    tokenizer,
                    model,
                    hparams,
                    generated=None,
                    speaker_aware=False):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if generated is None:
        generated = []

    for i in range(hparams['max_response_length']):
        instance, sequence = build_input_from_segments(
            history,
            generated,
            tokenizer,
            with_eos=False,
            speaker_aware=speaker_aware)
        input_ids = torch.tensor(
            instance['input_ids'], dtype=torch.long).unsqueeze(0)

        token_type_ids = None
        if 'token_type_ids' in instance:
            token_type_ids = torch.tensor(
                instance['token_type_ids'], dtype=torch.long).unsqueeze(0)

        outputs, _, _ = model(
            context_input_ids=input_ids, context_segment_ids=token_type_ids)
        next_token_logits = outputs['logits'][0, -1, :]

        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for id in set(generated):
            next_token_logits[id] /= hparams['repetition_penalty']

        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[tokenizer.convert_tokens_to_ids(
            '[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(
            next_token_logits,
            top_k=hparams['decode_topk'],
            top_p=hparams['decode_topp'],
        )

        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        generated.append(next_token.item())

    return generated


class ChatBot:

    def __init__(self):
        self.history = []
        self.responses = []
        self.inputs = []
        # self.models = []
        self.model = None
        self.tokenizer = None
        self.hparams = None

    def set_model(self, model, tokenizer, hparams):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams

    def get_bot_response(self, input_query: str, model_name: str):

        self.inputs.append(input_query)

        self.history.append(self.tokenizer.encode(input_query))

        history = self.history[-(self.hparams['max_history'] + 1):]

        with torch.no_grad():
            response_ids = sample_sequence(
                history,
                self.tokenizer,
                self.model,
                self.hparams,
                speaker_aware=False)

        self.history.append(response_ids)

        response = self.tokenizer.decode(
            response_ids, skip_special_tokens=True)
        self.responses.append(response)
        # print(colored("chatbot: " + "".join(response_text.split(" ")), "green"))
        return response

    def clear_history(self):
        self.inputs.clear()
        self.responses.clear()
        self.history.clear()


def main():
    logger = helper.set_logger('dialog_gpt2', verbose=True)

    st.title(':tv: &nbsp; 虚拟主播直播间')

    st.sidebar.title('虚拟主播对话模型')
    seleted_model = st.sidebar.selectbox(
        '选择模型', MODELS, format_func=lambda m: m['model_label'])
    model_name = seleted_model['model_name']
    model_label = seleted_model['model_label']
    resume_from = seleted_model['resume_from']

    # live_mode = st.sidebar.checkbox("直播模式", False)
    # dialog_mode = st.sidebar.checkbox("对话模式", True)

    st.sidebar.title('使用说明')
    st.sidebar.info(
        ':airplane: &nbsp; **目前支持的模型**\n'
        '- :one: &nbsp; **基础预训练模型**: 公开通用领域对话模型\n'
        '- :two: &nbsp; **正恒YJ-画心**: 御姐型人格对话模型\n'
        '\n:bulb: &nbsp; **注意事项**\n'
        '- :icecream: &nbsp; **单例模式**: 目前该系统只支持单例模式，不同用户同时访问时会看到其他用户的输入内容\n'
        '- :beers: &nbsp; **多轮对话**：目前默认记录最近**20轮**的对话信息\n'
        '- :clap: &nbsp; **清空按钮**: 点击该按钮后会清空所有对话记录，重新开启对话。\n'
        '&nbsp;&nbsp;&nbsp;&nbsp; :warning: 点击清空按钮后，**用户第一次的输入系统不会响应**，第二次输入才能正常工作 (这是一个 :bug:，后边会修复)。'
    )

    @st.cache(allow_output_mutation=True)
    def create_chatbot():
        return ChatBot()

    @st.cache(allow_output_mutation=True)
    def load_model(model_name, resume_from=None):
        from transformers import BertTokenizer

        print('loding model: %s' % model_name)
        hparams = helper.parse_args('app.yaml', model_name, 'default')
        model_pkg = importlib.import_module(hparams['model_package'])
        model_cls_ = getattr(model_pkg, hparams.get('model_class', 'Model'))

        model = model_cls_(**hparams)

        # restore model
        if resume_from:
            if os.path.isfile(resume_from):
                logger.info("=> loading checkpoint '{}'".format(resume_from))
                checkpoint = torch.load(
                    resume_from, map_location=torch.device('cpu'))
                model.load_state_dict(checkpoint['state_dict'])
            else:
                logger.error(
                    "=> no checkpoint found at '{}'".format(resume_from))
                # sys.exit(1)

        model.eval()

        tokenizer = BertTokenizer(vocab_file=hparams['tokenizer_name_or_path'])

        return model, tokenizer, hparams

    bot = create_chatbot()

    model, tokenizer, hparams = load_model(model_name, resume_from)
    bot.set_model(model, tokenizer, hparams)

    chat_container = st.markdown('### :rocket: 正在和 [%s] 聊天 ...' % model_label)

    # chat_placeholder = st.empty()
    # user_query = chat_placeholder.text_input("input_query", "")
    # user_query = user_query.strip()

    chat_input_ = st.empty()
    input_value = ''
    if st.button('清空历史'):
        input_value = ' '
        bot.clear_history()

    user_query = chat_input_.text_input('请输入 ...', input_value)
    user_query = str(user_query.strip())

    if user_query != '':

        resp = bot.get_bot_response(user_query, 'chitchat')
        print('query: %s' % user_query)
        print('resp: %s' % resp)

        for i in range(len(bot.inputs) - 1, -1, -1):
            st.markdown(f':space_invader:  **机器人**: {bot.responses[i]}')
            st.markdown(f':santa:  **用户** : {bot.inputs[i]} \n ---')

    # if app_mode == "Show instructions":
    #     st.sidebar.success('To continue select "Run the app".')
    # elif app_mode == "Show the source code":
    #     readme_text.empty()
    #     st.code(get_file_content_as_string("app.py"))
    # elif app_mode == "Run the app":
    #     readme_text.empty()
    #     # run_the_app()

    # hparams = helper.parse_args("app.yaml", args.model_name, "default")


if __name__ == '__main__':
    main()
