import argparse
import importlib
from itertools import chain
import os
import sys

from rosetta import __version__, helper
import streamlit as st


sys.path.append(".")

# from termcolor import colored
# import torch
# import torch.nn.functional as F


MODEL_NAMES = ["gpt2_chitchat"]


class ChatBot:
    def __init__(self):
        # self.history = []
        self.responses = []
        self.inputs = []
        self.models = []

    def get_bot_response(self, input_query: str, model_name: str):
        self.inputs.append(input_query)
        response = "傻了吧！"
        self.responses.append(response)
        return response

    def clear_history(self):
        self.inputs.clear()
        self.responses.clear()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    pass


def main():
    # logger = helper.set_logger("dialog_gpt2", verbose=True)

    st.sidebar.title("智能对话")
    model_name = st.sidebar.selectbox("选择模型", MODEL_NAMES)
    st.sidebar.success('To continue select "Run the app".')

    @st.cache(allow_output_mutation=True)
    def create_chatbot():
        return ChatBot()

    @st.cache(allow_output_mutation=True)
    def load_model(model_name):
        hparams = helper.parse_args("app.yaml", model_name, "default")
        model_pkg = importlib.import_module(hparams["model_package"])
        model_cls_ = getattr(model_pkg, hparams.get("model_class", "Model"))

        model = model_cls_(**hparams)

        tokenizer = BertTokenizer(vocab_file=hparams["tokenizer_name_or_path"])

        # turn on eval mode
        model.eval()

        return model, tokenizer

    bot = create_chatbot()

    # model, tokenizer = load_model(model_name)

    chat_container = st.markdown("## 开始和 [%s] 对话 ..." % model_name)

    chat_placeholder = st.empty()
    user_query = chat_placeholder.text_input("input_query", "")
    user_query = user_query.strip()

    if user_query != "":
        resp = bot.get_bot_response(user_query, "chitchat")

        for i in range(len(bot.inputs) - 1, -1, -1):
            st.markdown(f"**Bot**: {bot.responses[i]}")
            st.markdown(f"**You**: {bot.inputs[i]} \n ---")

    # if app_mode == "Show instructions":
    #     st.sidebar.success('To continue select "Run the app".')
    # elif app_mode == "Show the source code":
    #     readme_text.empty()
    #     st.code(get_file_content_as_string("app.py"))
    # elif app_mode == "Run the app":
    #     readme_text.empty()
    #     # run_the_app()

    # hparams = helper.parse_args("app.yaml", args.model_name, "default")


if __name__ == "__main__":
    main()
