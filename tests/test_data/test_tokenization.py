import logging
import re

from rosetta.data.tokenization import (
    Tokenizer,
    tokenize_with_metadata,
    truncate_sequences,
)
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer


def test_basic_loading(caplog):
    caplog.set_level(logging.CRITICAL)
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-cased", do_lower_case=True
    )
    assert type(tokenizer) == BertTokenizer
    assert tokenizer.basic_tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="xlnet-base-cased", do_lower_case=True
    )
    assert type(tokenizer) == XLNetTokenizer
    assert tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(pretrained_model_name_or_path="roberta-base")
    assert type(tokenizer) == RobertaTokenizer


def test_bert_tokenizer_all_meta(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model, do_lower_case=False
    )

    basic_text = (
        "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"
    )

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == [
        "Some",
        "Text",
        "with",
        "never",
        "##see",
        "##nto",
        "##ken",
        "##s",
        "plus",
        "!",
        "215",
        "?",
        "#",
        ".",
        "and",
        "a",
        "combined",
        "-",
        "token",
        "_",
        "with",
        "/",
        "ch",
        "##ars",
    ]

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] == [
        0,
        5,
        10,
        15,
        20,
        23,
        26,
        29,
        31,
        36,
        37,
        40,
        41,
        42,
        44,
        48,
        50,
        58,
        59,
        64,
        65,
        69,
        70,
        72,
    ]
    assert tokenized_meta["start_of_word"] == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
