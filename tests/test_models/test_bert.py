import os

import pytest
from rosetta.models.language_model import BertLM
import torch


def test_model_from_scratch():
    with pytest.raises(ValueError):
        model = BertLM()

    params = {
        "bert_config_file": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
    }
    model = BertLM(**params)
