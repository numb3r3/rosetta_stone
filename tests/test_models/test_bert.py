import os

import pytest
from rosetta.models.bert import Bert
import torch


def test_model_from_scratch():
    with pytest.raises(ValueError):
        model = Bert()

    params = {
        "bert_config_path": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json"
    }

    model = Bert(**params)
