# -*- coding: utf-8 -*-

from transformers import (
    BertTokenizer,
    BertConfig,
)

from models.ft import FT
from models.kd import KD

def get_model_class(model_type):
    if model_type == "ft":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = FT
    elif model_type == "kd":
        tokenizer_class = BertTokenizer
        config_class = BertConfig
        model_class = KD
    else:
        raise KeyError(f"Unknown model type {model_type}.")

    return tokenizer_class, config_class, model_class