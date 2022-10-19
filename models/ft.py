# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from modules.modeling_sparsebert import SparseBertModel

import collections


class FT(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            "logit",
            "prediction", 
            "label",
        )
    )
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseBertModel(config)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.init_weights()

    def forward(self, inputs, head_mask=None, neuron_mask=None):
        text_indices, text_mask, text_segments, label = inputs

        # Gather knowledge.
        if neuron_mask is None:
            hidden_states = \
                self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        else:
            hidden_states = \
                self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments, head_mask=head_mask, neuron_mask=neuron_mask)[0]

        # Mapping.
        # There is no mapping for KD.
        # Logit.
        
        logit = self.cls(hidden_states[:, 0])
        # Mask and reshape.
        # There is no mask or reshape for KD.

        if logit.shape[-1] == 1:
            prediction = logit.squeeze(-1)
        else:
            prediction = logit.argmax(-1)

        return FT.Output(
            logit=logit,
            prediction=prediction, 
            label=label,
        )

    @staticmethod
    def loss_fn(output):
        if output.logit.shape[-1] == 1:
            loss = F.mse_loss(output.logit.squeeze(-1), output.label, reduction="mean")
        else:
            loss = F.cross_entropy(output.logit, output.label, reduction="mean")
        return loss

        
