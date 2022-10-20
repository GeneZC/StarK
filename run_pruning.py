# -*- coding: utf-8 -*-

import os
import re
import json
import math
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers

from tqdm.auto import tqdm
import numpy as np
from data import get_reader_class, get_pipeline_class, Dataset
from metrics import get_metric_fn
from models import get_model_class
from utils import add_kwargs_to_config, Logger


logger = Logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Prune a transformers model.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained model.",    
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The task to train on, for indexing data reader.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Where to load a glue dataset.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/prune", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Use CPU or not.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.task_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    args.data_dir = os.path.join(args.data_dir, args.task_name)

    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")
    
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.add_stream_handler()
    logger.add_file_handler(args.output_dir)
    logger.set_verbosity_info() 
    
        
    # Load metric functin and data reader.
    metric_fn = get_metric_fn(args.task_name)
    data_reader = get_reader_class(args.task_name)(args.data_dir)
    label_map, reverse_label_map, num_labels = data_reader.get_label_map()

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # Pruning.
    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.model_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, label_map, args.max_length)

    dev_examples = data_reader.get_dev_examples()
    dev_examples = data_pipeline.build(dev_examples)

    dev_dataset = Dataset(dev_examples, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)

    config = config_class.from_pretrained(args.model_path)
    add_kwargs_to_config(
        config,
        num_labels=num_labels,
    )
    model = model_class.from_pretrained(
        args.model_path,
        config=config,
    )
    model = model.to(device)


    # Prune!
    logger.info("***** Running pruning (w. sanity check) *****")
    # Set student to pruned student with dev set.
    num_layers, num_heads, num_neurons = \
            config.num_hidden_layers, config.num_attention_heads, config.intermediate_size

    head_score = torch.zeros(num_layers, num_heads).to(device)
    head_mask = torch.ones(num_layers, num_heads).to(device)
    head_mask.requires_grad_(True)
    neuron_score = torch.zeros(num_layers, num_neurons).to(device)
    neuron_mask = torch.ones(num_layers, num_neurons).to(device)
    neuron_mask.requires_grad_(True)

    # Compute importance.
    model.eval()
    for batch in dev_loader:
        batch = [v.to(device) for k, v in batch._asdict().items()]
        output = model(batch, head_mask=head_mask, neuron_mask=neuron_mask)
        # Expressive score.
        if output.logit.shape[-1] == 1:
            loss = F.mse_loss(output.logit.squeeze(-1), output.label, reduction="mean")
        else:
            loss = F.cross_entropy(output.logit, output.label, reduction="mean")
        loss.backward()
        head_score += head_mask.grad.abs().detach()
        neuron_score += neuron_mask.grad.abs().detach()
        # Clear the gradients in case of potential overflow.
        head_mask.grad = None
        neuron_mask.grad = None
        model.zero_grad()
    
    # Normalize score.
    norm_per_layer = torch.pow(torch.pow(head_score, 2).sum(-1), 0.5)
    head_score /= norm_per_layer.unsqueeze(-1) + 1e-7
    norm_per_layer = torch.pow(torch.pow(neuron_score, 2).sum(-1), 0.5)
    neuron_score /= norm_per_layer.unsqueeze(-1) + 1e-7
            
    # Reorder for efficient indexing with module-wise sparsity.
    base_model = getattr(model, model.base_model_prefix, model)
    head_score, head_indices = torch.sort(head_score, dim=1, descending=True)
    neuron_score, neuron_indices = torch.sort(neuron_score, dim=1, descending=True)
    head_indices = {layer_idx: indices for layer_idx, indices in enumerate(head_indices)}
    neuron_indices = {layer_idx: indices for layer_idx, indices in enumerate(neuron_indices)}
    base_model.reorder(head_indices, neuron_indices)

    # Compute module-wise sparsity from overall sparsity.
    head_sort = [
        (layer_idx, head_score[layer_idx, head_idx].item())
        for layer_idx in range(num_layers)
        for head_idx in range(num_heads)
    ]

    head_sort = sorted(head_sort, key=lambda x: x[1])
    neuron_sort = [
        (layer_idx, neuron_score[layer_idx, neuron_idx].item())
        for layer_idx in range(num_layers)
        for neuron_idx in range(num_neurons)
    ]
    neuron_sort = sorted(neuron_sort, key=lambda x: x[1])
    
    num_total_heads = num_layers * num_heads
    num_total_neurons = num_layers * num_neurons
    
    sparsity_map = {str(s): {"head": {}, "neuron": {}} for s in range(0, 100, 10)}
    for sparsity in sparsity_map:
        heads_sparsified = head_sort[:round(float(sparsity) / 100 * num_total_heads)]
        for (layer_idx, _) in heads_sparsified:
            if str(layer_idx) not in sparsity_map[sparsity]["head"]:
                sparsity_map[sparsity]["head"][str(layer_idx)] = 0
            sparsity_map[sparsity]["head"][str(layer_idx)] += 1
        neurons_sparsified = neuron_sort[:round(float(sparsity) / 100 * num_total_neurons)]
        for (layer_idx, _) in neurons_sparsified:
            if str(layer_idx) not in sparsity_map[sparsity]["neuron"]:
                sparsity_map[sparsity]["neuron"][str(layer_idx)] = 0
            sparsity_map[sparsity]["neuron"][str(layer_idx)] += 1


    logger.info("***** Finalizing pruning *****")
    
    logger.info("***** Adding sparsity & sparsity map to config *****")
    config.sparsity = "0"
    config.sparsity_map = sparsity_map

    preds, labels = {s: [] for s in sparsity_map}, {s: [] for s in sparsity_map}
    with torch.no_grad():
        for batch in dev_loader:
            batch = [v.to(device) for k, v in batch._asdict().items()]
            for sparsity in sparsity_map:
                base_model.sparsify(sparsity)
                output = model(batch)
                pred, label = output.prediction, output.label
                preds[sparsity].extend(pred.cpu().numpy().tolist())
                labels[sparsity].extend(label.cpu().numpy().tolist())
    for sparsity in config.sparsity_map:
        dev_metric_at_sparsity = metric_fn(preds[sparsity], labels[sparsity])
        logger.info(f"  Verified dev metric at sparsity {sparsity} = {dev_metric_at_sparsity}")


    logger.info("***** Saving pruned model *****")
    save_path = os.path.join(args.output_dir, "ckpt")
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)
    model.save_pretrained(save_path)
    

if __name__ == "__main__":
    main()
