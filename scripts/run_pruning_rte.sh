#!/bin/bash

python run_pruning.py \
    --model_type ft \
    --model_path outputs/finetune/ft_rte/ckpt \
    --task_name rte \
    --data_type combined \
    --max_length 128 \
    --per_device_eval_batch_size 32
