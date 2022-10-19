#!/bin/bash

python run_finetuning.py \
    --model_type ft \
    --model_path path/to/bert-base-uncased \
    --task_name rte \
    --data_type combined \
    --selection_metric acc \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --log_interval 10 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_train