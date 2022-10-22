#!/bin/bash

# parameter-pruned
python run_rewinding.py \
    --model_type kd \
    --teacher_model_path outputs/stark/parameter_pruned/kd_70_rte_0.5/ckpt \
    --student_model_path outputs/prune/ft_rte/ckpt \
    --task_name rte \
    --data_type combined \
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
    --selection_metric acc \
    --seed 776 \
    --do_rewind \
    --student_sparsity 70 \
    --lam 0.5

# layer-dropped
python run_rewinding.py \
    --model_type kd \
    --teacher_model_path outputs/stark/layer_dropped/kd_4_rte_0.5/ckpt \
    --student_model_path outputs/finetune/ft_rte/ckpt \
    --task_name rte \
    --data_type combined \
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
    --selection_metric acc \
    --seed 776 \
    --do_rewind \
    --student_layer 4 \
    --lam 0.5