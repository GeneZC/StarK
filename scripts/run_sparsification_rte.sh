#!/bin/bash

# parameter-pruned
python run_sparsification.py \
    --model_type kd \
    --teacher_model_path outputs/finetune/ft_rte/ckpt \
    --student_model_path outputs/distil/parameter_pruned/kd_70_rte/ckpt \
    --task_name rte \
    --data_type combined \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --student_sparsity 70 \
    --lam 0.5

# layer-dropped
python run_sparsification.py \
    --model_type kd \
    --teacher_model_path outputs/finetune/ft_rte/ckpt \
    --student_model_path outputs/distil/layer_dropped/kd_4_rte/ckpt \
    --task_name rte \
    --data_type combined \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --student_layer 4 \
    --lam 0.5

