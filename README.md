## StarK <img src="./assets/ironman.png" width="22" height="22" alt="stark" align=center/>

This repository contains code for EMNLP 2022 paper titled [Sparse Teachers Can Be Dense with Knowledge](https://arxiv.org/abs/2210.03923).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* 10/19/22: We released our paper and code. Check it out!

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [GLUE Data](#glue-data)
    - [Training & Evaluation](#training&evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Recent advances in distilling pretrained language models have discovered that, besides the expressiveness of knowledge, the student-friendliness should be taken into consideration to realize a truly knowledgeable teacher. Based on a pilot study, we find that over-parameterized teachers can produce expressive yet student-unfriendly knowledge and are thus limited in overall knowledgeableness. To remove the parameters that result in student-unfriendliness, we propose a sparse teacher trick under the guidance of an overall knowledgeable score for each teacher parameter. The knowledgeable score is essentially an interpolation of the expressiveness and student-friendliness scores. The aim is to ensure that the expressive parameters are retained while the student-unfriendly ones are removed. Extensive experiments on the GLUE benchmark show that the proposed sparse teachers can be dense with knowledge and lead to students with compelling performance in comparison with a series of competitive baselines.

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### GLUE Data

Get GLUE data through the [link](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) and put it to the corresponding directory. For example, MRPC dataset should be placed into `datasets/mrpc`.

### Training & Evaluation

The training and evaluation are achieved in several scripts. We provide example scripts as follows.

**Finetuning**

We provide an example of finetuning `bert-base-uncased` on RTE in `scripts/run_finetuning_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `ft` in the case.
* `--model_path`: Pretrained language models to start with, should be `bert-base-uncased` in the case and can be others as you like.
* `--task_name`: Task to use, should be chosen from `rte`, `mrpc`, `stsb`, `sst2`, `qnli`, `qqp`, `mnli`, and `mnlimm`.
* `--data_type`: Input format to use, default to `combined`.

**Pruning**

We provide and example of pruning a finetuned checkpoint on RTE in `scripts/run_pruning_rte.sh`. The arguments should be self-contained.

**Distillation**

We provide an example of distilling a finetuned teacher to a layer-dropped or parameter-pruned student on RTE in `scripts/run_distillation_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.

**Teacher Sparsification**

We provide an example of sparsfying the teacher based on the student on RTE in `scripts/run_sparsification_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to use, should be the path to the distilled student checkpoint.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness.

**Rewinding**

We provide an example of rewinding the student on RTE in `scripts/run_rewinding_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the sparsified teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness. Here, it is just used for folder names.

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code in your work:

```bibtex
@inproceedings{yang2022sparse,
   title={Sparse Teachers Can Be Dense with Knowledge},
   author={Yang, Yi and Zhang, Chen and Song, Dawei},
   booktitle={EMNLP},
   year={2022}
}
```

