## StarK <img src="./assets/ironman.png" width="22" height="22" alt="stark" align=center/>

This repository contains code for EMNLP 2022 paper titled [Sparse Teachers Can Be Dense with Knowledge](https://arxiv.org/abs/2210.03923).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* 19/10/2022: We released our paper and code. Check it out!

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [GLUE Data](#glue-data)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Recent advances in distilling pretrained language models have discovered that, besides the expressiveness of knowledge, the student-friendliness should be taken into consideration to realize a truly knowledgeable teacher. Based on a pilot study, we find that over-parameterized teachers can produce expressive yet student-unfriendly knowledge and are thus limited in overall knowledgeableness. To remove the parameters that result in student-unfriendliness, we propose a sparse teacher trick under the guidance of an overall knowledgeable score for each teacher parameter. The knowledgeable score is essentially an interpolation of the expressiveness and student-friendliness scores. The aim is to ensure that the expressive parameters are retained while the student-unfriendly ones are removed. Extensive experiments on the GLUE benchmark show that the proposed sparse teachers can be dense with knowledge and lead to students with compelling performance in comparison with a series of competitive baselines.

<img src="./assets/stark.png" alt="stark" align=center/>

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### GLUE Data

Get GLUE data through the [link](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) and put them to the corresponding directories. For example, MRPC dataset should be placed into `datasets/mrpc`.

### Training

The training is achieved in several scripts. We provide example scripts as follows.

**Finetuning**

We provide an example of finetuning `bert-base-uncased` on RTE in `scripts/run_finetuning_rte.sh`. We explain some important arguments in the following:
* `--model_type`: variant to use, should be `ft` in the case.
* `--model_path`: pretrained language models to start with, should be `bert-base-uncased` in the case and can be others as you like.
* `--task_name`: task to use, should be chosen from `rte`, `mrpc`, `stsb`, `sst2`, `qnli`, `qqp`, `mnli`, and `mnlimm`.
* `--data_type`: input format to use, default to `combined`.

We also give the finetuned checkpoints from `bert-base-uncased` and `bert-large-uncased` as follows:

|Model|Checkpoint|Model|Checkpoint|
|--|--|--|--|
|bert-base-rte|[huggingface](https://huggingface.co/GeneZC/bert-base-rte)|bert-large-rte|[huggingface](https://huggingface.co/GeneZC/bert-large-rte)|
|bert-base-mrpc|[huggingface](https://huggingface.co/GeneZC/bert-base-mrpc)|bert-large-mrpc|[huggingface](https://huggingface.co/GeneZC/bert-large-mrpc)|
|bert-base-stsb|[huggingface](https://huggingface.co/GeneZC/bert-base-stsb)|bert-large-stsb|[huggingface](https://huggingface.co/GeneZC/bert-large-stsb)|
|bert-base-sst2|[huggingface](https://huggingface.co/GeneZC/bert-base-sst2)|bert-large-sst2|[huggingface](https://huggingface.co/GeneZC/bert-large-sst2)|
|bert-base-qnli|[huggingface](https://huggingface.co/GeneZC/bert-base-qnli)|bert-large-qnli|[huggingface](https://huggingface.co/GeneZC/bert-large-qnli)|
|bert-base-qqp|[huggingface](https://huggingface.co/GeneZC/bert-base-qqp)|bert-large-qqp|[huggingface](https://huggingface.co/GeneZC/bert-large-qqp)|
|bert-base-mnli|[huggingface](https://huggingface.co/GeneZC/bert-base-mnli)|bert-large-mnli|[huggingface](https://huggingface.co/GeneZC/bert-large-mnli)|
|bert-base-mnlimm|[huggingface](https://huggingface.co/GeneZC/bert-base-mnlimm)|bert-large-mnlimm|[huggingface](https://huggingface.co/GeneZC/bert-large-mnlimm)|

**Pruning**

We provide and example of pruning a finetuned checkpoint on RTE in `scripts/run_pruning_rte.sh`. The arguments should be self-contained.

**Distillation**

We provide an example of distilling a finetuned teacher to a layer-dropped or parameter-pruned student on RTE in `scripts/run_distillation_rte.sh`. We explain some important arguments in following:
* `--model_type`: variant to use, should be `kd` in the case.
* `--teacher_model_path`: teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: student layer, should be set if you would like to use layer-dropped student, e.g., 4.

**Teacher Sparsification**

We provide an example of sparsfying the teacher based on the student on RTE in `scripts/run_sparsification_rte.sh`. We explain some important arguments in following:
* `--model_type`: variant to use, should be `kd` in the case.
* `--teacher_model_path`: teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: student models to use, should be the path to the distilled student checkpoint.
* `--student_sparsity`: student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness.

**Rewinding**

We provide an example of rewinding the student on RTE in `scripts/run_rewinding_rte.sh`. We explain some important arguments in following:
* `--model_type`: variant to use, should be `kd` in the case.
* `--teacher_model_path`: teacher models to use, should be the path to the sparsified teacher checkpoint.
* `--student_model_path`: student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: student layer, should be set if you would like to use layer-dropped student, e.g., 4.
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

