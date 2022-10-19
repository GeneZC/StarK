# -*- coding: utf-8 -*-

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from functools import partial

import numpy as np

"""
Metric Facotry:
    Get metric function. [task-specific]
"""


def acc_and_f1(preds, labels, average="macro"):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"acc": acc, "f1": f1, "acc_and_f1": (acc + f1) / 2}


def acc(preds, labels):
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {"acc": acc}


def f1(preds, labels, average="macro"):
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"f1": f1}


def matthews(preds, labels):
    matthews_corrcoef = matthews_corrcoef(y_true=labels, y_pred=preds)
    return {"matthews_corrcoef": matthews_corrcoef}


def pearson_and_spearman(preds, labels):

    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {"pearson_corr": pearson_corr, "spearman_corr": spearman_corr, "corr": (pearson_corr + spearman_corr) / 2}


METRIC_FN = {
    "sst2": acc,
    "mrpc": partial(acc_and_f1, average="binary"),
    "stsb": pearson_and_spearman,
    "qqp": partial(acc_and_f1, average="binary"),
    "mnli": acc,
    "mnlimm": acc,
    "qnli": acc,
    "rte": acc,
}


def get_metric_fn(task_name):
    return METRIC_FN[task_name]