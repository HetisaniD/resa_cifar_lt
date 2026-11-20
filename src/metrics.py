# Evaluation metrics shared by every downstream script
# (eval_linear_cifar.py, eval_knn.py, eval_transfer.py).
#
# Why more than accuracy?  On a long-tailed problem a classifier can score a
# reasonable top-1 accuracy while almost never predicting the rare classes.
# The metrics below expose that:
#
#   per class      precision, recall, F1, one-vs-rest AUC, number of test
#                  images predicted as the class
#   macro          unweighted means of the per-class values
#   balanced acc.  mean per-class recall (equals top-1 accuracy when the test
#                  set is balanced, as it is for every dataset used here)
#   G-mean         geometric mean of per-class recall; collapses towards 0 as
#                  soon as one class is never recognised
#   shot groups    mean recall of many-/medium-/few-shot classes, defined by
#                  the number of training images (> 100, 20-100, < 20). This
#                  is the CIFAR-100-LT / ImageNet-LT convention.
#   frequency      mean recall of the head / medium / tail classes, defined by
#   groups         RANK: the 30% most frequent classes, the middle 40% and the
#                  30% rarest. Needed for CIFAR-10-LT, where the smallest
#                  class still has 50+ images, so the shot thresholds put
#                  nearly every class in "many".
#
# Both groupings are always computed from the LONG-TAILED training counts of
# the experiment (the reference distribution), not from the data the
# classifier happened to be trained on -- otherwise a classifier trained on
# balanced data would have every class in the same group.
#
# Units: acc1 / acc5 are percentages (0-100), as in the original ReSA code.
# Every other metric is a fraction in [0, 1].

import csv
import os

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

SUMMARY_KEYS = [
    "acc1", "acc5", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
    "balanced_accuracy", "gmean",
    "many_shot_acc", "medium_shot_acc", "few_shot_acc",
    "head_acc", "mid_acc", "tail_acc", "head_tail_gap",
    "min_class_recall", "n_zero_recall_classes",
]


# --------------------------------------------------------------------------
# Class groups
# --------------------------------------------------------------------------
def shot_groups(counts, many_thr=100, few_thr=20):
    """'many' / 'medium' / 'few' per class from training-image counts."""
    groups = []
    for c in counts:
        if c > many_thr:
            groups.append("many")
        elif c < few_thr:
            groups.append("few")
        else:
            groups.append("medium")
    return groups


def frequency_groups(counts, head_frac=0.3, tail_frac=0.3):
    """'head' / 'medium' / 'tail' per class by frequency RANK.

    The `head_frac` most frequent classes are the head, the `tail_frac`
    rarest are the tail (ties keep class-index order). For 10 classes this
    gives 3 / 4 / 3 and for 100 classes 30 / 40 / 30.
    """
    counts = np.asarray(counts)
    n = len(counts)
    order = np.argsort(-counts, kind="stable")  # most frequent first
    n_head = int(round(head_frac * n))
    n_tail = int(round(tail_frac * n))
    groups = np.array(["medium"] * n, dtype=object)
    groups[order[:n_head]] = "head"
    if n_tail:
        groups[order[n - n_tail:]] = "tail"
    return groups.tolist()


def _group_mean(values, groups, name):
    if groups is None:
        return float("nan")
    idx = np.where(np.asarray(groups) == name)[0]
    return float(values[idx].mean()) if len(idx) else float("nan")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def per_class_auc(y_true, y_score, num_classes):
    """One-vs-rest ROC AUC per class from the class scores (softmax
    probabilities, or vote shares for k-NN). NaN where undefined."""
    auc = np.full(num_classes, np.nan)
    if y_score is None:
        return auc
    for k in range(num_classes):
        pos = y_true == k
        if pos.any() and (~pos).any():
            auc[k] = roc_auc_score(pos, y_score[:, k])
    return auc


def compute_metrics(y_true, y_pred, y_score, num_classes, shot=None, freq=None, acc5=None):
    """All metrics for one set of predictions.

    y_true, y_pred: (N,) int arrays.
    y_score: (N, C) class scores, or None (AUC is then NaN).
    shot / freq: per-class group labels from shot_groups() / frequency_groups(),
        or None when the experiment has no long-tailed reference distribution.
    acc5: top-5 accuracy in percent if the caller computed it, else NaN.

    Returns (summary dict, confusion matrix, per-class dict).
    The confusion matrix has rows = true class, columns = predicted class.
    """
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    pc_precision, pc_recall, pc_f1, pc_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )  # precision of a class that is never predicted is reported as 0
    pc_auc = per_class_auc(y_true, y_score, num_classes)

    head = _group_mean(pc_recall, freq, "head")
    tail = _group_mean(pc_recall, freq, "tail")
    summary = {
        "acc1": float((y_true == y_pred).mean() * 100.0),
        "acc5": float("nan") if acc5 is None else float(acc5),
        "macro_precision": float(pc_precision.mean()),
        "macro_recall": float(pc_recall.mean()),
        "macro_f1": float(pc_f1.mean()),
        "macro_auc": float(np.nanmean(pc_auc)) if not np.all(np.isnan(pc_auc)) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "gmean": float(np.exp(np.mean(np.log(pc_recall + 1e-12)))),
        "many_shot_acc": _group_mean(pc_recall, shot, "many"),
        "medium_shot_acc": _group_mean(pc_recall, shot, "medium"),
        "few_shot_acc": _group_mean(pc_recall, shot, "few"),
        "head_acc": head,
        "mid_acc": _group_mean(pc_recall, freq, "medium"),
        "tail_acc": tail,
        "head_tail_gap": head - tail,
        "min_class_recall": float(pc_recall.min()),
        "n_zero_recall_classes": int((pc_recall == 0).sum()),
    }
    per_class = {
        "precision": pc_precision,
        "recall": pc_recall,
        "f1": pc_f1,
        "auc": pc_auc,
        "support": pc_support,
        "n_predicted": cm.sum(axis=0),
    }
    return summary, cm, per_class


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
def csv_num(v):
    """Round for a CSV cell; empty for NaN / None."""
    if v is None:
        return ""
    v = float(v)
    return "" if np.isnan(v) else round(v, 6)


def save_per_class_csv(path, per_class, class_names, train_counts, shot, lt_counts, freq, epoch=""):
    """One row per class (precision / recall / F1 / AUC are fractions).

    train_count    images of the class the classifier was trained on
    lt_train_count images of the class in the long-tailed reference set
    """
    n = len(per_class["recall"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "class_id", "class_name", "train_count", "shot_group",
            "test_count", "n_predicted", "precision", "recall", "f1", "auc",
            "lt_train_count", "freq_group",
        ])
        for c in range(n):
            writer.writerow([
                epoch, c, class_names[c] if c < len(class_names) else c,
                "" if train_counts is None else int(train_counts[c]),
                "" if shot is None else shot[c],
                int(per_class["support"][c]), int(per_class["n_predicted"][c]),
                csv_num(per_class["precision"][c]), csv_num(per_class["recall"][c]),
                csv_num(per_class["f1"][c]), csv_num(per_class["auc"][c]),
                "" if lt_counts is None else int(lt_counts[c]),
                "" if freq is None else freq[c],
            ])


def save_summary_csv(path, summary, epoch=""):
    """The macro metrics of one evaluation (acc1/acc5 in %, the rest in [0, 1])."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + SUMMARY_KEYS)
        writer.writerow([epoch] + [csv_num(summary.get(k)) for k in SUMMARY_KEYS])


def save_artifacts(metrics_dir, tag, summary, cm, per_class, y_true, y_pred, y_score,
                   class_names, train_counts, shot, lt_counts, freq, epoch=""):
    """Write everything needed to re-plot or re-analyse one evaluation.

    `tag` names the evaluation: 'final' / 'best' for the linear probe,
    'k5' / 'k10' / ... for k-NN, 'linear' for the transfer probe.

      confusion_matrix_<tag>.npy   (C, C) ints, rows = true, columns = predicted
      predictions_<tag>.npz        y_true, y_pred, y_prob (class scores)
      per_class_metrics_<tag>.csv  one row per class
      per_class_acc_<tag>.csv      per-class recall only (kept for older notebooks)
      summary_metrics_<tag>.csv    one row of macro metrics
    """
    os.makedirs(metrics_dir, exist_ok=True)
    np.save(os.path.join(metrics_dir, f"confusion_matrix_{tag}.npy"), cm)
    np.savetxt(os.path.join(metrics_dir, f"per_class_acc_{tag}.csv"), per_class["recall"], delimiter=",")
    arrays = dict(y_true=y_true, y_pred=y_pred)
    if y_score is not None:
        arrays["y_prob"] = y_score.astype(np.float32)
    np.savez(os.path.join(metrics_dir, f"predictions_{tag}.npz"), **arrays)
    save_per_class_csv(os.path.join(metrics_dir, f"per_class_metrics_{tag}.csv"),
                       per_class, class_names, train_counts, shot, lt_counts, freq, epoch)
    save_summary_csv(os.path.join(metrics_dir, f"summary_metrics_{tag}.csv"), summary, epoch)


def append_row(path, row, fieldnames):
    """Append one dict as a CSV row, writing the header if the file is new."""
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: (csv_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
