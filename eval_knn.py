# eval_knn.py
# -----------
# k-nearest-neighbour evaluation of a pretrained ReSA encoder on CIFAR-10 /
# CIFAR-100, for several values of k in one run.
#
# What it measures
#   The frozen momentum encoder turns every image into a feature vector. A
#   test image is given the class that wins a vote among its k nearest
#   training images (the "bank"). Nothing is trained, so the result reflects
#   the representation itself.
#
# The three probes (--probe) mirror the three linear probes of
# eval_linear_cifar.py and differ only in the bank:
#
#   long_tailed   bank = the long-tailed training subset, plain majority vote.
#                 Head classes dominate the bank, so they dominate the votes.
#   rebalanced    same long-tailed bank, but a neighbour of class c votes
#                 with weight 1 / n_c (see src/knn.py). Uses no extra labels.
#   balanced      bank = the full balanced training set, plain majority vote.
#                 A diagnostic of the representation with class imbalance
#                 removed from the classifier.
#
# The test set is always the official balanced test set.
#
# Output (in <dump_path>/metrics/)
#   knn_summary.csv              one row per k: accuracy and the long-tail
#                                metrics of src/metrics.py
#   per_class_metrics_k<K>.csv   precision / recall / F1 / AUC per class
#   confusion_matrix_k<K>.npy    rows = true class, columns = predicted
#   predictions_k<K>.npz         y_true, y_pred, y_prob (vote shares)
#   summary.json                 the settings and every result
# visualise_results.py plots any of them with --tag k<K>.
#
# Example
#   python eval_knn.py --dataset cifar10 --imb_ratio 100 --probe rebalanced \
#       --k 5 10 20 --seed 0 \
#       --pretrained /results/cifar10_lt100/checkpoint.pth.tar \
#       --data_path /datasets --dump_path /results/cifar10_lt100_knn_rebalanced

import argparse
import logging
import os
import time

import numpy as np

from src.datasets import (
    EVAL_STATS, NUM_CLASSES, base_name, load_arrays, long_tail_indices,
    reference_lt_counts, is_balanced_reference,
)
from src.features import build_frozen_encoder, cached_features
from src.knn import knn_classify
from src.metrics import (
    SUMMARY_KEYS, append_row, compute_metrics, frequency_groups, save_artifacts, shot_groups,
)
from src.utils import get_device, random_seed, setup_logging, write_json

PROBES = ("long_tailed", "rebalanced", "balanced")
ROW_FIELDS = (["dataset", "probe", "imb_ratio", "imb_type", "classifier", "k", "knn_metric", "seed",
               "n_train", "n_test"] + SUMMARY_KEYS + ["pretrained", "pretrain_epoch", "date"])


def get_args():
    p = argparse.ArgumentParser(description="ReSA k-NN evaluation on CIFAR-10/100 (balanced or long-tailed)")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "cifar10_lt", "cifar100_lt"],
                   help="evaluation dataset (the _lt suffix is accepted and ignored: the bank is set by --probe)")
    p.add_argument("--probe", type=str, default="long_tailed", choices=PROBES,
                   help="which training images vote, and how (see the header of this file)")
    p.add_argument("--imb_ratio", type=int, default=100,
                   help="imbalance ratio of the long-tailed reference distribution (largest / smallest "
                        "class). It defines the long-tailed bank and the head/medium/tail class groups. "
                        "Use the ratio the encoder was pretrained with; 1 = balanced.")
    p.add_argument("--imb_type", type=str, default="exp", choices=["exp", "step", "Dsample"])
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="neighbour counts to evaluate")
    p.add_argument("--knn_metric", type=str, default="euclidean", choices=["euclidean", "cosine"],
                   help="'euclidean' on raw features is the ReSA CIFAR protocol")
    p.add_argument("--pretrained", type=str, required=True, help="ReSA pretraining checkpoint (checkpoint.pth.tar)")
    p.add_argument("--pretrain_dataset", type=str, default=None,
                   help="dataset the encoder was pretrained on (default: --dataset); selects the network stem")
    p.add_argument("-a", "--arch", type=str, default="resnet18")
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--dump_path", type=str, required=True, help="output folder of this evaluation")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="keep extracted features here so other probes / runs reuse them (optional)")
    p.add_argument("--seed", type=int, default=0,
                   help="selects the images of the long-tailed subset; must equal the pretraining seed")
    p.add_argument("--many_shot_thr", type=int, default=100)
    p.add_argument("--few_shot_thr", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=500, help="images per forward pass")
    return p.parse_args()


def main():
    args = get_args()
    random_seed(args.seed)
    device = get_device()
    name = base_name(args.dataset)
    num_classes = NUM_CLASSES[name]
    metrics_dir = os.path.join(args.dump_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    setup_logging(os.path.join(args.dump_path, "out.log"), logging.INFO)
    logging.info(f"k-NN evaluation | dataset {name} | probe {args.probe} | IF {args.imb_ratio} | k {args.k}")

    # ---- data: full training set (bank candidates) and balanced test set ----
    x_train, y_train, class_names = load_arrays(name, args.data_path, train=True)
    x_test, y_test, _ = load_arrays(name, args.data_path, train=False)

    # ---- features of the frozen encoder (computed once, reused for every k) ----
    encoder, _, pretrain_epoch = build_frozen_encoder(
        args.arch, args.pretrained, args.pretrain_dataset or name, device)
    mean, std = EVAL_STATS[name]
    f_train = cached_features(args.cache_dir, encoder, args.pretrained, x_train, mean, std, device,
                              f"{name}_train", args.batch_size)
    f_test = cached_features(args.cache_dir, encoder, args.pretrained, x_test, mean, std, device,
                             f"{name}_test", args.batch_size)

    # ---- the bank for this probe ----
    lt_counts = np.array(reference_lt_counts(name, args.imb_ratio, args.imb_type))
    balanced_ref = is_balanced_reference(args.imb_ratio, args.imb_type)
    if args.probe == "balanced" or balanced_ref:
        bank_idx = np.arange(len(y_train))
    else:
        # exactly the images pretraining kept (same function, same seed)
        bank_idx = long_tail_indices(y_train, args.imb_ratio, args.imb_type, args.seed)
    bank_x, bank_y = f_train[bank_idx], y_train[bank_idx]
    bank_counts = np.bincount(bank_y, minlength=num_classes)
    class_weight = (1.0 / np.maximum(bank_counts, 1)) if args.probe == "rebalanced" else None
    logging.info(f"bank: {len(bank_y)} images, per-class counts {bank_counts.tolist()}")

    # class groups always come from the long-tailed reference distribution
    shot = shot_groups(lt_counts, args.many_shot_thr, args.few_shot_thr)
    freq = frequency_groups(lt_counts)

    # ---- k-NN for every k ----
    results = knn_classify(bank_x, bank_y, f_test, args.k, num_classes, metric=args.knn_metric,
                           class_weight=class_weight, device=device)
    summary_csv = os.path.join(metrics_dir, "knn_summary.csv")
    if os.path.isfile(summary_csv):
        os.remove(summary_csv)  # one invocation = one complete table
    all_results = {}
    for k, (y_pred, vote_share) in results.items():
        summary, cm, per_class = compute_metrics(y_test, y_pred, vote_share, num_classes, shot, freq)
        save_artifacts(metrics_dir, f"k{k}", summary, cm, per_class, y_test, y_pred, vote_share,
                       class_names, bank_counts, shot, lt_counts, freq)
        row = dict(dataset=name, probe=args.probe, imb_ratio=args.imb_ratio, imb_type=args.imb_type,
                   classifier=f"knn_k{k}", k=k, knn_metric=args.knn_metric, seed=args.seed,
                   n_train=len(bank_y), n_test=len(y_test), pretrained=args.pretrained,
                   pretrain_epoch=pretrain_epoch, date=time.strftime("%Y-%m-%d %H:%M:%S"), **summary)
        append_row(summary_csv, row, ROW_FIELDS)
        all_results[f"k{k}"] = summary
        logging.info(
            f" * k={k:<3d} acc {summary['acc1']:.2f}% | head {summary['head_acc']:.3f} "
            f"medium {summary['mid_acc']:.3f} tail {summary['tail_acc']:.3f} | "
            f"macroF1 {summary['macro_f1']:.3f} gmean {summary['gmean']:.3f} "
            f"never recognised: {summary['n_zero_recall_classes']} classes")

    write_json(os.path.join(metrics_dir, "summary.json"),
               dict(script="eval_knn", args=vars(args), pretrain_epoch=pretrain_epoch,
                    n_train=int(len(bank_y)), results=all_results))
    logging.info(f"Done. Results: {summary_csv}")


if __name__ == "__main__":
    main()
