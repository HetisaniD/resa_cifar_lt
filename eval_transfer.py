# eval_transfer.py
# ----------------
# Transfer evaluation: how well do the features of a ReSA encoder pretrained
# on one dataset (the SOURCE: CIFAR-10 or CIFAR-100, balanced or long-tailed)
# classify another dataset (the TARGET: CIFAR-10, CIFAR-100 or Tiny ImageNet)?
#
# The encoder stays frozen. Two classifiers are trained on its features of
# the target TRAINING set and scored on the target TEST set:
#
#   linear   one fully connected layer, with the same optimisation as the
#            linear probe of eval_linear_cifar.py: the CIFAR protocol of the
#            ReSA paper (Adam, 500 epochs, learning rate 1e-2 decayed
#            exponentially to 1e-6, weight decay 5e-6, batch 1000, no
#            augmentation, final epoch reported). --optimizer sgd gives the
#            earlier scaffold (SGD, momentum 0.9, cosine, 100 epochs, batch
#            256). Because the encoder is frozen and nothing is augmented,
#            each image has one fixed feature vector; the layer is trained on
#            those precomputed vectors, which is mathematically the same
#            procedure and takes seconds instead of hours.
#   knn      k-nearest-neighbour vote for every k in --k (default 5 10 20).
#
# All three targets are balanced, so the classifier sees balanced labels;
# any imbalance in the experiment lies in how the SOURCE encoder was
# pretrained.
#
# Preprocessing choices (both configurable):
#   --img_size 32  Tiny ImageNet's 64x64 images are resized to 32x32, the
#                  resolution the encoders were pretrained at.
#   --norm source  images are normalised with the SOURCE dataset's channel
#                  statistics, i.e. exactly as the encoder saw its inputs
#                  during pretraining ('target' uses the target's own).
#
# Output (in <dump_path>/metrics/)
#   transfer_summary.csv          one row per classifier (linear, knn_k5, ...)
#   linear_train_log.csv          loss / train / test accuracy per epoch
#   per_class_metrics_<tag>.csv, confusion_matrix_<tag>.npy,
#   predictions_<tag>.npz         tag = linear, k5, k10, k20
#   summary.json                  the settings and every result
#
# Example
#   python eval_transfer.py --source_dataset cifar10 --target tiny_imagenet \
#       --pretrained /results/cifar10_balanced/checkpoint.pth.tar \
#       --data_path /datasets --dump_path /results/transfer/cifar10_balanced__to__tiny_imagenet

import argparse
import csv
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.datasets import DATASETS, EVAL_STATS, NUM_CLASSES, base_name, load_arrays
from src.features import build_frozen_encoder, cached_features
from src.knn import knn_classify
from src.metrics import SUMMARY_KEYS, append_row, compute_metrics, save_artifacts
from src.utils import get_device, random_seed, setup_logging, write_json

# Same protocol defaults as eval_linear_cifar.py, so the transfer probe and
# the in-domain probe are trained the same way.
OPTIMIZER_DEFAULTS = {
    "adam": dict(epochs=500, batch_size=1000, lr_classifier=1e-2, lr_end=1e-6, wd=5e-6),
    "sgd":  dict(epochs=100, batch_size=256,  lr_classifier=1.0,  lr_end=0.0,  wd=0.0),
}

ROW_FIELDS = (["source_dataset", "target", "classifier", "k", "knn_metric", "img_size", "norm", "seed",
               "n_train", "n_test"] + SUMMARY_KEYS
              + ["optimizer", "epochs", "lr_classifier", "pretrained", "pretrain_epoch", "date"])


def get_args():
    p = argparse.ArgumentParser(description="ReSA transfer evaluation (linear + k-NN on frozen features)")
    p.add_argument("--source_dataset", type=str, required=True,
                   choices=["cifar10", "cifar100", "cifar10_lt", "cifar100_lt"],
                   help="dataset the encoder was pretrained on (network stem and --norm source statistics)")
    p.add_argument("--target", type=str, required=True, choices=list(DATASETS),
                   help="dataset the classifiers are trained and tested on")
    p.add_argument("--pretrained", type=str, required=True, help="ReSA pretraining checkpoint (checkpoint.pth.tar)")
    p.add_argument("-a", "--arch", type=str, default="resnet18")
    p.add_argument("--classifiers", type=str, nargs="+", default=["linear", "knn"], choices=["linear", "knn"])
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--knn_metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    # linear probe (defaults = eval_linear_cifar.py: the paper's CIFAR protocol)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                   help="'adam': the paper's protocol (500 epochs, lr 1e-2 -> 1e-6 exponential, "
                        "weight decay 5e-6, batch 1000); 'sgd': momentum 0.9 with a cosine schedule")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr_classifier", type=float, default=None)
    p.add_argument("--lr_end", type=float, default=None,
                   help="final learning rate of the exponential schedule (Adam only)")
    p.add_argument("--wd", type=float, default=None)
    # data
    p.add_argument("--data_path", type=str, default="./data",
                   help="folder holding the CIFAR folders and tiny-imagenet-200.zip (or its cache)")
    p.add_argument("--tiny_zip", type=str, default=None, help="path of tiny-imagenet-200.zip if not in --data_path")
    p.add_argument("--img_size", type=int, default=32, help="resolution Tiny ImageNet is resized to")
    p.add_argument("--norm", type=str, default="source", choices=["source", "target"])
    p.add_argument("--dump_path", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=None, help="keep extracted features here (optional)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature_batch_size", type=int, default=500)
    args = p.parse_args()
    for key, value in OPTIMIZER_DEFAULTS[args.optimizer].items():   # explicit values win
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.optimizer == "adam" and not 0 < args.lr_end <= args.lr_classifier:
        p.error(f"--lr_end ({args.lr_end}) must be > 0 and <= --lr_classifier ({args.lr_classifier})")
    return args


def train_linear(f_train, y_train, f_test, y_test, num_classes, args, device, log_path):
    """Train one linear layer on fixed features; returns (test predictions,
    test softmax scores, top-5 accuracy in %) of the final epoch."""
    x = torch.as_tensor(f_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xt = torch.as_tensor(f_test, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_test, dtype=torch.long, device=device)

    fc = nn.Linear(x.shape[1], num_classes).to(device)
    fc.weight.data.normal_(mean=0.0, std=0.01)
    fc.bias.data.zero_()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(fc.parameters(), lr=args.lr_classifier, weight_decay=args.wd)
        gamma = (args.lr_end / args.lr_classifier) ** (1.0 / max(args.epochs, 1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        optimizer = torch.optim.SGD(fc.parameters(), lr=args.lr_classifier, momentum=0.9,
                                    weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(args.seed)  # reproducible shuffling
    # Adam (the paper's protocol) uses every image of each epoch; the older
    # SGD scaffold dropped an incomplete last batch.
    drop_last = args.optimizer == "sgd"
    n_batches = len(x) // args.batch_size if drop_last else -(-len(x) // args.batch_size)
    if n_batches == 0:
        raise ValueError(f"the target training set ({len(x)} images) is smaller than --batch_size {args.batch_size}")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "train_acc1", "test_acc1", "lr"])
    for epoch in range(args.epochs):
        fc.train()
        perm = torch.randperm(len(x), generator=gen).to(device)
        loss_sum, correct = 0.0, 0
        seen = 0
        for b in range(n_batches):
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            seen += len(idx)
            out = fc(x[idx])
            loss = criterion(out, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == y[idx]).sum().item()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        fc.eval()
        with torch.no_grad():
            logits = fc(xt)
        test_acc = (logits.argmax(1) == yt).float().mean().item() * 100
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss_sum / max(n_batches, 1),
                                    100.0 * correct / max(seen, 1), test_acc, lr])
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logging.info(f"linear epoch {epoch:3d} | loss {loss_sum / max(n_batches, 1):.4f} | test acc {test_acc:.2f}%")

    top5 = logits.topk(min(5, num_classes), dim=1).indices
    acc5 = (top5 == yt[:, None]).any(dim=1).float().mean().item() * 100
    return logits.argmax(1).cpu().numpy(), torch.softmax(logits.float(), dim=1).cpu().numpy(), acc5


def main():
    args = get_args()
    random_seed(args.seed)
    device = get_device()
    source = base_name(args.source_dataset)
    target = args.target
    num_classes = NUM_CLASSES[target]
    metrics_dir = os.path.join(args.dump_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    setup_logging(os.path.join(args.dump_path, "out.log"), logging.INFO)
    logging.info(f"transfer | source {args.source_dataset} -> target {target} | "
                 f"classifiers {args.classifiers} | linear optimiser {args.optimizer} "
                 f"({args.epochs} epochs, lr {args.lr_classifier}, batch {args.batch_size})")

    # ---- target data ----
    x_train, y_train, class_names = load_arrays(target, args.data_path, train=True,
                                                img_size=args.img_size, tiny_zip=args.tiny_zip)
    x_test, y_test, _ = load_arrays(target, args.data_path, train=False,
                                    img_size=args.img_size, tiny_zip=args.tiny_zip)
    num_classes = max(num_classes, len(class_names)) if class_names else num_classes
    if len(class_names) != num_classes:  # e.g. a partial copy of the dataset
        logging.warning(f"expected {NUM_CLASSES[target]} classes for {target}, found {len(class_names)}")
        num_classes = len(class_names)
    logging.info(f"target data: {x_train.shape} train / {x_test.shape} test, {num_classes} classes")

    # ---- frozen source encoder -> features of the target images ----
    encoder, _, pretrain_epoch = build_frozen_encoder(args.arch, args.pretrained, source, device)
    mean, std = EVAL_STATS[source if args.norm == "source" else target]
    tag = f"{target}{x_train.shape[1]}px_norm-{args.norm}-{source}"
    f_train = cached_features(args.cache_dir, encoder, args.pretrained, x_train, mean, std, device,
                              f"{tag}_train", args.feature_batch_size)
    f_test = cached_features(args.cache_dir, encoder, args.pretrained, x_test, mean, std, device,
                             f"{tag}_test", args.feature_batch_size)
    del encoder
    train_counts = np.bincount(y_train, minlength=num_classes)

    summary_csv = os.path.join(metrics_dir, "transfer_summary.csv")
    if os.path.isfile(summary_csv):
        os.remove(summary_csv)  # one invocation = one complete table
    all_results = {}

    def record(tag_, classifier, k, summary, cm, per_class, y_pred, y_score):
        # the targets are balanced, so there are no long-tail class groups
        save_artifacts(metrics_dir, tag_, summary, cm, per_class, y_test, y_pred, y_score,
                       class_names, train_counts, None, None, None)
        row = dict(source_dataset=args.source_dataset, target=target, classifier=classifier, k=k,
                   knn_metric=args.knn_metric if k else "", img_size=x_train.shape[1], norm=args.norm,
                   seed=args.seed, n_train=len(y_train), n_test=len(y_test),
                   optimizer=args.optimizer if classifier == "linear" else "",
                   epochs=args.epochs if classifier == "linear" else "",
                   lr_classifier=args.lr_classifier if classifier == "linear" else "",
                   pretrained=args.pretrained,
                   pretrain_epoch=pretrain_epoch, date=time.strftime("%Y-%m-%d %H:%M:%S"), **summary)
        append_row(summary_csv, row, ROW_FIELDS)
        all_results[tag_] = summary
        logging.info(f" * {classifier:<8s} acc {summary['acc1']:.2f}% | macroF1 {summary['macro_f1']:.3f} "
                     f"| worst class recall {summary['min_class_recall']:.3f}")

    # ---- linear classifier ----
    if "linear" in args.classifiers:
        y_pred, y_prob, acc5 = train_linear(f_train, y_train, f_test, y_test, num_classes, args, device,
                                            os.path.join(metrics_dir, "linear_train_log.csv"))
        summary, cm, per_class = compute_metrics(y_test, y_pred, y_prob, num_classes, acc5=acc5)
        record("linear", "linear", "", summary, cm, per_class, y_pred, y_prob)

    # ---- k-NN ----
    if "knn" in args.classifiers:
        results = knn_classify(f_train, y_train, f_test, args.k, num_classes,
                               metric=args.knn_metric, device=device)
        for k, (y_pred, vote_share) in results.items():
            summary, cm, per_class = compute_metrics(y_test, y_pred, vote_share, num_classes)
            record(f"k{k}", f"knn_k{k}", k, summary, cm, per_class, y_pred, vote_share)

    write_json(os.path.join(metrics_dir, "summary.json"),
               dict(script="eval_transfer", args=vars(args), pretrain_epoch=pretrain_epoch,
                    n_train=int(len(y_train)), results=all_results))
    logging.info(f"Done. Results: {summary_csv}")


if __name__ == "__main__":
    main()
