# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# eval_linear_cifar.py
# --------------------
# Linear-probe evaluation of a pretrained ReSA encoder on CIFAR-10 / CIFAR-100.
#
# A single fully connected layer is trained on top of the frozen momentum
# encoder and scored on the official, balanced test set.
#
# PROTOCOL (the default): the CIFAR linear evaluation of the ReSA paper
# (Weng et al., ICML 2025, Appendix B.3), inherited from W-MSE (Ermolov et
# al., 2021) and INTL (Weng et al., 2024) and implemented upstream in
# eval/evaluate.py:
#
#   Adam, 500 epochs, no data augmentation, learning rate decayed
#   exponentially from 1e-2 to 1e-6, weight decay 5e-6, batch size 1000,
#   the FINAL epoch reported.
#
# Because the encoder is frozen and nothing is augmented, every image has one
# fixed feature vector: the features are extracted once and the layer is
# trained on them (--precompute_features, on by default). That is the same
# computation as running the encoder every epoch, and it is what makes 500
# epochs take about a minute instead of hours.
#
# --optimizer sgd keeps the earlier scaffold of this file (SGD, momentum 0.9,
# cosine schedule, 100 epochs, batch 256), which follows `eval_linear.py` of
# https://github.com/winci-ai/resa. Use it only to reproduce results that
# were produced with it; state which optimiser was used when reporting.
#
# THE THREE PROBES (--probe)
# The central finding of the study is that the score of a long-tailed
# encoder depends strongly on which labels the linear layer is trained on.
# The same frozen encoder is therefore evaluated three ways:
#
#   long_tailed   trained on the long-tailed training subset with plain
#                 cross-entropy. The layer learns the class prior and
#                 over-predicts head classes.
#   rebalanced    same long-tailed subset, but each image is drawn with
#                 probability 1 / n_c, so every class is seen equally often
#                 (class-balanced sampling, as in classifier re-training,
#                 cRT; Kang et al., 2020). Uses no extra labels.
#   balanced      trained on the full balanced training set. Not a fair
#                 comparison with methods that only see long-tailed labels;
#                 a diagnostic of what the representation contains.
#
# --imb_ratio names the long-tailed distribution the encoder was pretrained
# on. It builds the long-tailed subset (with --seed, which must equal the
# pretraining seed so that the same images are selected) and defines the
# head / medium / tail class groups used in the metrics for ALL three probes.
#
# OUTPUT (in <dump_path>/metrics/) -- see src/metrics.py for the metrics
#   train_log.csv / val_log.csv        one row per epoch
#   summary_metrics_final.csv          macro metrics of the FINAL epoch  <- report these
#   per_class_metrics_final.csv        precision / recall / F1 / AUC per class
#   confusion_matrix_final.npy         rows = true class, columns = predicted
#   predictions_final.npz              y_true, y_pred, y_prob
#   *_best.*                           the same for the epoch with the best
#                                      --best_metric. That epoch is chosen on
#                                      the TEST set, so these files are
#                                      optimistically biased: use them for
#                                      inspection only, never as results.
#   summary.json                       the settings plus both result sets
#
# USAGE
#   python eval_linear_cifar.py --dataset cifar10 --imb_ratio 100 --probe rebalanced --seed 0 \
#       --pretrained /results/cifar10_lt100/checkpoint.pth.tar \
#       --data_path /datasets --dump_path /results/cifar10_lt100_probe_rebalanced
#
# Every optimisation setting of the protocol above is the default, so the
# command needs none of them. The older SGD scaffold:
#   ... --optimizer sgd --epochs 100 --batch_size 256 --lr_classifier 10.0
#
# The older spelling still works and means the same thing:
#   --dataset cifar10_lt --imb_ratio 100                  (long_tailed)
#   --dataset cifar10_lt --imb_ratio 100 --class_balanced (rebalanced)
#   --dataset cifar10                                     (balanced; add
#       --imb_ratio to get head/medium/tail groups in the metrics)
#
# Every run needs its own --dump_path. Re-running a command resumes it; a
# folder that holds a run made with different settings is refused unless
# --overwrite is given.
#
# Modified for the long-tailed ReSA study -- see CHANGES.md.

import argparse
import csv
import json
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100

from src.cifar_lt import apply_long_tail
from src.datasets import EVAL_STATS, base_name, is_balanced_reference, reference_lt_counts
from src.metrics import compute_metrics, frequency_groups, save_artifacts, shot_groups
from src.model import get_encoder
from src.utils import (
    AverageMeter,
    accuracy,
    init_distributed_device,
    load_pretrained_encoder,
    random_seed,
    restart_from_checkpoint,
    save_checkpoint,
    setup_logging,
    write_json,
)

PROBES = ("long_tailed", "rebalanced", "balanced")

# Columns of val_log.csv (one row per evaluated epoch).
VAL_LOG_KEYS = [
    "loss", "acc1", "acc5",
    "many_shot_acc", "medium_shot_acc", "few_shot_acc",
    "macro_f1", "macro_recall", "macro_precision", "macro_auc",
    "balanced_accuracy", "gmean",
    "head_acc", "mid_acc", "tail_acc", "n_zero_recall_classes",
]

# Settings that define a probe run; compared with the checkpoint on resume.
RUN_DEFINING_ARGS = [
    "dataset", "probe", "imb_ratio", "imb_type", "img_max", "seed", "arch", "pretrained",
    "weights", "optimizer", "epochs", "batch_size", "lr_classifier", "lr_end", "lr_encoder",
    "wd", "scheduler", "milestones", "augment",
]

# The protocol of the ReSA paper for CIFAR (Appendix B.3) and the earlier SGD
# scaffold of this file. Whatever the user passes explicitly wins.
OPTIMIZER_DEFAULTS = {
    #            epochs  batch_size  lr_classifier  lr_end  wd      scheduler
    "adam": dict(epochs=500, batch_size=1000, lr_classifier=1e-2, lr_end=1e-6, wd=5e-6, scheduler="exp"),
    "sgd":  dict(epochs=100, batch_size=256,  lr_classifier=1.0,  lr_end=0.0,  wd=0.0,  scheduler="cos"),
}


# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="ReSA CIFAR linear-probe evaluation")

    # data
    parser.add_argument("--dataset", type=str, default="cifar10",
                         choices=["cifar10", "cifar100", "cifar10_lt", "cifar100_lt"],
                         help="evaluation dataset. With --probe the _lt suffix is optional and ignored.")
    parser.add_argument("--probe", type=str, default=None, choices=PROBES,
                         help="which labels the linear layer is trained on: long_tailed, rebalanced or "
                              "balanced (see the header of this file). The test set is always balanced.")
    parser.add_argument("--data_path", type=str, default="./data", help="path to dataset root")
    parser.add_argument("--dump_path", type=str, default=".", help="where to write logs/checkpoints/metrics")
    parser.add_argument("--num_classes", type=int, default=None,
                         help="overrides the number of classes inferred from --dataset")
    parser.add_argument("--augment", action="store_true",
                         help="use RandomCrop(32,pad=4)+HFlip for training (default: no augmentation, "
                              "matching the ReSA CIFAR eval protocol)")
    parser.add_argument("--workers", type=int, default=4)

    # model / pretrained weights
    parser.add_argument("-a", "--arch", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=str, default="", metavar="PATH",
                         help="path to the ReSA pretraining checkpoint (checkpoint.pth.tar)")
    parser.add_argument("--weights", type=str, default="freeze", choices=["freeze", "finetune"],
                         help="'freeze' = linear probe (the protocol of the study); "
                              "'finetune' also updates the encoder")

    # optimization. The defaults of epochs / batch_size / lr_classifier /
    # lr_end / wd / scheduler follow --optimizer (see OPTIMIZER_DEFAULTS);
    # any value given on the command line is used as given.
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                         help="'adam': the CIFAR linear evaluation protocol of the ReSA paper "
                              "(500 epochs, lr 1e-2 -> 1e-6 exponential, weight decay 5e-6, batch 1000). "
                              "'sgd': the earlier scaffold of this file (momentum 0.9, cosine, 100 "
                              "epochs, batch 256) -- for reproducing results made with it.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr_encoder", type=float, default=0.0002, metavar="LR",
                         help="learning rate of the encoder; only used with --weights finetune")
    parser.add_argument("--lr_classifier", type=float, default=None, metavar="LR",
                         help="learning rate of the linear layer (default: 1e-2 with Adam, as in the "
                              "paper); state the value you used when reporting.")
    parser.add_argument("--wd", type=float, default=None, help="weight decay of the linear layer")
    parser.add_argument("--lr_end", type=float, default=None,
                         help="final learning rate of the exponential schedule (default 1e-6)")
    parser.add_argument("--scheduler", type=str, default=None, choices=["step", "cos", "exp"],
                         help="'exp' decays the learning rate from --lr_classifier to --lr_end over "
                              "--epochs (the paper's schedule); default: exp with Adam, cos with SGD")
    parser.add_argument("--milestones", type=int, nargs="+", default=[60, 80],
                         help="used only when --scheduler step")
    parser.add_argument("--precompute_features", type=str, default="auto",
                         choices=["auto", "on", "off"],
                         help="extract the frozen encoder's features once and train the linear layer "
                              "on them instead of running the encoder every epoch. Identical result, "
                              "much faster. 'auto' = on unless --weights finetune or --augment.")
    parser.add_argument("--seed", type=int, default=None,
                         help="random seed; also selects the images of the long-tailed subset "
                              "(None behaves as 0 there). Must equal the pretraining seed.")

    # long-tailed reference distribution and class groups
    parser.add_argument("--imb_ratio", type=int, default=None,
                         help="imbalance ratio (largest class / smallest class) of the long-tailed "
                              "distribution the encoder was pretrained on: 20, 50, 100, ... "
                              "Use 1 for a balanced-pretrained encoder. Default: 100 for a *_lt "
                              "dataset without --probe.")
    parser.add_argument("--imb_type", type=str, default="exp", choices=["exp", "step", "Dsample"],
                         help="long-tail profile (see src/cifar_lt.py)")
    parser.add_argument("--img_max", type=int, default=None,
                         help="largest class size override (see args.py); normally unset")
    parser.add_argument("--many_shot_thr", type=int, default=100,
                         help="classes with > this many long-tailed training images are 'many-shot'")
    parser.add_argument("--few_shot_thr", type=int, default=20,
                         help="classes with < this many are 'few-shot'; the rest are 'medium-shot'")
    parser.add_argument("--class_balanced", action="store_true",
                         help="older spelling of --probe rebalanced (with a *_lt dataset)")

    # model selection / eval cadence / resuming
    parser.add_argument("--best_metric", type=str, default="balanced_accuracy",
                         choices=["acc1", "balanced_accuracy", "macro_f1", "gmean"],
                         help="metric used to pick the 'best' epoch (inspection only -- report 'final')")
    parser.add_argument("--eval_every", type=int, default=1, help="run full validation every N epochs")
    parser.add_argument("--overwrite", action="store_true",
                         help="delete whatever run is in --dump_path and start again")
    parser.add_argument("--force_resume", action="store_true",
                         help="resume even if the checkpoint was made with different settings")

    # dist params (required by src.utils.init_distributed_device)
    parser.add_argument("--dist-backend", default=None, type=str,
                         help="default: 'nccl' on GPU, 'gloo' on CPU")
    parser.add_argument("--dist-url", default="env://", type=str)
    parser.add_argument("--no-set-device-rank", default=False, action="store_true")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()
    resolve_probe(args, parser)
    resolve_optimizer(args, parser)
    if args.num_classes is None:
        args.num_classes = 10 if base_name(args.dataset) == "cifar10" else 100
    return args


def resolve_optimizer(args, parser):
    """Fill the optimisation settings the user did not give with the
    defaults of --optimizer, and decide whether features are precomputed."""
    for key, value in OPTIMIZER_DEFAULTS[args.optimizer].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.scheduler == "exp":
        if not 0 < args.lr_end <= args.lr_classifier:
            parser.error(f"--lr_end ({args.lr_end}) must be > 0 and <= --lr_classifier "
                         f"({args.lr_classifier}) for the exponential schedule")

    # Features can only be reused when the encoder is frozen and the images
    # it sees never change.
    cacheable = args.weights == "freeze" and not args.augment
    if args.precompute_features == "on" and not cacheable:
        parser.error("--precompute_features on needs --weights freeze and no --augment "
                     "(otherwise the features change during training)")
    args.precompute_features = args.precompute_features != "off" and cacheable


def resolve_probe(args, parser):
    """Turn (--dataset, --probe, --class_balanced, --imb_ratio) into one
    consistent description: args.probe, args.imb_ratio, args.train_long_tailed."""
    is_lt_name = args.dataset.endswith("_lt")
    if args.probe is None:
        # older spelling: the dataset name decides
        if is_lt_name:
            args.probe = "rebalanced" if args.class_balanced else "long_tailed"
            if args.imb_ratio is None:
                args.imb_ratio = 100
        else:
            if args.class_balanced:
                parser.error("--class_balanced needs a long-tailed dataset (cifar10_lt / cifar100_lt)")
            args.probe = "balanced"
            if args.imb_ratio is None:
                args.imb_ratio = 1  # no long-tailed reference: no class groups
    else:
        if args.imb_ratio is None:
            parser.error("--probe needs --imb_ratio: the imbalance ratio the encoder was pretrained "
                         "with (use 1 for a balanced-pretrained encoder)")
    args.class_balanced = args.probe == "rebalanced"
    balanced_ref = is_balanced_reference(args.imb_ratio, args.imb_type)
    # the classifier trains on the long-tailed subset for these two probes
    args.train_long_tailed = args.probe in ("long_tailed", "rebalanced") and not balanced_ref


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_datasets(args):
    name = base_name(args.dataset)
    normalize = transforms.Normalize(*EVAL_STATS[name])  # CIFAR-10 or CIFAR-100 statistics

    if args.augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])

    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    ds_cls = CIFAR10 if name == "cifar10" else CIFAR100
    train_dataset = ds_cls(root=args.data_path, train=True, download=True, transform=train_tf)
    val_dataset = ds_cls(root=args.data_path, train=False, download=True, transform=test_tf)

    if args.train_long_tailed:
        # Same function and same seed as pretraining (src/transform.py), so
        # the classifier sees exactly the images the encoder was trained on.
        seed = args.seed if args.seed is not None else 0
        num_per_cls = apply_long_tail(
            train_dataset, imb_ratio=args.imb_ratio, imb_type=args.imb_type, seed=seed,
            img_max=args.img_max,
        )
        logging.info(
            f"Long-tailed {name} ({args.imb_type}, imb_ratio={args.imb_ratio}): "
            f"{len(train_dataset.data)} train images kept, per-class counts: {num_per_cls}"
        )
    # val_dataset is intentionally left untouched -- always the official balanced test set

    return train_dataset, val_dataset


@torch.no_grad()
def encode_dataset(encoder, dataset, args, device, what):
    """Run the frozen encoder once over `dataset` and return a dataset of
    (feature, label) pairs in the same order.

    Valid only while the encoder is frozen and the images do not change
    (no augmentation): then every image has one fixed feature vector, so
    training the linear layer on these features is the same computation as
    running the encoder every epoch -- a few hundred times cheaper.
    """
    loader = DataLoader(dataset, batch_size=500, num_workers=args.workers,
                        pin_memory=True, shuffle=False, drop_last=False)
    encoder.eval()
    feats, labels = [], []
    start = time.time()
    for images, target in loader:
        feats.append(encoder(images.to(device, non_blocking=True)).float().cpu())
        labels.append(target.clone())
    features = torch.cat(feats)
    targets = torch.cat(labels)
    logging.info(f"encoded {what}: {tuple(features.shape)} features in {time.time() - start:.1f} s")
    return TensorDataset(features, targets), targets.numpy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LinearProbeModel(nn.Module):
    def __init__(self, encoder, fc):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, x):
        return self.fc(self.encoder(x))


# --------------------------------------------------------------------------
# Train / validate
# --------------------------------------------------------------------------
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    if args.weights == "finetune":
        model.train()
    else:
        model.eval()  # keep encoder (incl. BN running stats) frozen in eval mode

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            logging.info(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
            )
    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(val_loader, model, criterion, shot, freq, args):
    """Score the balanced test set. Returns (summary, confusion matrix,
    per-class metrics, y_true, y_pred, y_prob)."""
    batch_time, losses = AverageMeter(), AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()

    model.eval()

    all_targets, all_preds, all_probs = [], [], []
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        all_targets.append(target.cpu().numpy())
        all_preds.append(output.argmax(dim=1).cpu().numpy())
        all_probs.append(torch.softmax(output.float(), dim=1).cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            logging.info(
                f"Test: [{i}/{len(val_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs).astype(np.float64)
    summary, cm, per_class = compute_metrics(
        y_true, y_pred, y_prob, args.num_classes, shot, freq, acc5=top5.avg)
    summary["loss"] = losses.avg

    logging.info(
        f" * Acc@1 {summary['acc1']:.3f} Acc@5 {summary['acc5']:.3f} | "
        f"head {summary['head_acc']:.3f} medium {summary['mid_acc']:.3f} tail {summary['tail_acc']:.3f} | "
        f"many {summary['many_shot_acc']:.3f} medium-shot {summary['medium_shot_acc']:.3f} "
        f"few {summary['few_shot_acc']:.3f} | "
        f"macroF1 {summary['macro_f1']:.3f} macroRecall {summary['macro_recall']:.3f} "
        f"macroPrecision {summary['macro_precision']:.3f} macroAUC {summary['macro_auc']:.3f} "
        f"bacc {summary['balanced_accuracy']:.3f} gmean {summary['gmean']:.3f}"
    )
    return summary, cm, per_class, y_true, y_pred, y_prob


# --------------------------------------------------------------------------
# Log files
# --------------------------------------------------------------------------
def init_metrics_dir(args, lt_counts, shot, freq, resuming):
    metrics_dir = os.path.join(args.dump_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    train_log_path = os.path.join(metrics_dir, "train_log.csv")
    val_log_path = os.path.join(metrics_dir, "val_log.csv")

    # per-class group assignment (needed to label / group any plot)
    with open(os.path.join(metrics_dir, "class_shot_groups.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "train_count", "shot_group", "freq_group"])
        for c, (count, s, g) in enumerate(zip(lt_counts, shot, freq)):
            writer.writerow([c, int(count), s, g])

    # Start the per-epoch logs afresh for a new run; keep them when resuming
    # (they used to be truncated on every start, losing the earlier epochs).
    if not (resuming and os.path.isfile(train_log_path) and os.path.isfile(val_log_path)):
        with open(train_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "acc1", "acc5", "lr"])
        with open(val_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch"] + VAL_LOG_KEYS)

    return metrics_dir, train_log_path, val_log_path


def append_csv(path, values):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(values)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = get_args()
    random_seed(args.seed)
    device, args = init_distributed_device(args)
    name = base_name(args.dataset)

    ckp_path = os.path.join(args.dump_path, "checkpoint.pth.tar")
    if args.overwrite and args.local_rank == 0 and os.path.isdir(args.dump_path):
        for leftover in ("checkpoint.pth.tar", "checkpoint.pth.tar.prev", "best.pth.tar", "metrics"):
            p = os.path.join(args.dump_path, leftover)
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)
    resuming = os.path.isfile(ckp_path) or os.path.isfile(ckp_path + ".prev")

    os.makedirs(args.dump_path, exist_ok=True)
    setup_logging(os.path.join(args.dump_path, "out.log"), logging.INFO)
    if args.local_rank != 0:
        def log_pass(*a, **k): pass
        logging.info = log_pass

    run_args = {k: getattr(args, k) for k in RUN_DEFINING_ARGS}
    run_args["dataset"] = name  # "cifar10_lt" and "cifar10 --probe ..." describe the same run
    logging.info(f"Linear probe | dataset {name} | probe {args.probe} | imbalance ratio {args.imb_ratio} "
                 f"({args.imb_type}) | weights {args.weights} | optimiser {args.optimizer}")

    train_dataset, val_dataset = build_datasets(args)
    class_names = list(getattr(val_dataset, "classes", [])) or [str(c) for c in range(args.num_classes)]

    # Images per class the classifier is trained on ...
    train_counts = np.bincount(np.array(train_dataset.targets), minlength=args.num_classes)
    # ... and in the long-tailed reference distribution, which defines the
    # class groups for every probe (so the balanced probe is grouped the same
    # way as the long-tailed one).
    lt_counts = np.array(reference_lt_counts(name, args.imb_ratio, args.imb_type, args.img_max))
    shot = shot_groups(lt_counts, args.many_shot_thr, args.few_shot_thr)
    freq = frequency_groups(lt_counts)
    logging.info(
        f"classes: {args.num_classes} | shot groups -> many:{shot.count('many')} "
        f"medium:{shot.count('medium')} few:{shot.count('few')} | "
        f"frequency groups -> head:{freq.count('head')} medium:{freq.count('medium')} tail:{freq.count('tail')}"
    )

    metrics_dir = os.path.join(args.dump_path, "metrics")
    train_log_path = os.path.join(metrics_dir, "train_log.csv")
    val_log_path = os.path.join(metrics_dir, "val_log.csv")
    if args.local_rank == 0:
        init_metrics_dir(args, lt_counts, shot, freq, resuming)

    # build model
    logging.info(f"creating model '{args.arch}'")
    # NOTE: get_encoder() (from src.model) applies the same dataset-conditional
    # stem patch used during pretraining -- for any --dataset containing
    # 'cifar' it replaces the ImageNet-style 7x7/stride-2 conv1 + maxpool with
    # a 3x3/stride-1 conv1 + Identity (see src/model.py). Building the encoder
    # any other way would construct the wrong stem.
    encoder, out_size = get_encoder(args)
    load_pretrained_encoder(encoder, args.pretrained)  # raises if the file or any weight is missing

    fc = nn.Linear(out_size, args.num_classes)
    fc.weight.data.normal_(mean=0.0, std=0.01)
    fc.bias.data.zero_()

    if args.weights == "freeze":
        encoder.requires_grad_(False)
        fc.requires_grad_(True)

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ---- data loaders (features instead of images where possible) ----
    train_targets = np.array(train_dataset.targets)
    if args.precompute_features:
        encoder.to(device)
        train_dataset, train_targets = encode_dataset(encoder, train_dataset, args, device, "train set")
        val_dataset, _ = encode_dataset(encoder, val_dataset, args, device, "test set")
        encoder = nn.Identity()  # the features are the input now
        n_workers = 0            # the tensors are already in memory
    else:
        n_workers = args.workers
    model = LinearProbeModel(encoder, fc)

    if args.class_balanced:
        # Re-balanced probe: draw each image with probability 1 / n_c.
        assert args.world_size == 1, "the rebalanced probe supports a single process only"
        weights = torch.as_tensor(1.0 / train_counts[train_targets], dtype=torch.double)
        g = torch.Generator()
        g.manual_seed(args.seed if args.seed is not None else 0)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_dataset), replacement=True, generator=g)
        logging.info(f"Class-balanced sampling on {len(train_dataset)} images, per-class counts {train_counts.tolist()}")
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # The paper's protocol uses every image of each epoch; the older SGD
    # scaffold dropped an incomplete last batch.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=n_workers,
        pin_memory=not args.precompute_features, sampler=train_sampler,
        drop_last=args.optimizer == "sgd",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, num_workers=n_workers,
        pin_memory=not args.precompute_features, shuffle=False,
    )
    logging.info(f"Building data done: {len(train_dataset)} train / {len(val_dataset)} val images.")

    # ---- optimiser and schedule ----
    param_groups = [dict(params=fc.parameters(), lr=args.lr_classifier)]
    if args.weights == "finetune":
        param_groups.append(dict(params=encoder.parameters(), lr=args.lr_encoder))

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=args.wd)

    if args.scheduler == "step":
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    elif args.scheduler == "exp":
        # lr_classifier -> lr_end over args.epochs, as in the paper
        gamma = (args.lr_end / args.lr_classifier) ** (1.0 / max(args.epochs, 1))
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    logging.info(
        f"optimiser {args.optimizer} | epochs {args.epochs} | batch {args.batch_size} | "
        f"lr {args.lr_classifier}" + (f" -> {args.lr_end} (exponential)" if args.scheduler == "exp"
                                       else f" ({args.scheduler})") +
        f" | weight decay {args.wd} | features precomputed: {args.precompute_features}")

    model.to(device)
    ddp_device_ids = [device] if torch.cuda.is_available() else None
    model = nn.parallel.DistributedDataParallel(model, device_ids=ddp_device_ids)

    if args.weights == "freeze":
        trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(trainable) == 2  # fc.weight, fc.bias

    if args.local_rank == 0:
        logging.info(model)
        logging.info("Building model done.")

    criterion = nn.CrossEntropyLoss().to(device)

    to_restore = {"epoch": 0, "best_metric_value": -1.0, "best_epoch": -1}
    restart_from_checkpoint(
        ckp_path,
        run_variables=to_restore,
        expected_args=run_args,
        force=args.force_resume,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_metric_value = to_restore["best_metric_value"]
    best_epoch = to_restore["best_epoch"]
    if start_epoch >= args.epochs:
        logging.info("This run is already complete -- nothing to do. "
                     "(Use --overwrite to run it again, or another --dump_path for other settings.)")

    torch.backends.cudnn.benchmark = True
    summary_path = os.path.join(metrics_dir, "summary.json")

    def model_state():
        """Weights worth saving: only the linear layer when the encoder is
        frozen (it is reloaded from --pretrained), everything when finetuning."""
        state = model.state_dict()
        if args.weights == "freeze":
            state = {k: v for k, v in state.items() if ".fc." in k}
        return state

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"============ Starting epoch {epoch} ... ============")
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_acc1, train_acc5 = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, args
        )
        scheduler.step()

        if args.local_rank == 0:
            append_csv(train_log_path, [epoch, train_loss, train_acc1, train_acc5,
                                        optimizer.param_groups[0]["lr"]])

            is_last = (epoch + 1) == args.epochs
            if (epoch + 1) % args.eval_every == 0 or is_last:
                summary, cm, per_class, y_true, y_pred, y_prob = validate(
                    val_loader, model, criterion, shot, freq, args
                )
                append_csv(val_log_path, [epoch] + [summary[k] for k in VAL_LOG_KEYS])
                artifacts = dict(
                    summary=summary, cm=cm, per_class=per_class, y_true=y_true, y_pred=y_pred,
                    y_score=y_prob, class_names=class_names, train_counts=train_counts,
                    shot=shot, lt_counts=lt_counts, freq=freq, epoch=epoch,
                )

                is_best = summary[args.best_metric] > best_metric_value
                if is_best:
                    best_metric_value, best_epoch = summary[args.best_metric], epoch
                    save_artifacts(metrics_dir, "best", **artifacts)
                    best_summary = summary
                if is_last:
                    save_artifacts(metrics_dir, "final", **artifacts)

                if is_best or is_last:
                    record = dict(
                        script="eval_linear_cifar", probe=args.probe, args=vars(args),
                        train_counts=train_counts, lt_counts=lt_counts,
                        best_epoch=best_epoch, best_metric_name=args.best_metric,
                        best_metric_value=best_metric_value,
                    )
                    if is_best:
                        record["best"] = best_summary
                    elif os.path.isfile(summary_path):
                        with open(summary_path) as f:
                            record["best"] = json.load(f).get("best")
                    if is_last:
                        record["final"] = summary  # <- the result to report
                    write_json(summary_path, record)

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model_state(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": run_args,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
            }
            save_checkpoint(save_dict, ckp_path)
            if best_epoch == epoch:
                torch.save(save_dict, os.path.join(args.dump_path, "best.pth.tar"))

    # the backup copy of the checkpoint is only needed while the run is in progress
    if args.local_rank == 0 and os.path.isfile(ckp_path + ".prev"):
        os.remove(ckp_path + ".prev")
    logging.info(f"Done. Best {args.best_metric}: {best_metric_value:.4f} (epoch {best_epoch}; "
                 "selected on the test set -- report the *_final files).")
    logging.info(f"Metrics saved under: {metrics_dir}")


if __name__ == "__main__":
    main()
