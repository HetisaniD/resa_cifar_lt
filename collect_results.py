#!/usr/bin/env python3
"""
collect_results.py -- gather every evaluation under a results folder into CSV files.

Each evaluation script writes its numbers into <run>/metrics/. This script
walks a results root, reads all of them and writes three tables:

  all_results.csv      one row per (run, classifier): accuracy, macro
                       precision / recall / F1 / AUC, balanced accuracy,
                       head / medium / tail accuracy, ... -- the table to
                       paste into the dissertation or open in Excel
  all_per_class.csv    one row per (run, classifier, class): training count,
                       precision, recall, F1, AUC
  pretraining_status.csv  one row per pretraining run: epochs done / planned

What is recognised
  linear probe   (eval_linear_cifar.py)  metrics/summary_metrics_final.csv
  k-NN           (eval_knn.py)           metrics/knn_summary.csv
  transfer       (eval_transfer.py)      metrics/transfer_summary.csv
Runs produced by earlier versions of the code (no summary.json) are included
too; their dataset / probe / imbalance ratio are then read from the folder
name (e.g. cifar10_lt100_probe_rebalanced).

Units: acc1 and acc5 are percentages (0-100); every other metric is a
fraction between 0 and 1.

Usage
    python collect_results.py --results_root /content/drive/MyDrive/ReSA_Research/results
"""

import argparse
import csv
import json
import os
import re

# columns that come first in all_results.csv, in this order
LEAD = ["run", "kind", "pretrain_dataset", "dataset", "probe", "imb_ratio", "classifier", "k", "tag",
        "acc1", "acc5", "macro_precision", "macro_recall", "macro_f1", "macro_auc",
        "balanced_accuracy", "gmean", "head_acc", "mid_acc", "tail_acc", "head_tail_gap",
        "many_shot_acc", "medium_shot_acc", "few_shot_acc", "min_class_recall", "n_zero_recall_classes"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def guess_from_name(run):
    """Dataset, imbalance ratio and probe from a folder name such as
    'cifar100_lt50_probe_rebalanced' or 'cifar10_lt20_probeBalanced'."""
    info = {}
    m = re.match(r"(cifar100|cifar10|cinic10|tiny_imagenet)", run)
    if m:
        info["dataset"] = m.group(1)
    m = re.search(r"_lt(\d+)", run)
    info["imb_ratio"] = int(m.group(1)) if m else (1 if "balanced" in run.split("_probe")[0] else "")
    low = run.lower()
    if "rebalanced" in low:
        info["probe"] = "rebalanced"
    elif re.search(r"probe_?balanced|knn_balanced", low):
        info["probe"] = "balanced"
    elif "probe" in low or "knn" in low:
        info["probe"] = "long_tailed" if info["imb_ratio"] not in ("", 1) else "balanced"
    if "finetune" in low:
        info["weights"] = "finetune"
    return info


def collect_linear(run, mdir):
    """Linear probe: the final-epoch metrics (the ones to report)."""
    path = os.path.join(mdir, "summary_metrics_final.csv")
    if not os.path.isfile(path):
        return []
    meta = guess_from_name(run)
    js = read_json(os.path.join(mdir, "summary.json"))
    a = js.get("args", {})
    for key in ("dataset", "probe", "imb_ratio", "imb_type", "seed", "weights", "lr_classifier",
                "epochs", "batch_size", "pretrained"):
        if a.get(key) not in (None, ""):
            meta[key] = a[key]
    if "dataset" in meta:  # older runs stored e.g. 'cifar10_lt'
        meta["dataset"] = str(meta["dataset"]).replace("_lt", "")
    rows = []
    for r in read_csv(path)[-1:]:
        row = dict(run=run, kind="linear_probe", classifier="linear", tag="final", **meta)
        row.update({k: v for k, v in r.items() if k != "epoch"})
        row["epoch"] = r.get("epoch", "")
        row["pretrain_dataset"] = meta.get("dataset", "")
        rows.append(row)
    return rows


def collect_table(run, mdir, filename, kind):
    """k-NN and transfer runs already keep one row per classifier."""
    path = os.path.join(mdir, filename)
    if not os.path.isfile(path):
        return []
    rows = []
    for r in read_csv(path):
        row = dict(run=run, kind=kind)
        row.update(r)
        if kind == "transfer":
            row["pretrain_dataset"] = r.get("source_dataset", "")
            row["dataset"] = r.get("target", "")
            m = re.search(r"_lt(\d+)", run.split("__to__")[0])
            row["imb_ratio"] = int(m.group(1)) if m else 1   # imbalance ratio of the SOURCE pretraining
            row["probe"] = "balanced"                        # classifiers see the full target training set
        else:
            row["pretrain_dataset"] = r.get("dataset", "")
        c = row.get("classifier", "")
        row["tag"] = "linear" if c == "linear" else f"k{row.get('k', '')}"
        rows.append(row)
    return rows


def collect_per_class(run, kind, mdir, tag, classifier):
    path = os.path.join(mdir, f"per_class_metrics_{tag}.csv")
    if not os.path.isfile(path):
        return []
    return [dict(run=run, kind=kind, classifier=classifier, tag=tag, **{k: v for k, v in r.items() if k != "epoch"})
            for r in read_csv(path)]


def write_table(path, rows, lead):
    if not rows:
        return
    extra = []
    for r in rows:
        for k in r:
            if k not in lead and k not in extra:
                extra.append(k)
    cols = [c for c in lead if any(c in r for r in rows)] + extra
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print(f"wrote {path}  ({len(rows)} rows)")


def collect(results_root, out_dir=None):
    """Walk results_root and write the three tables. Returns the result rows."""
    out_dir = out_dir or results_root
    os.makedirs(out_dir, exist_ok=True)
    results, per_class, status = [], [], []

    for dirpath, dirnames, filenames in sorted(os.walk(results_root)):
        dirnames.sort()
        run = os.path.relpath(dirpath, results_root).replace(os.sep, "/")
        # pretraining runs
        if "status.json" in filenames:
            st = read_json(os.path.join(dirpath, "status.json"))
            ar = read_json(os.path.join(dirpath, "args.json"))
            status.append(dict(run=run, dataset=ar.get("dataset", ""), imb_ratio=ar.get("imb_ratio", ""),
                               seed=ar.get("seed", ""), epoch=st.get("epoch", ""), epochs=st.get("epochs", ""),
                               done=st.get("done", ""), updated=st.get("updated", "")))
        # evaluation runs
        if os.path.basename(dirpath) != "metrics":
            continue
        run = os.path.dirname(run) or "."
        found = (collect_linear(run, dirpath)
                 + collect_table(run, dirpath, "knn_summary.csv", "knn")
                 + collect_table(run, dirpath, "transfer_summary.csv", "transfer"))
        results += found
        for row in found:
            per_class += collect_per_class(run, row["kind"], dirpath, row["tag"], row["classifier"])

    write_table(os.path.join(out_dir, "all_results.csv"), results, LEAD)
    write_table(os.path.join(out_dir, "all_per_class.csv"), per_class,
                ["run", "kind", "classifier", "tag", "class_id", "class_name"])
    write_table(os.path.join(out_dir, "pretraining_status.csv"), status, ["run"])
    if not results:
        print(f"no evaluation results found under {results_root}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect all evaluation results into CSV files")
    p.add_argument("--results_root", required=True, help="folder that holds the run folders")
    p.add_argument("--out_dir", default=None, help="where to write the CSV files (default: --results_root)")
    a = p.parse_args()
    collect(a.results_root, a.out_dir)
