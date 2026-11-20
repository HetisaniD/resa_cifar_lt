#!/usr/bin/env python3
"""
visualise_results.py -- plots and raw-data CSVs for one evaluation run.

Reads the files an evaluation script wrote into <run>/metrics/ and produces,
for the chosen --tag (final, best, linear, k5, k10, k20, ...):

  Figures (PNG)
    precision_recall_<tag>.png        precision and recall of every class, one graph
    f1_<tag>.png                      F1 of every class
    auc_<tag>.png                     one-vs-rest ROC AUC of every class
    confusion_pairs_<tag>.png         the most frequent confusions (true -> predicted)
    confusion_matrix_<tag>.png        the confusion matrix (counts and row-normalised)
    prediction_distribution_<tag>.png how many test images were assigned to each class,
                                      against how many truly belong to it -- the clearest
                                      picture of a classifier that favours head classes

  Raw data (CSV) -- everything the figures show, ready for Excel
    raw_per_class_<tag>.csv           one row per class: counts, precision, recall, F1, AUC
    raw_confusion_pairs_<tag>.csv     every (true, predicted) pair with at least one error
    raw_confusion_matrix_<tag>.csv    the matrix with class names
    raw_summary_<tag>.csv             accuracy and the macro / long-tail metrics

Classes are drawn in class-index order, which for CIFAR-LT is frequency order
(class 0 has the most training images).

Usage
    python visualise_results.py --metrics_dir /results/cifar10_lt100_probe_rebalanced/metrics
    python visualise_results.py --metrics_dir /results/cifar10_lt100_knn_balanced/metrics --tag k10
    python visualise_results.py --metrics_dir ... --all_tags

Compare several runs in one figure (per-class recall, one line per run):
    python visualise_results.py --compare RUN_A/metrics RUN_B/metrics RUN_C/metrics \\
        --labels "Long-tailed" "Re-balanced" "Balanced" --out_dir /results/figures

Dependencies: numpy, matplotlib, scikit-learn (for AUC).
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")  # no display needed (Colab, servers)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# --- Style ------------------------------------------------------------------
# One small palette, used the same way in every figure: series colours in a
# fixed order, grey for reference lines, a single-hue ramp for the matrix.
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
GROUP_NAMES = {"head": "Head", "medium": "Medium", "tail": "Tail"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "font.size": 10,
})

COMPACT_THRESHOLD = 20  # above this many classes, bars become dots and tick labels are thinned

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


# --- Loading ----------------------------------------------------------------
def available_tags(metrics_dir):
    """Tags with a saved confusion matrix, e.g. ['final', 'best'] or ['k5', 'k10', 'k20']."""
    tags = [re.match(r"confusion_matrix_(.+)\.npy$", os.path.basename(p)).group(1)
            for p in glob.glob(os.path.join(metrics_dir, "confusion_matrix_*.npy"))]
    order = {"final": 0, "linear": 1, "best": 99}
    return sorted(tags, key=lambda t: (order.get(t, 10), int(t[1:]) if re.fullmatch(r"k\d+", t) else 0, t))


def per_class_from_matrix(cm):
    """Precision, recall and F1 of every class, straight from the matrix."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def per_class_auc(y_true, y_prob, n):
    from sklearn.metrics import roc_auc_score
    auc = np.full(n, np.nan)
    for k in range(n):
        pos = y_true == k
        if pos.any() and (~pos).any():
            auc[k] = roc_auc_score(pos, y_prob[:, k])
    return auc


def load_run(metrics_dir, tag):
    """Everything known about one evaluation, as a dict of arrays."""
    cm_path = os.path.join(metrics_dir, f"confusion_matrix_{tag}.npy")
    if not os.path.isfile(cm_path):
        raise FileNotFoundError(f"{cm_path} not found. Tags available in this folder: "
                                f"{available_tags(metrics_dir) or 'none'}")
    cm = np.load(cm_path)
    n = cm.shape[0]
    run = dict(cm=cm, n=n, tag=tag, metrics_dir=metrics_dir)
    run["precision"], run["recall"], run["f1"] = per_class_from_matrix(cm)
    run["test_count"] = cm.sum(axis=1)
    run["n_predicted"] = cm.sum(axis=0)
    run["auc"] = np.full(n, np.nan)
    run["names"] = CIFAR10_CLASSES if n == 10 else [f"class {i}" for i in range(n)]
    run["lt_count"], run["train_count"] = None, None
    run["freq_group"], run["shot_group"] = None, None

    # class names, training counts and groups, if the evaluation saved them
    pc_path = os.path.join(metrics_dir, f"per_class_metrics_{tag}.csv")
    if os.path.isfile(pc_path):
        with open(pc_path) as f:
            rows = list(csv.DictReader(f))
        if len(rows) == n:
            run["names"] = [r["class_name"] for r in rows]

            def column(key, cast):
                vals = [r.get(key, "") for r in rows]
                return [cast(v) for v in vals] if all(v not in ("", None) for v in vals) else None
            run["train_count"] = column("train_count", int)
            run["lt_count"] = column("lt_train_count", int) or run["train_count"]
            run["freq_group"] = column("freq_group", str)
            run["shot_group"] = column("shot_group", str)
            auc = column("auc", float)
            if auc:
                run["auc"] = np.array(auc)

    # AUC from the saved class scores (softmax probabilities or k-NN vote shares)
    pred_path = os.path.join(metrics_dir, f"predictions_{tag}.npz")
    if os.path.isfile(pred_path) and np.all(np.isnan(run["auc"])):
        with np.load(pred_path) as d:
            if "y_prob" in d:
                run["auc"] = per_class_auc(d["y_true"], d["y_prob"], n)

    run["accuracy"] = float(np.trace(cm) / cm.sum())
    return run


def top_confusions(cm, limit=None):
    """Off-diagonal cells, largest first, as (true index, predicted index, count)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None, kind="stable")[::-1]
    rows, cols = np.unravel_index(flat, off.shape)
    pairs = [(int(r), int(c), int(off[r, c])) for r, c in zip(rows, cols) if off[r, c] > 0]
    return pairs[:limit] if limit else pairs


# --- Shared plotting helpers ---------------------------------------------------
def _title(run, what, subtitle=None):
    name = run.get("title") or os.path.basename(os.path.dirname(os.path.abspath(run["metrics_dir"])))
    return f"{what}\n{name} · {run['tag']} · accuracy {run['accuracy']:.1%}" + (f" · {subtitle}" if subtitle else "")


def _class_axis(ax, run):
    """x axis = classes in index order; labels thinned when there are many."""
    n = run["n"]
    x = np.arange(n)
    if n <= COMPACT_THRESHOLD:
        labels = run["names"]
        if run["lt_count"] is not None and len(set(run["lt_count"])) > 1:
            labels = [f"{nm}\n{c:,}" for nm, c in zip(run["names"], run["lt_count"])]
            ax.set_xlabel("class · long-tailed training images")
        ax.set_xticks(x, labels, rotation=0 if n <= 10 else 45,
                      ha="center" if n <= 10 else "right", fontsize=9)
    else:
        step = 10 if n <= 120 else 20
        ax.set_xticks(x[::step], [str(i) for i in x[::step]])
        ax.set_xlabel("class index (0 = most training images)" if run["lt_count"] is not None
                      and len(set(run["lt_count"])) > 1 else "class index")
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis="x", visible=False)
    return x


def _group_bands(ax, run):
    """Label the head / medium / tail class groups along the top of the axes."""
    groups = run["freq_group"]
    if not groups or len(set(groups)) < 2:
        return
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if start > 0:
                ax.axvline(start - 0.5, color=GRID, linewidth=1.2, zorder=0)
            ax.text((start + i - 1) / 2, 1.01, GROUP_NAMES.get(groups[start], groups[start]),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9, color=INK_2)
            start = i


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _figure(n):
    return plt.subplots(figsize=(max(9, min(16, 0.9 * n + 2)) if n <= COMPACT_THRESHOLD else 14, 5.2))


# --- Figure 1: precision and recall in one graph ------------------------------
def plot_precision_recall(run, out_path):
    fig, ax = _figure(run["n"])
    x = _class_axis(ax, run)
    if run["n"] <= COMPACT_THRESHOLD:
        w = 0.38
        ax.bar(x - w / 2 - 0.01, run["precision"], w, color=SERIES[0], label="Precision")
        ax.bar(x + w / 2 + 0.01, run["recall"], w, color=SERIES[1], label="Recall")
    else:
        ax.plot(x, run["precision"], "o", color=SERIES[0], markersize=4, label="Precision")
        ax.plot(x, run["recall"], "o", color=SERIES[1], markersize=4, label="Recall")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    _group_bands(ax, run)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2 if run["n"] <= COMPACT_THRESHOLD else -0.13), ncol=2)
    ax.set_title(_title(run, "Precision and recall per class"), pad=22)
    _save(fig, out_path)


# --- Figures 2 and 3: one metric per class -------------------------------------
def _plot_single_metric(run, values, name, out_path, dots=False, ylim=(0, 1.0)):
    fig, ax = _figure(run["n"])
    x = _class_axis(ax, run)
    mean = float(np.nanmean(values))
    if dots or run["n"] > COMPACT_THRESHOLD:
        ax.plot(x, values, "o", color=SERIES[0], markersize=7 if run["n"] <= COMPACT_THRESHOLD else 4)
    else:
        ax.bar(x, values, 0.6, color=SERIES[0])
    ax.axhline(mean, color=INK_2, linestyle="--", linewidth=1)
    ax.text(run["n"] - 0.35, mean, f" macro {name} {mean:.3f}", va="bottom", ha="right", fontsize=9, color=INK_2)
    ax.set_ylim(*ylim)
    ax.set_ylabel(name)
    _group_bands(ax, run)
    ax.set_title(_title(run, f"{name} per class"), pad=22)
    _save(fig, out_path)


def plot_f1(run, out_path):
    _plot_single_metric(run, run["f1"], "F1", out_path)


def plot_auc(run, out_path):
    if np.all(np.isnan(run["auc"])):
        print(f"skipped {out_path}: no class scores were saved for this run")
        return
    # AUC values usually sit close to 1, so they are drawn as dots on a
    # zoomed axis (a bar must start at zero; a dot need not).
    low = float(np.nanmin(run["auc"]))
    floor = 0.5 if low >= 0.5 else max(0.0, np.floor(low * 10) / 10)
    if low > 0.8:
        floor = np.floor((low - 0.02) * 20) / 20
    _plot_single_metric(run, run["auc"], "AUC", out_path, dots=True, ylim=(floor, 1.003))


# --- Figure 4: confusion pairs ---------------------------------------------------
def plot_confusion_pairs(run, out_path, top_n=10):
    pairs = top_confusions(run["cm"], top_n)
    if not pairs:
        print(f"skipped {out_path}: there are no misclassifications")
        return
    labels = [f"{run['names'][t]} → {run['names'][p]}" for t, p, _ in pairs][::-1]
    counts = [c for _, _, c in pairs][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(pairs) + 1.8))
    y = np.arange(len(counts))
    ax.barh(y, counts, 0.62, color=SERIES[0])
    ax.set_yticks(y, labels)
    for yi, c, (t, _, _) in zip(y, counts, pairs[::-1]):
        share = c / max(run["test_count"][t], 1)
        ax.text(c + max(counts) * 0.012, yi, f"{c}  ({share:.0%} of the true class)", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(counts) * 1.38)
    ax.set_xlabel("misclassified test images")
    ax.grid(axis="y", visible=False)
    ax.set_title(_title(run, f"Top {len(counts)} confusion pairs (true → predicted)"))
    _save(fig, out_path)


# --- Figure 5: confusion matrix ----------------------------------------------------
def plot_confusion_matrix(run, out_path):
    """Counts on the left, row-normalised (each row sums to 1) on the right."""
    cm, n, names = run["cm"], run["n"], run["names"]
    compact = n > COMPACT_THRESHOLD
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_totals, out=np.zeros(cm.shape, dtype=float), where=row_totals > 0)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6) if not compact else (30, 14))
    panels = [(axes[0], cm, "Counts", None, "d"),
              (axes[1], cm_norm, "Row-normalised (diagonal = recall)", 1.0, ".2f")]
    for ax, data, title, vmax, fmt in panels:
        hi = float(vmax if vmax is not None else data.max())
        # with many classes a compressed colour scale keeps small confusions visible
        kwargs = {"norm": PowerNorm(gamma=0.35, vmin=0, vmax=hi)} if compact and hi > 0 else {"vmin": 0, "vmax": hi}
        im = ax.imshow(data, cmap=BLUES, aspect="equal", **kwargs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_xticks(range(n), names, rotation=45 if not compact else 90,
                      ha="right" if not compact else "center", fontsize=10 if not compact else 5.5)
        ax.set_yticks(range(n), names, fontsize=10 if not compact else 5.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if not compact:  # print every cell, light text on dark cells
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8,
                            color="white" if data[i, j] > hi * 0.55 else INK)
    note = "  (compressed colour scale)" if compact else ""
    fig.suptitle(_title(run, "Confusion matrix").replace("\n", " — ") + note, fontsize=13, fontweight="bold",
                 x=0.01, ha="left", y=0.995)
    fig.tight_layout()
    _save(fig, out_path)


# --- Figure 6: predicted vs actual images per class ---------------------------------
def plot_prediction_distribution(run, out_path):
    """How many test images the classifier assigned to each class, against how
    many really belong to it. A classifier biased towards head classes
    predicts them far more often than they occur and the tail far less."""
    fig, ax = _figure(run["n"])
    x = _class_axis(ax, run)
    if run["n"] <= COMPACT_THRESHOLD:
        ax.bar(x, run["n_predicted"], 0.6, color=SERIES[0], label="predicted as this class")
    else:
        ax.plot(x, run["n_predicted"], "o-", color=SERIES[0], markersize=3.5, linewidth=1,
                label="predicted as this class")
    actual = run["test_count"]
    if len(set(actual.tolist())) == 1:
        ax.axhline(actual[0], color=INK_2, linestyle="--", linewidth=1, label="actually in this class")
    else:
        ax.plot(x, actual, "--", color=INK_2, linewidth=1, label="actually in this class")
    ax.set_ylim(0, max(run["n_predicted"].max(), actual.max()) * 1.1)
    ax.set_ylabel("test images")
    _group_bands(ax, run)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2 if run["n"] <= COMPACT_THRESHOLD else -0.13), ncol=2)
    ax.set_title(_title(run, "Predicted versus actual test images per class"), pad=22)
    _save(fig, out_path)


# --- Comparison of several runs ---------------------------------------------------
def plot_comparison(runs, labels, out_path):
    """Per-class recall of up to four runs in one graph (e.g. the three probes)."""
    if len(runs) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs can be compared in one figure")
    n = runs[0]["n"]
    fig, ax = _figure(n)
    x = _class_axis(ax, runs[0])
    for run, label, colour in zip(runs, labels, SERIES):
        if run["n"] != n:
            raise ValueError("the runs have different numbers of classes")
        ax.plot(x, run["recall"], "o-", color=colour, linewidth=2, markersize=5 if n <= COMPACT_THRESHOLD else 3,
                label=f"{label} (accuracy {run['accuracy']:.1%})")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall")
    _group_bands(ax, runs[0])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2 if n <= COMPACT_THRESHOLD else -0.13), ncol=len(runs))
    ax.set_title("Recall per class", pad=22)
    _save(fig, out_path)


# --- Raw-data CSVs ---------------------------------------------------------------
def _fmt(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.6f}" if isinstance(v, float) else v)


def export_csvs(run, out_dir):
    tag, n, names = run["tag"], run["n"], run["names"]

    def col(key, i):
        return "" if run[key] is None else run[key][i]

    path = os.path.join(out_dir, f"raw_per_class_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "lt_train_count", "classifier_train_count", "freq_group",
                    "shot_group", "test_count", "n_predicted", "precision", "recall", "f1", "auc"])
        for i in range(n):
            w.writerow([i, names[i], col("lt_count", i), col("train_count", i), col("freq_group", i),
                        col("shot_group", i), int(run["test_count"][i]), int(run["n_predicted"][i]),
                        _fmt(float(run["precision"][i])), _fmt(float(run["recall"][i])),
                        _fmt(float(run["f1"][i])), _fmt(float(run["auc"][i]))])
    print(f"wrote {path}")

    path = os.path.join(out_dir, f"raw_confusion_pairs_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "true_class", "predicted_class", "count", "share_of_true_class",
                    "true_lt_train_count", "predicted_lt_train_count"])
        for rank, (t, p, c) in enumerate(top_confusions(run["cm"]), 1):
            w.writerow([rank, names[t], names[p], c, _fmt(c / max(int(run["test_count"][t]), 1)),
                        col("lt_count", t), col("lt_count", p)])
    print(f"wrote {path}")

    path = os.path.join(out_dir, f"raw_confusion_matrix_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + names)
        for i in range(n):
            w.writerow([names[i]] + run["cm"][i].tolist())
    print(f"wrote {path}")

    # summary: copy the evaluation's own table if present, else derive what the matrix allows
    path = os.path.join(out_dir, f"raw_summary_{tag}.csv")
    src = os.path.join(run["metrics_dir"], f"summary_metrics_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if os.path.isfile(src):
            with open(src) as g:
                for row in csv.reader(g):
                    w.writerow(row)
        else:
            w.writerow(["acc1", "macro_precision", "macro_recall", "macro_f1", "macro_auc"])
            w.writerow([_fmt(100 * run["accuracy"]), _fmt(float(run["precision"].mean())),
                        _fmt(float(run["recall"].mean())), _fmt(float(run["f1"].mean())),
                        _fmt(float(np.nanmean(run["auc"])) if not np.all(np.isnan(run["auc"])) else float("nan"))])
    print(f"wrote {path}")


# --- Entry points ------------------------------------------------------------------
def visualise(metrics_dir, tag="final", out_dir=None, title=None, top_pairs=10):
    """Write every figure and CSV for one tag. Returns the output folder."""
    run = load_run(metrics_dir, tag)
    run["title"] = title
    out_dir = out_dir or os.path.join(metrics_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{run['n']} classes, {int(run['cm'].sum()):,} test images, accuracy {run['accuracy']:.2%} [{tag}]")
    plot_precision_recall(run, os.path.join(out_dir, f"precision_recall_{tag}.png"))
    plot_f1(run, os.path.join(out_dir, f"f1_{tag}.png"))
    plot_auc(run, os.path.join(out_dir, f"auc_{tag}.png"))
    plot_confusion_pairs(run, os.path.join(out_dir, f"confusion_pairs_{tag}.png"), top_pairs)
    plot_confusion_matrix(run, os.path.join(out_dir, f"confusion_matrix_{tag}.png"))
    plot_prediction_distribution(run, os.path.join(out_dir, f"prediction_distribution_{tag}.png"))
    export_csvs(run, out_dir)
    return out_dir


def compare(metrics_dirs, labels=None, tag="final", out_dir=".", name="compare_recall"):
    """Per-class recall of several runs in one figure, plus the data as CSV."""
    runs = [load_run(d, tag) for d in metrics_dirs]
    labels = labels or [os.path.basename(os.path.dirname(os.path.abspath(d))) for d in metrics_dirs]
    os.makedirs(out_dir, exist_ok=True)
    plot_comparison(runs, labels, os.path.join(out_dir, f"{name}_{tag}.png"))
    path = os.path.join(out_dir, f"{name}_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "lt_train_count"] + [f"recall_{l}" for l in labels]
                   + [f"precision_{l}" for l in labels])
        for i in range(runs[0]["n"]):
            w.writerow([i, runs[0]["names"][i], "" if runs[0]["lt_count"] is None else runs[0]["lt_count"][i]]
                       + [_fmt(float(r["recall"][i])) for r in runs] + [_fmt(float(r["precision"][i])) for r in runs])
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Figures and raw-data CSVs for an evaluation run")
    p.add_argument("--metrics_dir", type=str, help="the metrics/ folder of one evaluation run")
    p.add_argument("--tag", type=str, default="final",
                   help="which evaluation in that folder: final / best (linear probe), linear (transfer), "
                        "k5 / k10 / k20 (k-NN)")
    p.add_argument("--all_tags", action="store_true", help="process every tag found in --metrics_dir")
    p.add_argument("--out_dir", type=str, default=None, help="default: <metrics_dir>/figures")
    p.add_argument("--title", type=str, default=None, help="run name shown in the figure titles")
    p.add_argument("--top_pairs", type=int, default=10, help="confusion pairs shown in the bar chart")
    p.add_argument("--compare", type=str, nargs="+", default=None,
                   help="metrics/ folders of 2-4 runs to compare (per-class recall)")
    p.add_argument("--labels", type=str, nargs="+", default=None, help="legend labels for --compare")
    a = p.parse_args()

    if a.compare:
        compare(a.compare, a.labels, a.tag, a.out_dir or ".")
        return
    if not a.metrics_dir:
        p.error("give --metrics_dir (or --compare)")
    tags = available_tags(a.metrics_dir) if a.all_tags else [a.tag]
    if not tags:
        p.error(f"no confusion_matrix_*.npy files in {a.metrics_dir}")
    for tag in tags:
        visualise(a.metrics_dir, tag, a.out_dir, a.title, a.top_pairs)


if __name__ == "__main__":
    main()
