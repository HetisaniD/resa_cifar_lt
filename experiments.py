"""
experiments.py -- one place that knows how every experiment of the study is launched.

The notebook (notebooks/ReSA_long_tail_experiments.ipynb) holds the settings
in a single configuration cell and calls the functions below; each function
builds the command line of one script, prints it, and runs it. The same
commands can be typed in a terminal -- see the README.

Folder names
------------
Everything is written under cfg["results_root"]:

  pretraining          cifar10_lt100/            (dataset + imbalance ratio)
                       cifar10_balanced/         (imbalance ratio 1)
                       cifar10_lt100_seed1/      (any seed other than 0)
  linear probe         cifar10_lt100_probe_long_tailed/   _probe_rebalanced/   _probe_balanced/
  k-NN                 cifar10_lt100_knn_long_tailed/     _knn_rebalanced/     _knn_balanced/
  transfer             transfer/cifar10_lt100__to__tiny_imagenet/
  collected tables     all_results.csv, all_per_class.csv, pretraining_status.csv

cfg["run_names"] maps these standard names to folders that already exist
under another name (for example {"cifar100_balanced": "cifar100"}), so
earlier runs are reused instead of repeated.
"""

import json
import os
import shlex
import subprocess
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    results_root="./results",
    data_root="./data",
    tiny_zip=None,            # path of tiny-imagenet-200.zip if it is not in data_root
    arch="resnet18",
    seed=0,
    pretrain_epochs=1000,
    pretrain_batch_size=256,
    workers=2,
    # Linear probe: the CIFAR protocol of the ReSA paper (Appendix B.3) --
    # Adam, 500 epochs, lr 1e-2 decayed exponentially to 1e-6, weight decay
    # 5e-6, batch 1000, no augmentation, final epoch reported.
    probe_optimizer="adam",   # "sgd" = the earlier scaffold (momentum 0.9, cosine)
    probe_epochs=500,
    probe_lr=1e-2,
    probe_lr_end=1e-6,
    probe_wd=5e-6,
    probe_batch_size=1000,
    knn_metric="euclidean",
    transfer_img_size=32,
    use_wandb=False,
    save_every=0,             # also keep checkpoint_ep<N>.pth.tar every N epochs (0 = off)
    run_names={},
)

PROBES = ("long_tailed", "rebalanced", "balanced")
TARGETS = ("cifar10", "cifar100", "tiny_imagenet")


def config(**overrides):
    """DEFAULTS with the given values replaced; unknown keys are rejected so a
    typo in the notebook is caught immediately."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown setting(s): {sorted(unknown)}")
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------
# Names and paths
# --------------------------------------------------------------------------
def run_name(cfg, dataset, imb_ratio, seed=None):
    """Folder name of a pretraining run, e.g. ('cifar10', 100) -> 'cifar10_lt100'."""
    seed = cfg["seed"] if seed is None else seed
    name = f"{dataset}_balanced" if int(imb_ratio) <= 1 else f"{dataset}_lt{int(imb_ratio)}"
    if seed not in (0, None):
        name += f"_seed{seed}"
    return cfg["run_names"].get(name, name)


def run_dir(cfg, dataset, imb_ratio, seed=None):
    return os.path.join(cfg["results_root"], run_name(cfg, dataset, imb_ratio, seed))


def checkpoint_path(cfg, dataset, imb_ratio, seed=None):
    return os.path.join(run_dir(cfg, dataset, imb_ratio, seed), "checkpoint.pth.tar")


def eval_dir(cfg, dataset, imb_ratio, kind, probe, seed=None):
    """kind is 'probe' (linear) or 'knn'."""
    return os.path.join(cfg["results_root"], f"{run_name(cfg, dataset, imb_ratio, seed)}_{kind}_{probe}")


def transfer_dir(cfg, source_dataset, source_imb_ratio, target, seed=None):
    return os.path.join(cfg["results_root"], "transfer",
                        f"{run_name(cfg, source_dataset, source_imb_ratio, seed)}__to__{target}")


# --------------------------------------------------------------------------
# Running a script
# --------------------------------------------------------------------------
def run(script, args, dry_run=False):
    """Run `python <script> <args>` from the repository folder, streaming its
    output. Raises RuntimeError if the script fails."""
    cmd = [sys.executable, "-u", script] + [str(a) for a in args]
    print("$ " + " ".join(shlex.quote(c) for c in cmd[2:]), flush=True)
    if dry_run:
        return
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{script} failed (exit code {proc.returncode}); see the output above")


def _need_checkpoint(cfg, dataset, imb_ratio, seed=None, dry_run=False):
    path = checkpoint_path(cfg, dataset, imb_ratio, seed)
    if not dry_run and not os.path.isfile(path):
        raise FileNotFoundError(f"no pretraining checkpoint at {path} -- pretrain "
                                f"{dataset} with imbalance ratio {imb_ratio} first")
    return path


# --------------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------------
def read_status(cfg, dataset, imb_ratio, seed=None):
    """{'epoch', 'epochs', 'done'} of a pretraining run, or None if it has not started."""
    path = os.path.join(run_dir(cfg, dataset, imb_ratio, seed), "status.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except ValueError:
            pass
    if os.path.isfile(checkpoint_path(cfg, dataset, imb_ratio, seed)):
        return dict(epoch="?", epochs="?", done=None)  # run from an earlier version of the code
    return None


def pretrain(cfg, dataset, imb_ratio, seed=None, dry_run=False):
    """Pretrain ReSA on `dataset` at the given imbalance ratio (1 = balanced).

    Safe to call again at any time: main.py continues from the last saved
    epoch if the folder holds a checkpoint, and returns at once if the run
    is already complete. This is what makes the 'resume' and 'run all' cells
    of the notebook work.
    """
    seed = cfg["seed"] if seed is None else seed
    st = read_status(cfg, dataset, imb_ratio, seed)
    if st and st.get("done") is True and st.get("epochs") == cfg["pretrain_epochs"]:
        print(f"[skip] {run_name(cfg, dataset, imb_ratio, seed)}: pretraining already complete "
              f"({st['epochs']} epochs)")
        return
    args = ["--dataset", dataset if int(imb_ratio) <= 1 else f"{dataset}_lt"]
    if int(imb_ratio) > 1:
        args += ["--imb_ratio", int(imb_ratio)]
    args += ["--data_path", cfg["data_root"],
             "--dump_path", run_dir(cfg, dataset, imb_ratio, seed),
             "-a", cfg["arch"], "--epochs", cfg["pretrain_epochs"],
             "--batch_size", cfg["pretrain_batch_size"],
             "--workers", cfg["workers"], "--seed", seed]
    if cfg["save_every"]:
        args += ["--save_every", cfg["save_every"]]
    if cfg["use_wandb"]:
        args += ["--use_wandb", "--wandb_project", "resa-cifar-lt",
                 "--env_name", run_name(cfg, dataset, imb_ratio, seed)]
    run("main.py", args, dry_run)


def pretraining_status(cfg, datasets, imb_ratios, seed=None):
    """Print one line per pretraining run and return the unfinished (dataset, ratio) pairs."""
    todo = []
    print(f"{'run':<28}{'progress':<18}state")
    for dataset in datasets:
        for ratio in imb_ratios:
            st = read_status(cfg, dataset, ratio, seed)
            name = run_name(cfg, dataset, ratio, seed)
            if st is None:
                progress, state = "-", "not started"
            elif st.get("done") is None:
                progress, state = "?", "checkpoint from an earlier code version (re-running resumes or confirms it)"
            else:
                progress = f"{st['epoch']}/{st['epochs']} epochs"
                state = "complete" if st["done"] and st["epochs"] == cfg["pretrain_epochs"] else \
                    f"INTERRUPTED -- last saved {st.get('updated', '')}"
                if st["done"] and st["epochs"] != cfg["pretrain_epochs"]:
                    state = f"finished with {st['epochs']} epochs, but the config asks for {cfg['pretrain_epochs']}"
            if state != "complete":
                todo.append((dataset, ratio))
            print(f"{name:<28}{progress:<18}{state}")
    return todo


# --------------------------------------------------------------------------
# Downstream evaluation on the pretraining dataset
# --------------------------------------------------------------------------
def linear_probe(cfg, dataset, imb_ratio, probe, seed=None, overwrite=False, dry_run=False):
    """Train a linear classifier on the frozen encoder.

    probe: 'long_tailed' (the imbalanced training subset, plain cross-entropy),
           'rebalanced'  (the same subset, class-balanced sampling) or
           'balanced'    (the full balanced training set).
    The test set is always the official balanced test set.
    """
    seed = cfg["seed"] if seed is None else seed
    args = ["--dataset", dataset, "--probe", probe, "--imb_ratio", int(imb_ratio),
            "--pretrained", _need_checkpoint(cfg, dataset, imb_ratio, seed, dry_run),
            "--data_path", cfg["data_root"],
            "--dump_path", eval_dir(cfg, dataset, imb_ratio, "probe", probe, seed),
            "-a", cfg["arch"], "--weights", "freeze",
            "--optimizer", cfg["probe_optimizer"],
            "--lr_classifier", cfg["probe_lr"], "--epochs", cfg["probe_epochs"],
            "--wd", cfg["probe_wd"], "--batch_size", cfg["probe_batch_size"],
            "--workers", cfg["workers"], "--seed", seed]
    if cfg["probe_optimizer"] == "adam":
        args += ["--lr_end", cfg["probe_lr_end"]]
    if overwrite:
        args.append("--overwrite")
    run("eval_linear_cifar.py", args, dry_run)


def knn(cfg, dataset, imb_ratio, probe, k_values=(5, 10, 20), seed=None, dry_run=False):
    """k-nearest-neighbour evaluation for every k in k_values (one feature
    extraction, one neighbour search). Results: <run>_knn_<probe>/metrics/knn_summary.csv"""
    seed = cfg["seed"] if seed is None else seed
    args = ["--dataset", dataset, "--probe", probe, "--imb_ratio", int(imb_ratio),
            "--k", *[int(k) for k in k_values], "--knn_metric", cfg["knn_metric"],
            "--pretrained", _need_checkpoint(cfg, dataset, imb_ratio, seed, dry_run),
            "--data_path", cfg["data_root"],
            "--dump_path", eval_dir(cfg, dataset, imb_ratio, "knn", probe, seed),
            "-a", cfg["arch"], "--seed", seed]
    run("eval_knn.py", args, dry_run)


# --------------------------------------------------------------------------
# Transfer to other datasets
# --------------------------------------------------------------------------
def transfer(cfg, source_dataset, source_imb_ratio, target, k_values=(5, 10, 20),
             classifiers=("linear", "knn"), seed=None, dry_run=False):
    """Freeze the encoder pretrained on `source_dataset` and train a linear
    classifier and k-NN classifiers on the full training set of `target`
    ('cifar10', 'cifar100' or 'tiny_imagenet'); test on the target's test set.
    Results: transfer/<source run>__to__<target>/metrics/transfer_summary.csv"""
    seed = cfg["seed"] if seed is None else seed
    args = ["--source_dataset", source_dataset, "--target", target,
            "--pretrained", _need_checkpoint(cfg, source_dataset, source_imb_ratio, seed, dry_run),
            "--classifiers", *classifiers, "--k", *[int(k) for k in k_values],
            "--knn_metric", cfg["knn_metric"],
            "--optimizer", cfg["probe_optimizer"],
            "--epochs", cfg["probe_epochs"], "--lr_classifier", cfg["probe_lr"],
            "--wd", cfg["probe_wd"],
            "--batch_size", cfg["probe_batch_size"], "--img_size", cfg["transfer_img_size"],
            "--data_path", cfg["data_root"],
            "--dump_path", transfer_dir(cfg, source_dataset, source_imb_ratio, target, seed),
            "-a", cfg["arch"], "--seed", seed]
    if cfg["probe_optimizer"] == "adam":
        args += ["--lr_end", cfg["probe_lr_end"]]
    if cfg["tiny_zip"]:
        args += ["--tiny_zip", cfg["tiny_zip"]]
    run("eval_transfer.py", args, dry_run)
