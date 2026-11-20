# Changes relative to winci-ai/resa_cifar

The ReSA method (`method/`), the network (`src/resnet.py`, `src/model.py`), the augmentations and the optimisation settings of pretraining are unchanged. Everything below was added or modified for the long-tailed study. (The Apache-2.0 licence asks for modified files to be marked; this file and the file headers do that.)

## New files

| File | Purpose |
|---|---|
| `src/cifar_lt.py` | long-tailed subsampling for CIFAR-10/100 and CINIC-10: exponential (the standard CIFAR-LT protocol), step, and `Dsample` (a balanced subset of a given size); optional `img_max` override, with augmented copies for CINIC-10 when a class has fewer images than requested |
| `src/cinic10.py` | CINIC-10 loader (train and valid pooled for pretraining; the test split, which contains the CIFAR-10 test set, is only ever used for evaluation) |
| `src/datasets.py` | datasets as arrays, evaluation normalisation statistics, long-tailed reference class sizes, and the exact indices of the long-tailed subset (same function and seed as pretraining) |
| `src/tiny_imagenet.py` | Tiny ImageNet read from the zip archive (copied to the local disk first, never unpacked on Drive), resized to 32×32 and cached as `.npz`; downloads the archive if missing, with time-outs, retries and progress messages |
| `src/features.py` | frozen momentum encoder and cached feature extraction |
| `src/knn.py` | k-NN classifier, several k in one pass, optional class-weighted (re-balanced) votes, ties broken by the nearest neighbour |
| `src/metrics.py` | accuracy, macro precision / recall / F1 / AUC, balanced accuracy, G-mean, head / medium / tail and many / medium / few-shot accuracies, worst-class recall, and their output files |
| `eval_linear_cifar.py` | linear probe with the three probes (`long_tailed`, `rebalanced`, `balanced`), long-tail metrics, resumable |
| `eval_knn.py` | k-NN evaluation with the three probes |
| `eval_transfer.py` | transfer to CIFAR-10 / CIFAR-100 / Tiny ImageNet (linear + k-NN) |
| `visualise_results.py` | figures and raw-data CSV files for one evaluation; `--compare` draws several runs (e.g. the three probes) in one graph |
| `collect_results.py` | one CSV for all evaluations (`all_results.csv`), one for all per-class values (`all_per_class.csv`) and the pretraining progress (`pretraining_status.csv`); also reads runs made with earlier versions of the code |
| `experiments.py` | builds, prints and runs the commands of the study; standard folder names (`<dataset>_lt<ratio>`, `<dataset>_balanced`, `_seed<N>`), a `run_names` mapping to reuse existing folders, skipping of finished pretraining runs, and pretraining status |
| `notebooks/ReSA_long_tail_experiments.ipynb` | Colab notebook (see below) |
| `scripts/` | optional learning-rate sweep for the linear probe (`lr_sweep.sh`, `pick_best_lr.py`) |

## The notebook

One configuration cell (0.2) holds every setting; the other cells call `experiments.py`.

- Setup: mount Drive, configuration, environment report (versions and GPU), dataset preparation (CIFAR downloaded to the local disk and copied to Drive; Tiny ImageNet cache, optional).
- Pretraining: one run, resume after a disconnect, all imbalance ratios in sequence, progress table.
- Evaluation: k-NN for one probe or for everything (also writes `knn_all_results.csv`), linear probe for one or for everything (resumable, `LIN_OVERWRITE` to repeat).
- Transfer: the source encoders are chosen with `TRANSFER_SOURCE_RATIOS` (balanced and every imbalance ratio by default) and the targets with `TRANSFER_TARGETS`; the tables are saved as `transfer_from_cifar10.csv` / `transfer_from_cifar100.csv`.
- Results: collect all CSV files, figures for one or for every evaluation, the three probes compared in one graph, and a download of every table and figure as one zip.
- Appendix: a two-minute smoke test of the whole pipeline.
- `RUN_NAMES` maps a standard run name to an existing folder (e.g. `{"cifar100_balanced": "cifar100"}`).
- Weights & Biases is off by default; the key is read from Colab's Secrets panel, never from a cell.

## Modified files

- **`main.py`**
  - `--dataset cifar10_lt / cifar100_lt` (and `cinic10`, `cinic10_lt`) with `--imb_ratio`, `--imb_type`, `--img_max` (via `src/transform.py`).
  - Checkpoint written atomically after every epoch, with one backup that is removed when the run finishes; `status.json` and `args.json` written next to it; optional `--save_every N` keeps `checkpoint_ep<N>.pth.tar`.
  - On resume the settings stored in the checkpoint are compared with the command line; a mismatch stops the run (`--force_resume` overrides). Re-running a finished run trains nothing more (and `experiments.py` skips it before starting).
  - Random-number state is saved and restored; the data sampler's shuffling now follows `--seed` (it was fixed to 0 upstream, so every seed saw the same order).
  - Weights & Biases is opt-in (`--use_wandb`); a resumed run continues the same W&B run (`wandb_id.txt`); `--disable_wandb` is accepted and ignored.
  - Runs with plain `python main.py` on one GPU, and on CPU for smoke tests: the distributed backend is chosen automatically (`nccl` on GPU, `gloo` on CPU), and synchronised batch norm is only applied when several GPUs are used (with one process it is numerically the same as ordinary batch norm).
  - A clear error when the training set is smaller than one batch.
- **`args.py`** — the new arguments above.
- **`eval_linear_cifar.py`**, **`eval_transfer.py`** (added by this study, listed here because the protocol changed during the study)
  - The linear probe follows the paper's CIFAR linear evaluation protocol (Appendix B.3): `--optimizer adam` (the default), 500 epochs, learning rate decayed exponentially from 1e-2 to 1e-6, weight decay 5e-6, batch size 1000, no augmentation, every image used in each epoch, final-epoch result reported — the same settings as upstream `eval/evaluate.py`, which implements that protocol for the quick monitor during pretraining.
  - `--optimizer sgd` keeps the earlier scaffold of these files (SGD, momentum 0.9, cosine schedule, 100 epochs, batch 256, incomplete last batch dropped), so results produced with it can be reproduced. Values given explicitly on the command line override the defaults of either optimiser.
  - `--precompute_features` (on by default when the encoder is frozen and augmentation is off): the features are extracted once instead of running the encoder every epoch. Same computation, a few hundred times faster, which is what makes 500 epochs practical.
  - `eval_linear_cifar.py` resumes an interrupted probe, does nothing for a finished one, and refuses a folder made with different settings (`--overwrite` starts again, `--force_resume` overrides). The `best` files are selected on the test set by `--best_metric` and are for diagnosis only. `--class_balanced` with a `*_lt` dataset is the older spelling of `--probe rebalanced`.
- **`src/utils.py`** — atomic checkpoint saving, safer resume with a settings check, `load_pretrained_encoder`, device helpers, automatic distributed backend; the cosine scheduler tolerates runs shorter than the warm-up.
- **`src/transform.py`** — dataset selection for the long-tailed and CINIC-10 variants.
- **`method/resa.py`**, **`eval/dataloader.py`**, **`eval/evaluate.py`** — inputs are moved to the model's device instead of a hard-coded `.cuda()`; comments.
- **`requirements.txt`** — version pins relaxed so that the code runs on the PyTorch that Colab provides; scikit-learn, pandas and matplotlib added; `wandb` optional.
- **`README.md`** — replaced; now also reports the results obtained with the paper's linear protocol.

## Removed

- The upstream README (replaced), and the helper scripts of earlier notebook versions (`combine_eval_metrics.py`, superseded by `collect_results.py`).
