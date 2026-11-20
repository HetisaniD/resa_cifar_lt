# Changes relative to winci-ai/resa_cifar

The ReSA method (`method/`), the network (`src/resnet.py`, `src/model.py`), the augmentations and the optimisation settings are unchanged. Everything below was added or modified for the long-tailed study. (The Apache-2.0 licence asks for modified files to be marked; this file and the file headers do that.)

## New files

| File | Purpose |
|---|---|
| `src/cifar_lt.py` | long-tailed subsampling (exponential, step, balanced subset) for CIFAR-10/100 and CINIC-10 |
| `src/cinic10.py` | CINIC-10 loader |
| `src/datasets.py` | datasets as arrays, evaluation normalisation, long-tailed reference class sizes |
| `src/tiny_imagenet.py` | Tiny ImageNet from the zip archive, cached as 32×32 arrays |
| `src/features.py` | frozen momentum encoder and cached feature extraction |
| `src/knn.py` | k-NN classifier, several k in one pass, optional class-weighted votes |
| `src/metrics.py` | accuracy, macro precision / recall / F1 / AUC, class-group accuracies, and their output files |
| `eval_linear_cifar.py` | linear probe with the three probes, long-tail metrics, resumable |
| `eval_knn.py` | k-NN evaluation with the three probes |
| `eval_transfer.py` | transfer to CIFAR-10 / CIFAR-100 / Tiny ImageNet (linear + k-NN) |
| `visualise_results.py` | figures and raw-data CSV files |
| `collect_results.py` | one CSV for all evaluations |
| `experiments.py` | builds and runs the commands of the study |
| `notebooks/ReSA_long_tail_experiments.ipynb` | Colab notebook |
| `scripts/` | optional learning-rate sweep |

## Modified files

- **`main.py`**
  - `--dataset cifar10_lt / cifar100_lt` with `--imb_ratio`, `--imb_type`, `--img_max` (via `src/transform.py`).
  - Checkpoint written atomically after every epoch, with one backup that is removed when the run finishes; `status.json` and `args.json` written next to it; optional `--save_every N`.
  - On resume the settings stored in the checkpoint are compared with the command line; a mismatch stops the run (`--force_resume` overrides).
  - Random-number state is saved and restored; the data sampler's shuffling now follows `--seed` (it was fixed to 0 upstream, so every seed saw the same order).
  - Weights & Biases is opt-in (`--use_wandb`); `--disable_wandb` is accepted and ignored.
  - Runs with plain `python main.py` on one GPU, and on CPU for smoke tests (synchronised batch norm is only applied when several GPUs are used; with one process it is numerically the same as ordinary batch norm).
- **`args.py`** — the new arguments above.
- **`eval_linear_cifar.py`**, **`eval_transfer.py`** (added by this study, listed here because the protocol changed)
  - The linear probe follows the paper's CIFAR linear evaluation protocol (Appendix B.3): `--optimizer adam` (the default), 500 epochs, learning rate decayed exponentially from 1e-2 to 1e-6, weight decay 5e-6, batch size 1000, no augmentation, final-epoch result reported — the same settings as upstream `eval/evaluate.py`, which implements that protocol for the quick monitor during pretraining.
  - `--optimizer sgd` keeps the earlier scaffold of these files (SGD, momentum 0.9, cosine schedule, 100 epochs, batch 256), so results produced with it can be reproduced.
  - `--precompute_features` (on by default when the encoder is frozen and augmentation is off): the features are extracted once instead of running the encoder every epoch. Same computation, a few hundred times faster, which is what makes 500 epochs practical.

- **`src/utils.py`** — atomic checkpoint saving, safer resume, `load_pretrained_encoder`, device helpers; the cosine scheduler tolerates runs shorter than the warm-up.
- **`src/transform.py`** — dataset selection for the long-tailed and CINIC-10 variants.
- **`method/resa.py`**, **`eval/dataloader.py`**, **`eval/evaluate.py`** — inputs are moved to the model's device instead of a hard-coded `.cuda()`; comments.
- **`requirements.txt`** — version pins relaxed so that the code runs on the PyTorch that Colab provides; scikit-learn, pandas and matplotlib added.

## Removed

- The upstream README (replaced), and the helper scripts of earlier notebook versions (`combine_eval_metrics.py`, superseded by `collect_results.py`).
