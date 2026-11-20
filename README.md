# ReSA under class imbalance

Code for the study of **ReSA self-supervised learning on long-tailed data**: ReSA is pretrained on long-tailed CIFAR-10 and CIFAR-100 and the frozen encoder is evaluated with k-NN and linear classifiers, on the pretraining dataset and on other datasets (transfer).

It is a modified copy of the official ReSA CIFAR code, [winci-ai/resa_cifar](https://github.com/winci-ai/resa_cifar) (Weng et al., *Clustering Properties of Self-Supervised Learning*, ICML 2025, [arXiv:2501.18452](https://arxiv.org/abs/2501.18452)). The ReSA method and its training procedure are unchanged; what was added is listed in [CHANGES.md](CHANGES.md).

> **Before publishing:** replace the placeholders in `CITATION.cff` and in the *Citing* section below with your name and repository URL.

## What the study does

ReSA's published results use balanced datasets. Real data is rarely balanced, so this study asks how ReSA behaves when a few classes dominate.

1. **Pretraining.** ReSA (ResNet-18) is trained without labels on CIFAR-10-LT and CIFAR-100-LT at imbalance ratios 20, 50 and 100, and on the balanced datasets as reference. The imbalance ratio is the size of the largest class divided by the size of the smallest. The subsets follow the standard exponential protocol (Cui et al., 2019; Cao et al., 2019): class *i* of *C* keeps `n_max · ratio^(−i/(C−1))` images.
2. **Evaluation on the pretraining dataset.** The encoder is frozen and a classifier is placed on its 512-dimensional features. Three *probes* separate what the encoder learned from how the classifier was trained:

   | Probe | The classifier is trained on | What it measures |
   |---|---|---|
   | `long_tailed` | the long-tailed subset used for pretraining | the end-to-end result under imbalance |
   | `rebalanced` | the same subset, every class weighted equally | how much of the loss is classifier bias |
   | `balanced` | the full balanced training set | the quality of the representation itself |

   Each probe is run with a **linear classifier** and with **k-NN (k = 5, 10, 20)**. The test set is always the official balanced test set.
3. **Transfer.** The frozen CIFAR-10 encoder is evaluated on Tiny ImageNet, CIFAR-100 and CIFAR-10, and the frozen CIFAR-100 encoder on Tiny ImageNet, CIFAR-10 and CIFAR-100, again with a linear classifier and k-NN.

## Repository layout

```
main.py                    ReSA pretraining (resumable)
eval_knn.py                k-NN evaluation, three probes, several k in one pass
eval_linear_cifar.py       linear probe, three probes (resumable)
eval_transfer.py           transfer: frozen encoder -> linear + k-NN on another dataset
visualise_results.py       figures and raw-data CSV files for one evaluation
collect_results.py         all evaluations under a folder -> one CSV
experiments.py             builds and runs the commands above (used by the notebook)
args.py                    pretraining arguments
method/                    ReSA (resa.py) and Sinkhorn-Knopp (base.py)
src/
  cifar_lt.py              long-tailed subsampling
  datasets.py              datasets as arrays, normalisation statistics, class groups
  tiny_imagenet.py         Tiny ImageNet loader (reads the .zip, caches 32x32 arrays)
  features.py              frozen encoder, cached feature extraction
  knn.py                   k-NN classifier (plain and class-weighted votes)
  metrics.py               all metrics and the files they are saved to
  transform.py, model.py, resnet.py, utils.py, cinic10.py
eval/                      the quick k-NN / linear check that monitors pretraining
notebooks/ReSA_long_tail_experiments.ipynb   the Colab notebook that runs everything
scripts/                   optional learning-rate sweep for the linear probe
```

## Running on Google Colab with Google Drive

1. **Code on Drive.** Upload this folder to Drive, for example to `MyDrive/ReSA_Research/resa_cifar_lt` (or, in a Colab cell, `!git clone <your repository URL> /content/drive/MyDrive/ReSA_Research/resa_cifar_lt`).
2. **Datasets on Drive.** Create a folder, for example `MyDrive/Datasets`. CIFAR-10 and CIFAR-100 are downloaded into it by the notebook if they are missing. For the transfer experiments put `tiny-imagenet-200.zip` there (from <http://cs231n.stanford.edu/tiny-imagenet-200.zip>; the notebook also downloads it if missing). Do **not** unzip it on Drive: the loader reads the zip once and stores two small array files next to it.

   ```
   MyDrive/Datasets/
     cifar-10-batches-py/
     cifar-100-python/
     tiny-imagenet-200.zip
     tiny_imagenet_train_32px.npz      (created on first use)
     tiny_imagenet_val_32px.npz        (created on first use)
   ```
3. **Open** `notebooks/ReSA_long_tail_experiments.ipynb` in Colab and choose a GPU runtime.
4. Run **0.1 Mount**, edit and run **0.2 Configuration** (paths, dataset, imbalance ratio, epochs, …), run **0.3 Environment** and, once, **0.4 Datasets**.
5. Run the cells you need:

   | Cell | Does |
   |---|---|
   | 1.1 | pretrain the configured dataset / imbalance ratio |
   | 1.2 | **resume after a Colab disconnect** (run 0.1–0.3 first) |
   | 1.3 | pretrain every imbalance ratio, one after the other |
   | 1.4 | show the progress of all pretraining runs |
   | 2.1 | k-NN for a chosen probe, dataset and list of k → `knn_summary.csv` |
   | 2.2 | k-NN for both datasets × all ratios × three probes |
   | 2.3 / 2.4 | linear probe: one / all |
   | 3.2 | CIFAR-10 encoder → Tiny ImageNet, CIFAR-100, CIFAR-10 |
   | 3.3 | CIFAR-100 encoder → Tiny ImageNet, CIFAR-10, CIFAR-100 |
   | 4.1 | collect every result into `all_results.csv` and `all_per_class.csv` |
   | 4.2 | figures and raw-data CSV for one evaluation |
   | 4.3 | figures and raw data for every finished evaluation |
   | 4.4 | the three probes compared in one graph |
   | Appendix | two-minute smoke test of the whole pipeline |

Pretraining saves a checkpoint to Drive after every epoch. When Colab disconnects, nothing but the epoch in progress is lost.

## Running from a terminal

```bash
pip install -r requirements.txt

# 1. Pretrain (re-run the same command to resume)
python main.py --dataset cifar10_lt --imb_ratio 100 --seed 0 --epochs 1000 \
    --data_path ./data --dump_path ./results/cifar10_lt100

# 2a. k-NN, k = 5, 10, 20, one of the three probes
python eval_knn.py --dataset cifar10 --probe rebalanced --imb_ratio 100 --k 5 10 20 \
    --pretrained ./results/cifar10_lt100/checkpoint.pth.tar \
    --data_path ./data --dump_path ./results/cifar10_lt100_knn_rebalanced

# 2b. Linear probe (the paper's CIFAR protocol is the default: Adam, 500
#     epochs, lr 1e-2 -> 1e-6, weight decay 5e-6, batch 1000)
python eval_linear_cifar.py --dataset cifar10 --probe rebalanced --imb_ratio 100 \
    --pretrained ./results/cifar10_lt100/checkpoint.pth.tar --weights freeze --seed 0 \
    --data_path ./data --dump_path ./results/cifar10_lt100_probe_rebalanced

# 3. Transfer: CIFAR-10 encoder -> Tiny ImageNet (linear + k-NN)
python eval_transfer.py --source_dataset cifar10 --target tiny_imagenet \
    --pretrained ./results/cifar10_lt100/checkpoint.pth.tar --k 5 10 20 \
    --data_path ./data --dump_path ./results/transfer/cifar10_lt100__to__tiny_imagenet

# 4. Tables and figures
python collect_results.py --results_root ./results
python visualise_results.py --metrics_dir ./results/cifar10_lt100_knn_rebalanced/metrics --tag k10
```

Balanced pretraining uses `--dataset cifar10` (no `--imb_ratio`). Every script explains its options with `--help`, and the header of each file describes the protocol it implements. `torchrun --nproc_per_node=1 <script> ...` is equivalent to `python <script> ...`.

## What is written where

All paths are below the results folder.

| Folder | Files |
|---|---|
| `<dataset>_lt<ratio>/` (pretraining) | `checkpoint.pth.tar` (every epoch), `status.json`, `args.json`, `out.log` |
| `<run>_knn_<probe>/metrics/` | `knn_summary.csv` (one row per k), and per k: `per_class_metrics_k<K>.csv`, `confusion_matrix_k<K>.npy`, `predictions_k<K>.npz`, `summary_metrics_k<K>.csv` |
| `<run>_probe_<probe>/metrics/` | `summary_metrics_final.csv`, `per_class_metrics_final.csv`, `confusion_matrix_final.npy`, `predictions_final.npz`, `train_log.csv`, `val_log.csv`, `summary.json` |
| `transfer/<source run>__to__<target>/metrics/` | `transfer_summary.csv` (linear and each k), `linear_train_log.csv`, and the per-classifier files as above (tags `linear`, `k5`, …) |
| `<any run>/metrics/figures/` | `precision_recall_<tag>.png`, `f1_<tag>.png`, `auc_<tag>.png`, `confusion_pairs_<tag>.png`, `confusion_matrix_<tag>.png`, `prediction_distribution_<tag>.png`, `raw_per_class_<tag>.csv`, `raw_confusion_pairs_<tag>.csv`, `raw_confusion_matrix_<tag>.csv`, `raw_summary_<tag>.csv` |
| results root | `all_results.csv`, `all_per_class.csv`, `pretraining_status.csv` |

### Metrics

`acc1` and `acc5` are percentages (0–100); all other metrics are fractions between 0 and 1.

| Column | Meaning |
|---|---|
| `acc1`, `acc5` | top-1 / top-5 accuracy on the balanced test set |
| `macro_precision`, `macro_recall`, `macro_f1` | unweighted mean over classes |
| `macro_auc` | mean one-vs-rest ROC AUC (from softmax scores, or vote shares for k-NN) |
| `balanced_accuracy` | mean per-class recall (equal to accuracy on a balanced test set) |
| `gmean` | geometric mean of per-class recall; 0 if any class is never recognised |
| `head_acc`, `mid_acc`, `tail_acc` | mean recall of the 30 % most frequent classes, the middle 40 %, the 30 % least frequent |
| `head_tail_gap` | `head_acc − tail_acc` |
| `many_shot_acc`, `medium_shot_acc`, `few_shot_acc` | classes with more than 100, 20–100, fewer than 20 training images |
| `min_class_recall`, `n_zero_recall_classes` | the worst class, and how many classes are never predicted correctly |

Class groups are always defined by the **long-tailed** class sizes at the given imbalance ratio, also for the balanced probe, so the three probes are compared on the same groups.

## Protocol details that matter for replication

- **Which network is evaluated:** the momentum (EMA) encoder stored in `checkpoint.pth.tar`.
- **Long-tailed subset:** determined by dataset, imbalance ratio and seed. Seed 0 is used throughout. Pretraining and evaluation must use the same seed; the notebook guarantees this.
- **k-NN:** Euclidean distance on the raw 512-d features (the ReSA CIFAR protocol); majority vote, ties broken by the nearest neighbour. `rebalanced` divides each neighbour's vote by its class size in the long-tailed subset. `--knn_metric cosine` is available.
- **Linear probe:** the CIFAR linear evaluation protocol of the ReSA paper (Weng et al., Appendix B.3), inherited from W-MSE (Ermolov et al., 2021) and INTL (Weng et al., 2024): **one linear layer, Adam, 500 epochs, learning rate decayed exponentially from 1e-2 to 1e-6, weight decay 5e-6, batch size 1000, no augmentation, final-epoch result reported.** The same settings are used for the transfer probe, so in-domain and transfer numbers are comparable. `rebalanced` draws training images with probability inversely proportional to class size. The files tagged `best` are selected on the test set and are for diagnosis only.
- **Why 500 epochs are cheap:** the encoder is frozen and nothing is augmented, so every image has one fixed feature vector. The features are extracted once and the linear layer is trained on them (`--precompute_features`, on by default) — the same computation as running the encoder every epoch, but a few hundred times faster.
- **The earlier protocol:** `--optimizer sgd` (notebook: `PROBE_OPTIMIZER = "sgd"`) keeps this repository's previous scaffold — SGD with momentum 0.9, cosine schedule, 100 epochs, batch 256, no weight decay, and `--lr_classifier 10.0` as used for the numbers in the table below. Keep it only to reproduce those numbers, and state in the write-up which protocol a number comes from.
- **Transfer:** the linear layer is trained on features extracted once (no augmentation), with the same optimiser settings. Tiny ImageNet is resized from 64×64 to 32×32 (bicubic) and its 10,000 validation images are the test set. Images are normalised with the source dataset's statistics (`--norm source`).
- **Known deviations from a textbook setup, kept so that results stay comparable:** the CIFAR ResNet stem applies one extra pixel of padding (inherited from the upstream code); the CIFAR-10 evaluation normalisation uses standard deviation (0.247, 0.243, 0.261).
- **Randomness:** seeds are fixed, but GPU training is not bit-for-bit deterministic; expect differences of a few tenths of a percent between repetitions.

## Reference results

Linear-probe top-1 accuracy (%) on the balanced test set, ResNet-18, 1000 pretraining epochs, obtained in this study **with the earlier SGD protocol** (`--optimizer sgd --lr_classifier 10.0 --epochs 100`). Numbers produced with the paper's Adam protocol, now the default, will differ; re-run the probes to report a single consistent protocol:

| Pretraining data | long-tailed probe | re-balanced probe | balanced probe |
|---|---|---|---|
| CIFAR-10-LT, ratio 20 | 79.5 | 85.4 | 87.1 |
| CIFAR-10-LT, ratio 50 | 69.7 | 81.9 | 84.9 |
| CIFAR-10-LT, ratio 100 | 59.6 | 78.7 | 83.4 |
| CIFAR-100-LT, ratio 20 | 44.0 | 55.3 | 61.2 |
| CIFAR-100-LT, ratio 50 | 34.2 | 48.7 | 57.4 |
| CIFAR-100-LT, ratio 100 | 30.5 | 45.9 | 57.1 |

The gap between the first and the last column shows that most of the accuracy lost under imbalance is lost in the classifier, not in the representation.

## Weights & Biases

Logging to Weights & Biases is off by default. To enable it pass `--use_wandb` (notebook: `USE_WANDB = True`) and provide the API key through the `WANDB_API_KEY` environment variable or Colab's *Secrets* panel. Never write a key into a notebook or commit the `wandb/` folder; `.gitignore` excludes it.

## Publishing on GitHub

```bash
cd resa_cifar_lt
git init && git add . && git commit -m "ReSA under class imbalance: code for the study"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/resa_cifar_lt.git
git push -u origin main
```

Tag the version used for the dissertation (`git tag v1.0 && git push --tags`) and cite that tag or its commit hash, so readers get exactly the code behind the results.

## Citing

This repository (replace the placeholders):

```bibtex
@misc{YOURKEY2026resalt,
  author = {YOUR NAME},
  title  = {ReSA under class imbalance: long-tailed pretraining and evaluation on CIFAR-10/100},
  year   = {2026},
  howpublished = {\url{https://github.com/YOUR-USERNAME/resa_cifar_lt}},
  note   = {Version v1.0}
}
```

The method (check the entry against the published version):

```bibtex
@inproceedings{weng2025clustering,
  title     = {Clustering Properties of Self-Supervised Learning},
  author    = {Weng, Xi and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2501.18452}
}
```

Long-tailed CIFAR protocol: Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019; Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss*, NeurIPS 2019. Tiny ImageNet: Le and Yang, *Tiny ImageNet Visual Recognition Challenge*, 2015.

## Licence and attribution

Apache License 2.0 (see `LICENSE`), the licence of the upstream ReSA code. Files taken from [winci-ai/resa_cifar](https://github.com/winci-ai/resa_cifar) keep their copyright notice; the modifications are described in `CHANGES.md` and in the header of each modified file.

