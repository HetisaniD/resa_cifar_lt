#!/bin/bash
# Learning-rate sweep for the linear probe (optional).
#
# Trains short probes with several values of --lr_classifier so that a
# sensible value can be chosen before the full runs; compare them afterwards
# with scripts/pick_best_lr.py. The sweep selects on the TEST set, so use it
# only to find the right order of magnitude, and report the value you used.
#
# Usage (from the repository root):
#   bash scripts/lr_sweep.sh <checkpoint.pth.tar> <dataset> <imb_ratio> <data_path> <sweep_root>
# Example:
#   bash scripts/lr_sweep.sh /results/cifar100_lt100/checkpoint.pth.tar cifar100 100 /data /results/lr_sweep
set -e
PRETRAINED=${1:?path of the pretraining checkpoint}
DATASET=${2:-cifar10}
IMB_RATIO=${3:-100}
DATA_PATH=${4:-./data}
SWEEP_ROOT=${5:-./results/lr_sweep}
PROBE=${PROBE:-long_tailed}       # long_tailed | rebalanced | balanced
SWEEP_EPOCHS=${SWEEP_EPOCHS:-50}  # short budget: this is a sweep, not a final run

for LR in 0.01 0.03 0.1 0.3 1.0 3.0 10.0; do
  echo ">>> lr_classifier=$LR"
  python eval_linear_cifar.py \
    --dataset "$DATASET" --probe "$PROBE" --imb_ratio "$IMB_RATIO" \
    --data_path "$DATA_PATH" \
    --dump_path "$SWEEP_ROOT/freeze_lr${LR}" \
    --pretrained "$PRETRAINED" \
    --weights freeze --lr_classifier "$LR" --epochs "$SWEEP_EPOCHS"
done
