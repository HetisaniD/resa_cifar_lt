# Datasets for the downstream evaluation scripts (eval_knn.py,
# eval_transfer.py): CIFAR-10, CIFAR-100 and Tiny ImageNet as plain arrays,
# their normalisation statistics, and the long-tailed reference distribution.
#
# Pretraining has its own multi-crop pipeline in src/transform.py; the
# linear probe has its own loaders in eval_linear_cifar.py. All of them build
# the long-tailed subset with the same function (src/cifar_lt.py) and the
# same seed, so they select exactly the same images.

from types import SimpleNamespace

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100

from src.cifar_lt import apply_long_tail, get_img_num_per_cls
from src.tiny_imagenet import load_tiny_imagenet

# Channel mean / std used to normalise images at EVALUATION time.
# CIFAR-10's std differs from the pretraining value (0.2470, 0.2435, 0.2616)
# in the third decimal. That is inherited from the original resa_cifar
# evaluation code and is kept so results stay comparable with it; the effect
# is about 0.2% of the input scale.
EVAL_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tiny_imagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}
NATIVE_PER_CLASS = {"cifar10": 5000, "cifar100": 500, "tiny_imagenet": 500}  # training images per class
DATASETS = tuple(NUM_CLASSES)


def base_name(dataset):
    """'cifar10_lt' -> 'cifar10'."""
    return dataset[:-3] if dataset.endswith("_lt") else dataset


def is_balanced_reference(imb_ratio, imb_type="exp"):
    """True when (imb_ratio, imb_type) describes the full balanced set."""
    return imb_type != "Dsample" and (imb_ratio is None or imb_ratio <= 1)


def reference_lt_counts(dataset, imb_ratio, imb_type="exp", img_max=None):
    """Training images per class in the long-tailed reference distribution.

    This is the distribution the encoder was pretrained on. Class groups
    (head / medium / tail, many / medium / few shot) are always derived from
    it, whichever data the downstream classifier is trained on.
    """
    name = base_name(dataset)
    native = NATIVE_PER_CLASS[name]
    if is_balanced_reference(imb_ratio, imb_type):
        return [native] * NUM_CLASSES[name]
    return get_img_num_per_cls(NUM_CLASSES[name], img_max or native, imb_type, imb_ratio)


def long_tail_indices(targets, imb_ratio, imb_type="exp", seed=0, img_max=None):
    """Indices (into the full training set) of the images that the
    long-tailed subset keeps.

    Calls the same apply_long_tail() used for pretraining on an index array,
    so the selection -- including the random choice of images inside each
    class -- is identical to pretraining with the same seed. This lets the
    k-NN script extract features for the full training set once and take the
    long-tailed bank as a subset of them.
    """
    shim = SimpleNamespace(data=np.arange(len(targets)), targets=list(targets))
    apply_long_tail(shim, imb_ratio=imb_ratio, imb_type=imb_type, seed=seed, img_max=img_max)
    return np.asarray(shim.data, dtype=np.int64)


def load_arrays(dataset, root, train, img_size=32, download=True, tiny_zip=None):
    """Load a dataset split fully into memory.

    Returns (images uint8 [N, H, W, 3], labels int64 [N], class names).
    The 'test' split of Tiny ImageNet is its labelled validation set (the
    official test labels are not public).
    """
    name = base_name(dataset)
    if name == "cifar10":
        ds = CIFAR10(root=root, train=train, download=download)
    elif name == "cifar100":
        ds = CIFAR100(root=root, train=train, download=download)
    elif name == "tiny_imagenet":
        return load_tiny_imagenet(root, "train" if train else "val", img_size=img_size,
                                  zip_path=tiny_zip, download=download)
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected one of {DATASETS})")
    return np.asarray(ds.data), np.asarray(ds.targets, dtype=np.int64), list(ds.classes)
