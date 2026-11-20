# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# CINIC-10 (Darlow et al., 2018, https://arxiv.org/abs/1810.03505) loader.
#
# CINIC-10 extends CIFAR-10 with downsampled ImageNet images for the same 10
# classes at the same 32x32 resolution. It ships as three equal 90,000-image
# splits (train/valid/test), one PNG folder per class -- not as torchvision's
# pickled-batch CIFAR format, so there is no built-in torchvision dataset for
# it. This class decodes those PNGs once into the same `.data` (ndarray) /
# `.targets` (list) layout torchvision.datasets.CIFAR10 uses, and also
# implements `__getitem__`/`__len__`/`transform` so it works as a plain
# Dataset too. That dual interface means it drops in unchanged wherever this
# repo currently uses CIFAR10: MultiVisionDataset (src/transform.py) reads
# `.data` directly for its multi-crop pipeline, apply_long_tail
# (src/cifar_lt.py) mutates `.data`/`.targets` in place, and eval/dataloader.py
# uses it as a normal transform-applying Dataset for the linear/knn probe.
#
# IMPORTANT: CINIC-10's `test` split contains the *entire* original CIFAR-10
# test set plus extra CIFAR-10 train images (see the paper, Section 2). It is
# therefore only ever used here as the held-out evaluation split -- never
# pooled into pretraining -- to avoid leaking test data into training.

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

CINIC10_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
CINIC10_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC10_STD = (0.24205776, 0.23828046, 0.25874835)
CINIC10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

# Default `augment` transform for apply_long_tail (src/cifar_lt.py) when
# --img_max is set above CINIC-10's native ~18k/class (train+valid): used only
# to pad the shortfall for whichever classes need it, never applied to every
# image. Same recipe as a standard CIFAR-style augmentation pipeline -- crop/
# flip/color-jitter -- since the "extra" images are recycled, transformed
# copies of that class's own native images, not new information.
DEFAULT_AUGMENT = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
])

# Folder names a fresh extraction (or a manually-placed download) commonly
# uses for the CINIC-10 root, tried case-insensitively under --data_path
# whenever --data_path itself isn't already the root.
_NESTED_NAME_CANDIDATES = ["cinic-10", "CINIC-10", "cinic10", "CINIC10", "cinic_10"]


def _missing_split_reason(path, split):
    """None if `path/split` looks like a complete CINIC-10 split (a directory
    containing all 10 class subfolders), otherwise a short human-readable
    reason it doesn't. Used purely for diagnostics: when an existing folder
    is rejected, the log says *why* instead of silently falling through to a
    multi-hundred-MB re-download."""
    split_dir = os.path.join(path, split)
    if not os.path.isdir(split_dir):
        return f"'{split}/' does not exist"
    entries = set(os.listdir(split_dir))
    missing = [c for c in CINIC10_CLASSES if c not in entries]
    if missing:
        return f"'{split}/' is missing class folder(s): {missing}"
    return None


def _is_cinic10_root(path, splits):
    """True if `path` already contains a complete CINIC-10 folder for every
    split in `splits`. Only the splits actually requested are checked, so a
    folder with just train/ and valid/ (no test/ -- common when test wasn't
    kept, since pretraining never uses it) still counts as usable for
    pretraining or for the clf/probe loader, which only ever request
    ("train", "valid")."""
    if not os.path.isdir(path):
        return False
    return all(_missing_split_reason(path, split) is None for split in splits)


def _find_existing_cinic10_root(root, splits):
    """Look for an already-extracted CINIC-10 that covers `splits`: checks
    `root` itself first, then common nested-folder names a download or manual
    extraction could have produced (case-sensitive filesystems mean "CINIC-10"
    and "cinic-10" are different paths, so both -- and a couple of other
    common spellings -- are tried). Returns (resolved_path, reasons) on a
    match, or (None, reasons) with one human-readable rejection reason per
    candidate that existed but didn't qualify."""
    candidates = [root] + [os.path.join(root, name) for name in _NESTED_NAME_CANDIDATES]
    reasons = []
    seen = set()
    for cand in candidates:
        real = os.path.normpath(cand)
        if real in seen or not os.path.isdir(cand):
            continue
        seen.add(real)
        if _is_cinic10_root(cand, splits):
            return cand, reasons
        split_reasons = [r for s in splits for r in [_missing_split_reason(cand, s)] if r]
        reasons.append(f"  {cand}: {'; '.join(split_reasons)}")
    return None, reasons


def _download_cinic10(root):
    os.makedirs(root, exist_ok=True)
    print(f"Downloading CINIC-10 (~650MB, one-off) to {root} ...")
    try:
        # No solid published MD5 for this exact .tar.gz (DataShare's listed
        # checksum is for a separate .zip packaging of the same data), so
        # checksum verification is skipped rather than risk a false mismatch.
        download_and_extract_archive(
            CINIC10_URL, download_root=root, extract_root=root,
            filename="CINIC-10.tar.gz", md5=None, remove_finished=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Automatic CINIC-10 download failed ({e}). Download it manually from "
            "https://github.com/BayesWatch/cinic-10 or "
            "https://datashare.ed.ac.uk/handle/10283/3192 and extract it so that "
            f"{root}/train, {root}/valid, and {root}/test each contain one "
            "PNG folder per class."
        ) from e


def _resolve_cinic10_root(root, splits, download):
    """Find where an already-extracted CINIC-10 covering `splits` lives
    before downloading anything (see _find_existing_cinic10_root for the
    search order). Only downloads (into `root/cinic-10`) if nothing already
    qualifies and `download=True`; otherwise raises, listing why every
    candidate that did exist was rejected.
    """
    found, reasons = _find_existing_cinic10_root(root, splits)
    if found is not None:
        print(f"Found existing CINIC-10 (splits={list(splits)}) at {found} -- skipping download.")
        return found

    if not download:
        detail = ("\n" + "\n".join(reasons)) if reasons else ""
        raise FileNotFoundError(
            f"CINIC-10 (splits={list(splits)}) not found under {root}, and download=False."
            f"{detail}\n"
            "Pass download=True, or point --data_path at a folder that already "
            "contains the needed split(s), each with all 10 class subfolders."
        )

    if reasons:
        print(f"--data_path {root!r} exists but isn't a usable CINIC-10 root for "
              f"splits={list(splits)}, so downloading a fresh copy instead. Why the "
              f"existing folder(s) were rejected:")
        print("\n".join(reasons))

    nested = os.path.join(root, "cinic-10")
    _download_cinic10(nested)
    return nested


def _load_split(root, split):
    """Decode one CINIC-10 split into (data, targets), caching the result to
    an .npz next to the PNGs so repeat runs skip re-decoding ~90,000 images."""
    cache = os.path.join(root, f"{split}_cache.npz")
    if os.path.exists(cache):
        npz = np.load(cache)
        return npz["data"], npz["targets"].tolist()

    folder = ImageFolder(os.path.join(root, split))
    if folder.classes != CINIC10_CLASSES:
        raise RuntimeError(
            f"unexpected CINIC-10 class folders under {os.path.join(root, split)}: "
            f"{folder.classes} (expected {CINIC10_CLASSES})"
        )
    data = np.empty((len(folder.samples), 32, 32, 3), dtype=np.uint8)
    targets = np.empty(len(folder.samples), dtype=np.int64)
    for i, (path, label) in enumerate(tqdm(folder.samples, desc=f"decoding CINIC-10 {split}")):
        data[i] = np.array(Image.open(path).convert("RGB"))
        targets[i] = label
    np.savez(cache, data=data, targets=targets)
    return data, targets.tolist()


class CINIC10(torch.utils.data.Dataset):
    """Behaves like torchvision.datasets.CIFAR10, backed by CINIC-10.

    `splits`: which CINIC-10 splits to concatenate into `.data`/`.targets`.
    Defaults to ('train', 'valid') -- the combination the CINIC-10 paper
    itself suggests for a larger training pool -- keeping 'test' free for
    evaluation (see the module docstring for why).
    """

    def __init__(self, root, splits=("train", "valid"), download=True, transform=None):
        self.transform = transform
        cinic_root = _resolve_cinic10_root(root, splits, download)

        all_data, all_targets = [], []
        for split in splits:
            d, t = _load_split(cinic_root, split)
            all_data.append(d)
            all_targets.extend(t)
        self.data = np.concatenate(all_data, axis=0)
        self.targets = all_targets
        self.classes = CINIC10_CLASSES

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
