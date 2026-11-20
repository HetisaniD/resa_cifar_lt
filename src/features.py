# Frozen-encoder feature extraction for eval_knn.py and eval_transfer.py.
#
# Both scripts evaluate a FROZEN encoder without data augmentation, so every
# image maps to one fixed feature vector. The features are therefore computed
# once and reused for every classifier (each k of the k-NN sweep, and the
# linear probe), which turns minutes of repeated forward passes into seconds.
# With --cache_dir they are also kept on disk between runs.

import hashlib
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import torch

from src.model import get_encoder
from src.utils import load_pretrained_encoder


def build_frozen_encoder(arch, pretrained, source_dataset, device):
    """The pretrained ReSA momentum encoder, in eval mode, gradients off.

    source_dataset: the dataset the encoder was PRETRAINED on. It selects the
        network stem (get_encoder builds the CIFAR stem -- 3x3 convolution,
        no max-pool -- for any CIFAR/CINIC name), which must match the
        checkpoint whatever dataset is being evaluated.
    Returns (encoder, feature dimension, pretraining epoch of the checkpoint).
    """
    encoder, out_size = get_encoder(SimpleNamespace(arch=arch, dataset=source_dataset))
    epoch = load_pretrained_encoder(encoder, pretrained)
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    return encoder, out_size, epoch


@torch.no_grad()
def extract_features(encoder, images, mean, std, device, batch_size=500):
    """Encode uint8 images [N, H, W, 3] -> float32 features [N, D].

    Applies the same preprocessing as the linear probe: scale to [0, 1], then
    normalise each channel with (mean, std). No augmentation.
    """
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size])).to(device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean_t) / std_t
        out.append(encoder(x).float().cpu())
    return torch.cat(out).numpy()


def _signature(pretrained, extra):
    """Short hash identifying (checkpoint file, preprocessing)."""
    st = os.stat(pretrained)
    key = dict(path=os.path.abspath(pretrained), size=st.st_size, mtime=int(st.st_mtime), **extra)
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_features(cache_dir, encoder, pretrained, images, mean, std, device, tag, batch_size=500):
    """extract_features() with an optional on-disk cache.

    The cache key covers the checkpoint (path, size, modification time), the
    normalisation and `tag` (dataset / split / resolution), so a retrained
    checkpoint or a different setting never reuses stale features.
    """
    path = None
    if cache_dir:
        sig = _signature(pretrained, dict(mean=list(mean), std=list(std), tag=tag, n=len(images)))
        path = os.path.join(cache_dir, f"features_{tag}_{sig}.npy")
        if os.path.isfile(path):
            logging.info(f"features: loaded from cache {path}")
            return np.load(path)
    logging.info(f"features: encoding {len(images)} images ({tag}) ...")
    feats = extract_features(encoder, images, mean, std, device, batch_size)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp.npy"
        np.save(tmp, feats)
        os.replace(tmp, path)
    return feats
