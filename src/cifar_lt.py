# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# CIFAR-10-LT / CIFAR-100-LT / CINIC-10-LT long-tailed subsampling.
#
# Implements the standard exponential-imbalance protocol introduced by
# Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
# (CVPR 2019) and used by Cao et al., "Learning Imbalanced Datasets with
# Label-Distribution-Aware Margin Loss" (LDAM-DRW, NeurIPS 2019), and
# essentially every CIFAR-LT paper since (BBN, cRT/LWS, RIDE, ...). Using
# this exact formula keeps results comparable with the published literature
# at the same --imb_ratio.
#
# Definition: for `imb_ratio` R (e.g. 100), class i in {0, ..., C-1} keeps
#   n_i = n_max * R^(-i / (C-1))
# samples, where n_max = (total samples) / C is the per-class count of the
# original balanced dataset. This gives n_0 = n_max (largest class) and
# n_{C-1} = n_max / R (smallest class) -- i.e. R is literally the ratio
# between the largest and smallest class sizes.
#
# For CIFAR-10 (n_max=5000) this gives, e.g.:
#   R=20  -> class sizes from 5000 down to 250   (roughly)
#   R=50  -> class sizes from 5000 down to 100
#   R=100 -> class sizes from 5000 down to 50
#
# R=1 is a no-op (returns the original balanced counts), so this function is
# safe to call unconditionally with --imb_ratio 1 if you ever want that.
#
# n_max defaults to the dataset's own native per-class count, so the same R
# means different absolute smallest-class sizes on different datasets (e.g.
# 50 for CIFAR-10 vs 180 for CINIC-10 train+valid's native 18000/class, both
# at R=100) -- that's expected, not a bug: R is what's comparable across
# datasets, not the raw counts. To instead target a *specific* n_max
# regardless of a dataset's native size (e.g. to reproduce the same
# [40000..400] class sizes on CINIC-10 that a CIFAR-10 run got by first
# padding CIFAR-10 up to 40000/class with extra images), pass `img_max`
# explicitly below. A class whose target then exceeds how many native images
# it actually has gets padded with augmented copies of its own images via
# `augment`, since apply_long_tail no longer has any other source to draw
# from.

import numpy as np
from PIL import Image


def get_img_num_per_cls(num_classes, img_max, imb_type, imb_ratio):
    """Per-class sample counts for a given imbalance type/ratio.

    imb_type='exp'  : smooth exponential decay (the standard CIFAR-LT protocol)
    imb_type='step' : first half of classes at img_max, second half at
                       img_max/imb_ratio (a secondary protocol some papers
                       also report; included for completeness)
    """
    imb_factor = 1.0 / imb_ratio
    img_num_per_cls = []
    if imb_type == "exp":
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == "step":
        half = num_classes // 2
        img_num_per_cls.extend([int(img_max)] * half)
        img_num_per_cls.extend([int(img_max * imb_factor)] * (num_classes - half))
    elif imb_type == "Dsample":
        for cls_idx in range(num_classes):
            img_num_per_cls.append(int(imb_ratio))
    else:
        raise ValueError(f"Unknown imb_type: {imb_type!r} (expected 'exp' or 'step')")
    return img_num_per_cls


def apply_long_tail(dataset, imb_ratio, imb_type="exp", seed=0, img_max=None, augment=None):
    """
    Mutates a torchvision-style dataset (must expose `.data` as an ndarray
    indexable along axis 0, and `.targets` as a list/array of int labels --
    true for torchvision.datasets.CIFAR10/CIFAR100, and for this repo's
    CINIC10) into its long-tailed subset, in place.

    img_max: largest class's target count. Defaults to None, which derives
        it from the dataset itself (len(dataset.data) / num_classes) -- the
        standard CIFAR-LT convention, and the only behavior this function had
        before this parameter existed; every existing call site that doesn't
        pass img_max is completely unaffected. Pass an explicit img_max to
        target a specific class-0 size regardless of the dataset's native
        per-class count.

    augment: optional callable (PIL.Image -> PIL.Image), e.g. a
        torchvision.transforms.Compose of crop/flip/color-jitter. Used ONLY
        for a class whose target count (from img_max) exceeds how many
        native images that class actually has -- resampled native images are
        passed through it to manufacture the shortfall. Required if img_max
        is large enough to trigger this for any class; a class that doesn't
        need it never calls this, and it's fine to leave as None otherwise.
        This is recycled, augmented duplication, not new information --
        kept as an explicit, opt-in argument (rather than silently kicking
        in) so it's never an undisclosed part of the resulting dataset.

    Per-class samples are chosen uniformly at random (seeded, so the split is
    reproducible) rather than just taking the first n_i in dataset order.

    Returns {class_id: kept_sample_count} for logging. Also sets
    dataset.num_augmented_dict = {class_id: augmented_count}, all zero
    unless img_max forced augmentation for some class.
    """
    targets_np = np.array(dataset.targets, dtype=np.int64)
    classes = np.unique(targets_np)
    num_classes = len(classes)
    if img_max is None:
        img_max = len(dataset.data) / num_classes

    img_num_per_cls = get_img_num_per_cls(num_classes, img_max, imb_type, imb_ratio)

    rng = np.random.default_rng(seed)
    new_data = []
    new_targets = []
    num_per_cls_dict = {}
    num_augmented_dict = {}
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        rng.shuffle(idx)
        native_pool = dataset.data[idx, ...]

        if the_img_num <= len(native_pool):
            selected = native_pool[:the_img_num]
            num_augmented_dict[int(the_class)] = 0
        else:
            if augment is None:
                raise ValueError(
                    f"apply_long_tail: class {int(the_class)} needs {the_img_num} images "
                    f"(img_max={img_max}, imb_ratio={imb_ratio}) but only {len(native_pool)} "
                    "native images are available, and no `augment` transform was given to "
                    "manufacture the rest. Pass img_max <= the dataset's native per-class "
                    "count, or supply `augment`."
                )
            n_extra = the_img_num - len(native_pool)
            extra = np.empty((n_extra, *native_pool.shape[1:]), dtype=native_pool.dtype)
            for i in range(n_extra):
                img = Image.fromarray(native_pool[i % len(native_pool)])
                extra[i] = np.array(augment(img))
            selected = np.concatenate([native_pool, extra], axis=0)
            num_augmented_dict[int(the_class)] = n_extra

        new_data.append(selected)
        new_targets.extend([int(the_class)] * len(selected))
        num_per_cls_dict[int(the_class)] = len(selected)

    dataset.data = np.concatenate(new_data, axis=0)
    dataset.targets = new_targets
    dataset.num_per_cls_dict = num_per_cls_dict
    dataset.num_augmented_dict = num_augmented_dict
    return num_per_cls_dict
