# Data loaders for the lightweight evaluation that MONITORS pretraining
# (eval/evaluate.py, called every --eval_every epochs from main.py).
# The reported results come from eval_knn.py / eval_linear_cifar.py /
# eval_transfer.py, not from here.

import torch
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100
import torchvision.transforms as transforms
from abc import ABCMeta
from functools import lru_cache
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.cifar_lt import apply_long_tail
from src.cinic10 import CINIC10, CINIC10_MEAN, CINIC10_STD, DEFAULT_AUGMENT


@torch.no_grad()
def get_data(model, loader, output_size, device):
    """ encodes whole dataset into embeddings.

    Builds the result by concatenating per-batch outputs rather than writing
    into a pre-allocated (len(loader), batch_size, output_size) tensor, since
    that assumes every batch -- including the last -- has exactly
    `loader.batch_size` samples. With drop_last=False (used for both the
    clf/probe and test loaders), the final batch is smaller whenever the
    dataset size isn't a multiple of batch_size -- always true for a
    long-tailed (_lt) split, and only accidentally true for a balanced one.
    """
    model_device = next(model.parameters()).device
    xs, ys = [], []
    for x, y in loader:
        x = x.to(model_device)
        xs.append(model(x).to(device))
        ys.append(y.to(device))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, args, bs_clf=1000, bs_test=1000):
        self.bs_clf, self.bs_test = bs_clf, bs_test
        self.workers = args.workers
        self.data_path = args.data_path
        self.long_tailed = args.dataset.endswith('_lt')
        self.imb_ratio = getattr(args, 'imb_ratio', 100)
        self.imb_type = getattr(args, 'imb_type', 'exp')
        self.img_max = getattr(args, 'img_max', None)
        self.lt_seed = args.seed if getattr(args, 'seed', None) is not None else 0

    @property
    @lru_cache()
    def clf(self):
        return DataLoader(dataset=self.ds_clf(), batch_size=self.bs_clf, shuffle=True,
                           num_workers=self.workers, pin_memory=True, drop_last=False)

    @property
    @lru_cache()
    def test(self):
        return DataLoader(dataset=self.ds_test(), batch_size=self.bs_test, shuffle=False,
                           num_workers=self.workers, pin_memory=True, drop_last=False)


class CIFAR10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def ds_clf(self):
        ds = C10(root=self.data_path, train=True, download=True, transform=self.base_transform())
        if self.long_tailed:
            # No augment=: matches src/transform.py's cifar10_lt path -- an
            # --img_max above CIFAR-10's native 5000/class raises rather than
            # silently padding with augmented copies.
            apply_long_tail(ds, imb_ratio=self.imb_ratio, imb_type=self.imb_type,
                             seed=self.lt_seed, img_max=self.img_max)
        return ds

    def ds_test(self):
        return C10(root=self.data_path, train=False, download=True, transform=self.base_transform())


class CIFAR100_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    def ds_clf(self):
        ds = C100(root=self.data_path, train=True, download=True, transform=self.base_transform())
        if self.long_tailed:
            apply_long_tail(ds, imb_ratio=self.imb_ratio, imb_type=self.imb_type,
                             seed=self.lt_seed, img_max=self.img_max)
        return ds

    def ds_test(self):
        return C100(root=self.data_path, train=False, download=True, transform=self.base_transform())


class CINIC10_clf(BaseDataset):
    def base_transform(self):
        return transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(CINIC10_MEAN, CINIC10_STD)])

    def ds_clf(self):
        ds = CINIC10(root=self.data_path, splits=("train", "valid"), download=True,
                     transform=self.base_transform())
        if self.long_tailed:
            # augment=DEFAULT_AUGMENT: mirrors src/transform.py's cinic10_lt
            # path, so the eval/probe pool is built the same way (same
            # img_max, same seed -> same per-class selection/augmentation)
            # as the pretraining pool, not a differently-sized long tail.
            apply_long_tail(ds, imb_ratio=self.imb_ratio, imb_type=self.imb_type,
                             seed=self.lt_seed, img_max=self.img_max, augment=DEFAULT_AUGMENT)
        return ds

    def ds_test(self):
        return CINIC10(root=self.data_path, splits=("test",), download=True,
                        transform=self.base_transform())


def get_clf(args):
    if args.dataset in ('cifar10', 'cifar10_lt'):
        return CIFAR10_clf(args)
    elif args.dataset in ('cifar100', 'cifar100_lt'):
        return CIFAR100_clf(args)
    elif args.dataset in ('cinic10', 'cinic10_lt'):
        return CINIC10_clf(args)
    else:
        raise ValueError(f"get_clf: unsupported --dataset {args.dataset!r} "
                          "(expected one of: cifar10, cifar10_lt, cifar100, cifar100_lt, "
                          "cinic10, cinic10_lt)")
