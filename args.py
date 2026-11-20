# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

#
# Modified for the long-tailed ReSA study (see CHANGES.md): W&B is opt-in,
# plus --save_every / --force_resume and an automatic distributed backend.

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Implementation of ReSA")

    parser.add_argument("--wandb_project", type=str, default="resa_cifar",
                    help="name of the run for wandb project" )

    parser.add_argument("--env_name", type=str, default="resa_cifar",
                    help="name of the run for wandb env" )

    parser.add_argument("--use_wandb", action="store_true",
                help="log to Weights & Biases (off by default so the code runs without "
                     "an account; needs `pip install wandb` and WANDB_API_KEY in the "
                     "environment -- never write the key into a notebook or script)")

    parser.add_argument("--disable_wandb", action="store_true",
                help="deprecated, kept so old commands still run: W&B is already off "
                     "unless --use_wandb is given")

    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")

    parser.add_argument('--seed', default=None, type=int,
                    help='random seed. Seeds the network initialisation, the augmentations '
                         'and the data order, AND selects which images the long-tailed '
                         'subset keeps (None behaves as 0 for the subset). Use the same '
                         'seed for pretraining and for the downstream scripts so they '
                         'see the same subset.')

    parser.add_argument("--save_every", type=int, default=0,
                    help="additionally keep checkpoint_ep<N>.pth.tar every N epochs "
                         "(0 = only the rolling checkpoint.pth.tar used for resuming)")

    parser.add_argument("--force_resume", action="store_true",
                    help="resume from the checkpoint in --dump_path even if it was created "
                         "with different settings (by default this is an error)")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--dataset", type=str, default="cifar10",
                    help="cifar10, cifar100, stl10, cinic10, or the long-tailed variants "
                         "cifar10_lt / cifar100_lt / cinic10_lt (see --imb_ratio, --imb_type). "
                         "cinic10(_lt) is pretrained on CINIC-10's train+valid splits pooled "
                         "together (~18k images/class before any long-tail subsampling); "
                         "see src/cinic10.py.")

    parser.add_argument("--imb_ratio", type=int, default=100,
                    help="imbalance ratio (largest class size / smallest class size) "
                         "used only when --dataset is cifar10_lt, cifar100_lt, or cinic10_lt. "
                         "Common literature values: 20, 50, 100.")

    parser.add_argument("--imb_type", type=str, default="exp", choices=["exp", "step", "Dsample"],
                    help="long-tail sampling profile used only when --dataset is "
                         "cifar10_lt, cifar100_lt, or cinic10_lt. 'exp' (exponential decay) is "
                         "the standard CIFAR-LT protocol; 'step' is a secondary variant; "
                         "'Dsample' keeps a BALANCED subset with --imb_ratio images in every "
                         "class (used for the training-set-size experiment).")

    parser.add_argument("--img_max", type=int, default=None,
                    help="override the largest (class-0) target image count used to derive "
                         "per-class counts for cifar10_lt / cifar100_lt / cinic10_lt, together "
                         "with --imb_ratio. Defaults to the dataset's own native per-class "
                         "count (e.g. 5000 for CIFAR-10, 18000 for CINIC-10 train+valid) -- "
                         "the standard CIFAR-LT convention, unaffected unless this is set. Set "
                         "it higher (e.g. 40000) to reproduce the same [img_max .. img_max/"
                         "imb_ratio] class sizes across datasets with different native counts. "
                         "Only cinic10_lt can actually satisfy an img_max above its native "
                         "count: it pads the shortfall with augmented copies (see "
                         "src/cifar_lt.py's `augment` argument to apply_long_tail). "
                         "cifar10_lt / cifar100_lt raise a clear error instead, since no "
                         "augmentation path is wired for them.")

    parser.add_argument("--data_path", type=str, default="./data",
                    help="path to dataset repository")

    parser.add_argument("--crops_nmb", type=int, default=[1], nargs="+",
                    help="list of number of crops")

    parser.add_argument("--crops_size", type=int, default=[32], nargs="+",
                    help="crops resolutions")

    parser.add_argument("--solarization_prob", type=float, default=[0.2], nargs="+",
                    help="solarization prob")

    parser.add_argument("--size_dataset", type=int, default=-1,
                    help="size of dataset")

    parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")

    #########################
    ## resa specific params #
    #########################

    parser.add_argument("--temperature", default=0.4, type=float,
                    help="temperature parameter in training loss")

    parser.add_argument("--momentum", type=float, default=0.996,
                    help="momentum")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=1000, type=int,
                    help="number of total epochs to run")

    parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument('--lr', default=0.3, type=float,
                    help='initial (base) learning rate for train')

    parser.add_argument('--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

    parser.add_argument("--warmup_epochs", default=2, type=int,
                    help="number of warmup epochs")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--world_size", default=1, type=int,
                    help="""number of processes: it is set automatically and
                            should not be passed as argument""")

    parser.add_argument("--rank", default=0, type=int,
                    help="rank of this process: it is set automatically and should not be passed as argument")

    parser.add_argument('--dist-backend', default=None, type=str,
                    help="distributed backend (default: 'nccl' on GPU, 'gloo' on CPU)")

    parser.add_argument("--no-set-device-rank", default=False, action="store_true",
                    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")

    parser.add_argument("--dist-url", default="env://", type=str,
                    help="url used to set up distributed training")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (e.g. resnet18, resnet50, resnet200, resnet50x2)')

    parser.add_argument("--mlp_layers", type=int, default=3,
                    help="number of FC layers in projection")

    parser.add_argument("--mlp_dim", type=int, default=2048,
                    help="size of FC layers in projection/predictor")

    parser.add_argument("--emb", type=int, default=512,
                    help="embedding size")

    parser.add_argument(
        "--eval_every", type=int, default=5, help="how often to evaluate"
    )

    return parser.parse_args()
