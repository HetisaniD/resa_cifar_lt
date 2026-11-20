# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# main.py -- ReSA self-supervised pretraining on CIFAR-10/100 (balanced or
# long-tailed).
#
# Modified for the long-tailed ReSA study (see CHANGES.md). The training
# procedure itself is unchanged from https://github.com/winci-ai/resa_cifar;
# the changes make long runs on Google Colab safe to interrupt and resume:
#
#   * `checkpoint.pth.tar` is rewritten after EVERY epoch, atomically, with
#     one backup. Re-running the same command with the same --dump_path
#     continues from the last finished epoch (model, optimiser, schedules and
#     random-number state are all restored).
#   * The settings of the run are stored in the checkpoint and in
#     `args.json`; resuming with different settings is refused, so a folder
#     can never silently mix two experiments.
#   * `status.json` records progress (epoch reached / finished) so a
#     notebook can show which runs are complete without loading checkpoints.
#   * Weights & Biases logging is opt-in (--use_wandb).
#
# Usage (single GPU):
#   python main.py --dataset cifar10_lt --imb_ratio 100 --seed 0 \
#       --data_path /path/to/datasets --dump_path /path/to/results/cifar10_lt100
# `torchrun --nproc_per_node=1 main.py ...` is equivalent.

import os
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from args import get_args
from src.transform import MultiVisionDataset
from method import get_method
from eval.evaluate import get_acc
from eval.dataloader import get_clf

from src.utils import (
    setup_logging,
    init_distributed_device,
    restart_from_checkpoint,
    save_checkpoint,
    write_json,
    random_seed,
    AverageMeter,
    cosine_scheduler,
    )

# Settings that define a pretraining run. They are saved in every checkpoint
# and compared on resume (see restart_from_checkpoint).
RUN_DEFINING_ARGS = [
    "dataset", "imb_ratio", "imb_type", "img_max", "size_dataset", "seed",
    "arch", "mlp_layers", "mlp_dim", "emb", "temperature", "momentum",
    "epochs", "batch_size", "lr", "wd", "warmup_epochs",
    "crops_nmb", "crops_size", "solarization_prob",
]


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def run_defining_args(args):
    """The subset of the command-line settings that must not change between
    the start of a run and a resume."""
    out = {k: getattr(args, k) for k in RUN_DEFINING_ARGS}
    if not args.dataset.endswith("_lt"):
        # imbalance settings are ignored for balanced datasets
        for k in ("imb_ratio", "imb_type", "img_max"):
            out.pop(k)
    return out


def init_wandb(args):
    """Start (or resume) a W&B run. Returns the wandb module, or None when
    logging is off. The run id is kept in <dump_path>/wandb_id.txt so that a
    resumed training continues the same W&B run instead of creating a new one."""
    if not args.use_wandb or args.rank != 0:
        return None
    import wandb  # imported lazily: only needed with --use_wandb
    id_file = os.path.join(args.dump_path, "wandb_id.txt")
    if os.path.isfile(id_file):
        run_id = open(id_file).read().strip()
    else:
        run_id = wandb.util.generate_id()
        with open(id_file, "w") as f:
            f.write(run_id)
    wandb.init(project=args.wandb_project, name=args.env_name, config=vars(args),
               id=run_id, resume="allow")
    return wandb


def main():
    args = get_args()
    random_seed(args.seed)

    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    os.makedirs(args.dump_path, exist_ok=True)
    setup_logging(os.path.join(args.dump_path, 'out.log'), logging.INFO)

    if args.local_rank != 0:
        def log_pass(*args): pass
        logging.info = log_pass

    # Snapshot of the settings before anything below rescales them.
    run_args = run_defining_args(args)
    status_path = os.path.join(args.dump_path, "status.json")
    wandb = init_wandb(args)

    # build data
    train_dataset = MultiVisionDataset(
        args.data_path,
        args,
        dataset_type=args.dataset,
        download=True,
        return_index=False,
    )
    ds = get_clf(args)

    # The sampler's shuffling seed follows --seed (it was fixed to 0 before,
    # which gave every seed the same data order).
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, seed=(args.seed if args.seed is not None else 0))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"the training set has {len(train_dataset)} images, fewer than one batch of "
            f"{args.batch_size}; lower --batch_size.")

    logging.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = get_method(args)

    # Synchronised batch norm is only needed (and only supported) when
    # several GPUs share a batch. With one process it computes exactly what
    # ordinary batch norm does, so it is skipped -- which also lets the code
    # run on CPU for smoke tests. Parameter names are identical either way.
    use_cuda = device.type == "cuda"
    if use_cuda and args.world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to the device
    if use_cuda:
        torch.cuda.set_device(device)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if use_cuda else None)

    logging.info(model)
    logging.info("Building model done.")

    # build optimizer
    args.lr = args.lr * args.batch_size * args.world_size / 256

    # ============ init schedulers ... ============
    # Both schedules are per-iteration arrays computed from --epochs and the
    # number of batches per epoch, so a resumed run picks up exactly where
    # the interrupted one stopped.
    args.lr_schedule = cosine_scheduler(
        args.lr,
        0.1 * args.lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    args.momentum_schedule = cosine_scheduler(
            args.momentum, 1,
            args.epochs, len(train_loader)
    )

    optimizer = torch.optim.SGD(model.parameters(), 0, momentum=0.9, weight_decay=args.wd)

    logging.info("Building SGD optimizer done.")

    # optionally resume from a checkpoint
    ckp_path = os.path.join(args.dump_path, "checkpoint.pth.tar")
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        ckp_path,
        run_variables=to_restore,
        expected_args=run_args,
        force=args.force_resume,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]
    if args.rank == 0:
        # written only after the resume check passed, so it always describes
        # the run that is actually in this folder
        write_json(os.path.join(args.dump_path, "args.json"),
                   {k: v for k, v in vars(args).items() if not k.endswith("_schedule")})
    if start_epoch > 0:
        logging.info(f"Resuming from epoch {start_epoch} of {args.epochs}.")
    if start_epoch >= args.epochs:
        logging.info("This run is already complete -- nothing to do.")

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logging.info("============ Starting epoch %i ... ============" % epoch)
        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        unwrap_model(model).current_epoch = epoch
        loss = train(train_loader, model, optimizer, epoch, args)

        # evaluate, log and save a checkpoint
        if args.rank == 0:
            if epoch == 0 or (epoch + 1) % args.eval_every == 0:
                acc_knn, acc = get_acc(model, ds, args)
                logging.info("acc: {}, acc_5: {}, acc_knn: {}".format(acc[1], acc[5], acc_knn))
                if wandb is not None:
                    wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)

            if wandb is not None:
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr'],
                           "momentum": unwrap_model(model).momentum,
                           "loss": loss, "ep": epoch})

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": run_args,
                "py_rng_state": random.getstate(),
                "np_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            save_checkpoint(save_dict, ckp_path)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                torch.save(save_dict, os.path.join(args.dump_path, f"checkpoint_ep{epoch + 1}.pth.tar"))

            write_json(status_path, {
                "epoch": epoch + 1, "epochs": args.epochs,
                "done": (epoch + 1) >= args.epochs, "last_loss": loss,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    if args.rank == 0 and start_epoch >= args.epochs:
        write_json(status_path, {"epoch": start_epoch, "epochs": args.epochs, "done": True,
                                 "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    if args.rank == 0 and os.path.isfile(ckp_path + ".prev"):
        os.remove(ckp_path + ".prev")  # the backup is only needed while training
    if wandb is not None:
        wandb.finish()
    logging.info("Pretraining finished. Checkpoint: %s" % ckp_path)


def train(loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    end = time.time()
    for it, samples in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update parameters
        iters = len(loader) * epoch + it  # global training iteration
        adjust_parameters(model, optimizer, args, iters)

        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss = model(samples)
        loss.backward()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), samples[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logging.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return losses.avg

def adjust_parameters(model, optimizer, args, iters):
    """Set the learning rate and the EMA momentum for this iteration."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr_schedule[iters]

    unwrap_model(model).momentum = args.momentum_schedule[iters]

if __name__ == "__main__":
    main()
