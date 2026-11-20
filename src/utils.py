# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Modified for the long-tailed ReSA study (see CHANGES.md):
#   - checkpoints are written atomically and keep one backup, so a Colab
#     disconnect in the middle of a save cannot destroy a run;
#   - resuming checks that the run settings match the checkpoint;
#   - loading a pretrained encoder fails loudly instead of silently
#     evaluating a randomly initialised network;
#   - everything works on CPU (for smoke tests) as well as on GPU, and with
#     plain `python script.py` as well as `torchrun`.

import json
import logging
import os
import random
import socket

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(log_file, level, include_host=False):
    """Log to the console and (appending) to `log_file`."""
    if include_host:
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def random_seed(seed):
    """Seed Python, NumPy and PyTorch. `None` leaves them unseeded."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def json_safe(obj):
    """Recursively make `obj` serialisable as strict JSON (NaN/Inf -> null,
    NumPy scalars/arrays -> Python types, anything else -> str)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path, obj):
    """Write strict JSON atomically (temp file + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Pretrained encoder
# --------------------------------------------------------------------------
def load_pretrained_encoder(encoder, pretrained_path):
    """Load the ReSA *momentum* encoder from a pretraining checkpoint.

    The momentum (EMA) encoder is the network ReSA uses for every downstream
    task. Raises if the file is missing or if any encoder weight is absent
    from it -- the previous behaviour (log a line and carry on) meant a
    wrong path silently evaluated a randomly initialised encoder.
    """
    if not pretrained_path:
        raise ValueError("no --pretrained checkpoint given")
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: '{pretrained_path}'")

    logging.info(f"loading checkpoint '{pretrained_path}'")
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Keep only the momentum encoder's weights and strip the prefix
    # ('module.' is added by DistributedDataParallel).
    encoder_state = {}
    for prefix in ("module.momentum_encoder.", "momentum_encoder."):
        for k, v in state_dict.items():
            if k.startswith(prefix):
                encoder_state[k[len(prefix):]] = v
        if encoder_state:
            break
    if not encoder_state:
        raise RuntimeError(
            f"'{pretrained_path}' contains no 'momentum_encoder' weights -- is it a ReSA "
            "pretraining checkpoint (the checkpoint.pth.tar written by main.py)?")

    msg = encoder.load_state_dict(encoder_state, strict=False)
    if msg.missing_keys:
        raise RuntimeError(
            f"checkpoint '{pretrained_path}' is missing encoder weights {msg.missing_keys[:5]}... "
            "(wrong --arch, or a checkpoint from a different dataset stem?)")
    if msg.unexpected_keys:
        logging.info(f"ignored unexpected keys: {msg.unexpected_keys[:5]}...")

    epoch = checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
    logging.info(f"loaded pre-trained momentum encoder (epoch {epoch})")
    del checkpoint, state_dict
    return epoch


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Per-iteration cosine schedule with optional linear warm-up."""
    if niter_per_ep < 1:
        raise ValueError("the training set is smaller than one batch; lower --batch_size")
    warmup_epochs = min(warmup_epochs, epochs)  # very short test runs
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(state, path):
    """Write a checkpoint so that an interruption can never leave the run
    without a loadable file.

    1. write to `<path>.tmp`;
    2. move the current checkpoint (if any) to `<path>.prev`;
    3. move the temp file into place.
    If the process dies during step 1 the old checkpoint is untouched; if it
    dies between 2 and 3, `restart_from_checkpoint` falls back to `.prev`.
    """
    tmp, prev = path + ".tmp", path + ".prev"
    torch.save(state, tmp)
    if os.path.isfile(path):
        os.replace(path, prev)
    os.replace(tmp, path)


def _load_checkpoint_file(ckp_path):
    """Load `ckp_path`, falling back to its `.prev` backup if it is missing
    or unreadable (e.g. truncated by a disconnect). Returns (checkpoint, path)
    or (None, None)."""
    for candidate in (ckp_path, ckp_path + ".prev"):
        if not os.path.isfile(candidate):
            continue
        try:
            return torch.load(candidate, map_location="cpu", weights_only=False), candidate
        except Exception as e:  # corrupt / truncated file
            logging.warning(f"could not read checkpoint '{candidate}' ({type(e).__name__}: {e})")
    return None, None


def _args_mismatch(saved, current):
    """Settings whose saved value differs from the current one."""
    diffs = []
    for k, v in current.items():
        if k in saved and json_safe(saved[k]) != json_safe(v):
            diffs.append(f"{k}: checkpoint={saved[k]!r} vs now={v!r}")
    return diffs


def restart_from_checkpoint(ckp_paths, run_variables=None, expected_args=None,
                            force=False, **kwargs):
    """Resume from a checkpoint if one exists.

    ckp_paths: a path, or a list of paths (the first that exists is used).
    run_variables: dict of variables to restore from the checkpoint, e.g.
        {"epoch": 0}.
    expected_args: dict of the settings that define this run. If the
        checkpoint stores its own settings (key "args") and any of them
        differs, resuming is refused: continuing e.g. an IF=50 run with
        IF=100 data would silently produce a mislabelled result. `force`
        turns that error into a warning.
    kwargs: objects to restore with load_state_dict, keyed by their name in
        the checkpoint, e.g. state_dict=model, optimizer=optimizer.
    """
    if isinstance(ckp_paths, list):
        ckp_path = ckp_paths[0]
        for p in ckp_paths:
            if os.path.isfile(p):
                ckp_path = p
                break
    else:
        ckp_path = ckp_paths

    checkpoint, used_path = _load_checkpoint_file(ckp_path)
    if checkpoint is None:
        return
    logging.info(f"Found checkpoint at {used_path}")

    # --- refuse to resume a run made with different settings ---------------
    if expected_args is not None:
        saved_args = checkpoint.get("args")
        if saved_args is None:
            logging.warning("checkpoint stores no run settings (written by an older version of "
                            "this code) -- cannot verify that it belongs to this run.")
        else:
            diffs = _args_mismatch(saved_args, expected_args)
            if diffs:
                msg = (f"the checkpoint in '{os.path.dirname(used_path)}' was created with "
                       "different settings:\n  " + "\n  ".join(diffs) +
                       "\nUse a different --dump_path for the new settings (or pass "
                       "--force_resume / --overwrite if this is intended).")
                if force:
                    logging.warning(msg)
                else:
                    raise RuntimeError(msg)

    # --- restore RNG state so a resumed run continues the same random stream
    if "py_rng_state" in checkpoint:
        random.setstate(checkpoint["py_rng_state"])
    if "np_rng_state" in checkpoint:
        np.random.set_state(checkpoint["np_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state"]])
        except Exception as e:  # e.g. resumed on a machine with fewer GPUs
            logging.warning(f"could not restore CUDA RNG state: {e}")

    # --- restore model / optimizer / scheduler ----------------------------
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                logging.info(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logging.info(f"=> loaded {key} from checkpoint '{used_path}'")
        else:
            logging.warning(f"=> failed to load {key} from checkpoint '{used_path}'")

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


# --------------------------------------------------------------------------
# Distributed set-up
# --------------------------------------------------------------------------
def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_single_process_env():
    """Let the scripts run with plain `python main.py` (no torchrun): if the
    launcher's environment variables are absent, describe a one-process job."""
    if 'RANK' in os.environ or 'SLURM_PROCID' in os.environ:
        return
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(_free_port()))
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')


def init_distributed_device(args):
    """Initialise torch.distributed and pick the device.

    Works under torchrun, SLURM, or plain `python` (single process). The
    backend defaults to NCCL on GPU and Gloo on CPU.
    """
    if getattr(args, 'dist_backend', None) in (None, '', 'auto'):
        args.dist_backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    if 'SLURM_PROCID' in os.environ:
        # DDP via SLURM
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        # DDP via torchrun, or a single plain-python process
        _ensure_single_process_env()
        args.local_rank, _, _ = world_info_from_env()
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
    args.distributed = True

    if torch.cuda.is_available():
        if not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device, args


def get_device():
    """Device for the single-process evaluation scripts."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Meters
# --------------------------------------------------------------------------
class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
