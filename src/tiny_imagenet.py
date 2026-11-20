# Tiny ImageNet (200 classes, 64x64 colour images; 500 training and 50
# validation images per class) for the transfer experiments.
#
# The archive (tiny-imagenet-200.zip, ~240 MB, http://cs231n.stanford.edu/)
# holds 120,000 small JPEG files. Reading those one by one from Google Drive
# is extremely slow, so this loader:
#   1. copies the .zip from Drive to the local disk of the machine (one
#      sequential read, about a minute) and reads the images out of that
#      local copy, in the order they are stored, without unpacking it;
#   2. resizes them to the requested resolution -- 32x32 by default, the
#      resolution the CIFAR-pretrained encoders were trained at;
#   3. saves the result as ONE .npz file next to the archive.
# Later runs load that single file in seconds.
#
# Every step reports what it is doing and how far it is (roughly every ten
# seconds), so a slow step can be told apart from a stuck one. The download
# has a time-out and resumes after a stalled connection.
#
# The 10,000 validation images (which have labels) are used as the test set;
# the labels of the official test split were never released.
#
# Prepare the cache once from the command line (optional -- the evaluation
# scripts do it on first use):
#   python -m src.tiny_imagenet --root /content/drive/MyDrive/Datasets

import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image

TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
FOLDER = "tiny-imagenet-200"
SPLITS = ("train", "val")
MIN_ZIP_BYTES = 200 * 1024 ** 2   # the real archive is ~248 MB; less means a broken download
REPORT_EVERY = 10.0               # seconds between progress lines
NET_TIMEOUT = 60                  # seconds without data before a download attempt is abandoned
NET_RETRIES = 8

log = logging.getLogger("tiny_imagenet")


def _make_visible():
    """Show this module's messages even where logging was never configured
    (a notebook cell): otherwise a long step looks like a hang."""
    if not log.isEnabledFor(logging.INFO):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


class _Progress:
    """Log 'done/total, rate, time left' at most every REPORT_EVERY seconds."""
    def __init__(self, what, total, unit="images", scale=1):
        self.what, self.total, self.unit, self.scale = what, total, unit, scale
        self.start = self.last = time.time()

    def update(self, done):
        now = time.time()
        if now - self.last < REPORT_EVERY and done < self.total:
            return
        self.last = now
        rate = done / max(now - self.start, 1e-6)
        left = (self.total - done) / rate if rate > 0 else float("inf")
        left = f"{left / 60:.1f} min left" if done < self.total else f"done in {(now - self.start) / 60:.1f} min"
        log.info(f"  {self.what}: {done / self.scale:,.0f}/{self.total / self.scale:,.0f} {self.unit} "
                 f"({100 * done / max(self.total, 1):.0f}%), {left}")


def _cache_path(root, split, img_size):
    return os.path.join(root, f"tiny_imagenet_{split}_{img_size}px.npz")


def _scratch_dir():
    """Folder on the local disk of the machine (never on Drive)."""
    d = os.path.join(tempfile.gettempdir(), "resa_tiny_imagenet")
    os.makedirs(d, exist_ok=True)
    return d


def _copy(src, dst, what):
    """Copy a file in chunks, with progress; atomic at the destination."""
    size = os.path.getsize(src)
    prog = _Progress(what, size, unit="MB", scale=1024 ** 2)
    tmp = dst + ".part"
    done = 0
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            chunk = fi.read(8 * 1024 ** 2)
            if not chunk:
                break
            fo.write(chunk)
            done += len(chunk)
            prog.update(done)
    os.replace(tmp, dst)


def _check_zip(path):
    """Raise a readable error if the archive is incomplete or not Tiny ImageNet."""
    size = os.path.getsize(path)
    try:
        with zipfile.ZipFile(path) as zf:
            ok = f"{FOLDER}/wnids.txt" in zf.namelist()
    except zipfile.BadZipFile:
        ok = False
    if not ok:
        raise RuntimeError(
            f"'{path}' ({size / 1024 ** 2:.0f} MB) is not a complete tiny-imagenet-200.zip "
            f"(the real file is about 248 MB) -- probably an interrupted download or upload. "
            f"Delete it and run again, or download {TINY_URL} on your computer and upload it.")


def _download(dst, url=None, what="Tiny ImageNet (~248 MB)", min_bytes=None):
    """Download url to dst (local disk). Time-out, retries, resume."""
    url = url or TINY_URL
    min_bytes = MIN_ZIP_BYTES if min_bytes is None else min_bytes
    name = os.path.basename(dst)
    part = dst + ".part"
    log.info(f"downloading {what} from {url}")
    prog = None
    for attempt in range(1, NET_RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=NET_TIMEOUT) as r:
                if have and r.status != 206:      # server ignored the range: start again
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                prog = prog or _Progress("download", total or 1, unit="MB", scale=1024 ** 2)
                prog.total = total or prog.total
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1024 ** 2)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        prog.update(min(have, prog.total))
            if total and have < total:
                raise OSError(f"connection closed after {have} of {total} bytes")
            os.replace(part, dst)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have >= min_bytes:       # nothing left to fetch
                os.replace(part, dst)
                return
            err = e
        except (OSError, urllib.error.URLError) as e:     # includes time-outs
            err = e
        log.warning(f"  download attempt {attempt}/{NET_RETRIES} stopped ({err}); "
                    + ("retrying from where it stopped..." if attempt < NET_RETRIES else "giving up."))
        time.sleep(min(5 * attempt, 30))
    raise RuntimeError(
        f"{name} could not be downloaded from {url} (the server is often slow or "
        f"unreachable). Download it on your computer, upload it to the "
        f"datasets folder on Drive, and run this again.")


def _save_cache(cache, data, targets, classes):
    """Write the array cache: on the local disk first, then one sequential
    copy to its place (which may be on Drive)."""
    try:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        local = os.path.join(_scratch_dir(), os.path.basename(cache))
        np.savez(local, data=data, targets=targets, classes=np.array(classes))
        log.info(f"saving {cache} ({os.path.getsize(local) / 1024 ** 2:.0f} MB) ...")
        _copy(local, cache, "save cache")
        os.remove(local)
        log.info(f"saved {cache}")
    except OSError as e:  # read-only location: carry on without the cache
        log.warning(f"could not write the cache file {cache}: {e}")


class _ZipSource:
    """Read archive members by their path inside tiny-imagenet-200/."""
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        self.offset = {i.filename: i.header_offset for i in self.zf.infolist()}

    def read(self, rel):
        return self.zf.read(f"{FOLDER}/{rel}")

    def list(self, prefix):
        full = f"{FOLDER}/{prefix}"
        return sorted(n[len(FOLDER) + 1:] for n in self.offset
                      if n.startswith(full) and not n.endswith("/"))

    def position(self, rel):
        """Where the member sits in the archive (to read it front to back)."""
        return self.offset.get(f"{FOLDER}/{rel}", 0)


class _DirSource:
    """Same interface for an extracted tiny-imagenet-200/ folder."""
    def __init__(self, path):
        self.path = path

    def read(self, rel):
        with open(os.path.join(self.path, rel), "rb") as f:
            return f.read()

    def list(self, prefix):
        out = []
        base = os.path.join(self.path, prefix)
        for n, (dirpath, _, files) in enumerate(os.walk(base), 1):
            for fn in files:
                out.append(os.path.relpath(os.path.join(dirpath, fn), self.path).replace(os.sep, "/"))
            if n % 50 == 0:
                log.info(f"  listing the extracted folder: {n} folders, {len(out):,} files so far")
        return sorted(out)

    def position(self, rel):
        return 0


def _local_zip(zip_on_disk):
    """Return a copy of the archive on the local disk (made once per session)."""
    scratch = _scratch_dir()
    if os.path.abspath(zip_on_disk).startswith(os.path.abspath(tempfile.gettempdir()) + os.sep):
        return zip_on_disk
    local = os.path.join(scratch, FOLDER + ".zip")
    size = os.path.getsize(zip_on_disk)
    if os.path.isfile(local) and os.path.getsize(local) == size:
        return local
    if shutil.disk_usage(scratch).free < size + 1024 ** 3:
        log.warning("not enough local disk space for a copy; reading the archive in place (slower)")
        return zip_on_disk
    log.info(f"copying {zip_on_disk} to the local disk ({size / 1024 ** 2:.0f} MB) ...")
    _copy(zip_on_disk, local, "copy to local disk")
    return local


def _find_source(root, zip_path, download):
    # 1. the archive (preferred: one file, read from a local copy)
    candidates = [zip_path] if zip_path else []
    candidates.append(os.path.join(root, FOLDER + ".zip"))
    for z in candidates:
        if z and os.path.isfile(z):
            local = _local_zip(z)
            _check_zip(local)
            return _ZipSource(local)

    # 2. an extracted folder
    folder = os.path.join(root, FOLDER)
    if os.path.isfile(os.path.join(folder, "wnids.txt")):
        log.warning(
            f"no tiny-imagenet-200.zip in '{root}'; reading the extracted folder '{folder}' instead. "
            "On Google Drive this means opening 120,000 files one by one and can take HOURS. "
            "Stop this cell and put tiny-imagenet-200.zip in the datasets folder to do it in minutes.")
        return _DirSource(folder)

    # 3. download: to the local disk first, then a copy to root for next time
    if download:
        local = os.path.join(_scratch_dir(), FOLDER + ".zip")
        if not (os.path.isfile(local) and os.path.getsize(local) >= MIN_ZIP_BYTES):
            _download(local)
        _check_zip(local)
        try:
            os.makedirs(root, exist_ok=True)
            log.info(f"saving the archive to {root} for later sessions ...")
            _copy(local, os.path.join(root, FOLDER + ".zip"), "copy to the datasets folder")
        except OSError as e:
            log.warning(f"could not store the archive in {root}: {e}")
        return _ZipSource(local)

    raise FileNotFoundError(
        f"Tiny ImageNet not found. Put tiny-imagenet-200.zip (from {TINY_URL}) in '{root}', "
        "or pass its location with --tiny_zip.")


def _decode(raw, img_size):
    img = Image.open(io.BytesIO(raw)).convert("RGB")  # a few images are greyscale
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def _read_split(src, split, img_size):
    wnids = sorted(src.read("wnids.txt").decode().split())
    class_to_idx = {w: i for i, w in enumerate(wnids)}

    # human-readable class names (first synonym of each WordNet id)
    words = {}
    for line in src.read("words.txt").decode().splitlines():
        if "\t" in line:
            wnid, names = line.split("\t", 1)
            if wnid in class_to_idx:
                words[wnid] = names.split(",")[0].strip()
    classes = [words.get(w, w) for w in wnids]

    if split == "train":
        items = []
        for rel in src.list("train/"):
            if rel.lower().endswith((".jpeg", ".jpg", ".png")):
                items.append((rel, class_to_idx[rel.split("/")[1]]))
    else:
        items = []
        for line in src.read("val/val_annotations.txt").decode().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                items.append((f"val/images/{parts[0]}", class_to_idx[parts[1]]))
        items.sort()
    if not items:
        raise RuntimeError(f"no {split} images found in the Tiny ImageNet source")

    # The arrays keep the sorted order above (so the cache is always the same),
    # but the files are READ in the order they are stored in the archive.
    data = np.empty((len(items), img_size, img_size, 3), dtype=np.uint8)
    targets = np.empty(len(items), dtype=np.int64)
    read_order = sorted(range(len(items)), key=lambda i: src.position(items[i][0]))
    prog = _Progress(f"Tiny ImageNet {split}", len(items))
    for n, i in enumerate(read_order, 1):
        rel, label = items[i]
        data[i] = _decode(src.read(rel), img_size)
        targets[i] = label
        prog.update(n)
    return data, targets, classes


def load_tiny_imagenet(root, split="train", img_size=32, zip_path=None, download=True):
    """Return (images uint8 [N, S, S, 3], labels int64 [N], class names).

    root: folder that holds the cache file, the .zip, or the extracted
        tiny-imagenet-200/ folder (checked in that order).
    split: 'train' (100,000 images) or 'val' (10,000 images, used as test).
    img_size: side length the 64x64 images are resized to (bicubic).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    cache = _cache_path(root, split, img_size)
    if os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            return z["data"], z["targets"], [str(c) for c in z["classes"]]

    _make_visible()
    log.info(f"building the Tiny ImageNet {split} cache at {img_size}px (one-off, a few minutes)...")
    src = _find_source(root, zip_path, download)
    data, targets, classes = _read_split(src, split, img_size)
    _save_cache(cache, data, targets, classes)
    return data, targets, classes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the Tiny ImageNet array cache")
    p.add_argument("--root", required=True, help="folder with tiny-imagenet-200.zip (the cache is written here)")
    p.add_argument("--tiny_zip", default=None, help="path of the .zip if it is not in --root")
    p.add_argument("--img_size", type=int, default=32)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for s in SPLITS:
        d, t, c = load_tiny_imagenet(a.root, s, a.img_size, a.tiny_zip)
        print(f"{s}: {d.shape} images, {len(c)} classes, {np.bincount(t).min()}-{np.bincount(t).max()} per class")
