"""
Pretrained BrainPFN variants: download, cache and load.

Weights are downloaded on first use to ``~/.cache/brainpfn`` (or
``$BRAINPFN_CACHE_DIR``) and checked against their SHA-256.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import torch

from sklearn.decomposition import PCA

from brainpfn.features import pca_from_state
from brainpfn.network import PFNTransformer

RELEASE_FORMAT = "brainpfn-release/v1"

DEFAULT_VARIANT = "4.7M"

# name -> download URL and SHA-256 of the released file.
VARIANTS = {
    "0.8M": {
        "url": "https://huggingface.co/cbarbano/brainpfn/resolve/d64d6aadb44c41ca9a84da86ee905fe7943fb9b4/brainpfn-0.8M.pt",
        "sha256": "d69fbf3bb37811f24e7680081c01dbda31cf7fe3db7a924fecdabb5cd63fc7cd",
    },
    "4.7M": {
        "url": "https://huggingface.co/cbarbano/brainpfn/resolve/d64d6aadb44c41ca9a84da86ee905fe7943fb9b4/brainpfn-4.7M.pt",
        "sha256": "e479d33cb7b87a0c3e0aecdce6530b171c6f014cd87c09d924e6e0a9706d1d94",
    },
}


def cache_dir() -> Path:
    return Path(os.environ.get("BRAINPFN_CACHE_DIR", Path.home() / ".cache" / "brainpfn"))


def _download(url: str, dst: Path, sha256: Optional[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    print(f"Downloading BrainPFN weights from {url} to {dst} ...")
    # hash_prefix makes torch verify the SHA-256 and delete the file on mismatch.
    torch.hub.download_url_to_file(url, str(tmp), hash_prefix=sha256, progress=True)
    os.replace(tmp, dst)


def resolve_variant(variant: str) -> Path:
    """Local path of a variant's weights, downloading them if needed.

    `variant` is a released variant name (see VARIANTS), an http(s) URL, or a
    local path to a .pt file.
    """
    variant = str(variant)
    if variant in VARIANTS:
        entry = VARIANTS[variant]
        dst = cache_dir() / f"brainpfn-{variant}.pt"
        if not dst.exists():
            if entry["url"] is None:
                raise RuntimeError(
                    f"BrainPFN variant {variant!r} has no download URL yet. Put the file at "
                    f"{dst}, or pass its local path as `variant`.")
            _download(entry["url"], dst, entry["sha256"])
        return dst
    if urlparse(variant).scheme in ("http", "https"):
        digest = hashlib.sha256(variant.encode()).hexdigest()[:16]
        name = Path(urlparse(variant).path).name or "brainpfn.pt"
        dst = cache_dir() / f"{digest}_{name}"
        if not dst.exists():
            _download(variant, dst, None)
        return dst
    path = Path(variant).expanduser()
    if not path.is_file():
        raise ValueError(f"Unknown BrainPFN variant {variant!r}: expected one of "
                         f"{sorted(VARIANTS)}, an http(s) URL, or a path to a .pt file.")
    return path


@functools.lru_cache(maxsize=8)
def _load(path: str, device: str) -> tuple[PFNTransformer, dict, PCA]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != RELEASE_FORMAT:
        fmt = ckpt.get("format") if isinstance(ckpt, dict) else type(ckpt).__name__
        raise ValueError(f"{path} is not a BrainPFN release checkpoint "
                         f"(format={fmt!r}, expected {RELEASE_FORMAT!r}).")
    net = PFNTransformer(**ckpt["net"])
    net.load_state_dict(ckpt["net_state"], strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, dict(ckpt["transform"]), pca_from_state(ckpt["pretrain_pca"])


def load_network(variant: str, device: str) -> tuple[PFNTransformer, dict, PCA]:
    """The network of `variant` on `device` (in eval mode), its transform settings, and
    the PCA basis of its pretraining data.

    Loaded networks are cached per (file, device), so refitting or cloning an
    estimator does not reload the weights. Both are shared: do not modify them.
    """
    return _load(str(resolve_variant(variant).resolve()), str(device))
