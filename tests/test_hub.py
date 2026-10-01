import hashlib

import pytest

from brainpfn import hub


def test_unknown_variant():
    with pytest.raises(ValueError, match="Unknown BrainPFN variant"):
        hub.resolve_variant("12M")


def test_missing_url(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINPFN_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(hub.VARIANTS, "test", {"url": None, "sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="no download URL"):
        hub.resolve_variant("test")


def test_download_and_hash_check(tmp_path, monkeypatch):
    src = tmp_path / "weights.pt"
    src.write_bytes(b"not really weights")
    cache = tmp_path / "cache"
    monkeypatch.setenv("BRAINPFN_CACHE_DIR", str(cache))

    monkeypatch.setitem(hub.VARIANTS, "bad", {"url": src.as_uri(), "sha256": "0" * 64})
    with pytest.raises(RuntimeError):
        hub.resolve_variant("bad")
    assert not (cache / "brainpfn-bad.pt").exists()

    good = hashlib.sha256(src.read_bytes()).hexdigest()
    monkeypatch.setitem(hub.VARIANTS, "good", {"url": src.as_uri(), "sha256": good})
    assert hub.resolve_variant("good").read_bytes() == src.read_bytes()


def test_local_path(tmp_path):
    p = tmp_path / "w.pt"
    p.write_bytes(b"x")
    assert hub.resolve_variant(str(p)) == p


def test_wrong_format(tmp_path):
    import torch
    p = tmp_path / "w.pt"
    torch.save({"format": "something-else"}, p)
    with pytest.raises(ValueError, match="not a BrainPFN release checkpoint"):
        hub.load_network(str(p), "cpu")
