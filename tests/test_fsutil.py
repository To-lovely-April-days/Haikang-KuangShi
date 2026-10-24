"""原子写：写入中断不损坏旧文件。"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from oresorter import fsutil


def test_atomic_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "a" / "b.json"
    fsutil.atomic_write_json(p, {"x": 1, "名": "中"})
    assert fsutil.read_json(p) == {"x": 1, "名": "中"}
    assert not list(p.parent.glob(".*.tmp"))


def test_atomic_write_keeps_old_on_failure(tmp_path: Path):
    p = tmp_path / "cfg.json"
    fsutil.atomic_write_text(p, "old")
    with mock.patch.object(os, "replace", side_effect=OSError("boom")), pytest.raises(OSError):
        fsutil.atomic_write_text(p, "new")
    assert p.read_text() == "old"
    assert not list(tmp_path.glob(".*.tmp")), "失败后临时文件应被清理"


def test_read_json_default(tmp_path: Path):
    assert fsutil.read_json(tmp_path / "nope.json", default={"d": 1}) == {"d": 1}


def test_atomic_replace_dir(tmp_path: Path):
    dst = tmp_path / "models" / "1"
    dst.mkdir(parents=True)
    (dst / "old.txt").write_text("old")
    src = tmp_path / "staging"
    src.mkdir()
    (src / "new.txt").write_text("new")
    fsutil.atomic_replace_dir(src, dst)
    assert (dst / "new.txt").exists()
    assert not (dst / "old.txt").exists()
    assert not src.exists()
    assert not (tmp_path / "models" / "1.old").exists()
