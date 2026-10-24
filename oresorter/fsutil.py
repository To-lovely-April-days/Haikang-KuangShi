"""文件系统工具：原子写、目录保证、JSON 读写。

所有落盘（配置、元数据、样本、模型产物）都必须经过这里的原子写，
保证写入中断时旧文件完好，不出现半成品。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | Path, data: bytes, *, fsync: bool = True) -> None:
    """原子写入字节：先写同目录临时文件，再 ``os.replace``。

    Args:
        path: 目标路径（父目录不存在时自动创建）。
        data: 要写入的内容。
        fsync: 是否在替换前 fsync 临时文件（默认开，掉电安全）。
    """
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            pass
        raise


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    """原子写入文本。"""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """原子写入 JSON（UTF-8，保留中文）。"""
    atomic_write_text(
        path, json.dumps(obj, ensure_ascii=False, indent=indent, default=_json_default) + "\n"
    )


def read_json(path: str | Path, default: Any = None) -> Any:
    """读取 JSON；文件不存在时返回 ``default``。损坏的 JSON 会抛出 ``json.JSONDecodeError``。"""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_replace_dir(src_tmp: str | Path, dst: str | Path) -> None:
    """把临时目录原子地换到目标位置（目标存在则先改名为 ``.old`` 再删除）。

    用于“备份至”“部署切换”等整目录替换：先在同一文件系统准备好 ``src_tmp``，
    再调用本函数，任何时刻 ``dst`` 要么是旧版本要么是新版本。
    """
    src = Path(src_tmp)
    target = Path(dst)
    ensure_dir(target.parent)
    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        os.replace(target, old)
    try:
        os.replace(src, target)
    except BaseException:
        if old.exists() and not target.exists():
            os.replace(old, target)
        raise
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def disk_free_mb(path: str | Path) -> float:
    """返回路径所在文件系统的剩余空间（MB）。路径不存在时向上找最近的存在的父目录。"""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    usage = shutil.disk_usage(str(p))
    return usage.free / (1024 * 1024)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"无法序列化为 JSON: {type(obj).__name__}")
