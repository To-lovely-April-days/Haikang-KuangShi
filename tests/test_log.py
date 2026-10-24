"""日志：结构化 extra、保留键兜底、源码静态扫描（禁止在 extra 中使用 LogRecord 保留键）。"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from oresorter.log import RESERVED_EXTRA_KEYS, KVFormatter, SafeLogger, get_logger

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = (ROOT / "oresorter", ROOT / "scripts")


def test_logger_class_is_safe():
    assert isinstance(get_logger("oresorter.test.safe"), SafeLogger)


def test_reserved_extra_key_does_not_raise(caplog):
    log = get_logger("oresorter.test.reserved")
    with caplog.at_level(logging.INFO, logger="oresorter.test.reserved"):
        log.info("任务结束", extra={"message": "x", "thread": "t", "job_id": "j-1"})
    rec = caplog.records[-1]
    assert rec.getMessage() == "任务结束"
    assert rec.x_message == "x" and rec.x_thread == "t" and rec.job_id == "j-1"  # type: ignore[attr-defined]


def test_kv_formatter_appends_extra():
    log = get_logger("oresorter.test.kv")
    rec = log.makeRecord(
        "oresorter.test.kv",
        logging.INFO,
        "f",
        1,
        "块处理完成",
        (),
        None,
        extra={"seq": 12, "ms": 31.2},
    )
    out = KVFormatter("%(message)s").format(rec)
    assert out.startswith("块处理完成") and "seq=12" in out and "ms=31.200" in out


def _extra_literal_keys(tree: ast.AST):
    """找出所有 ``log.xxx(..., extra={...})`` 调用里字典字面量的字符串键。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        yield node.lineno, k.value


@pytest.mark.parametrize("scan_dir", SCAN_DIRS, ids=lambda p: p.name)
def test_no_reserved_keys_in_extra_literals(scan_dir: Path):
    """静态扫描：源码中 ``extra={...}`` 不得使用 LogRecord 保留键（标准库会抛 KeyError 炸掉业务线程）。"""
    offenders = []
    for path in sorted(scan_dir.rglob("*.py")):
        if "hik_sdk" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, key in _extra_literal_keys(tree):
            if key in RESERVED_EXTRA_KEYS:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} extra 键 {key!r}")
    assert not offenders, "\n".join(offenders)
