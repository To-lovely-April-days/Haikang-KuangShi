"""分类指标（纯 numpy，不依赖 sklearn）：混淆矩阵、每类 P/R/F1/support、准确率、macro-F1 与控制台报告。

``report()`` 的返回结构是 ``train/metrics.json`` 的共享契约（docs/01 §6.2、docs/03 §7），
全部为 Python 原生类型，可直接 ``json.dumps``。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> np.ndarray:
    """计算 ``n × n`` 混淆矩阵（行 = 真实类别，列 = 预测类别），``int64``。

    Args:
        y_true: 真实标签索引序列。
        y_pred: 预测标签索引序列，长度须与 ``y_true`` 一致。
        n: 类别数（≥ 1）。

    Raises:
        ValueError: 长度不一致、``n`` 非法或标签越界。
    """
    if n < 1:
        raise ValueError(f"类别数必须 ≥ 1，得到 {n}")
    t = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if t.shape[0] != p.shape[0]:
        raise ValueError(f"y_true 与 y_pred 长度不一致: {t.shape[0]} vs {p.shape[0]}")
    cm = np.zeros((n, n), dtype=np.int64)
    if t.shape[0] == 0:
        return cm
    if t.min() < 0 or t.max() >= n or p.min() < 0 or p.max() >= n:
        raise ValueError(f"标签越界，必须在 0~{n - 1} 之间")
    np.add.at(cm, (t, p), 1)
    return cm


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """逐元素除法，分母为 0 时结果取 0。"""
    out = np.zeros_like(num, dtype=np.float64)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def report(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]) -> dict[str, Any]:
    """生成完整指标报告。

    Returns:
        ``{"labels": [...], "matrix": [[...]], "accuracy": float, "macro_f1": float,
        "per_class": {cls: {"precision", "recall", "f1", "support"}}}``；除零按 0 处理，
        macro-F1 为全部类别 F1 的算术平均（含 support 为 0 的类别）。
    """
    labels = [str(c) for c in classes]
    if not labels:
        raise ValueError("classes 不能为空")
    if len(set(labels)) != len(labels):
        raise ValueError(f"classes 含重复类别: {labels}")
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, n)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    total = float(cm.sum())
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    per_class: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(labels):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "labels": labels,
        "matrix": [[int(v) for v in row] for row in cm.tolist()],
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()) if n > 0 else 0.0,
        "per_class": per_class,
    }


def summary(rep: dict[str, Any]) -> dict[str, float]:
    """从报告中抽取 meta.json 的 ``metrics_summary`` 段。"""
    return {
        "val_accuracy": float(rep.get("accuracy", 0.0)),
        "macro_f1": float(rep.get("macro_f1", 0.0)),
    }


# --------------------------------------------------------------------------------------
# 控制台格式化
# --------------------------------------------------------------------------------------


def _display_width(s: str) -> int:
    """按终端显示宽度计算（全角/宽字符算 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int, *, right: bool = False) -> str:
    fill = max(0, width - _display_width(s))
    return (" " * fill + s) if right else (s + " " * fill)


def _table(rows: list[list[str]], right_cols: set[int]) -> list[str]:
    """按列对齐一个字符串表格：``right_cols`` 中的列右对齐，其余左对齐。"""
    ncol = max(len(r) for r in rows)
    widths = [0] * ncol
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], _display_width(cell))
    lines = []
    for r in rows:
        cells = [_pad(c, widths[j], right=j in right_cols) for j, c in enumerate(r)]
        lines.append("  ".join(cells).rstrip())
    return lines


def format_report(rep: dict[str, Any]) -> str:
    """把 ``report()`` 的结果渲染成控制台可读文本：混淆矩阵（行=真实、列=预测）+ 每类指标表。"""
    labels: list[str] = [str(x) for x in rep.get("labels", [])]
    matrix: list[list[int]] = rep.get("matrix", [])
    per_class: dict[str, dict[str, Any]] = rep.get("per_class", {})
    out: list[str] = ["混淆矩阵（行=真实，列=预测）"]

    head = ["真实\\预测", *labels, "合计"]
    rows = [head]
    for i, name in enumerate(labels):
        row = matrix[i] if i < len(matrix) else [0] * len(labels)
        rows.append([name, *[str(int(v)) for v in row], str(int(sum(row)))])
    out.extend(_table(rows, right_cols=set(range(1, len(labels) + 2))))

    out.append("")
    rows = [["类别", "精确率", "召回率", "F1", "样本数"]]
    for name in labels:
        m = per_class.get(name, {})
        rows.append(
            [
                name,
                f"{float(m.get('precision', 0.0)):.3f}",
                f"{float(m.get('recall', 0.0)):.3f}",
                f"{float(m.get('f1', 0.0)):.3f}",
                str(int(m.get("support", 0))),
            ]
        )
    out.extend(_table(rows, right_cols={1, 2, 3, 4}))

    total = int(sum(int(m.get("support", 0)) for m in per_class.values()))
    out.append("")
    out.append(
        f"准确率 {float(rep.get('accuracy', 0.0)):.4f}  "
        f"宏平均F1 {float(rep.get('macro_f1', 0.0)):.4f}  样本总数 {total}"
    )
    return "\n".join(out)
