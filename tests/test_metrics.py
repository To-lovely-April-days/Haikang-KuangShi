"""指标：手算 3 类小例子、除零、报告格式、summary。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from oresorter.ml import metrics as M

CLASSES = ["A", "B", "C"]
# 手算用例：
#   真实 A×3: 预测 A,A,B      → 行 A = [2,1,0]
#   真实 B×2: 预测 B,B        → 行 B = [0,2,0]
#   真实 C×4: 预测 C,C,A,C    → 行 C = [1,0,3]
Y_TRUE = [0, 0, 0, 1, 1, 2, 2, 2, 2]
Y_PRED = [0, 0, 1, 1, 1, 2, 2, 0, 2]


def test_confusion_matrix_hand_computed():
    cm = M.confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.dtype == np.int64
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 3]]
    # 空输入 → 全零；越界 / 长度不一致 → 拒绝
    assert M.confusion_matrix([], [], 2).tolist() == [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        M.confusion_matrix([0, 3], [0, 0], 3)
    with pytest.raises(ValueError):
        M.confusion_matrix([0, 1], [0], 3)
    with pytest.raises(ValueError):
        M.confusion_matrix([0], [0], 0)


def test_report_hand_computed_values():
    rep = M.report(Y_TRUE, Y_PRED, CLASSES)
    assert rep["labels"] == CLASSES
    assert rep["matrix"] == [[2, 1, 0], [0, 2, 0], [1, 0, 3]]
    pc = rep["per_class"]
    # precision = 列和分母；recall = 行和分母
    assert pc["A"]["precision"] == pytest.approx(2 / 3)
    assert pc["A"]["recall"] == pytest.approx(2 / 3)
    assert pc["A"]["f1"] == pytest.approx(2 / 3)
    assert pc["B"]["precision"] == pytest.approx(2 / 3)
    assert pc["B"]["recall"] == pytest.approx(1.0)
    assert pc["B"]["f1"] == pytest.approx(0.8)
    assert pc["C"]["precision"] == pytest.approx(1.0)
    assert pc["C"]["recall"] == pytest.approx(0.75)
    assert pc["C"]["f1"] == pytest.approx(1.5 / 1.75)
    assert [pc[c]["support"] for c in CLASSES] == [3, 2, 4]
    assert rep["accuracy"] == pytest.approx(7 / 9)
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1.5 / 1.75) / 3)


def test_report_is_json_native():
    rep = M.report(np.array(Y_TRUE), np.array(Y_PRED), CLASSES)
    text = json.dumps(rep)  # numpy 标量会在这里抛 TypeError
    back = json.loads(text)
    assert back["per_class"]["C"]["support"] == 4
    assert isinstance(rep["accuracy"], float) and isinstance(rep["per_class"]["A"]["support"], int)
    assert all(isinstance(v, int) for row in rep["matrix"] for v in row)


def test_report_division_by_zero_is_zero():
    # 类 B 从未被预测（precision 分母 0），类 C 没有样本（recall 分母 0，support 0）
    rep = M.report([0, 0, 1], [0, 0, 0], CLASSES)
    pc = rep["per_class"]
    assert pc["A"] == {
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(1.0),
        "f1": pytest.approx(0.8),
        "support": 2,
    }
    assert pc["B"]["precision"] == 0.0 and pc["B"]["recall"] == 0.0 and pc["B"]["f1"] == 0.0
    assert pc["C"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    assert rep["accuracy"] == pytest.approx(2 / 3)
    assert rep["macro_f1"] == pytest.approx(0.8 / 3)
    # 完全没有样本
    empty = M.report([], [], CLASSES)
    assert empty["accuracy"] == 0.0 and empty["macro_f1"] == 0.0
    assert empty["matrix"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_report_rejects_bad_classes():
    with pytest.raises(ValueError):
        M.report([0], [0], [])
    with pytest.raises(ValueError):
        M.report([0], [0], ["A", "A"])


def test_format_report_contains_names_and_numbers():
    rep = M.report(Y_TRUE, Y_PRED, ["铜矿", "B", "C"])
    text = M.format_report(rep)
    lines = text.splitlines()
    for name in ("铜矿", "B", "C"):
        assert name in text
    assert "混淆矩阵" in lines[0] and "真实" in lines[0] and "预测" in lines[0]
    for header in ("类别", "精确率", "召回率", "F1", "样本数"):
        assert header in text
    # 混淆矩阵行：类名 + 3 个计数 + 合计
    row_a = next(ln for ln in lines if ln.startswith("铜矿"))
    assert row_a.split()[1:] == ["2", "1", "0", "3"]
    # 指标行
    metric_rows = [ln for ln in lines if ln.startswith("C ") or ln.startswith("C\t")]
    assert any(
        "1.000" in ln and "0.750" in ln and "0.857" in ln and ln.split()[-1] == "4"
        for ln in metric_rows
    )
    assert "准确率 0.7778" in text and "宏平均F1 0.7746" in text and "样本总数 9" in text


def test_format_report_columns_aligned():
    rep = M.report(Y_TRUE, Y_PRED, ["A", "BBBB", "C"])
    lines = M.format_report(rep).splitlines()
    matrix_rows = lines[1:5]  # 表头 + 3 行
    # 每行"合计"列的右边界对齐（右对齐数字列；中文按 2 列宽计）
    assert len({M._display_width(ln) for ln in matrix_rows}) == 1
    metric_rows = lines[6:10]
    assert len({M._display_width(ln) for ln in metric_rows}) == 1


def test_summary():
    rep = M.report(Y_TRUE, Y_PRED, CLASSES)
    s = M.summary(rep)
    assert set(s) == {"val_accuracy", "macro_f1"}
    assert s["val_accuracy"] == pytest.approx(rep["accuracy"])
    assert s["macro_f1"] == pytest.approx(rep["macro_f1"])
    assert M.summary({}) == {"val_accuracy": 0.0, "macro_f1": 0.0}
