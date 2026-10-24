"""吹气决策：阈值 / 类别 / 粒度 / 截断 / 无分类器各分支，以及 EjectItem 映射（offset_x、expand、越界跳过）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oresorter.config import BlowParams
from oresorter.mcu.link import EjectItem
from oresorter.runtime import decision as D
from oresorter.runtime.decision import (
    Decision,
    decide,
    decide_batch,
    eject_items,
    eject_items_counted,
    size_norm,
)
from oresorter.vision.nozzle import NozzleMap
from oresorter.vision.segment import Particle

CLASSES = ["A", "B", "C"]
MAX_AREA = 10_000


def make_particle(
    *,
    x: int = 100,
    w: int = 20,
    area: int = 400,
    truncated: bool = False,
    trailing_line: int = 1234,
) -> Particle:
    crop = np.zeros((10, 10, 4), np.uint8)
    return Particle(
        bbox=(x, 50, w, 20),
        centroid_x=x + w / 2,
        centroid_y=60.0,
        area=area,
        trailing_y=69,
        crop_rgba=crop,
        truncated=truncated,
        trailing_line=trailing_line,
    )


def probs_for(cls: str, conf: float) -> np.ndarray:
    p = np.full(len(CLASSES), (1.0 - conf) / (len(CLASSES) - 1), np.float32)
    p[CLASSES.index(cls)] = conf
    return p


def blow_params(**over) -> BlowParams:
    base = {"classes": ["C"], "min_confidence": 128, "size_min": 0, "size_max": 255}
    base.update(over)
    return BlowParams(**base)


# --------------------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------------------


def test_blow_when_all_conditions_hold():
    d = decide(make_particle(), probs_for("C", 0.9), CLASSES, blow_params(), max_area_px=MAX_AREA)
    assert d == Decision(True, "C", pytest.approx(0.9), D.REASON_BLOW)
    assert d.to_dict() == {"blow": True, "cls": "C", "conf": 0.9, "reason": "blow"}


def test_class_not_selected():
    d = decide(make_particle(), probs_for("A", 0.95), CLASSES, blow_params(), max_area_px=MAX_AREA)
    assert d.blow is False and d.cls == "A" and d.reason == D.REASON_CLASS_NOT_SELECTED
    assert d.conf == pytest.approx(0.95)


def test_low_confidence_threshold_is_min_confidence_over_255():
    blow = blow_params(min_confidence=128)  # 阈值 0.50196
    low = decide(make_particle(), probs_for("C", 0.5), CLASSES, blow, max_area_px=MAX_AREA)
    assert low.blow is False and low.reason == D.REASON_LOW_CONFIDENCE
    ok = decide(make_particle(), probs_for("C", 0.51), CLASSES, blow, max_area_px=MAX_AREA)
    assert ok.blow is True
    # 0 → 任何置信度都过；255 → 几乎不可能
    assert decide(
        make_particle(),
        probs_for("C", 0.34),
        CLASSES,
        blow_params(min_confidence=0),
        max_area_px=MAX_AREA,
    ).blow
    assert (
        decide(
            make_particle(),
            probs_for("C", 0.999),
            CLASSES,
            blow_params(min_confidence=255),
            max_area_px=MAX_AREA,
        ).reason
        == D.REASON_LOW_CONFIDENCE
    )


def test_size_norm_is_sqrt_of_area_ratio():
    assert size_norm(0, MAX_AREA) == 0
    assert size_norm(MAX_AREA, MAX_AREA) == 255
    assert size_norm(MAX_AREA * 5, MAX_AREA) == 255, "超过上限按上限计（滑块满刻度）"
    assert size_norm(MAX_AREA // 4, MAX_AREA) == round(255 * 0.5)
    assert size_norm(100, MAX_AREA) == round(255 * math.sqrt(0.01))
    vals = [size_norm(a, MAX_AREA) for a in range(0, MAX_AREA + 1, 250)]
    assert vals == sorted(vals), "单调"
    with pytest.raises(ValueError):
        size_norm(10, 0)
    with pytest.raises(ValueError):
        size_norm(-1, 10)


def test_size_out_of_range_branches():
    small = make_particle(area=100)  # size_norm ≈ 26
    big = make_particle(area=MAX_AREA)  # 255
    blow = blow_params(size_min=50, size_max=200)
    d1 = decide(small, probs_for("C", 0.9), CLASSES, blow, max_area_px=MAX_AREA)
    assert d1.blow is False and d1.reason == D.REASON_SIZE_OUT_OF_RANGE
    d2 = decide(big, probs_for("C", 0.9), CLASSES, blow, max_area_px=MAX_AREA)
    assert d2.blow is False and d2.reason == D.REASON_SIZE_OUT_OF_RANGE
    mid = make_particle(area=MAX_AREA // 4)  # 128
    assert decide(mid, probs_for("C", 0.9), CLASSES, blow, max_area_px=MAX_AREA).blow is True
    # 边界含端点
    edge = blow_params(size_min=128, size_max=128)
    assert decide(mid, probs_for("C", 0.9), CLASSES, edge, max_area_px=MAX_AREA).blow is True


def test_truncated_particle_not_blown_unless_allowed():
    p = make_particle(truncated=True)
    d = decide(p, probs_for("C", 0.99), CLASSES, blow_params(), max_area_px=MAX_AREA)
    assert d.blow is False and d.reason == D.REASON_TRUNCATED and d.cls == "C"
    d2 = decide(
        p, probs_for("C", 0.99), CLASSES, blow_params(), max_area_px=MAX_AREA, blow_truncated=True
    )
    assert d2.blow is True and d2.reason == D.REASON_BLOW


def test_reason_precedence_truncated_then_class_then_conf_then_size():
    # 截断 + 类别不选 + 低置信度 + 粒度越界：reason 报 truncated
    p = make_particle(truncated=True, area=1)
    blow = blow_params(size_min=200)
    assert (
        decide(p, probs_for("A", 0.4), CLASSES, blow, max_area_px=MAX_AREA).reason
        == D.REASON_TRUNCATED
    )
    p2 = make_particle(area=1)
    assert (
        decide(p2, probs_for("A", 0.4), CLASSES, blow, max_area_px=MAX_AREA).reason
        == D.REASON_CLASS_NOT_SELECTED
    )
    assert (
        decide(p2, probs_for("C", 0.4), CLASSES, blow, max_area_px=MAX_AREA).reason
        == D.REASON_LOW_CONFIDENCE
    )
    assert (
        decide(p2, probs_for("C", 0.9), CLASSES, blow, max_area_px=MAX_AREA).reason
        == D.REASON_SIZE_OUT_OF_RANGE
    )


def test_no_classifier_decision():
    d = Decision.no_classifier()
    assert d.blow is False and d.cls == "" and d.conf == 0.0 and d.reason == D.REASON_NO_CLASSIFIER
    assert D.REASON_NO_CLASSIFIER in D.REASONS and len(set(D.REASONS)) == 6


def test_decide_validates_shapes():
    p = make_particle()
    with pytest.raises(ValueError):
        decide(p, np.array([0.5, 0.5], np.float32), CLASSES, blow_params(), max_area_px=MAX_AREA)
    with pytest.raises(ValueError):
        decide(p, np.array([1.0], np.float32), [], blow_params(), max_area_px=MAX_AREA)
    with pytest.raises(ValueError):
        decide(p, probs_for("C", 0.9), CLASSES, blow_params(), max_area_px=0)


def test_decide_batch():
    parts = [make_particle(), make_particle(area=100), make_particle(truncated=True)]
    probs = np.stack([probs_for("C", 0.9), probs_for("C", 0.9), probs_for("C", 0.9)])
    blow = blow_params(size_min=50)
    out = decide_batch(parts, probs, CLASSES, blow, max_area_px=MAX_AREA)
    assert [d.reason for d in out] == [
        D.REASON_BLOW,
        D.REASON_SIZE_OUT_OF_RANGE,
        D.REASON_TRUNCATED,
    ]
    assert decide_batch([], np.zeros((0, 3)), CLASSES, blow, max_area_px=MAX_AREA) == []
    with pytest.raises(ValueError):
        decide_batch(parts, probs[:2], CLASSES, blow, max_area_px=MAX_AREA)
    with pytest.raises(ValueError):
        decide_batch(parts, probs[:, :2], CLASSES, blow, max_area_px=MAX_AREA)
    with pytest.raises(ValueError):
        decide_batch(parts, probs.reshape(-1), CLASSES, blow, max_area_px=MAX_AREA)


# --------------------------------------------------------------------------------------
# eject_items
# --------------------------------------------------------------------------------------


def _blown(n: int) -> list[Decision]:
    return [Decision(True, "C", 0.9, D.REASON_BLOW) for _ in range(n)]


def test_eject_items_offset_and_expand():
    nm = NozzleMap(0, 4096, 64)  # 每喷头 64 px
    p = make_particle(x=100, w=20, trailing_line=777)  # x_range (100, 119) → 喷头 2
    items = eject_items([p], _blown(1), nm, offset_x=0)
    assert items == [EjectItem(line=777, nozzle_first=2, nozzle_last=2, flags=0)]
    # offset_x 是传感器坐标偏移：+64 → 喷头 3
    assert eject_items([p], _blown(1), nm, offset_x=64)[0].nozzle_first == 3
    # 跨喷头：x 120~199 → 2~4（含 3 的边界 128、192）
    wide = make_particle(x=120, w=80, trailing_line=5)
    it = eject_items([wide], _blown(1), nm, offset_x=0)[0]
    assert (it.nozzle_first, it.nozzle_last) == (2, 4)
    # expand=1 向两侧各扩一个
    it2 = eject_items([wide], _blown(1), nm, offset_x=0, expand=1)[0]
    assert (it2.nozzle_first, it2.nozzle_last) == (1, 5)
    # 两侧裁到 1..n
    left = make_particle(x=0, w=10, trailing_line=1)
    it3 = eject_items([left], _blown(1), nm, offset_x=0, expand=3)[0]
    assert (it3.nozzle_first, it3.nozzle_last) == (1, 4)


def test_eject_items_global_numbering_for_second_camera():
    nm = NozzleMap(0, 4096, 64, camera_index=1)
    p = make_particle(x=0, w=10, trailing_line=9)
    it = eject_items([p], _blown(1), nm, offset_x=0)[0]
    assert (it.nozzle_first, it.nozzle_last) == (65, 65)


def test_eject_items_skips_out_of_range_unknown_line_and_not_blown():
    nm = NozzleMap(512, 1024, 8)  # 有效区 [512, 1024)
    outside = make_particle(x=100, w=20, trailing_line=1)
    inside = make_particle(x=600, w=20, trailing_line=2)
    unknown_line = make_particle(x=600, w=20, trailing_line=-1)
    not_blown = make_particle(x=700, w=20, trailing_line=3)
    decisions = [*_blown(3), Decision(False, "A", 0.9, D.REASON_CLASS_NOT_SELECTED)]
    items, skipped = eject_items_counted(
        [outside, inside, unknown_line, not_blown], decisions, nm, offset_x=0
    )
    assert [it.line for it in items] == [2]
    assert skipped == 2, "越界与无行号各算一次跳过；未吹的不算"
    # offset_x 把越界颗粒移进有效区
    items2, skipped2 = eject_items_counted([outside], _blown(1), nm, offset_x=500)
    assert len(items2) == 1 and skipped2 == 0
    with pytest.raises(ValueError):
        eject_items([inside], [], nm, offset_x=0)
    # 部分相交只取相交部分
    straddle = make_particle(x=500, w=100, trailing_line=4)  # 500~599 ∩ [512,1024) → 喷头 1~2
    it = eject_items([straddle], _blown(1), nm, offset_x=0)[0]
    assert (it.nozzle_first, it.nozzle_last) == (1, 2)
    assert it.as_tuple() == (4, 1, 2, 0)
