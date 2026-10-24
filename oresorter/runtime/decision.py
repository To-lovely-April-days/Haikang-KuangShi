"""吹气决策（docs/01 §4.2、§9 blow 分组）：分类概率 → ``Decision``，再映射为下位机 ``EjectItem``。

规则（``decide``）：取 top-1 类别与置信度；**吹气** 当且仅当

1. 颗粒未被截断（``particle.truncated`` 为真时半颗矿石的颜色/粒度都不可信，默认不吹，
   可用 ``blow_truncated=True`` 放开）；
2. ``cls ∈ blow.classes``；
3. ``conf ≥ blow.min_confidence / 255``；
4. ``blow.size_min ≤ size_norm ≤ blow.size_max``，其中

   ``size_norm = round(255 · sqrt(min(area, max_area_px) / max_area_px))``

   用面积的平方根（≈ 等效直径）而不是面积本身做归一化，使 HMI 调整页 0~255 的"粒度"滑块与
   颗粒的线尺度近似线性对应（面积是尺度的平方，直接按面积映射时滑块前半段几乎全是小颗粒）。
   ``max_area_px`` 取采集设置的 ``capture.max_area_px``。**滑块的确切含义待客户确认**（docs/01 §11 #5）。

``reason`` 取值：``blow / class_not_selected / low_confidence / size_out_of_range / truncated / no_classifier``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from oresorter.log import get_logger
from oresorter.mcu.link import EjectItem

if TYPE_CHECKING:
    from oresorter.config import BlowParams
    from oresorter.vision.nozzle import NozzleMap
    from oresorter.vision.segment import Particle

log = get_logger(__name__)

REASON_BLOW = "blow"
REASON_CLASS_NOT_SELECTED = "class_not_selected"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_SIZE_OUT_OF_RANGE = "size_out_of_range"
REASON_TRUNCATED = "truncated"
REASON_NO_CLASSIFIER = "no_classifier"
REASONS: tuple[str, ...] = (
    REASON_BLOW,
    REASON_CLASS_NOT_SELECTED,
    REASON_LOW_CONFIDENCE,
    REASON_SIZE_OUT_OF_RANGE,
    REASON_TRUNCATED,
    REASON_NO_CLASSIFIER,
)

SIZE_SCALE = 255
"""粒度滑块满刻度。"""


@dataclass(slots=True)
class Decision:
    """一颗矿石的决策结果。``cls`` / ``conf`` 为 top-1 类别与置信度（无分类器时为 ``""`` / 0）。"""

    blow: bool
    cls: str
    conf: float
    reason: str

    @classmethod
    def no_classifier(cls) -> Decision:
        """未部署分类器时的占位决策：不吹。"""
        return cls(blow=False, cls="", conf=0.0, reason=REASON_NO_CLASSIFIER)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["conf"] = round(float(self.conf), 4)
        return d


def size_norm(area: int, max_area_px: int) -> int:
    """面积 → 0~255 粒度刻度：``round(255 · sqrt(min(area, max) / max))``。

    Raises:
        ValueError: ``max_area_px ≤ 0`` 或 ``area < 0``。
    """
    if max_area_px <= 0:
        raise ValueError(f"max_area_px 必须 > 0，得到 {max_area_px}")
    if area < 0:
        raise ValueError(f"area 不能为负，得到 {area}")
    ratio = min(int(area), int(max_area_px)) / float(max_area_px)
    return round(SIZE_SCALE * math.sqrt(ratio))


def confidence_threshold(blow: BlowParams) -> float:
    """``min_confidence``（0~255）→ 概率阈值（0~1）。"""
    return float(blow.min_confidence) / 255.0


def _top1(probs: np.ndarray, classes: Sequence[str]) -> tuple[int, float]:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if len(classes) == 0:
        raise ValueError("classes 不能为空")
    if p.shape[0] != len(classes):
        raise ValueError(f"probs 长度 {p.shape[0]} 与 classes 数量 {len(classes)} 不一致")
    idx = int(np.argmax(p))
    return idx, float(p[idx])


def decide(
    particle: Particle,
    probs: np.ndarray,
    classes: Sequence[str],
    blow: BlowParams,
    *,
    max_area_px: int,
    blow_truncated: bool = False,
) -> Decision:
    """单颗决策（规则见模块 docstring）。

    Args:
        particle: 分割得到的颗粒。
        probs: ``(C,)`` 概率。
        classes: 长度 C 的类别名（模型输出顺序）。
        blow: 吹气参数（类别集合、置信度阈值、粒度范围）。
        max_area_px: 粒度归一化的面积上限（``capture.max_area_px``）。
        blow_truncated: 截断颗粒是否允许吹（默认不吹）。

    Raises:
        ValueError: 形状不符或 ``max_area_px ≤ 0``。
    """
    idx, conf = _top1(probs, classes)
    cls = str(classes[idx])
    if particle.truncated and not blow_truncated:
        return Decision(False, cls, conf, REASON_TRUNCATED)
    if cls not in blow.classes:
        return Decision(False, cls, conf, REASON_CLASS_NOT_SELECTED)
    if conf < confidence_threshold(blow):
        return Decision(False, cls, conf, REASON_LOW_CONFIDENCE)
    s = size_norm(particle.area, max_area_px)
    if not (int(blow.size_min) <= s <= int(blow.size_max)):
        return Decision(False, cls, conf, REASON_SIZE_OUT_OF_RANGE)
    return Decision(True, cls, conf, REASON_BLOW)


def decide_batch(
    particles: Sequence[Particle],
    probs: np.ndarray,
    classes: Sequence[str],
    blow: BlowParams,
    *,
    max_area_px: int,
    blow_truncated: bool = False,
) -> list[Decision]:
    """批量决策：``probs`` 为 ``(N, C)``，``N == len(particles)``。空输入返回空列表。"""
    n = len(particles)
    p = np.asarray(probs, dtype=np.float32)
    if n == 0:
        return []
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError(f"probs 形状应为 ({n}, C)，得到 {p.shape}")
    if p.shape[1] != len(classes):
        raise ValueError(f"probs 列数 {p.shape[1]} 与 classes 数量 {len(classes)} 不一致")
    return [
        decide(
            particle,
            p[i],
            classes,
            blow,
            max_area_px=max_area_px,
            blow_truncated=blow_truncated,
        )
        for i, particle in enumerate(particles)
    ]


def eject_items_counted(
    particles: Sequence[Particle],
    decisions: Sequence[Decision],
    nozzle_map: NozzleMap,
    *,
    offset_x: int,
    expand: int = 0,
    flags: int = 0,
) -> tuple[list[EjectItem], int]:
    """``blow=True`` 的颗粒 → ``EjectItem`` 列表，并返回被跳过的数量。

    颗粒 ``x_range`` 是 **图像列号**，加上相机 ``offset_x`` 才是传感器坐标（喷头映射的坐标系）。
    跳过：``nozzle_map.global_span`` 为 ``None``（颗粒完全落在起始/结束像素之外）、
    ``trailing_line < 0``（拼接器未填写全局行号，无法给下位机定时）。

    Raises:
        ValueError: ``particles`` 与 ``decisions`` 长度不一致。
    """
    if len(particles) != len(decisions):
        raise ValueError(f"particles({len(particles)}) 与 decisions({len(decisions)}) 长度不一致")
    items: list[EjectItem] = []
    skipped = 0
    off = int(offset_x)
    for p, d in zip(particles, decisions, strict=True):
        if not d.blow:
            continue
        if p.trailing_line < 0:
            skipped += 1
            log.debug("颗粒无全局行号，跳过喷阀", extra={"bbox": list(p.bbox)})
            continue
        x0, x1 = p.x_range
        span = nozzle_map.global_span(x0 + off, x1 + off, expand)
        if span is None:
            skipped += 1
            log.debug(
                "颗粒不在喷头映射有效区内，跳过",
                extra={"x_range": [x0 + off, x1 + off], "trailing_line": p.trailing_line},
            )
            continue
        items.append(
            EjectItem(
                line=int(p.trailing_line),
                nozzle_first=int(span[0]),
                nozzle_last=int(span[1]),
                flags=int(flags),
            )
        )
    return items, skipped


def eject_items(
    particles: Sequence[Particle],
    decisions: Sequence[Decision],
    nozzle_map: NozzleMap,
    *,
    offset_x: int,
    expand: int = 0,
    flags: int = 0,
) -> list[EjectItem]:
    """同 :func:`eject_items_counted`，只返回 ``EjectItem`` 列表。"""
    items, _ = eject_items_counted(
        particles, decisions, nozzle_map, offset_x=offset_x, expand=expand, flags=flags
    )
    return items
