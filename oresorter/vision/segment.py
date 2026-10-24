"""分割：背景差分 → 阈值 → 形态学 → 连通域 → 带 mask 裁切，输出 ``Particle`` 列表（docs/01 §5.3）。

**本文件的 ``Particle`` 是 vision / runtime / data / api 共用契约**，字段只增不删。
坐标系：bbox / 质心 / trailing_y 均在传入 ``segment()`` 的图像坐标系内（可能是 stitcher 拼好的处理图），
``trailing_line`` 为全局行号，由 ``stitch.ChunkStitcher.select`` 填写。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oresorter.config import CaptureSettings
    from oresorter.vision.background import ColumnBackground


@dataclass(slots=True)
class Particle:
    """一颗矿石。"""

    bbox: tuple[int, int, int, int]
    """``(x, y, w, h)``，不含 margin 外扩，图像坐标。"""
    centroid_x: float
    centroid_y: float
    area: int
    """mask 像素数。"""
    trailing_y: int
    """块内尾沿行 = ``y + h - 1``（最后被扫到的一行）。"""
    crop_rgba: np.ndarray
    """``(h + 2·margin, w + 2·margin, 4) uint8``，alpha 即 mask（255 前景）。"""
    touches_top: bool = False
    touches_bottom: bool = False
    truncated: bool = False
    """对象长度超过 overlap 被截断（stitch 标记）。"""
    trailing_line: int = -1
    """全局尾沿行号（stitch.select 填写；-1 = 未知）。"""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def x_range(self) -> tuple[int, int]:
        """``(x0, x1)`` 闭区间，用于喷头 span。"""
        return self.bbox[0], self.bbox[0] + self.bbox[2] - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid_x, 1), round(self.centroid_y, 1)],
            "area_px": int(self.area),
            "trailing_line": int(self.trailing_line),
        }


def segment(
    image: np.ndarray,
    bg: ColumnBackground | None,
    settings: CaptureSettings,
) -> list[Particle]:
    """把一幅 RGB 图分割成矿石列表。P1 实现（见 docs/02 §3.3）。

    Args:
        image: ``(h, w, 3) uint8`` RGB。
        bg: 逐列背景模型；``settings.bg_mode == "color"`` 时可为 None（用 ``settings.bg_color``）。
        settings: 采集设置（阈值、形态学、面积过滤、margin、分水岭）。
    """
    raise NotImplementedError("segment() 在 P1 实现")
