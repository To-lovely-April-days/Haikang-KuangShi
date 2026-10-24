"""像素 → 喷头映射（原始需求 §3、docs/01 §5.4、docs/03 ``GET /cameras/{id}/nozzle-map``）。

规则
----
设起始像素 ``start_px``、结束像素 ``end_px``（左闭右开，有效像素为 ``[start_px, end_px)``），
喷头数 ``n``（默认 64）。有效像素 ``(end_px − start_px) = n·q + r``，则：

    前 ⌈r/2⌉ 个喷头各分 q+1 个像素，后 ⌊r/2⌋ 个喷头各分 q+1 个像素，中间喷头各分 q 个像素。

喷头局部编号 1..n 自 ``start_px`` 一侧（x 小的一端）起。全局编号 = ``camera_index · n + 局部编号``，
对应 HMI "孔位 0~768"（12 × 64）。

本模块只做纯整数几何运算，不涉及 IO。
"""

from __future__ import annotations

import operator
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

from oresorter.log import get_logger

if TYPE_CHECKING:
    from oresorter.config import CameraConfig

log = get_logger(__name__)


def _as_int(value: Any, name: str) -> int:
    """把参数转成 Python int；非整数类型（float/str 等）直接拒绝，不做静默取整。"""
    try:
        return operator.index(value)
    except TypeError as e:
        raise TypeError(f"{name} 必须是整数，收到 {type(value).__name__}: {value!r}") from e


class NozzleMap:
    """单台相机的像素 → 喷头映射表。构造后不可变，查询为 O(log n)。

    Args:
        start_px: 起始像素（含）。
        end_px: 结束像素（不含），有效像素为 ``[start_px, end_px)``。
        n: 该相机对应的喷头数，默认 64。
        camera_index: 相机在全局喷头编号中的序号（0 起），用于 ``to_global``。

    Raises:
        ValueError: ``n < 1``、``start_px ≥ end_px``、有效像素少于喷头数、``camera_index < 0``。
        TypeError: 参数不是整数。
    """

    __slots__ = ("_bounds", "_camera_index", "_end_px", "_n", "_q", "_r", "_start_px")

    def __init__(self, start_px: int, end_px: int, n: int = 64, *, camera_index: int = 0) -> None:
        start_px = _as_int(start_px, "start_px")
        end_px = _as_int(end_px, "end_px")
        n = _as_int(n, "n")
        camera_index = _as_int(camera_index, "camera_index")

        if n < 1:
            raise ValueError(f"喷头数 n 必须 ≥ 1，收到 {n}")
        if start_px >= end_px:
            raise ValueError(f"起始像素 start_px({start_px}) 必须小于结束像素 end_px({end_px})")
        total = end_px - start_px
        if total < n:
            raise ValueError(
                f"有效像素 {total}（{start_px}~{end_px}）少于喷头数 {n}，每个喷头至少需要 1 个像素"
            )
        if camera_index < 0:
            raise ValueError(f"camera_index 必须 ≥ 0，收到 {camera_index}")

        q, r = divmod(total, n)
        head = (r + 1) // 2  # ⌈r/2⌉ 个前部喷头多分 1 像素
        tail = r // 2  # ⌊r/2⌋ 个后部喷头多分 1 像素

        # bounds[i] = 第 i+1 号喷头的起始像素；bounds[n] = end_px（哨兵）
        bounds = [start_px]
        for i in range(1, n + 1):
            width = q + 1 if (i <= head or i > n - tail) else q
            bounds.append(bounds[-1] + width)
        assert bounds[-1] == end_px, "像素分配总和与有效像素不一致（内部错误）"

        self._start_px = start_px
        self._end_px = end_px
        self._n = n
        self._camera_index = camera_index
        self._q = q
        self._r = r
        self._bounds = bounds

        log.debug(
            "喷头映射表已建立",
            extra={
                "camera_index": camera_index,
                "start_px": start_px,
                "end_px": end_px,
                "nozzles": n,
                "quotient": q,
                "remainder": r,
            },
        )

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_camera(cls, cfg: CameraConfig, camera_index: int = 0) -> NozzleMap:
        """从相机配置构造：取 ``cfg.start_px`` / ``cfg.end_px`` / ``cfg.nozzles``。"""
        return cls(cfg.start_px, cfg.end_px, cfg.nozzles, camera_index=camera_index)

    # ------------------------------------------------------------------ 属性
    @property
    def start_px(self) -> int:
        """起始像素（含）。"""
        return self._start_px

    @property
    def end_px(self) -> int:
        """结束像素（不含）。"""
        return self._end_px

    @property
    def n(self) -> int:
        """喷头数。"""
        return self._n

    @property
    def camera_index(self) -> int:
        """相机序号（0 起）。"""
        return self._camera_index

    @property
    def quotient(self) -> int:
        """商 q：中间喷头每个分到的像素数。"""
        return self._q

    @property
    def remainder(self) -> int:
        """余数 r：多分 1 像素的喷头个数。"""
        return self._r

    @property
    def valid_px(self) -> int:
        """有效像素数 ``end_px − start_px``。"""
        return self._end_px - self._start_px

    # ------------------------------------------------------------------ 查询
    def ranges(self) -> list[tuple[int, int]]:
        """第 1..n 号喷头的像素闭区间 ``(px_from, px_to)``。

        首个从 ``start_px`` 起，最后一个到 ``end_px − 1``，连续无缝无重叠。每次调用返回新列表。
        """
        b = self._bounds
        return [(b[i], b[i + 1] - 1) for i in range(self._n)]

    def widths(self) -> list[int]:
        """第 1..n 号喷头各自分到的像素数（原始需求 64 行表的一行）。"""
        b = self._bounds
        return [b[i + 1] - b[i] for i in range(self._n)]

    def range_of(self, nozzle: int) -> tuple[int, int]:
        """第 ``nozzle`` 号（1..n）喷头的像素闭区间。越界抛 ``ValueError``。"""
        nozzle = _as_int(nozzle, "nozzle")
        self._check_local(nozzle)
        return self._bounds[nozzle - 1], self._bounds[nozzle] - 1

    def pixel_to_nozzle(self, x: int) -> int | None:
        """像素列 ``x`` 所属的局部喷头编号 1..n；``x`` 不在 ``[start_px, end_px)`` 内返回 ``None``。"""
        x = _as_int(x, "x")
        if x < self._start_px or x >= self._end_px:
            return None
        # bounds 中 ≤ x 的元素个数恰为所属喷头编号（bounds[k-1] ≤ x < bounds[k] ⇒ k）
        return bisect_right(self._bounds, x)

    def span(self, x0: int, x1: int, expand: int = 0) -> tuple[int, int] | None:
        """颗粒 x 范围（闭区间 ``[x0, x1]``）覆盖的局部喷头区间 ``(first, last)``。

        - 向两侧各扩 ``expand`` 个喷头，再裁到 ``1..n``；
        - 与有效区 ``[start_px, end_px)`` 完全不相交 → ``None``；部分相交 → 只取相交部分；
        - ``x0 > x1`` 时**自动交换**两端（视作同一闭区间的两种写法），不报错。

        Raises:
            ValueError: ``expand < 0``。
            TypeError: 参数不是整数。
        """
        x0 = _as_int(x0, "x0")
        x1 = _as_int(x1, "x1")
        expand = _as_int(expand, "expand")
        if expand < 0:
            raise ValueError(f"expand 必须 ≥ 0，收到 {expand}")
        if x0 > x1:
            x0, x1 = x1, x0
        if x1 < self._start_px or x0 >= self._end_px:
            return None
        lo = max(x0, self._start_px)
        hi = min(x1, self._end_px - 1)
        first = bisect_right(self._bounds, lo)
        last = bisect_right(self._bounds, hi)
        first = max(1, first - expand)
        last = min(self._n, last + expand)
        return first, last

    # ------------------------------------------------------------------ 全局编号
    def to_global(self, local: int) -> int:
        """局部编号 1..n → 全局编号 ``camera_index · n + local``（HMI 孔位）。越界抛 ``ValueError``。"""
        local = _as_int(local, "local")
        self._check_local(local)
        return self._camera_index * self._n + local

    def global_span(self, x0: int, x1: int, expand: int = 0) -> tuple[int, int] | None:
        """同 ``span``，但返回全局喷头编号区间。"""
        s = self.span(x0, x1, expand)
        if s is None:
            return None
        return self.to_global(s[0]), self.to_global(s[1])

    # ------------------------------------------------------------------ 序列化
    def to_dict(self) -> dict[str, Any]:
        """docs/03 ``GET /cameras/{id}/nozzle-map`` 的响应体。"""
        return {
            "start_px": self._start_px,
            "end_px": self._end_px,
            "nozzles": self._n,
            "quotient": self._q,
            "remainder": self._r,
            "ranges": [
                {"nozzle": i + 1, "px_from": a, "px_to": b}
                for i, (a, b) in enumerate(self.ranges())
            ],
        }

    # ------------------------------------------------------------------ 内部
    def _check_local(self, nozzle: int) -> None:
        if nozzle < 1 or nozzle > self._n:
            raise ValueError(f"喷头编号 {nozzle} 越界，有效范围 1~{self._n}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NozzleMap):
            return NotImplemented
        return (self._start_px, self._end_px, self._n, self._camera_index) == (
            other._start_px,
            other._end_px,
            other._n,
            other._camera_index,
        )

    def __hash__(self) -> int:
        return hash((self._start_px, self._end_px, self._n, self._camera_index))

    def __repr__(self) -> str:
        return (
            f"NozzleMap(start_px={self._start_px}, end_px={self._end_px}, n={self._n}, "
            f"camera_index={self._camera_index}, q={self._q}, r={self._r})"
        )
