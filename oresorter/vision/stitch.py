"""跨块拼接与去重（docs/01 §5.2、§5.5）。

运行时相机按 ``chunk_lines`` 行分块出帧，一颗矿石可能横跨两块。``ChunkStitcher`` 把皮带上紧邻当前块
之前的 ``overlap`` 行（取自上一块**处理图**的尾部）拼到当前块前面得到“处理图”，分割后按以下规则挑出
本块应处理的颗粒，保证每颗矿石**恰好处理一次**（以尾沿行号为准）：

- ``max_y < overlap_used - 1``：尾沿落在上一块末行之前，上一块已完整处理过 → 丢弃（``skipped_seen``）；
- 触及处理图底边：留给下一块，那时它会完整出现在重叠区 → 延后（``deferred``）；
- 其余保留。保留的颗粒若顶到处理图顶边且 ``overlap_used > 0``，说明它在上一块就触底、长度 ≥ overlap_used；
  再用上一块延后颗粒的位置判断它是否真的越过了处理图顶边 → ``truncated=True`` 并告警，按截断处理
  （尾沿行号仍然正确；详见 :meth:`ChunkStitcher.select`）。

去重规则不依赖尾部长度：记矿石尾沿全局行号为 ``M``，则 ``M + 1`` 唯一落在某一块的行区间
``[line_start, line_end)`` 内，只有那一块会保留它（第 k 块保留 ⟺ ``M ≥ line_end_{k-1} - 1`` 且
``M ≤ line_end_k - 2``）。

尾部缓存取自**处理图**（而不是当前块）的最后 ``min(overlap, 处理图高)`` 行：块高小于 overlap
（合法配置，例如 ``runtime_chunk_lines=128`` / ``overlap_lines=256``）时，重叠区在起始或断块后的头几块
逐步增长（``overlap_used = min(overlap, 自上次断块起的累计行数)``），之后稳定为 ``overlap``，因此长度
≤ overlap 的矿石不会因块高不足而被截断。

块内坐标 → 全局行号：处理图第 0 行的全局行号 ``y_offset = frame.line_start - overlap_used``，
``trailing_line = y_offset + trailing_y``（docs/01 §5.5 的下位机时间轴）。

内存：不保存整块引用，只拷贝尾部（≤ ``overlap`` 行）；处理图用一次 ``np.concatenate`` 生成。
本类非线程安全，由处理线程独占使用；每次 ``push`` 后应恰好调用一次 ``select``。
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from oresorter.camera.base import Frame
from oresorter.log import get_logger
from oresorter.vision.segment import Particle

log = get_logger(__name__)

SegmentFn = Callable[[np.ndarray], list[Particle]]
"""分割回调：输入处理图 ``(h, w, 3) uint8``，输出处理图坐标系下的颗粒列表。"""

_DeferredSpan = tuple[int, int, int]
"""延后颗粒的 ``(x0, x1, 全局顶行)``，用于下一块判断顶到顶边的颗粒是否真被截断。"""


@dataclass(frozen=True, slots=True)
class ChunkInfo:
    """最近一块的元数据快照（刻意不持有图像，避免整块驻留内存）。"""

    seq: int
    line_start: int
    lines: int
    width: int
    ts: float
    camera_id: int

    @property
    def line_end(self) -> int:
        """末行全局行号 + 1（开区间），与 ``Frame.line_end`` 同义。"""
        return self.line_start + self.lines

    @classmethod
    def from_frame(cls, frame: Frame) -> ChunkInfo:
        """从 ``Frame`` 抽取元数据。"""
        return cls(
            seq=int(frame.seq),
            line_start=int(frame.line_start),
            lines=int(frame.image.shape[0]),
            width=int(frame.image.shape[1]),
            ts=float(frame.ts),
            camera_id=int(frame.camera_id),
        )


class ChunkStitcher:
    """跨块拼接器：``push`` 拼处理图，``select`` 去重并换算全局行号，``process`` 一步到位。

    Args:
        overlap: 携带的上一块处理图尾部行数（``PipelineConfig.overlap_lines``），应 ≥ 最大矿石长度。
        max_object_lines: 预期的最大矿石长度（行）；给出且大于 ``overlap`` 时在构造时告警，
            并对高度超过它的颗粒打 ``extra["oversize"]`` 标记与告警。``None`` 表示不检查。
    """

    def __init__(self, overlap: int, *, max_object_lines: int | None = None) -> None:
        try:
            overlap = operator.index(overlap)
        except TypeError as e:
            raise ValueError(f"overlap 必须是整数，得到 {overlap!r}") from e
        if overlap < 0:
            raise ValueError(f"overlap 不能为负数，得到 {overlap}")
        if max_object_lines is not None:
            try:
                max_object_lines = operator.index(max_object_lines)
            except TypeError as e:
                raise ValueError(
                    f"max_object_lines 必须是整数或 None，得到 {max_object_lines!r}"
                ) from e
            if max_object_lines < 1:
                raise ValueError(f"max_object_lines 必须 ≥ 1，得到 {max_object_lines}")
            if max_object_lines > overlap:
                log.warning(
                    "overlap 小于最大矿石长度，超长矿石将被截断处理",
                    extra={"overlap": overlap, "max_object_lines": max_object_lines},
                )
        self.overlap: int = overlap
        self.max_object_lines: int | None = max_object_lines
        self._tail: np.ndarray | None = None
        self._last: ChunkInfo | None = None
        self._overlap_used = 0
        self._y_offset = 0
        self._proc_h = 0
        # 上一块 select 记录的延后颗粒（本块判断截断用）；None = 上一块没有 select 过或已断块
        self._deferred_prev: list[_DeferredSpan] | None = None
        # 本块 select 记录的延后颗粒；None = 本块尚未 select
        self._deferred_cur: list[_DeferredSpan] | None = None
        self._frames = 0
        self._dropped = 0
        self._resets = 0
        self._deferred = 0
        self._skipped_seen = 0
        self._truncated = 0
        self._selected = 0
        self._lost = 0

    # ---- 属性 ----
    @property
    def overlap_used(self) -> int:
        """最近一次 ``push`` 实际拼接的尾部行数（首块 / 断块后为 0，之后 ``min(overlap, 累计行数)``）。"""
        return self._overlap_used

    @property
    def y_offset(self) -> int:
        """最近一次 ``push`` 得到的处理图第 0 行的全局行号。"""
        return self._y_offset

    @property
    def last_chunk(self) -> ChunkInfo | None:
        """最近一块的元数据（:class:`ChunkInfo`，**不含图像**）；尚未 push 过时为 None。"""
        return self._last

    @property
    def last_frame(self) -> ChunkInfo | None:
        """``last_chunk`` 的别名（docs/02 §3.3 用词）。注意返回的是 ``ChunkInfo`` 而非 ``Frame``，
        没有 ``image`` 字段——拼接器刻意不保留整块图像。"""
        return self._last

    # ---- 拼接 ----
    def push(self, frame: Frame) -> tuple[np.ndarray, int]:
        """接收一块，返回 ``(处理图, y_offset)``。

        处理图 = 缓存尾部 ``overlap_used`` 行 + 当前块；``y_offset = frame.line_start - overlap_used``。
        首块、``seq`` 不连续（丢块 / 回退）、``line_start`` 与上一块 ``line_end`` 不接、图像宽度变化时
        没有可用尾部：``overlap_used = 0``，处理图就是当前块本身（不拷贝）。
        丢块时 ``dropped += seq 差 - 1``；除首块外的所有断块情形 ``resets += 1``、把上一块延后且再也无法
        完整处理的颗粒数累计到 ``lost``，并记 warning。

        push 后缓存处理图尾部 ``min(overlap, 处理图高)`` 行的拷贝供下一块使用。

        Raises:
            ValueError: ``frame.image`` 不是 ``(h, w, 3) uint8`` 或高 / 宽为 0。
        """
        img = frame.image
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"frame.image 必须是 (h, w, 3) 数组，得到 shape={getattr(img, 'shape', None)}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"frame.image 必须是 uint8，得到 {img.dtype}")
        if img.shape[0] < 1 or img.shape[1] < 1:
            raise ValueError(f"frame.image 高和宽都必须 ≥ 1，得到 shape={img.shape}")
        info = ChunkInfo.from_frame(frame)
        prev, tail = self._last, self._tail

        reason: str | None = None
        if prev is None or tail is None:
            reason = "first"
        elif info.seq != prev.seq + 1:
            if info.seq > prev.seq + 1:
                self._dropped += info.seq - prev.seq - 1
                reason = "dropped"
            else:
                reason = "seq_rewind"
        elif info.line_start != prev.line_end:
            reason = "line_jump"
        elif tail.shape[1] != img.shape[1]:
            reason = "shape_changed"

        if reason is None:
            assert tail is not None  # 上面的分支保证
            used = int(tail.shape[0])
            proc = np.concatenate((tail, img), axis=0) if used > 0 else img
            self._deferred_prev = self._deferred_cur
        else:
            used = 0
            proc = img
            self._deferred_prev = None
            if prev is not None:
                lost = len(self._deferred_cur) if self._deferred_cur is not None else 0
                self._lost += lost
                self._resets += 1
                log.warning(
                    "块不连续，放弃缓存尾部重叠区",
                    extra={
                        "reason": reason,
                        "camera": info.camera_id,
                        "seq": info.seq,
                        "prev_seq": prev.seq,
                        "line_start": info.line_start,
                        "prev_line_end": prev.line_end,
                        "lost": lost,
                        "lost_total": self._lost,
                        "dropped_total": self._dropped,
                    },
                )
        self._deferred_cur = None

        self._overlap_used = used
        self._y_offset = info.line_start - used
        proc_h = int(proc.shape[0])
        self._proc_h = proc_h
        keep = min(self.overlap, proc_h)
        # 只保留处理图尾部拷贝，不引用整块；overlap == 0 时为 (0, w, 3) 的空数组（仍视为“有尾部”）
        self._tail = proc[proc_h - keep :].copy()
        self._last = info
        self._frames += 1
        return proc, self._y_offset

    # ---- 去重 ----
    def _crossed_top(self, p: Particle, y_offset: int) -> tuple[bool, str]:
        """判断顶到处理图顶边、且 ``overlap_used > 0`` 的颗粒是否真的越过了顶边。

        返回 ``(truncated, basis)``。上一块 ``select`` 过时（``basis="tracked"``）：取 x 区间与它重叠的
        上一块延后颗粒，若全部从 ``y_offset`` 行（含）之后开始，说明它恰好完整可见（高度 == overlap_used
        的边界情形）→ 不截断；任一从 ``y_offset`` 之前开始 → 截断。找不到对应的延后颗粒
        （``basis="unmatched"``，分割结果不一致）或上一块没有 ``select`` 信息（``basis="untracked"``）时，
        退回“顶到顶边即截断”的启发式——此时高度恰好等于 overlap_used 的完整颗粒会被误报为截断。
        """
        prev = self._deferred_prev
        if prev is None:
            return True, "untracked"
        x0, x1 = p.x_range
        tops = [top for a, b, top in prev if a <= x1 and x0 <= b]
        if not tops:
            return True, "unmatched"
        return any(top < y_offset for top in tops), "tracked"

    def select(self, particles: list[Particle], image_h: int) -> list[Particle]:
        """按 §5.2 规则筛出本块应处理的颗粒，填写全局行号，按 ``trailing_line`` 升序返回。

        Args:
            particles: 对最近一次 ``push`` 返回的处理图做分割得到的颗粒（处理图坐标系）。
            image_h: 处理图高度，必须等于最近一次 ``push`` 返回的处理图高度。

        对每个返回的颗粒：``trailing_line = y_offset + trailing_y``、``extra["seq"]``、
        ``extra["y_offset"]``。顶到处理图顶边且 ``overlap_used > 0`` 的颗粒按 :meth:`_crossed_top`
        判定 ``truncated``（有上一块延后信息时精确；否则按启发式，``h == overlap_used`` 的完整颗粒可能误报）。
        被延后的颗粒标 ``touches_bottom=True`` 但不返回，其位置被记录供下一块判定截断。

        Raises:
            RuntimeError: 尚未 ``push`` 过任何块。
            ValueError: ``image_h`` 不是整数，或与最近处理图高度不一致。
        """
        if self._last is None:
            raise RuntimeError("select() 之前必须先 push() 一块")
        try:
            image_h = operator.index(image_h)
        except TypeError as e:
            raise ValueError(f"image_h 必须是整数，得到 {image_h!r}") from e
        if image_h != self._proc_h:
            raise ValueError(
                f"image_h={image_h} 与最近一次 push 的处理图高度 {self._proc_h} 不一致"
            )
        used = self._overlap_used
        y_offset = self._y_offset
        seq = self._last.seq
        camera = self._last.camera_id
        deferred_now: list[_DeferredSpan] = []
        out: list[Particle] = []
        for p in particles:
            x, y, w, h = p.bbox
            max_y = y + h - 1
            if max_y < used - 1:
                # 尾沿在上一块末行之前且整颗可见 → 上一块已处理
                self._skipped_seen += 1
                continue
            if y + h >= image_h:
                # 触底：下一块的重叠区会完整包含它
                p.touches_bottom = True
                self._deferred += 1
                deferred_now.append((x, x + w - 1, y_offset + y))
                continue
            if y <= 0:
                p.touches_top = True
                if used > 0:
                    truncated, basis = self._crossed_top(p, y_offset)
                    if truncated:
                        p.truncated = True
                        self._truncated += 1
                        log.warning(
                            "矿石长度超过 overlap，顶部被截断，按截断处理",
                            extra={
                                "camera": camera,
                                "seq": seq,
                                "bbox": list(p.bbox),
                                "overlap_used": used,
                                "basis": basis,
                            },
                        )
            if self.max_object_lines is not None and h > self.max_object_lines:
                p.extra["oversize"] = True
                log.warning(
                    "矿石高度超过 max_object_lines",
                    extra={
                        "camera": camera,
                        "seq": seq,
                        "bbox": list(p.bbox),
                        "max_object_lines": self.max_object_lines,
                    },
                )
            p.trailing_line = y_offset + int(p.trailing_y)
            p.extra["seq"] = seq
            p.extra["y_offset"] = y_offset
            out.append(p)
        self._deferred_cur = deferred_now
        out.sort(key=lambda q: q.trailing_line)
        self._selected += len(out)
        return out

    def process(self, frame: Frame, segment_fn: SegmentFn) -> list[Particle]:
        """``push`` → ``segment_fn(处理图)`` → ``select``，返回本块应处理的颗粒。"""
        proc, _ = self.push(frame)
        particles = segment_fn(proc)
        return self.select(particles, int(proc.shape[0]))

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        """累计计数。

        - ``frames`` / ``dropped`` / ``resets``：收到的块数、丢块数、断块次数；
        - ``deferred`` / ``skipped_seen`` / ``truncated`` / ``selected``：select 的四种结果计数；
        - ``lost``：断块时上一块已延后、因此再也无法被完整处理的颗粒数（上界：其中特别长的可能在
          断块后的首块以未标记截断的形式出现）。
        """
        return {
            "frames": self._frames,
            "dropped": self._dropped,
            "resets": self._resets,
            "deferred": self._deferred,
            "skipped_seen": self._skipped_seen,
            "truncated": self._truncated,
            "selected": self._selected,
            "lost": self._lost,
        }

    def reset(self) -> None:
        """清空尾部缓存、块记录与全部计数；下一块按首块处理（不计 resets / lost）。"""
        self._tail = None
        self._last = None
        self._overlap_used = 0
        self._y_offset = 0
        self._proc_h = 0
        self._deferred_prev = None
        self._deferred_cur = None
        self._frames = 0
        self._dropped = 0
        self._resets = 0
        self._deferred = 0
        self._skipped_seen = 0
        self._truncated = 0
        self._selected = 0
        self._lost = 0
