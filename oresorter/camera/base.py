"""相机抽象接口（docs/02 §3.2）。

所有相机实现（海康 SDK、Mock）都在这个接口后面；运行时流水线、采样任务、预览只依赖本模块。
坐标约定：``Frame.image`` 为 ``(h, w, 3) uint8 RGB``，第 0 行是该块最早扫到的行，行号随时间增大；
``line_start`` 是第 0 行的全局行号（自 ``reset_line_counter`` / SYNC 起），第 ``y`` 行的全局行号 = ``line_start + y``。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from oresorter.config import CameraConfig


class CameraError(Exception):
    """相机通用错误。"""


class CameraTimeout(CameraError):
    """在超时时间内没有收到帧。"""


class CameraDisconnected(CameraError):
    """相机掉线/未打开。"""


@dataclass(slots=True)
class Frame:
    """一个图像块（运行时按 chunk_lines 切块；采样时可为整帧）。"""

    seq: int
    """块序号（自 start_stream 起，单调递增；跳号表示 SDK 丢块）。"""
    line_start: int
    """该块首行的全局行号（自 reset_line_counter / SYNC 起）。"""
    image: np.ndarray
    """``(h, w, 3) uint8`` RGB。"""
    ts: float
    """单调时钟（``time.monotonic()``）。"""
    camera_id: int = 0

    @property
    def lines(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def line_end(self) -> int:
        """末行全局行号 + 1（开区间）。"""
        return self.line_start + self.lines


@dataclass
class CameraParams:
    """可下发到相机的参数（与 ``CameraConfig`` 的可编辑字段一一对应）。"""

    awb_mode: str = "once"
    exposure_us: float = 50.0
    width: int = 4096
    offset_x: int = 0
    height: int = 4096
    line_rate_hz: int = 9000
    trigger: str = "internal"

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> CameraParams:
        return cls(
            awb_mode=cfg.awb_mode,
            exposure_us=cfg.exposure_us,
            width=cfg.width,
            offset_x=cfg.offset_x,
            height=cfg.height,
            line_rate_hz=cfg.line_rate_hz,
            trigger=cfg.trigger,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraInfo:
    id: int
    name: str
    model: str
    serial: str | None
    ip: str | None
    nozzles: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CameraStats:
    chunks_total: int = 0
    dropped_chunks: int = 0
    reconnects: int = 0
    errors: int = 0
    last_frame_ts: float | None = None
    chunks_per_s: float = 0.0
    _window: list[float] = field(default_factory=list, repr=False)

    def on_frame(self, ts: float) -> None:
        self.chunks_total += 1
        self.last_frame_ts = ts
        self._window.append(ts)
        cutoff = ts - 2.0
        while self._window and self._window[0] < cutoff:
            self._window.pop(0)
        if len(self._window) >= 2:
            span = self._window[-1] - self._window[0]
            self.chunks_per_s = (len(self._window) - 1) / span if span > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks_total": self.chunks_total,
            "dropped_chunks": self.dropped_chunks,
            "reconnects": self.reconnects,
            "errors": self.errors,
            "chunks_per_s": round(self.chunks_per_s, 2),
        }


class Camera(ABC):
    """线阵相机抽象。实现必须线程安全地支持“一个生产线程 + 多个只读查询”。

    生命周期：``open()`` → ``set_params()`` → ``start_stream()`` → ``frames()`` 消费 → ``stop_stream()`` → ``close()``。
    所有方法都不得无限阻塞；SDK 调用要带超时。
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.id: int = cfg.id
        self.stats = CameraStats()

    # ---- 连接 ----
    @abstractmethod
    def open(self) -> None:
        """打开相机并应用配置中的参数。失败抛 CameraError。"""

    @abstractmethod
    def close(self) -> None:
        """停止流并释放资源，可重复调用。"""

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @property
    @abstractmethod
    def streaming(self) -> bool: ...

    # ---- 参数 ----
    @abstractmethod
    def get_params(self) -> CameraParams:
        """从相机实时读取参数。"""

    @abstractmethod
    def set_params(self, p: CameraParams) -> CameraParams:
        """下发参数，返回相机实际接受的值（相机可能对齐步进）。"""

    @abstractmethod
    def awb_once(self) -> None:
        """做一次白平衡并锁定（awb_mode → off）。"""

    # ---- 采集 ----
    @abstractmethod
    def snapshot(self, height: int) -> np.ndarray:
        """单次采集整帧 ``(height, width, 3)``；流式采集中也可调用（实现可复用流数据）。"""

    @abstractmethod
    def start_stream(self, chunk_lines: int) -> None:
        """开始按 ``chunk_lines`` 行分块连续采集。"""

    @abstractmethod
    def stop_stream(self) -> None: ...

    @abstractmethod
    def get_frame(self, timeout: float) -> Frame | None:
        """取下一块；超时返回 None。相机掉线抛 CameraDisconnected。"""

    def frames(self, timeout: float = 1.0) -> Iterator[Frame]:
        """块迭代器：流停止时结束；超时抛 CameraTimeout。"""
        while self.streaming:
            f = self.get_frame(timeout)
            if f is None:
                if not self.streaming:
                    return
                raise CameraTimeout(f"相机 {self.id} 在 {timeout}s 内无帧")
            yield f

    @abstractmethod
    def reset_line_counter(self) -> None:
        """把全局行号归零（与下位机 SYNC 配合）。下一块的 line_start 从 0 开始。"""

    # ---- 信息 ----
    def info(self) -> CameraInfo:
        return CameraInfo(
            id=self.cfg.id,
            name=self.cfg.name,
            model=self.cfg.model,
            serial=self.cfg.serial,
            ip=self.cfg.ip,
            nozzles=self.cfg.nozzles,
        )

    def status(self) -> dict[str, Any]:
        """GET /state 里的相机段。"""
        d: dict[str, Any] = {
            "id": self.id,
            "connected": self.connected,
            "streaming": self.streaming,
            "line_rate_hz": self.cfg.line_rate_hz,
        }
        d.update(self.stats.to_dict())
        return d

    @staticmethod
    def now() -> float:
        return time.monotonic()
