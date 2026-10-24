"""运行统计（docs/03 §3 ``state.stats`` / ``GET /run/stats``）。

:class:`RunStats` 累计自本次 ``start``（或 :meth:`RunStats.reset`）以来的计数，并维护一个滑动窗口
（默认 60 s）计算每分钟吞吐，以及最近 N 块的处理延迟分位数。

- ``on_chunk`` 为 O(1) 摊销（窗口过期项在 ``on_chunk`` 与 ``snapshot`` 时从队头弹出）；
- ``snapshot`` 返回全部原生 Python 类型，可直接 ``json.dumps``；每分钟值为四舍五入后的 ``int``
  （与 docs/03 §3 示例一致），延迟分位数为 ``float``（毫秒，最近秩法返回真实样本值）；
- ``per_class_*`` 对所有已配置的类别（``class_names``）都给出键（未出现为 0），HMI 不必补零；
  未配置但实际出现的类别追加在后面（按名称排序）；
- 时钟可注入（默认 ``time.monotonic``），便于测试。
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

Clock = Callable[[], float]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _percentile(sorted_values: list[float], pct: float) -> float:
    """最近秩法分位数：返回实际样本值；空列表返回 0.0。"""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = max(0, min(n - 1, math.ceil(pct / 100.0 * n) - 1))
    return sorted_values[idx]


class LatencyTimer:
    """计时上下文管理器::

    with stats.timer() as t:
        ...
    stats.on_chunk(t.ms, n_particles=7, blown=2)
    """

    def __init__(self, clock: Clock):
        self._clock = clock
        self._t0 = 0.0
        self.ms = 0.0

    def __enter__(self) -> LatencyTimer:
        self._t0 = self._clock()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (self._clock() - self._t0) * 1000.0


class RunStats:
    """线程安全的运行统计。

    Args:
        window_s: 吞吐滑动窗口长度（秒）。
        latency_samples: 保留的延迟样本数（分位数计算范围）。
        clock: 单调时钟（秒）。
        class_names: 已配置的类别名（``AppConfig.class_names``）；快照的 ``per_class_*`` 总是包含
            这些键（缺省 0），顺序与配置一致。None 表示只输出实际出现过的类别。
    """

    def __init__(
        self,
        *,
        window_s: float = 60.0,
        latency_samples: int = 2000,
        clock: Clock = time.monotonic,
        class_names: Iterable[str] | None = None,
    ):
        self._window_s = float(window_s)
        self._clock = clock
        self._class_names: list[str] = list(dict.fromkeys(str(c) for c in (class_names or ())))
        self._lock = threading.Lock()
        self._latency: deque[float] = deque(maxlen=max(1, int(latency_samples)))
        self._window: deque[tuple[float, int, int, dict[str, int] | None]] = deque()
        self._win_particles = 0
        self._win_blown = 0
        self._win_per_class: Counter[str] = Counter()
        self._chunks_total = 0
        self._dropped = 0
        self._particles_total = 0
        self._blown_total = 0
        self._per_class_total: Counter[str] = Counter()
        self._since_mono = self._clock()
        self._since_iso = _now_iso()

    # ---- 记录 ----
    def on_chunk(
        self,
        latency_ms: float,
        n_particles: int,
        blown: int,
        per_class: dict[str, int] | None = None,
        dropped: int = 0,
    ) -> None:
        """记录一个处理完的块。

        Args:
            latency_ms: 该块端到端处理延迟（毫秒）。
            n_particles: 块内颗粒数。
            blown: 块内吹气颗粒数。
            per_class: 各类别颗粒数。
            dropped: 该块之前丢弃的块数（一并计入 ``dropped_chunks``）。
        """
        now = self._clock()
        with self._lock:
            self._chunks_total += 1
            self._particles_total += int(n_particles)
            self._blown_total += int(blown)
            self._dropped += int(dropped)
            self._win_particles += int(n_particles)
            self._win_blown += int(blown)
            cls_copy: dict[str, int] | None = None
            if per_class:
                cls_copy = {str(k): int(v) for k, v in per_class.items()}
                self._per_class_total.update(cls_copy)
                self._win_per_class.update(cls_copy)
            self._window.append((now, int(n_particles), int(blown), cls_copy))
            self._latency.append(float(latency_ms))
            self._evict_locked(now)

    def on_dropped(self, n: int = 1) -> None:
        """记录丢块。"""
        with self._lock:
            self._dropped += int(n)

    def timer(self) -> LatencyTimer:
        """返回一个使用本实例时钟的 :class:`LatencyTimer`。"""
        return LatencyTimer(self._clock)

    def reset(self) -> None:
        """清零累计与窗口（每次 ``start`` 时调用）。"""
        with self._lock:
            self._latency.clear()
            self._window.clear()
            self._win_particles = 0
            self._win_blown = 0
            self._win_per_class.clear()
            self._chunks_total = 0
            self._dropped = 0
            self._particles_total = 0
            self._blown_total = 0
            self._per_class_total.clear()
            self._since_mono = self._clock()
            self._since_iso = _now_iso()

    # ---- 查询 ----
    def snapshot(self) -> dict[str, Any]:
        """返回统计快照（全部原生类型）。

        每分钟值按窗口内实际时长折算：自 reset 起不足 ``window_s`` 时用实际 elapsed，
        不足 1 s 时按 1 s；结果四舍五入为 ``int``。
        """
        now = self._clock()
        with self._lock:
            self._evict_locked(now)
            elapsed = max(0.0, now - self._since_mono)
            span = max(1.0, min(elapsed, self._window_s))
            factor = 60.0 / span
            lat = sorted(self._latency)
            classes = self._class_keys_locked()
            per_class_pm = {cls: round(self._win_per_class.get(cls, 0) * factor) for cls in classes}
            return {
                "particles_per_min": round(self._win_particles * factor),
                "blown_per_min": round(self._win_blown * factor),
                "per_class_per_min": per_class_pm,
                "chunk_latency_ms_p50": round(_percentile(lat, 50), 2),
                "chunk_latency_ms_p99": round(_percentile(lat, 99), 2),
                "chunk_latency_ms_max": round(lat[-1], 2) if lat else 0.0,
                "chunks_total": int(self._chunks_total),
                "dropped_chunks": int(self._dropped),
                "particles_total": int(self._particles_total),
                "blown_total": int(self._blown_total),
                "per_class_total": {cls: int(self._per_class_total.get(cls, 0)) for cls in classes},
                "since": self._since_iso,
                "elapsed_s": round(elapsed, 1),
            }

    def _class_keys_locked(self) -> list[str]:
        """快照的类别键：已配置类别（按配置顺序）+ 未配置但出现过的类别（按名称排序）。"""
        configured = set(self._class_names)
        extra = sorted(c for c in self._per_class_total if c not in configured)
        return [*self._class_names, *extra]

    # ---- 内部 ----
    def _evict_locked(self, now: float) -> None:
        cutoff = now - self._window_s
        window = self._window
        while window and window[0][0] < cutoff:
            _, n, blown, per_class = window.popleft()
            self._win_particles -= n
            self._win_blown -= blown
            if per_class:
                self._win_per_class.subtract(per_class)
                for k in [k for k, v in self._win_per_class.items() if v <= 0]:
                    del self._win_per_class[k]
