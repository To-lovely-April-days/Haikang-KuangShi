"""进程内事件总线 ``EventBus``（docs/03 §11 的 WS 事件源）。

``publish(type, data)`` 生成 ``{"type", "ts", "data"}`` 并 **同步** 分发给全部订阅者（回调异常被捕获记录，
不影响其他订阅者与发布方）；最近 :data:`HISTORY_LEN` 条事件保留在环形缓冲里（``recent()``），
WS 层连接建立后可据此补发。线程安全：订阅表与缓冲由一把锁保护，回调在锁外执行。

事件类型取 docs/03 §11：``state / stats / job / capture_sample / param_sync / fault / mcu``；
发布未知类型只记 warning，不拒绝（便于扩展）。
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from oresorter.log import get_logger

log = get_logger(__name__)

HISTORY_LEN = 200
"""环形缓冲保留的最近事件条数。"""

EVENT_TYPES: tuple[str, ...] = (
    "state",
    "stats",
    "job",
    "capture_sample",
    "param_sync",
    "fault",
    "mcu",
)
"""docs/03 §11 定义的事件类型。"""

Subscriber = Callable[[dict[str, Any]], None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class EventBus:
    """线程安全的同步事件总线。

    Args:
        history: 环形缓冲长度（默认 :data:`HISTORY_LEN`）。
    """

    def __init__(self, history: int = HISTORY_LEN):
        self._lock = threading.RLock()
        self._subs: list[Subscriber] = []
        self._recent: deque[dict[str, Any]] = deque(maxlen=max(1, int(history)))
        self._published = 0
        self._unknown_types: set[str] = set()

    # ---- 订阅 ----
    def subscribe(self, cb: Subscriber) -> Callable[[], None]:
        """注册订阅者 ``cb(event)``；返回取消订阅函数（可重复调用）。"""
        with self._lock:
            self._subs.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs:
                    self._subs.remove(cb)

        return unsubscribe

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    # ---- 发布 ----
    def publish(self, type: str, data: dict[str, Any]) -> dict[str, Any]:
        """生成事件、写入环形缓冲并同步分发；返回事件字典。

        Args:
            type: 事件类型（见 :data:`EVENT_TYPES`）。
            data: 事件数据（应可 JSON 序列化；总线不做深拷贝，发布方不要事后修改）。
        """
        event = {"type": str(type), "ts": _now_iso(), "data": data}
        with self._lock:
            if type not in EVENT_TYPES and type not in self._unknown_types:
                self._unknown_types.add(type)
                log.warning("发布了未定义的事件类型", extra={"event_type": type})
            self._recent.append(event)
            self._published += 1
            subs = list(self._subs)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                log.exception("事件订阅者异常", extra={"event_type": type})
        return event

    # ---- 查询 ----
    def recent(self, n: int | None = None, type: str | None = None) -> list[dict[str, Any]]:
        """最近的事件（按发布顺序）；``type`` 可过滤类型，``n`` 限制条数（取最后 n 条）。"""
        with self._lock:
            items = list(self._recent)
        if type is not None:
            items = [e for e in items if e["type"] == type]
        if n is not None:
            items = items[-max(0, int(n)) :] if n > 0 else []
        return items

    @property
    def published_total(self) -> int:
        """累计发布条数。"""
        with self._lock:
            return self._published

    def clear(self) -> None:
        """清空环形缓冲（测试用）。"""
        with self._lock:
            self._recent.clear()
