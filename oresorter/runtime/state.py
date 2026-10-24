"""运行状态机（docs/01 §4.3）。

状态：``stopped / running / paused / sampling / fault``；事件：``start / pause / stop /
capture_begin / capture_end / fault_raised / fault_cleared``。

迁移规则：

- ``stopped --start--> running``；``running --pause--> paused``；``paused --start--> running``；
- ``running / paused / sampling --stop--> stopped``（sampling 下允许 stop，调用方负责取消采样任务）；
- ``stopped / paused --capture_begin--> sampling``（记住来源状态）；``sampling --capture_end--> 来源状态``；
- 任意状态 ``--fault_raised--> fault``；``fault --fault_cleared--> stopped``（需人工再次 start）；
- 幂等：``running+start``、``paused+pause``、``stopped+stop``、``fault+stop``、``fault+fault_raised``、
  非 fault 状态的 ``fault_cleared`` 均不报错、不触发监听；
  其中 ``fault+stop`` 返回 fault（故障不会因"停止"而消失），但 **不是 409**：routes_run 仍应向下位机
  下发 STOP（stopped = 全停的安全语义，docs/01 §4.3），HMI 的停止按钮在故障态不置灰；
- 非法迁移抛 :class:`IllegalTransition`，``reasons`` 用本模块的常量（docs/03 §4 的 409 原因）。

监听回调（``on_change`` / ``subscribe``）在 **不持有状态锁** 的情况下按迁移发生的先后顺序串行调用
（内部有独立的通知锁），因此回调收到的 ``new`` 与 ``sm.state`` 的变化顺序一致；回调里不要阻塞等待
另一个线程的 ``transition()``，否则会死锁。

本模块只管状态与迁移合法性，不操作设备；设备/任务动作由调用方（routes_run、pipeline）在迁移前后执行。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple

from oresorter.log import get_logger

log = get_logger(__name__)

# ---- 409 CONFLICT 原因常量（docs/03 §4）----
NO_MODEL_DEPLOYED = "NO_MODEL_DEPLOYED"
CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
MCU_DISCONNECTED = "MCU_DISCONNECTED"
SAMPLING_IN_PROGRESS = "SAMPLING_IN_PROGRESS"
IN_FAULT = "IN_FAULT"
ALREADY_RUNNING = "ALREADY_RUNNING"
NOT_RUNNING = "NOT_RUNNING"
NOT_SAMPLING = "NOT_SAMPLING"

REASON_MESSAGES: dict[str, str] = {
    NO_MODEL_DEPLOYED: "未部署模型",
    CAMERA_DISCONNECTED: "相机未连接",
    MCU_DISCONNECTED: "下位机未连接",
    SAMPLING_IN_PROGRESS: "采样进行中",
    IN_FAULT: "处于故障状态",
    ALREADY_RUNNING: "正在运行",
    NOT_RUNNING: "未在运行",
    NOT_SAMPLING: "未在采样",
}

HISTORY_LEN = 100


class RunState(str, Enum):
    """运行状态（值与 docs/03 的 ``run_state`` 枚举一致）。"""

    stopped = "stopped"
    running = "running"
    paused = "paused"
    sampling = "sampling"
    fault = "fault"


class Event(str, Enum):
    """状态机事件。"""

    start = "start"
    pause = "pause"
    stop = "stop"
    capture_begin = "capture_begin"
    capture_end = "capture_end"
    fault_raised = "fault_raised"
    fault_cleared = "fault_cleared"


class IllegalTransition(Exception):
    """非法迁移。``code`` 固定为 ``CONFLICT``，对应 HTTP 409。"""

    code = "CONFLICT"

    def __init__(self, state: RunState, event: Event, reasons: list[str]):
        self.state = state
        self.event = event
        self.reasons = list(reasons)
        why = "、".join(REASON_MESSAGES.get(r, r) for r in self.reasons) or "状态不允许"
        self.message = f"当前状态 {state.value} 不允许执行 {event.value}：{why}"
        super().__init__(self.message)

    @property
    def detail(self) -> dict[str, Any]:
        return {"state": self.state.value, "event": self.event.value, "reasons": list(self.reasons)}

    def to_dict(self) -> dict[str, Any]:
        """与 docs/03 错误体 ``error`` 字段一致。"""
        return {"code": self.code, "message": self.message, "detail": self.detail}


class Transition(NamedTuple):
    """一条迁移历史：``ts`` 为 ``time.time()``。"""

    ts: float
    src: RunState
    dst: RunState
    event: Event


ChangeListener = Callable[[RunState, RunState, Event], None]

_SAMPLING_SOURCES = (RunState.stopped, RunState.paused)


def _resolve(
    state: RunState, event: Event, previous: RunState | None
) -> tuple[RunState | None, list[str]]:
    """纯函数：给定当前状态与事件，返回 ``(新状态或 None 表示幂等无操作, 非法原因列表)``。"""
    if event is Event.fault_raised:
        return (None if state is RunState.fault else RunState.fault), []
    if event is Event.fault_cleared:
        return (RunState.stopped if state is RunState.fault else None), []
    if state is RunState.fault:
        if event is Event.capture_end:
            return None, [NOT_SAMPLING]
        if event is Event.stop:
            return None, []  # 故障态允许"停止"（幂等，不清故障；路由层仍下发 STOP）
        return None, [IN_FAULT]
    if event is Event.start:
        if state in _SAMPLING_SOURCES:
            return RunState.running, []
        if state is RunState.running:
            return None, []
        return None, [SAMPLING_IN_PROGRESS]
    if event is Event.pause:
        if state is RunState.running:
            return RunState.paused, []
        if state is RunState.paused:
            return None, []
        if state is RunState.stopped:
            return None, [NOT_RUNNING]
        return None, [SAMPLING_IN_PROGRESS]
    if event is Event.stop:
        if state is RunState.stopped:
            return None, []
        return RunState.stopped, []
    if event is Event.capture_begin:
        if state in _SAMPLING_SOURCES:
            return RunState.sampling, []
        if state is RunState.running:
            return None, [ALREADY_RUNNING]
        return None, [SAMPLING_IN_PROGRESS]
    if event is Event.capture_end:
        if state is RunState.sampling:
            return previous or RunState.stopped, []
        return None, [NOT_SAMPLING]
    raise ValueError(f"未知事件: {event!r}")  # pragma: no cover


class StateMachine:
    """线程安全的运行状态机。

    Args:
        initial: 初始状态。
        on_change: 迁移成功后的回调 ``(old, new, event)``，不持有状态锁、按迁移顺序串行调用；
            可用 :meth:`subscribe` 再加。

    锁顺序：``_notify_lock`` → ``_lock``；持有 ``_lock`` 的代码不得去取 ``_notify_lock``。
    """

    def __init__(
        self,
        initial: RunState = RunState.stopped,
        *,
        on_change: ChangeListener | None = None,
    ):
        self._state = RunState(initial)
        self._previous: RunState | None = None
        self._lock = threading.RLock()  # 保护状态 / previous / history / listeners
        self._notify_lock = threading.RLock()  # 串行化"迁移 + 通知"，保证回调顺序 == 迁移顺序
        self._listeners: list[ChangeListener] = []
        self._history: deque[Transition] = deque(maxlen=HISTORY_LEN)
        if on_change is not None:
            self._listeners.append(on_change)

    # ---- 查询 ----
    @property
    def state(self) -> RunState:
        """当前状态。"""
        with self._lock:
            return self._state

    @property
    def previous(self) -> RunState | None:
        """进入 sampling 前的来源状态（capture_end 回到该状态）；未采样过为 None。"""
        with self._lock:
            return self._previous

    @property
    def history(self) -> list[Transition]:
        """最近 :data:`HISTORY_LEN` 条实际发生的迁移（幂等无操作不计）。"""
        with self._lock:
            return list(self._history)

    def can(self, event: Event | str) -> bool:
        """事件在当前状态下是否合法（幂等无操作也算合法）。"""
        ev = Event(event)
        with self._lock:
            _, reasons = _resolve(self._state, ev, self._previous)
        return not reasons

    # ---- 迁移 ----
    def transition(self, event: Event | str) -> RunState:
        """执行迁移，返回新状态；非法时抛 :class:`IllegalTransition`。

        幂等事件（如 running+start、fault+stop）返回当前状态、不记历史、不触发监听。
        监听回调在状态锁外、通知锁内按迁移顺序串行调用。
        """
        ev = Event(event)
        with self._notify_lock:
            with self._lock:
                old = self._state
                new, reasons = _resolve(old, ev, self._previous)
                if reasons:
                    raise IllegalTransition(old, ev, reasons)
                if new is None:
                    return old
                if ev is Event.capture_begin:
                    self._previous = old
                self._state = new
                self._history.append(Transition(time.time(), old, new, ev))
                listeners = list(self._listeners)
            log.info("状态迁移", extra={"from": old.value, "to": new.value, "event": ev.value})
            for cb in listeners:
                try:
                    cb(old, new, ev)
                except Exception:
                    log.exception("状态监听回调异常", extra={"event": ev.value})
        return new

    def subscribe(self, cb: ChangeListener) -> Callable[[], None]:
        """注册迁移监听；返回取消注册的函数。"""
        with self._lock:
            self._listeners.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._listeners:
                    self._listeners.remove(cb)

        return unsubscribe

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "previous": self._previous.value if self._previous else None,
            }

    def __repr__(self) -> str:  # pragma: no cover
        return f"StateMachine(state={self.state.value})"


def check_start_preconditions(
    *,
    deployed_model_id: int | None,
    cameras_connected: bool,
    mcu_connected: bool,
) -> list[str]:
    """返回阻止 ``start`` 的原因列表（docs/03 §4），空列表表示可以启动。"""
    reasons: list[str] = []
    if deployed_model_id is None:
        reasons.append(NO_MODEL_DEPLOYED)
    if not cameras_connected:
        reasons.append(CAMERA_DISCONNECTED)
    if not mcu_connected:
        reasons.append(MCU_DISCONNECTED)
    return reasons
