"""下位机链路抽象 ``McuLink`` 与数据类型（docs/02 §3.6）。

``TcpMcuLink``（心跳线程、指数退避重连、按 SEQ 匹配 ACK、超时重发 3 次、链路故障回调）在 P2 实现；
本阶段提供接口、类型与基于 ``send`` 的高层便捷方法，Mock 与 TCP 实现共用。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from oresorter.mcu import protocol as P

if TYPE_CHECKING:
    from oresorter.config import AppConfig


class McuError(Exception):
    """下位机链路通用错误。"""


class McuTimeout(McuError):
    """等待 ACK 超时（已重发到上限）。"""


class McuDisconnected(McuError):
    """链路未连接。"""


class McuNack(McuError):
    """下位机拒绝（ACK result ≠ 0）。"""

    def __init__(self, result: int, cmd: int, detail: str = ""):
        self.result = result
        self.cmd = cmd
        text = P.ACK_RESULT_TEXT.get(result, f"result={result}")
        super().__init__(f"下位机拒绝 cmd=0x{cmd:02X}: {text}{(' ' + detail) if detail else ''}")


@dataclass(slots=True)
class Ack:
    ok: bool
    result: int
    data: bytes = b""
    seq: int = -1
    elapsed_ms: float = 0.0
    retries: int = 0

    @property
    def result_text(self) -> str:
        return P.ACK_RESULT_TEXT.get(self.result, f"result={self.result}")


@dataclass(slots=True)
class EjectItem:
    """一条喷阀指令：颗粒尾沿行号 + 全局喷头区间（1 起）。"""

    line: int
    nozzle_first: int
    nozzle_last: int
    flags: int = 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.line, self.nozzle_first, self.nozzle_last, self.flags)


@dataclass
class McuStatus:
    """GET /state 的 ``mcu`` 段。"""

    connected: bool = False
    state: str = "unknown"
    encoder_count: int = 0
    last_heartbeat_ms: float | None = None
    rtt_ms: float | None = None
    faults: list[str] = field(default_factory=list)
    queue_len: int = 0
    belt_speed: int = 0
    pressure: int | None = None
    counters: dict[str, int] = field(default_factory=dict)
    reconnects: int = 0
    ack_timeouts: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["last_heartbeat_ms"] is not None:
            d["last_heartbeat_ms"] = round(d["last_heartbeat_ms"], 1)
        if d["rtt_ms"] is not None:
            d["rtt_ms"] = round(d["rtt_ms"], 1)
        return d


@dataclass(slots=True)
class LinkEvent:
    """链路事件：``kind`` ∈ connected | disconnected | status | event | counters | fault | nack。"""

    kind: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.monotonic)


EventCallback = Callable[[LinkEvent], None]


class McuLink(ABC):
    """下位机链路。后端是唯一主机：所有控制与参数都经这里下发。

    线程安全：``send`` 可被多个线程调用（内部串行化）；``send_eject`` 由发送线程高频调用，不等 ACK。
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._listeners: list[EventCallback] = []
        self._listeners_lock = threading.Lock()

    # ---- 连接 ----
    @abstractmethod
    def connect(self) -> None:
        """建立连接（TCP 实现为启动后台连接/心跳线程，不阻塞等待成功）。"""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def connected(self) -> bool: ...

    @property
    @abstractmethod
    def status(self) -> McuStatus: ...

    # ---- 收发 ----
    @abstractmethod
    def send(
        self, cmd: int, payload: bytes = b"", *, need_ack: bool = True, timeout: float = 0.2
    ) -> Ack:
        """发送一帧。``need_ack`` 时阻塞等待 ACK（含重发），超时抛 McuTimeout，被拒抛 McuNack，
        未连接抛 McuDisconnected。``need_ack=False`` 立即返回 ``Ack(ok=True, result=0)``。"""

    @abstractmethod
    def send_eject(self, batch: list[EjectItem]) -> None:
        """发送喷阀批（不等 ACK；超过 100 条自动拆帧）。未连接时静默丢弃并计数。"""

    # ---- 事件 ----
    def on_event(self, cb: EventCallback) -> Callable[[], None]:
        with self._listeners_lock:
            self._listeners.append(cb)

        def off() -> None:
            with self._listeners_lock:
                if cb in self._listeners:
                    self._listeners.remove(cb)

        return off

    def _emit(self, kind: str, **data: Any) -> None:
        with self._listeners_lock:
            cbs = list(self._listeners)
        ev = LinkEvent(kind, data)
        for cb in cbs:
            try:
                cb(ev)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("下位机事件回调异常", extra={"kind": kind})

    # ---- 高层便捷方法（基于 send）----
    def heartbeat(self, host_ms: int | None = None, timeout: float = 0.2) -> P.StatusPayload:
        if host_ms is None:
            host_ms = int(time.monotonic() * 1000)
        ack = self.send(P.Cmd.HEARTBEAT, P.pack_heartbeat(host_ms), timeout=timeout)
        return P.StatusPayload.unpack(ack.data)

    def sync(self, clear_encoder: bool = True) -> Ack:
        return self.send(P.Cmd.SYNC, P.pack_sync(clear_encoder))

    def run(self) -> Ack:
        return self.send(P.Cmd.RUN)

    def pause(self) -> Ack:
        return self.send(P.Cmd.PAUSE)

    def stop(self) -> Ack:
        return self.send(P.Cmd.STOP)

    def set_params(self, params: dict[int, int]) -> Ack:
        """批量写参数；越界由下位机拒绝（result=3），整批不生效。"""
        if not params:
            return Ack(ok=True, result=0)
        return self.send(P.Cmd.SET_PARAM, P.pack_set_param(params))

    def get_params(self, ids: list[int]) -> dict[int, int]:
        ack = self.send(P.Cmd.GET_PARAM, P.pack_get_param(ids))
        return P.unpack_param_values(ack.data)

    def push_all_params(self) -> Ack:
        """重连后/启动时全量下发配置中的下位机参数。"""
        return self.set_params(P.params_from_config(self.cfg))

    def valve_test_single(self, hole: int) -> Ack:
        return self.send(P.Cmd.VALVE_TEST_SINGLE, P.pack_valve_test_single(hole))

    def valve_test_cycle(self, frm: int, to: int, interval_ms: int) -> Ack:
        return self.send(P.Cmd.VALVE_TEST_CYCLE, P.pack_valve_test_cycle(frm, to, interval_ms))

    def valve_test_stop(self) -> Ack:
        return self.send(P.Cmd.VALVE_TEST_STOP)

    def valve_reset(self) -> Ack:
        return self.send(P.Cmd.VALVE_RESET)

    def clean_now(self) -> Ack:
        return self.send(P.Cmd.CLEAN_NOW)

    def reset_link(self) -> Ack:
        return self.send(P.Cmd.RESET_LINK)
