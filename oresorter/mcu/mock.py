"""Mock 下位机。

- ``MockMcuLink``：进程内实现 ``McuLink``，直接解释指令负载，维护参数表 / 状态 / 编码器计数 / 喷阀队列，
  可注入 ACK 超时、NACK、掉线。开发机与大部分测试用它。
- ``MockMcuServer``（P2）：真正的 asyncio TCP server，讲 docs/04 的帧协议，用来测 ``TcpMcuLink``。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from oresorter.log import get_logger
from oresorter.mcu import protocol as P
from oresorter.mcu.link import (
    Ack,
    EjectItem,
    McuDisconnected,
    McuLink,
    McuNack,
    McuStatus,
    McuTimeout,
)

if TYPE_CHECKING:
    from oresorter.config import AppConfig

log = get_logger(__name__)


class McuModel:
    """下位机内部状态模型（协议无关），MockMcuLink 与 MockMcuServer 共用。"""

    def __init__(self, pulses_per_s: float = 9000.0, queue_capacity: int = 2000):
        self.lock = threading.RLock()
        self.params: dict[int, int] = {pid: lo for pid, (lo, _hi) in P.PARAM_LIMITS.items()}
        self.params[P.ParamId.LINES_PER_PULSE] = 1
        self.params[P.ParamId.NOZZLE_COUNT] = 64
        self.state = P.McuState.STOPPED
        self.pulses_per_s = pulses_per_s
        self.queue_capacity = queue_capacity
        self._enc_base = 0
        self._enc_t0: float | None = None
        self.eject_queue: deque[tuple[int, int, int, int]] = deque()
        self.fired: list[tuple[int, int, int, int]] = []
        self.eject_rx = 0
        self.eject_done = 0
        self.eject_dropped = 0
        self.faults = 0
        self.valve_tests: list[tuple[str, tuple[int, ...]]] = []
        self.clean_count = 0
        self.reset_link_count = 0
        self.sync_count = 0
        self.last_host_ms = 0

    # ---- 编码器 ----
    @property
    def encoder_count(self) -> int:
        with self.lock:
            if self._enc_t0 is None:
                return self._enc_base
            return self._enc_base + int((time.monotonic() - self._enc_t0) * self.pulses_per_s)

    def _freeze_encoder(self) -> None:
        self._enc_base = self.encoder_count
        self._enc_t0 = None

    def _run_encoder(self) -> None:
        if self._enc_t0 is None:
            self._enc_t0 = time.monotonic()

    # ---- 指令处理：返回 (result, data) ----
    def handle(self, cmd: int, payload: bytes) -> tuple[int, bytes]:
        with self.lock:
            try:
                return self._handle(cmd, payload)
            except P.ProtocolError:
                return P.AckResult.UNKNOWN_CMD, b""

    def _handle(self, cmd: int, payload: bytes) -> tuple[int, bytes]:
        res = P.AckResult
        if cmd == P.Cmd.HEARTBEAT:
            if len(payload) >= 4:
                self.last_host_ms = int.from_bytes(payload[:4], "little")
            return res.OK, self.status_payload().pack()
        if cmd == P.Cmd.SYNC:
            clear = bool(payload[0] & 1) if payload else True
            if clear:
                self._enc_base = 0
                if self._enc_t0 is not None:
                    self._enc_t0 = time.monotonic()
            self.sync_count += 1
            return res.OK, b""
        if cmd == P.Cmd.RUN:
            if self.state == P.McuState.FAULT:
                return res.BAD_STATE, b""
            self.state = P.McuState.RUNNING
            self._run_encoder()
            return res.OK, b""
        if cmd == P.Cmd.PAUSE:
            if self.state not in (P.McuState.RUNNING, P.McuState.PAUSED):
                return res.BAD_STATE, b""
            self.state = P.McuState.PAUSED
            self._freeze_encoder()
            return res.OK, b""
        if cmd == P.Cmd.STOP:
            self.state = P.McuState.STOPPED
            self._freeze_encoder()
            self.eject_queue.clear()
            return res.OK, b""
        if cmd == P.Cmd.SET_PARAM:
            params = P.unpack_set_param(payload)
            if P.check_param_range(params):
                return res.OUT_OF_RANGE, b""
            self.params.update(params)
            return res.OK, b""
        if cmd == P.Cmd.GET_PARAM:
            ids = P.unpack_get_param(payload)
            if any(i not in self.params for i in ids):
                return res.OUT_OF_RANGE, b""
            return res.OK, P.pack_param_values({i: self.params[i] for i in ids})
        if cmd == P.Cmd.VALVE_TEST_SINGLE:
            if self.state == P.McuState.RUNNING:
                return res.BAD_STATE, b""
            (hole,) = P.struct.unpack("<H", payload[:2])
            if not (1 <= hole <= self.params[P.ParamId.NOZZLE_COUNT]):
                return res.OUT_OF_RANGE, b""
            self.valve_tests.append(("single", (hole,)))
            return res.OK, b""
        if cmd == P.Cmd.VALVE_TEST_CYCLE:
            if self.state == P.McuState.RUNNING:
                return res.BAD_STATE, b""
            frm, to, interval = P.struct.unpack("<HHH", payload[:6])
            if not (1 <= frm <= to <= self.params[P.ParamId.NOZZLE_COUNT]):
                return res.OUT_OF_RANGE, b""
            self.state = P.McuState.VALVE_TEST
            self.valve_tests.append(("cycle", (frm, to, interval)))
            return res.OK, b""
        if cmd == P.Cmd.VALVE_TEST_STOP:
            if self.state == P.McuState.VALVE_TEST:
                self.state = P.McuState.STOPPED
            self.valve_tests.append(("stop", ()))
            return res.OK, b""
        if cmd == P.Cmd.VALVE_RESET:
            self.valve_tests.append(("reset", ()))
            return res.OK, b""
        if cmd == P.Cmd.CLEAN_NOW:
            self.clean_count += 1
            return res.OK, b""
        if cmd == P.Cmd.EJECT:
            items = P.unpack_eject(payload)
            enc = self.encoder_count
            delay = self.params[P.ParamId.DELAY]
            lpp = max(1, self.params[P.ParamId.LINES_PER_PULSE])
            for it in items:
                self.eject_rx += 1
                fire_at = it[0] * lpp + delay
                if fire_at < enc or len(self.eject_queue) >= self.queue_capacity:
                    self.eject_dropped += 1
                else:
                    self.eject_queue.append(it)
            self.settle()
            return res.OK, b""
        if cmd == P.Cmd.RESET_LINK:
            self.eject_queue.clear()
            self.reset_link_count += 1
            return res.OK, b""
        return res.UNKNOWN_CMD, b""

    def settle(self) -> None:
        """把编码器已经走过的喷阀条目标记为已执行。"""
        with self.lock:
            enc = self.encoder_count
            delay = self.params[P.ParamId.DELAY]
            lpp = max(1, self.params[P.ParamId.LINES_PER_PULSE])
            keep: deque[tuple[int, int, int, int]] = deque()
            for it in self.eject_queue:
                if it[0] * lpp + delay <= enc:
                    self.eject_done += 1
                    self.fired.append(it)
                else:
                    keep.append(it)
            self.eject_queue = keep

    def status_payload(self) -> P.StatusPayload:
        with self.lock:
            self.settle()
            return P.StatusPayload(
                state=int(self.state),
                encoder_count=self.encoder_count,
                belt_speed=int(self.pulses_per_s) if self.state == P.McuState.RUNNING else 0,
                faults=self.faults,
                queue_len=len(self.eject_queue),
                pressure=0xFFFF,
                host_ms_echo=self.last_host_ms,
            )

    def counters_payload(self) -> P.CountersPayload:
        with self.lock:
            self.settle()
            return P.CountersPayload(self.eject_rx, self.eject_done, self.eject_dropped, 1000)


class MockMcuLink(McuLink):
    """进程内 Mock 链路：``send`` 直接调用 ``McuModel.handle``，无网络。"""

    def __init__(self, cfg: AppConfig, *, pulses_per_s: float | None = None):
        super().__init__(cfg)
        pps = pulses_per_s if pulses_per_s is not None else float(cfg.cameras[0].line_rate_hz)
        self.model = McuModel(pulses_per_s=pps)
        self._connected = False
        self._lock = threading.RLock()
        self._seq = 0
        self._last_hb: float | None = None
        self._rtt_ms: float | None = None
        self._reconnects = 0
        self._ack_timeouts = 0
        self.eject_dropped_offline = 0
        self.sent_log: deque[tuple[int, bytes]] = deque(maxlen=1000)
        # 故障注入
        self.fail_acks_remaining = 0
        self.nack_next: int | None = None
        self.latency_s = 0.0

    # ---- 连接 ----
    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            self._connected = True
            self._reconnects += 1 if self._seq else 0
        self._emit("connected")

    def close(self) -> None:
        with self._lock:
            self._connected = False
        self._emit("disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    def set_offline(self, offline: bool) -> None:
        """模拟掉线/恢复。"""
        if offline:
            self.close()
        else:
            self.connect()

    # ---- 收发 ----
    def send(
        self, cmd: int, payload: bytes = b"", *, need_ack: bool = True, timeout: float = 0.2
    ) -> Ack:
        if not self._connected:
            raise McuDisconnected("Mock 下位机未连接")
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFF
            seq = self._seq
            self.sent_log.append((cmd, payload))
            if need_ack and self.fail_acks_remaining > 0:
                self.fail_acks_remaining -= 1
                self._ack_timeouts += 1
                raise McuTimeout(f"Mock 下位机未应答 cmd=0x{cmd:02X}（注入故障）")
            if need_ack and self.nack_next is not None:
                result, self.nack_next = self.nack_next, None
                self._emit("nack", cmd=cmd, result=result)
                raise McuNack(result, cmd)
        if self.latency_s > 0:
            time.sleep(self.latency_s)
        t0 = time.monotonic()
        result, data = self.model.handle(cmd, payload)
        elapsed = (time.monotonic() - t0) * 1000
        if not need_ack:
            return Ack(ok=True, result=0, seq=seq, elapsed_ms=elapsed)
        if cmd == P.Cmd.HEARTBEAT and result == P.AckResult.OK:
            self._last_hb = time.monotonic()
            self._rtt_ms = elapsed
        if result != P.AckResult.OK:
            self._emit("nack", cmd=cmd, result=result)
            raise McuNack(result, cmd)
        return Ack(ok=True, result=result, data=data, seq=seq, elapsed_ms=elapsed)

    def send_eject(self, batch: list[EjectItem]) -> None:
        if not batch:
            return
        if not self._connected:
            self.eject_dropped_offline += len(batch)
            return
        items = [b.as_tuple() for b in batch]
        for i in range(0, len(items), P.EJECT_MAX_ITEMS):
            self.send(P.Cmd.EJECT, P.pack_eject(items[i : i + P.EJECT_MAX_ITEMS]), need_ack=False)

    # ---- 状态 ----
    @property
    def status(self) -> McuStatus:
        sp = self.model.status_payload()
        hb_ms = (time.monotonic() - self._last_hb) * 1000 if self._last_hb is not None else None
        return McuStatus(
            connected=self._connected,
            state=sp.state_text if self._connected else "unknown",
            encoder_count=sp.encoder_count,
            last_heartbeat_ms=hb_ms,
            rtt_ms=self._rtt_ms,
            faults=[
                P.EVENT_TEXT.get(P.EventCode(c), f"0x{c:04X}")
                for c in sp.fault_codes()
                if c in P.EventCode._value2member_map_
            ],
            queue_len=sp.queue_len,
            belt_speed=sp.belt_speed,
            pressure=None if sp.pressure == 0xFFFF else sp.pressure,
            counters=self.model.counters_payload().to_dict(),
            reconnects=self._reconnects,
            ack_timeouts=self._ack_timeouts,
        )

    # ---- 测试辅助 ----
    def inject_ack_timeouts(self, n: int) -> None:
        with self._lock:
            self.fail_acks_remaining = int(n)

    def inject_nack(self, result: int) -> None:
        with self._lock:
            self.nack_next = int(result)

    def inject_fault(self, code: int, active: bool = True) -> None:
        with self.model.lock:
            if active:
                self.model.faults |= 1 << int(code)
            else:
                self.model.faults &= ~(1 << int(code))
        self._emit("fault", code=int(code), active=active)

    def debug_state(self) -> dict[str, Any]:
        m = self.model
        return {
            "state": P.MCU_STATE_TEXT[m.state],
            "params": dict(m.params),
            "encoder": m.encoder_count,
            "queue_len": len(m.eject_queue),
            "fired": len(m.fired),
            "counters": m.counters_payload().to_dict(),
        }
