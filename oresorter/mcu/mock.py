"""Mock 下位机。

- ``MockMcuLink``：进程内实现 ``McuLink``，直接解释指令负载，维护参数表 / 状态 / 编码器计数 / 喷阀队列，
  可注入 ACK 超时、NACK、掉线。开发机与大部分测试用它。
- ``MockMcuServer``：真正的 TCP server（threading），讲 docs/04 的帧协议，复用 ``McuModel``，
  带故障注入（丢 ACK、延迟 ACK、静默、CRC 错误、主动断线、推送 EVENT），用来测 ``TcpMcuLink``。
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections import OrderedDict, deque
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


# --------------------------------------------------------------------------------------
# TCP Mock 下位机（讲帧协议）
# --------------------------------------------------------------------------------------

_ACK_CACHE_SIZE = 64
_SERVER_RECV_CHUNK = 65536
_SERVER_SOCK_TIMEOUT = 0.5
_SERVER_JOIN_S = 2.0


class _ClientConn:
    """一条客户端连接：socket、帧解析器、最近 64 个 (seq → ACK 字节) 幂等缓存。"""

    def __init__(self, sock: socket.socket, addr: tuple[Any, ...], index: int):
        self.sock = sock
        self.addr = addr
        self.index = index
        self.parser = P.FrameParser()
        self.ack_cache: OrderedDict[int, bytes] = OrderedDict()
        self.send_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.closed = False

    def remember_ack(self, seq: int, ack: bytes) -> None:
        self.ack_cache[seq] = ack
        while len(self.ack_cache) > _ACK_CACHE_SIZE:
            self.ack_cache.popitem(last=False)

    def close(self) -> None:
        self.closed = True
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.sock.close()


class MockMcuServer:
    """讲 docs/04 帧协议的 TCP Mock 下位机（threading）。

    - 每个连接一个线程：``FrameParser.feed`` → 对每帧：SEQ 在最近 64 个已处理缓存里 → 只重发缓存的 ACK
      （幂等，不重复执行）；否则 ``model.handle`` → ``encode_ack``（HEARTBEAT 的应答即 STATUS 0x81，
      data = StatusPayload）。EJECT 与广播地址（0xFF）不回 ACK。
    - 每 ``counters_interval`` 秒向所有客户端推 COUNTERS（0xC1，服务端 seq 自增）。
    - 故障注入（线程安全）：``drop_acks(n)``、``ack_delay_s``、``silent``、``corrupt_next_ack()``、
      ``disconnect_clients()``、``send_event(code, level, detail)``、``heartbeat_result``（≠0 时
      HEARTBEAT 只回 1 字节 result 的 NACK，不带 StatusPayload，用于测心跳看门狗对 NACK 的容忍）。

    注意：``drop_acks`` **不作用于 HEARTBEAT**（否则后台心跳会不确定地消耗计数，测试无法断言重发次数）；
    心跳丢失请用 ``silent``。``silent`` 期间指令仍被执行，只是不回应，COUNTERS 推送也暂停。
    """

    def __init__(
        self,
        model: McuModel | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        counters_interval: float = 1.0,
    ):
        self.model = model if model is not None else McuModel()
        self.host = host
        self._port_requested = int(port)
        self.port = 0
        self.counters_interval = float(counters_interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._srv_sock: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._counters_thread: threading.Thread | None = None
        self._clients: dict[int, _ClientConn] = {}
        self._client_index = 0
        self._push_seq = 0
        self.received: deque[tuple[int, int, bytes]] = deque(maxlen=1000)
        self.duplicate_frames = 0
        self.accepted_total = 0
        # 故障注入
        self._drop_acks_remaining = 0
        self.ack_delay_s = 0.0
        self.silent = False
        self.heartbeat_result = int(P.AckResult.OK)
        self._corrupt_next_ack = False

    # ---- 生命周期 ----
    def start(self) -> None:
        """绑定端口并启动 accept / counters 线程。``port`` 属性为实际绑定端口。"""
        if self._srv_sock is not None:
            return
        self._stop.clear()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self._port_requested))
        srv.listen(8)
        srv.settimeout(_SERVER_SOCK_TIMEOUT)
        self._srv_sock = srv
        self.port = srv.getsockname()[1]
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="mock-mcu-accept", daemon=True
        )
        self._counters_thread = threading.Thread(
            target=self._counters_loop, name="mock-mcu-counters", daemon=True
        )
        self._accept_thread.start()
        self._counters_thread.start()
        log.info("Mock 下位机 TCP server 启动", extra={"host": self.host, "port": self.port})

    def stop(self) -> None:
        """关闭监听、断开所有客户端并 join 全部线程（各 ≤ 2 s）。可重复调用。"""
        self._stop.set()
        srv, self._srv_sock = self._srv_sock, None
        if srv is not None:
            with contextlib.suppress(OSError):
                srv.close()
        clients = self.disconnect_clients()
        threads = [t for t in (self._accept_thread, self._counters_thread) if t is not None]
        threads += [c.thread for c in clients if c.thread is not None]
        self._accept_thread = None
        self._counters_thread = None
        for t in threads:
            t.join(_SERVER_JOIN_S)
            if t.is_alive():
                log.error("Mock 下位机线程未能在时限内退出", extra={"thread": t.name})
        if srv is not None:
            log.info("Mock 下位机 TCP server 已停止", extra={"port": self.port})

    @property
    def clients(self) -> int:
        """当前连接数。"""
        with self._lock:
            return len(self._clients)

    def recent(self, cmd: int | None = None) -> list[tuple[int, int, bytes]]:
        """``received`` 的线程安全快照，可按指令码过滤。"""
        with self._lock:
            items = list(self.received)
        return items if cmd is None else [r for r in items if r[0] == cmd]

    # ---- 故障注入 ----
    def drop_acks(self, n: int) -> None:
        """接下来 n 个需 ACK 的帧（HEARTBEAT 除外）不回应但仍执行，用于测重发与幂等。"""
        with self._lock:
            self._drop_acks_remaining = int(n)

    def corrupt_next_ack(self) -> None:
        """下一帧 ACK 的 CRC 故意写错（客户端应丢弃并重发）。"""
        with self._lock:
            self._corrupt_next_ack = True

    def disconnect_clients(self) -> list[_ClientConn]:
        """服务器主动断开所有客户端连接；返回被断开的连接对象（供 stop 时 join 线程）。"""
        with self._lock:
            clients = list(self._clients.values())
        for c in clients:
            c.close()
        return clients

    def send_event(self, code: int, level: int, detail: int = 0) -> None:
        """向所有客户端推送 EVENT 帧（0xC0）。"""
        frame = P.encode(
            P.Report.EVENT, P.EventPayload(code, level, detail).pack(), seq=self._next_push_seq()
        )
        self._broadcast(frame)

    # ---- 内部 ----
    def _next_push_seq(self) -> int:
        with self._lock:
            self._push_seq = self._push_seq % 0xFFFF + 1
            return self._push_seq

    def _accept_loop(self) -> None:
        srv = self._srv_sock
        while not self._stop.is_set() and srv is not None:
            try:
                sock, addr = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            sock.settimeout(_SERVER_SOCK_TIMEOUT)
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._client_index += 1
                idx = self._client_index
                client = _ClientConn(sock, addr, idx)
                self._clients[idx] = client
                self.accepted_total += 1
            client.thread = threading.Thread(
                target=self._serve_client,
                args=(client,),
                name=f"mock-mcu-client-{idx}",
                daemon=True,
            )
            client.thread.start()
            log.info("Mock 下位机：客户端接入", extra={"client": idx, "peer": str(addr)})

    def _serve_client(self, c: _ClientConn) -> None:
        try:
            while not self._stop.is_set() and not c.closed:
                try:
                    data = c.sock.recv(_SERVER_RECV_CHUNK)
                except TimeoutError:
                    continue
                except OSError:
                    break
                if not data:
                    break
                for frame in c.parser.feed(data):
                    self._handle_frame(c, frame)
        finally:
            c.close()
            with self._lock:
                self._clients.pop(c.index, None)
            log.info("Mock 下位机：客户端断开", extra={"client": c.index})

    def _handle_frame(self, c: _ClientConn, fr: P.McuFrame) -> None:
        with self._lock:
            self.received.append((fr.cmd, fr.seq, fr.payload))
        if fr.cmd == P.Cmd.EJECT or fr.is_broadcast:
            self.model.handle(fr.cmd, fr.payload)  # 不回 ACK
            return
        ack = c.ack_cache.get(fr.seq)
        if ack is None:
            result, data = self.model.handle(fr.cmd, fr.payload)
            with self._lock:
                hb_result = self.heartbeat_result
            if fr.cmd == P.Cmd.HEARTBEAT and hb_result != P.AckResult.OK:
                result, data = hb_result, b""  # 注入：心跳只回 result 字节
            ack = P.encode_ack(fr, result, data)
            c.remember_ack(fr.seq, ack)
        else:
            with self._lock:
                self.duplicate_frames += 1
        with self._lock:
            if self.silent:
                return
            if self._drop_acks_remaining > 0 and fr.cmd != P.Cmd.HEARTBEAT:
                self._drop_acks_remaining -= 1
                return
            corrupt, self._corrupt_next_ack = self._corrupt_next_ack, False
            delay = self.ack_delay_s
        if delay > 0 and self._stop.wait(delay):
            return
        if corrupt:
            ack = ack[:-2] + bytes([ack[-2] ^ 0xFF, ack[-1]])
        self._send_to(c, ack)

    def _send_to(self, c: _ClientConn, data: bytes) -> bool:
        with c.send_lock:
            try:
                c.sock.sendall(data)
                return True
            except OSError:
                return False

    def _broadcast(self, frame: bytes) -> None:
        with self._lock:
            clients = list(self._clients.values())
        for c in clients:
            self._send_to(c, frame)

    def _counters_loop(self) -> None:
        while not self._stop.wait(self.counters_interval):
            if self.silent:
                continue
            frame = P.encode(
                P.Report.COUNTERS, self.model.counters_payload().pack(), seq=self._next_push_seq()
            )
            self._broadcast(frame)
