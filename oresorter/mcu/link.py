"""下位机链路抽象 ``McuLink`` 与数据类型（docs/02 §3.6），以及 TCP 实现 ``TcpMcuLink``。

- ``McuLink``：接口、类型与基于 ``send`` 的高层便捷方法，Mock 与 TCP 实现共用；
- ``TcpMcuLink``：三个后台线程（连接管理 / 读取 / 心跳），指数退避重连、按 SEQ 匹配 ACK、
  超时重发同一 SEQ、心跳看门狗、链路故障事件（docs/01 §7.2、docs/04 §1 §3.3 §8）。
"""

from __future__ import annotations

import contextlib
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from oresorter.log import get_logger
from oresorter.mcu import protocol as P

if TYPE_CHECKING:
    from oresorter.config import AppConfig

log = get_logger(__name__)


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
        self, cmd: int, payload: bytes = b"", *, need_ack: bool = True, timeout: float | None = None
    ) -> Ack:
        """发送一帧。``need_ack`` 时阻塞等待 ACK（含重发），超时抛 McuTimeout，被拒抛 McuNack，
        未连接抛 McuDisconnected。``timeout=None`` 用实现自己的默认 ACK 超时（TCP 为 ``ack_timeout``）。
        ``need_ack=False`` 立即返回 ``Ack(ok=True, result=0)``。"""

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
                log.exception("下位机事件回调异常", extra={"kind": kind})

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


# --------------------------------------------------------------------------------------
# TCP 实现
# --------------------------------------------------------------------------------------

_STATUS_SIZE = struct.calcsize(P.StatusPayload.FMT)
_RECV_CHUNK = 65536
_THREAD_JOIN_S = 2.0
_RTT_SANE_MS = 60_000


class _Waiter:
    """一次需 ACK 的请求的等待者：按 SEQ 登记，读线程收到应答后填结果并唤醒。"""

    __slots__ = ("cmd", "data", "disconnected", "event", "result", "seq")

    def __init__(self, seq: int, cmd: int):
        self.seq = seq
        self.cmd = cmd
        self.event = threading.Event()
        self.result: int = -1
        self.data: bytes = b""
        self.disconnected = False


class _Conn:
    """一次 TCP 连接的句柄（socket + 建立时刻），用身份比较判断是否仍是当前连接。"""

    __slots__ = ("sock", "t_connected")

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.t_connected = time.monotonic()


def _close_socket(sock: socket.socket) -> None:
    """先 shutdown 再 close：shutdown 能可靠唤醒阻塞在 recv 的线程。"""
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        sock.close()


class TcpMcuLink(McuLink):
    """TCP 下位机链路（算力盒作 client，下位机作 server）。

    线程模型（全部 daemon，``close()`` 在 2 s 内 join）：

    - ``mcu-conn`` 连接管理：未连接时 ``create_connection``（带超时），失败按 ``backoff_min→backoff_max``
      指数退避；连上后发 ``connected`` 事件，``auto_push_params`` 时全量下发参数 + RESET_LINK；
      之后等待断开通知，断开后等 ``backoff_min`` 再重连。
    - ``mcu-reader`` 读取：``recv``（socket 超时 ``recv_timeout`` 以便可中断）→ ``FrameParser`` →
      分发：ACK 按 SEQ 匹配等待者；STATUS 同时更新状态与心跳时刻；EVENT / COUNTERS 发事件。
      ``recv`` 返回空或异常 → 断开。
    - ``mcu-heartbeat`` 心跳：每 ``heartbeat_interval`` 发一帧 HEARTBEAT。**心跳不等 ACK、不重发**
      （下一拍自然补上；STATUS 由读线程匹配），若距上次收到 STATUS 超过 ``heartbeat_timeout`` →
      ``fault(code=MCU_HEARTBEAT_TIMEOUT)`` 并断开，交给连接管理线程重连。

    ``send``：发送锁内分配 SEQ（1..65535 回绕）并 ``sendall``；需 ACK 时等待 ``ack_timeout``，
    超时重发**同一 SEQ**（下位机按 SEQ 幂等）最多 ``ack_retries`` 次，仍无应答 → ``ack_timeouts += 1``、
    ``fault(code=MCU_ACK_TIMEOUT)``、断开重连并抛 ``McuTimeout``（docs/04 §3.3：3 次失败置链路故障）。
    等待期间链路断开 → 抛 ``McuDisconnected``。

    重连退避：连接失败按 ``backoff_min → backoff_max`` 翻倍；**连上但不稳定**（从未收到 STATUS，或存活
    不足 ``2 × heartbeat_timeout`` 就被判故障）的连接断开后同样沿用并翻倍退避，只有稳定连接断开才把退避
    重置为 ``backoff_min``——避免"连上→不应答→断开"的重连风暴刷屏 HMI。

    心跳看门狗：任何 CRC 正确的 STATUS(0x81) 帧（包括只有 result 字节的 NACK）都视为对端活着并刷新
    看门狗；只有带完整 StatusPayload 的帧才更新 ``status`` 快照。心跳被拒绝时 ``status.faults`` 含
    "心跳被拒绝: …"，链路保持连接不重连。

    事件时序约定：``connected`` 属性先于 ``disconnected`` 事件翻转为 False（标志位在锁内翻转，事件在锁外
    回调），集成者不要依赖二者同步；``close()`` 可在事件回调内调用（不会 join 当前线程）。
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        host: str | None = None,
        port: int | None = None,
        heartbeat_interval: float = 0.2,
        heartbeat_timeout: float = 1.0,
        ack_timeout: float = 0.2,
        ack_retries: int = 3,
        backoff_min: float = 0.5,
        backoff_max: float = 5.0,
        connect_timeout: float = 1.0,
        auto_push_params: bool = True,
        recv_timeout: float = 0.5,
    ):
        super().__init__(cfg)
        self.host = host or cfg.system.mcu_host
        self.port = int(port or cfg.system.mcu_port)
        self.heartbeat_interval = float(heartbeat_interval)
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.ack_timeout = float(ack_timeout)
        self.ack_retries = int(ack_retries)
        self.backoff_min = float(backoff_min)
        self.backoff_max = float(backoff_max)
        self.connect_timeout = float(connect_timeout)
        self.auto_push_params = bool(auto_push_params)
        self.recv_timeout = float(recv_timeout)

        self._lock = threading.RLock()  # 保护连接句柄、等待者表、状态缓存与计数
        self._send_lock = threading.Lock()  # 串行化 SEQ 分配 + sendall
        self._stop = threading.Event()
        self._wake_conn = threading.Event()  # 通知连接线程：当前连接已断开
        self._conn_ready = threading.Event()  # 通知读线程：有新连接
        self._threads: list[threading.Thread] = []
        self._conn: _Conn | None = None
        self._seq = 0
        self._waiters: dict[int, _Waiter] = {}

        self._ever_connected = False
        self._reconnects = 0
        self._ack_timeouts = 0
        self._heartbeat_timeouts = 0
        self._late_acks = 0
        self._crc_errors = 0
        self._hb_sent = 0
        self._status_recv = 0
        self._last_status: P.StatusPayload | None = None
        self._last_status_t: float | None = None
        self._rtt_ms: float | None = None
        self._counters: dict[str, int] = {}
        self._event_faults: dict[int, str] = {}  # EVENT level=2 的文字（重连后清空）
        self._hb_fault = False
        self._hb_nack_result = int(P.AckResult.OK)  # 最近一次心跳应答的 result（≠0 表示被拒绝）
        self._backoff = self.backoff_min  # 最近一次断开/失败后采用的退避秒数（排障用）
        self.eject_dropped_offline = 0
        self.eject_frames_sent = 0
        self.eject_items_sent = 0

    # ---- 生命周期 ----
    def connect(self) -> None:
        """启动后台线程（不阻塞等待连上）。可在 ``close()`` 之后再次调用。"""
        with self._lock:
            if any(t.is_alive() for t in self._threads):
                return
            # 每次 connect 新建 stop 事件并传给线程：上一代线程（例如回调内 close() 时尚未退出的
            # 读线程）持有的旧事件保持置位，不会被这次 connect 复活去抢同一条连接。
            stop = threading.Event()
            self._stop = stop
            self._wake_conn.clear()
            self._conn_ready.clear()
            self._threads = [
                threading.Thread(
                    target=self._conn_loop, args=(stop,), name="mcu-conn", daemon=True
                ),
                threading.Thread(
                    target=self._reader_loop, args=(stop,), name="mcu-reader", daemon=True
                ),
                threading.Thread(
                    target=self._heartbeat_loop, args=(stop,), name="mcu-heartbeat", daemon=True
                ),
            ]
            for t in self._threads:
                t.start()
        log.info("下位机链路启动", extra={"host": self.host, "port": self.port})

    def close(self) -> None:
        """停止线程、关闭 socket、唤醒所有等待者（抛 McuDisconnected）、发 ``disconnected`` 事件。可重复调用。

        可在事件回调内调用（回调运行在链路自己的线程上）：此时跳过 join 当前线程，当前线程在回调返回后
        自行退出；其余线程仍在 2 s 内 join 完。
        """
        # 先置 stop 再读连接句柄：连接线程装配新连接与这里读取都在 _lock 内，且装配前会检查 stop，
        # 因此要么这里读到它并关掉，要么连接线程看到 stop 直接关掉新 socket，不会漏掉一条连接。
        self._stop.set()
        with self._lock:
            threads, self._threads = self._threads, []
            conn = self._conn
            self._wake_conn.set()
        if conn is not None:
            self._drop_conn(conn, "close", emit=False)
        me = threading.current_thread()
        for t in threads:
            if t is me:
                continue  # 回调内 close()：不能 join 自己，当前线程随 stop 自行退出
            t.join(_THREAD_JOIN_S)
            if t.is_alive():
                log.error("下位机线程未能在时限内退出", extra={"thread": t.name})
        # 兜底：线程都退出后再查一次，确保不残留打开的 socket
        with self._lock:
            conn = self._conn
        if conn is not None:
            self._drop_conn(conn, "close", emit=False)
        if threads:
            log.info("下位机链路已关闭", extra={"host": self.host, "port": self.port})
            self._emit("disconnected", reason="close")

    @property
    def connected(self) -> bool:
        return self._conn is not None

    # ---- 连接管理线程 ----
    def _open_socket(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self._tune_socket(sock)
        sock.settimeout(self.recv_timeout)
        return sock

    @staticmethod
    def _tune_socket(sock: socket.socket) -> None:
        """TCP_NODELAY + 内核 keepalive（idle 2 s / interval 1 s / count 3，按平台可用项设置）。"""
        opts: list[tuple[int, int, int]] = [
            (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        ]
        if hasattr(socket, "TCP_KEEPIDLE"):  # Linux
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 2))
        elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 2))
        if hasattr(socket, "TCP_KEEPINTVL"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1))
        if hasattr(socket, "TCP_KEEPCNT"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
        for level, name, value in opts:
            try:
                sock.setsockopt(level, name, value)
            except OSError as exc:  # 个别平台不支持某项，不致命
                log.debug("setsockopt 失败", extra={"opt": name, "err": str(exc)})

    def _conn_loop(self, stop: threading.Event) -> None:
        backoff = self.backoff_min
        while not stop.is_set():
            try:
                sock = self._open_socket()
            except OSError as exc:
                log.warning(
                    "下位机连接失败，退避重连",
                    extra={
                        "host": self.host,
                        "port": self.port,
                        "err": str(exc),
                        "backoff_s": round(backoff, 2),
                    },
                )
                with self._lock:
                    self._backoff = backoff
                if stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self.backoff_max)
                continue
            conn = _Conn(sock)
            with self._lock:
                if stop.is_set():
                    _close_socket(sock)
                    break
                self._conn = conn
                self._event_faults.clear()
                self._hb_fault = False
                self._hb_nack_result = int(P.AckResult.OK)
                self._last_status = None
                self._last_status_t = None
                self._rtt_ms = None
                self._wake_conn.clear()
                reconnect = self._ever_connected
                self._ever_connected = True
                if reconnect:
                    self._reconnects += 1
                reconnects = self._reconnects
                status_before = self._status_recv
                self._conn_ready.set()
            log.info(
                "下位机已连接" if not reconnect else "下位机已重连",
                extra={"host": self.host, "port": self.port, "reconnects": reconnects},
            )
            self._emit("connected", host=self.host, port=self.port, reconnects=reconnects)
            if self.auto_push_params:
                self._initial_push()
            # 等待断开通知（或 stop）
            while not stop.is_set():
                self._wake_conn.wait(1.0)
                with self._lock:
                    if self._conn is not conn:
                        break
                    self._wake_conn.clear()  # 误唤醒
            if stop.is_set():
                break
            # 只有"稳定"连接（收到过 STATUS 且存活 ≥ 2×heartbeat_timeout）断开才重置退避；
            # 连上即被判故障（不应答 / 立刻被断）的连接视同连接失败：沿用上一轮退避并继续翻倍。
            lived = time.monotonic() - conn.t_connected
            with self._lock:
                saw_status = self._status_recv > status_before
            stable = saw_status and lived >= 2 * self.heartbeat_timeout
            if stable:
                backoff = self.backoff_min
            else:
                log.warning(
                    "下位机连接不稳定，退避重连",
                    extra={
                        "lived_ms": round(lived * 1000, 1),
                        "saw_status": saw_status,
                        "backoff_s": round(backoff, 2),
                    },
                )
            with self._lock:
                self._backoff = backoff
            if stop.wait(backoff):
                break
            if not stable:
                backoff = min(backoff * 2, self.backoff_max)

    def _initial_push(self) -> None:
        """连接建立后：全量下发参数 + RESET_LINK（docs/04 §8）。失败只记日志，由上层决定是否重试。"""
        try:
            ack = self.push_all_params()
            log.info("下位机参数已全量下发", extra={"retries": ack.retries})
        except McuError as exc:
            log.warning("下位机参数全量下发失败", extra={"err": str(exc)})
            return
        try:
            self.reset_link()
            log.info("下位机 RESET_LINK 完成")
        except McuError as exc:
            log.warning("下位机 RESET_LINK 失败", extra={"err": str(exc)})

    def _drop_conn(self, conn: _Conn, reason: str, *, emit: bool = True) -> bool:
        """断开指定连接（若仍是当前连接）：关 socket、唤醒等待者、通知连接线程重连。

        ``connected`` 在锁内立刻变 False，``disconnected`` 事件在锁外回调时才发出（回调不能在锁内执行），
        二者之间有微小间隙，调用方不要假设同步。
        """
        with self._lock:
            if self._conn is not conn:
                return False
            self._conn = None
            self._conn_ready.clear()
            waiters = list(self._waiters.values())
            self._waiters.clear()
            self._last_status_t = None
            self._rtt_ms = None
            self._wake_conn.set()
        for w in waiters:
            w.disconnected = True
            w.event.set()
        _close_socket(conn.sock)
        log.warning(
            "下位机链路断开",
            extra={
                "host": self.host,
                "port": self.port,
                "reason": reason,
                "pending_acks": len(waiters),
            },
        )
        if emit:
            self._emit("disconnected", reason=reason)
        return True

    # ---- 读线程 ----
    def _reader_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self._conn_ready.wait(0.5):
                continue
            conn = self._conn
            if conn is None:
                continue
            parser = P.FrameParser()
            sock = conn.sock
            crc_seen = 0
            while not stop.is_set():
                try:
                    data = sock.recv(_RECV_CHUNK)
                except TimeoutError:
                    if self._conn is not conn:
                        break
                    continue
                except OSError as exc:
                    self._drop_conn(conn, f"recv 错误: {exc}")
                    break
                if not data:
                    self._drop_conn(conn, "对端关闭连接")
                    break
                for frame in parser.feed(data):
                    self._dispatch(frame)
                if parser.crc_errors != crc_seen:
                    with self._lock:
                        self._crc_errors += parser.crc_errors - crc_seen
                    log.warning(
                        "下位机帧 CRC 错误",
                        extra={"crc_errors": parser.crc_errors, "resyncs": parser.resyncs},
                    )
                    crc_seen = parser.crc_errors

    def _dispatch(self, frame: P.McuFrame) -> None:
        if frame.is_report:
            if frame.cmd == P.Report.EVENT:
                self._on_event(frame)
            elif frame.cmd == P.Report.COUNTERS:
                self._on_counters(frame)
            else:
                log.debug("未知上报帧", extra={"cmd": f"0x{frame.cmd:02X}", "seq": frame.seq})
            return
        if frame.is_ack:
            if frame.cmd == P.Report.STATUS:
                self._on_status(frame)
            elif frame.payload:
                self._resolve_waiter(frame.seq, frame.cmd, frame.payload[0], frame.payload[1:])
            else:
                log.warning("ACK 帧无 result 字节", extra={"cmd": f"0x{frame.cmd:02X}"})
            return
        log.debug("忽略非应答/非上报帧", extra={"cmd": f"0x{frame.cmd:02X}", "seq": frame.seq})

    def _on_status(self, frame: P.McuFrame) -> None:
        """STATUS(0x81)：既是 HEARTBEAT 的 ACK（payload = result + StatusPayload），也可能是主动上报（裸 StatusPayload）。

        任何 CRC 正确的 0x81 帧（包括只有 result 字节的 NACK）都说明对端活着，先刷新心跳看门狗；
        只有带完整 StatusPayload 的帧才更新 ``last_status`` 快照与 RTT。
        """
        payload = frame.payload
        now = time.monotonic()
        with self._lock:
            if self._conn is not None:
                self._last_status_t = now
                self._hb_fault = False
        if len(payload) >= 1 + _STATUS_SIZE:
            result, data = payload[0], payload[1:]
        elif len(payload) == _STATUS_SIZE:  # 主动上报，无 result 字节
            result, data = int(P.AckResult.OK), payload
        elif payload:  # 只有 result：下位机对 HEARTBEAT 回 NACK（例如 fault 态回 DEVICE_FAULT）
            self._note_heartbeat_result(payload[0])
            self._resolve_waiter(frame.seq, frame.cmd, payload[0], b"")
            return
        else:
            log.warning("STATUS 帧为空", extra={"seq": frame.seq})
            return
        try:
            sp = P.StatusPayload.unpack(data)
        except P.ProtocolError as exc:
            log.warning("STATUS 负载解析失败", extra={"err": str(exc)})
            return
        self._note_heartbeat_result(result)
        changed_from: str | None = None
        with self._lock:
            prev = self._last_status
            self._last_status = sp
            self._status_recv += 1
            if sp.host_ms_echo:
                rtt = (self._host_ms(now) - sp.host_ms_echo) & 0xFFFFFFFF
                if rtt < _RTT_SANE_MS:
                    self._rtt_ms = float(rtt)
            if prev is None or prev.state != sp.state:
                changed_from = prev.state_text if prev is not None else None
        self._resolve_waiter(frame.seq, frame.cmd, result, data)
        if changed_from is not None or prev is None:
            self._emit(
                "status",
                state=sp.state_text,
                prev_state=changed_from,
                encoder_count=sp.encoder_count,
                queue_len=sp.queue_len,
                faults=sp.fault_codes(),
            )

    def _note_heartbeat_result(self, result: int) -> None:
        """记录心跳应答 result 的变化（被拒绝 ↔ 正常），只在变化时打日志，避免每 200 ms 刷屏。"""
        with self._lock:
            if result == self._hb_nack_result:
                return
            prev, self._hb_nack_result = self._hb_nack_result, int(result)
        if result != P.AckResult.OK:
            log.warning(
                "下位机对心跳返回错误应答（链路保持连接）",
                extra={"result": result, "text": P.ACK_RESULT_TEXT.get(result, f"result={result}")},
            )
        else:
            log.info("下位机心跳应答恢复正常", extra={"prev_result": prev})

    def _resolve_waiter(self, seq: int, cmd: int, result: int, data: bytes) -> None:
        with self._lock:
            w = self._waiters.get(seq)
            if w is not None and (w.cmd & 0x7F) != (cmd & 0x7F):
                log.warning(
                    "ACK 指令码与等待者不匹配",
                    extra={"seq": seq, "cmd": f"0x{cmd:02X}", "expect": f"0x{w.cmd:02X}"},
                )
                w = None
            if w is not None:
                del self._waiters[seq]
            elif cmd != P.Report.STATUS:
                self._late_acks += 1  # 重发超时后迟到的 ACK，或下位机多发
        if w is not None:
            w.result = result
            w.data = data
            w.event.set()

    def _on_event(self, frame: P.McuFrame) -> None:
        try:
            ev = P.EventPayload.unpack(frame.payload)
        except P.ProtocolError as exc:
            log.warning("EVENT 负载解析失败", extra={"err": str(exc)})
            return
        if ev.level >= 2:
            with self._lock:
                self._event_faults[ev.code] = ev.text
            log.warning(
                "下位机故障事件",
                extra={"code": f"0x{ev.code:04X}", "text": ev.text, "detail": ev.detail},
            )
        else:
            log.info(
                "下位机事件",
                extra={
                    "code": f"0x{ev.code:04X}",
                    "level": ev.level,
                    "text": ev.text,
                    "detail": ev.detail,
                },
            )
        self._emit("event", code=ev.code, level=ev.level, detail=ev.detail, text=ev.text)

    def _on_counters(self, frame: P.McuFrame) -> None:
        try:
            cp = P.CountersPayload.unpack(frame.payload)
        except P.ProtocolError as exc:
            log.warning("COUNTERS 负载解析失败", extra={"err": str(exc)})
            return
        d = cp.to_dict()
        with self._lock:
            self._counters = d
        self._emit("counters", **d)

    # ---- 心跳线程 ----
    @staticmethod
    def _host_ms(now: float | None = None) -> int:
        return int((time.monotonic() if now is None else now) * 1000) & 0xFFFFFFFF

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_interval):
            conn = self._conn
            if conn is None:
                continue
            now = time.monotonic()
            with self._lock:
                last = self._last_status_t if self._last_status_t is not None else conn.t_connected
            age = now - last
            if age > self.heartbeat_timeout:
                with self._lock:
                    self._hb_fault = True
                    self._heartbeat_timeouts += 1
                log.warning(
                    "心跳超时，置链路故障并重连",
                    extra={"age_ms": round(age * 1000, 1), "timeout_s": self.heartbeat_timeout},
                )
                self._emit("fault", code="MCU_HEARTBEAT_TIMEOUT", age_ms=round(age * 1000, 1))
                self._drop_conn(conn, "心跳超时")
                continue
            try:
                self._request(
                    P.Cmd.HEARTBEAT,
                    P.pack_heartbeat(self._host_ms(now)),
                    need_ack=False,
                    timeout=self.ack_timeout,
                    retries=0,
                    fault_on_timeout=False,
                )
                with self._lock:
                    self._hb_sent += 1
            except McuError as exc:
                log.debug("心跳发送失败", extra={"err": str(exc)})

    # ---- 收发 ----
    def _alloc_seq(self) -> int:
        """1..65535 回绕（0 不用，便于区分"未分配"）。调用方须持有 ``_send_lock``。"""
        self._seq = self._seq % 0xFFFF + 1
        return self._seq

    def _sendall(self, conn: _Conn, data: bytes) -> None:
        try:
            conn.sock.sendall(data)
        except OSError as exc:
            self._drop_conn(conn, f"send 错误: {exc}")
            raise McuDisconnected(f"下位机发送失败: {exc}") from exc

    def send(
        self, cmd: int, payload: bytes = b"", *, need_ack: bool = True, timeout: float | None = None
    ) -> Ack:
        """发送一帧；``timeout`` 为 None 时用 ``ack_timeout``。详见类文档。"""
        return self._request(
            cmd,
            payload,
            need_ack=need_ack,
            timeout=self.ack_timeout if timeout is None else float(timeout),
            retries=self.ack_retries,
            fault_on_timeout=True,
        )

    def _request(
        self,
        cmd: int,
        payload: bytes,
        *,
        need_ack: bool,
        timeout: float,
        retries: int,
        fault_on_timeout: bool,
    ) -> Ack:
        t0 = time.monotonic()
        waiter: _Waiter | None = None
        with self._send_lock:
            seq = self._alloc_seq()
            data = P.encode(cmd, payload, seq=seq)
            # "取当前连接"与"登记等待者"在同一个 _lock 临界区内：_drop_conn 清空等待者表也在 _lock 内，
            # 因此等待者要么被 _drop_conn 标记为 disconnected，要么这里直接看到未连接，不会漏标。
            with self._lock:
                conn = self._conn
                if conn is None:
                    raise McuDisconnected(f"下位机未连接 ({self.host}:{self.port})")
                if need_ack:
                    waiter = _Waiter(seq, cmd)
                    self._waiters[seq] = waiter
            try:
                self._sendall(conn, data)
            except McuDisconnected:
                if waiter is not None:
                    with self._lock:
                        self._waiters.pop(seq, None)
                raise
        if waiter is None:
            return Ack(ok=True, result=0, seq=seq, elapsed_ms=(time.monotonic() - t0) * 1000)

        attempt = 0
        timed_out = False
        try:
            while True:
                if waiter.event.wait(timeout):
                    break
                if attempt >= retries:
                    timed_out = True
                    break
                attempt += 1
                log.warning(
                    "ACK 超时，重发同一 SEQ",
                    extra={"cmd": f"0x{cmd:02X}", "seq": seq, "attempt": attempt},
                )
                with self._send_lock:
                    self._sendall(conn, data)  # 失败直接抛 McuDisconnected（_sendall 已断开连接）
        finally:
            with self._lock:
                self._waiters.pop(seq, None)
        elapsed = (time.monotonic() - t0) * 1000
        if waiter.disconnected or (not timed_out and not waiter.event.is_set()):
            # 后一种情形理论上不可达（等待者未被唤醒也未超时），兜底按断开处理，绝不能误报成 NACK
            raise McuDisconnected(f"等待 ACK 期间链路断开 cmd=0x{cmd:02X} seq={seq}")
        if timed_out:
            with self._lock:
                self._ack_timeouts += 1
            log.warning(
                "ACK 超时，重发已达上限",
                extra={"cmd": f"0x{cmd:02X}", "seq": seq, "retries": attempt},
            )
            if fault_on_timeout:
                self._emit("fault", code="MCU_ACK_TIMEOUT", cmd=cmd, seq=seq, retries=attempt)
                self._drop_conn(conn, f"ACK 超时 cmd=0x{cmd:02X}")
            raise McuTimeout(f"下位机未应答 cmd=0x{cmd:02X} seq={seq}（重发 {attempt} 次）")
        if waiter.result != P.AckResult.OK:
            self._emit("nack", cmd=cmd, result=waiter.result, seq=seq)
            raise McuNack(waiter.result, cmd)
        return Ack(
            ok=True,
            result=waiter.result,
            data=waiter.data,
            seq=seq,
            elapsed_ms=elapsed,
            retries=attempt,
        )

    def send_eject(self, batch: list[EjectItem]) -> None:
        """喷阀批：按 100 条拆帧，发送锁内一次 ``sendall``；不等 ACK；未连接时计 ``eject_dropped_offline``。"""
        if not batch:
            return
        items = [b.as_tuple() for b in batch]
        with self._send_lock:
            with self._lock:
                conn = self._conn
            if conn is None:
                self.eject_dropped_offline += len(batch)
                return
            frames = []
            for i in range(0, len(items), P.EJECT_MAX_ITEMS):
                chunk = items[i : i + P.EJECT_MAX_ITEMS]
                frames.append(P.encode(P.Cmd.EJECT, P.pack_eject(chunk), seq=self._alloc_seq()))
            try:
                self._sendall(conn, b"".join(frames))
            except McuDisconnected:
                self.eject_dropped_offline += len(batch)
                return
            self.eject_frames_sent += len(frames)
            self.eject_items_sent += len(items)

    # ---- 状态 ----
    @property
    def status(self) -> McuStatus:
        now = time.monotonic()
        with self._lock:
            connected = self._conn is not None
            sp = self._last_status
            last_t = self._last_status_t
            faults: list[str] = []
            if sp is not None:
                for c in sp.fault_codes():
                    if c in P.EventCode._value2member_map_:
                        faults.append(P.EVENT_TEXT[P.EventCode(c)])
            for txt in self._event_faults.values():
                if txt not in faults:
                    faults.append(txt)
            if self._hb_fault:
                faults.append("心跳超时")
            if self._hb_nack_result != P.AckResult.OK:
                text = P.ACK_RESULT_TEXT.get(self._hb_nack_result, f"result={self._hb_nack_result}")
                faults.append(f"心跳被拒绝: {text}")
            return McuStatus(
                connected=connected,
                state=sp.state_text if (connected and sp is not None) else "unknown",
                encoder_count=sp.encoder_count if sp is not None else 0,
                last_heartbeat_ms=(now - last_t) * 1000 if (connected and last_t) else None,
                rtt_ms=self._rtt_ms,
                faults=faults,
                queue_len=sp.queue_len if sp is not None else 0,
                belt_speed=sp.belt_speed if sp is not None else 0,
                pressure=None if (sp is None or sp.pressure == 0xFFFF) else sp.pressure,
                counters=dict(self._counters),
                reconnects=self._reconnects,
                ack_timeouts=self._ack_timeouts,
            )

    def debug_counters(self) -> dict[str, Any]:
        """链路内部计数（排障用）；``backoff_s`` 为最近一次断开/连接失败后采用的退避秒数。"""
        with self._lock:
            return {
                "reconnects": self._reconnects,
                "ack_timeouts": self._ack_timeouts,
                "heartbeat_timeouts": self._heartbeat_timeouts,
                "heartbeat_nack_result": self._hb_nack_result,
                "backoff_s": self._backoff,
                "late_acks": self._late_acks,
                "crc_errors": self._crc_errors,
                "hb_sent": self._hb_sent,
                "status_recv": self._status_recv,
                "eject_frames_sent": self.eject_frames_sent,
                "eject_items_sent": self.eject_items_sent,
                "eject_dropped_offline": self.eject_dropped_offline,
            }
