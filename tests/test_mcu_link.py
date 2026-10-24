"""TcpMcuLink 对 MockMcuServer（127.0.0.1 随机端口）：连接/心跳/参数下发、ACK 重发幂等、
超时置故障重连、服务端断线重连、心跳超时、EVENT 故障、并发 ACK 匹配、COUNTERS 推送、线程不泄漏、
回调内 close、close 与连接竞争、重发失败不误报 NACK、心跳 NACK 不断链、不应答时的退避。"""

from __future__ import annotations

import itertools
import socket
import threading
import time
from collections.abc import Callable

import pytest

from oresorter.mcu import protocol as P
from oresorter.mcu.link import (
    EjectItem,
    LinkEvent,
    McuDisconnected,
    McuNack,
    McuTimeout,
    TcpMcuLink,
)
from oresorter.mcu.mock import MockMcuServer
from tests.conftest import make_cfg

FAST = {
    "heartbeat_interval": 0.05,
    "heartbeat_timeout": 0.3,
    "ack_timeout": 0.1,
    "backoff_min": 0.05,
    "backoff_max": 0.2,
    "connect_timeout": 0.5,
}


def wait_for(pred: Callable[[], bool], timeout: float = 2.0, step: float = 0.01) -> bool:
    """轮询直到 ``pred()`` 为真或超时；返回最终结果。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(step)
    return pred()


def wait_threads(baseline: int, timeout: float = 2.0) -> None:
    assert wait_for(
        lambda: threading.active_count() <= baseline, timeout
    ), f"线程泄漏: {[t.name for t in threading.enumerate()]}"
    assert threading.active_count() == baseline


@pytest.fixture
def baseline():
    n = threading.active_count()
    yield n
    wait_threads(n)


@pytest.fixture
def server(baseline):
    srv = MockMcuServer(counters_interval=0.2)
    srv.start()
    assert srv.port > 0
    yield srv
    srv.stop()


@pytest.fixture
def make_link(server, tmp_path):
    links: list[TcpMcuLink] = []

    def _make(**over) -> TcpMcuLink:
        kw = {**FAST, **over}
        lk = TcpMcuLink(make_cfg(tmp_path), host="127.0.0.1", port=server.port, **kw)
        links.append(lk)
        return lk

    yield _make
    for lk in links:
        lk.close()


@pytest.fixture
def link(make_link, server) -> TcpMcuLink:
    """已连接、且初始参数下发 + RESET_LINK 已完成的链路。"""
    lk = make_link()
    lk.connect()
    assert wait_for(lambda: lk.connected and server.model.reset_link_count >= 1, 2.0)
    return lk


# ---- 连接与心跳 ----


def test_connect_status_heartbeat_and_auto_push(make_link, server):
    lk = make_link()
    t0 = time.monotonic()
    lk.connect()
    assert wait_for(lambda: lk.connected and lk.status.state == "stopped", 1.0)
    assert wait_for(lambda: lk.status.last_heartbeat_ms is not None, 1.0)
    assert time.monotonic() - t0 < 1.0
    st = lk.status
    assert st.connected and st.last_heartbeat_ms is not None and st.last_heartbeat_ms < 500
    assert st.reconnects == 0 and st.ack_timeouts == 0 and st.faults == []
    expected = P.params_from_config(lk.cfg)
    assert wait_for(lambda: all(server.model.params.get(k) == v for k, v in expected.items()))
    assert server.model.reset_link_count >= 1
    assert wait_for(lambda: lk.status.rtt_ms is not None and lk.status.rtt_ms < 200, 1.0)
    d = st.to_dict()
    assert d["connected"] is True and d["state"] == "stopped"


def test_no_auto_push(make_link, server):
    lk = make_link(auto_push_params=False)
    lk.connect()
    assert wait_for(lambda: lk.connected and lk.status.last_heartbeat_ms is not None, 1.0)
    time.sleep(0.1)
    assert server.model.reset_link_count == 0
    assert server.recent(P.Cmd.SET_PARAM) == []


def test_params_roundtrip_nack_and_state(link: TcpMcuLink, server):
    ack = link.set_params({P.ParamId.DELAY: 123, P.ParamId.CLEAN_TIME: 7})
    assert ack.ok and ack.retries == 0 and ack.seq > 0
    assert link.get_params([P.ParamId.DELAY, P.ParamId.CLEAN_TIME]) == {
        P.ParamId.DELAY: 123,
        P.ParamId.CLEAN_TIME: 7,
    }
    nacks = []
    link.on_event(lambda e: nacks.append(e) if e.kind == "nack" else None)
    with pytest.raises(McuNack) as ei:
        link.set_params({P.ParamId.VALVE_OPEN_TIME: 300})
    assert ei.value.result == P.AckResult.OUT_OF_RANGE == 3
    assert server.model.params[P.ParamId.VALVE_OPEN_TIME] == 255, "整批不生效"
    assert len(nacks) == 1 and nacks[0].data["result"] == 3

    link.run()
    assert server.model.state == P.McuState.RUNNING
    assert wait_for(lambda: link.status.state == "running", 1.0)
    link.pause()
    assert server.model.state == P.McuState.PAUSED
    link.stop()
    assert server.model.state == P.McuState.STOPPED
    server.model._enc_base = 999
    link.sync()
    assert server.model.encoder_count == 0 and server.model.sync_count == 1


def test_send_eject_splits_frames(link: TcpMcuLink, server):
    link.run()
    link.send_eject([EjectItem(line=10**6 + i, nozzle_first=1, nozzle_last=2) for i in range(250)])
    assert wait_for(lambda: server.model.eject_rx == 250, 1.0)
    ejects = server.recent(P.Cmd.EJECT)
    assert len(ejects) == 3
    assert [len(P.unpack_eject(p)) for _, _, p in ejects] == [100, 100, 50]
    assert len({seq for _, seq, _ in ejects}) == 3
    assert link.eject_frames_sent == 3 and link.eject_items_sent == 250
    link.send_eject([])
    assert link.eject_frames_sent == 3


def test_eject_offline_is_counted(make_link):
    lk = make_link()
    lk.send_eject([EjectItem(1, 1, 1)] * 5)
    assert lk.eject_dropped_offline == 5
    with pytest.raises(McuDisconnected):
        lk.run()


# ---- ACK 重发 / 幂等 / 超时 ----


def test_drop_acks_retransmits_same_seq_and_executes_once(link: TcpMcuLink, server):
    server.drop_acks(2)
    ack = link.sync()
    assert ack.ok and ack.retries == 2
    syncs = server.recent(P.Cmd.SYNC)
    assert len(syncs) == 3, "同一指令被收到三次"
    assert len({seq for _, seq, _ in syncs}) == 1, "三次都是同一 SEQ"
    assert server.model.sync_count == 1, "下位机按 SEQ 幂等，只执行一次"
    assert server.duplicate_frames == 2
    assert link.status.ack_timeouts == 0 and link.connected


def test_corrupt_ack_is_discarded_and_retransmitted(link: TcpMcuLink, server):
    server.corrupt_next_ack()
    ack = link.run()
    assert ack.ok and ack.retries == 1
    assert server.model.state == P.McuState.RUNNING
    assert link.debug_counters()["crc_errors"] >= 1


def test_ack_delay_beyond_timeout_retries(link: TcpMcuLink, server):
    server.ack_delay_s = 0.15
    ack = link.sync()
    assert ack.ok and ack.retries >= 1 and server.model.sync_count == 1
    server.ack_delay_s = 0.0


def test_ack_timeout_sets_fault_and_reconnects(link: TcpMcuLink, server):
    events: list[LinkEvent] = []
    link.on_event(events.append)
    before = link.status.reconnects
    server.drop_acks(10)
    t0 = time.monotonic()
    with pytest.raises(McuTimeout):
        link.run()
    assert 0.3 < time.monotonic() - t0 < 1.0, "初发 + 3 次重发各等 0.1 s"
    assert link.status.ack_timeouts >= 1
    assert any(e.kind == "fault" and e.data["code"] == "MCU_ACK_TIMEOUT" for e in events)
    # 自动重连并恢复（重连期间参数下发还会吃掉剩余的丢 ACK 计数）
    assert wait_for(lambda: link.connected and server.model.reset_link_count >= 2, 3.0)
    assert link.status.reconnects > before
    assert link.run().ok
    kinds = [e.kind for e in events if e.kind in ("connected", "disconnected")]
    assert kinds[0] == "disconnected" and "connected" in kinds


# ---- 断线 / 心跳超时 ----


def test_server_disconnect_triggers_reconnect_and_repush(link: TcpMcuLink, server):
    events: list[str] = []
    link.on_event(lambda e: events.append(e.kind))
    r0 = link.status.reconnects
    c0 = server.model.reset_link_count
    server.disconnect_clients()
    assert wait_for(lambda: "connected" in events, 2.0)
    kinds = [k for k in events if k in ("connected", "disconnected")]
    assert kinds[:2] == ["disconnected", "connected"]
    assert wait_for(lambda: server.model.reset_link_count == c0 + 1, 2.0)
    assert link.status.reconnects == r0 + 1
    assert wait_for(lambda: link.status.last_heartbeat_ms is not None, 1.0)
    assert server.clients == 1


def test_heartbeat_timeout_faults_then_recovers(make_link, server):
    lk = make_link(backoff_min=0.3, backoff_max=0.3)
    lk.connect()
    assert wait_for(lambda: lk.connected and server.model.reset_link_count >= 1, 2.0)
    events: list[LinkEvent] = []
    lk.on_event(events.append)
    server.silent = True
    t0 = time.monotonic()
    assert wait_for(
        lambda: any(e.kind == "fault" and "HEARTBEAT" in str(e.data.get("code")) for e in events),
        1.0,
    )
    assert time.monotonic() - t0 <= 1.0
    assert wait_for(lambda: not lk.connected, 0.5)
    assert "心跳超时" in lk.status.faults and lk.status.state == "unknown"
    assert lk.status.last_heartbeat_ms is None
    # connected 标志在锁内翻转，disconnected 事件在锁外回调时才发出：等事件到达而不是立刻断言
    assert wait_for(lambda: any(e.kind == "disconnected" for e in events), 1.0)
    server.silent = False
    assert wait_for(
        lambda: lk.connected
        and lk.status.state == "stopped"
        and lk.status.last_heartbeat_ms is not None
        and lk.status.last_heartbeat_ms < 500,
        3.0,
    )
    assert "心跳超时" not in lk.status.faults
    assert lk.debug_counters()["heartbeat_timeouts"] >= 1


def test_connects_when_server_comes_up_later(baseline, tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    lk = TcpMcuLink(make_cfg(tmp_path), host="127.0.0.1", port=port, **FAST)
    try:
        lk.connect()
        time.sleep(0.3)
        assert not lk.connected and lk.status.state == "unknown"
        with pytest.raises(McuDisconnected):
            lk.run()
        srv = MockMcuServer(port=port)
        srv.start()
        try:
            assert wait_for(lambda: lk.connected and srv.model.reset_link_count >= 1, 2.0)
            assert lk.status.reconnects == 0, "首次连上不计重连"
        finally:
            lk.close()
            srv.stop()
    finally:
        lk.close()


# ---- 上报 ----


def test_event_level2_becomes_fault(link: TcpMcuLink, server):
    events: list[LinkEvent] = []
    link.on_event(events.append)
    server.send_event(P.EventCode.LOW_PRESSURE, 2, 0)
    assert wait_for(lambda: any(e.kind == "event" for e in events), 1.0)
    ev = next(e for e in events if e.kind == "event")
    assert ev.data == {"code": 1, "level": 2, "detail": 0, "text": "气压低"}
    assert "气压低" in link.status.faults
    server.send_event(P.EventCode.VALVE_OVERCURRENT, 1, 17)
    assert wait_for(lambda: sum(e.kind == "event" for e in events) == 2, 1.0)
    assert "喷阀驱动过流/短路" not in link.status.faults, "level 1 只是警告"
    assert link.status.faults == ["气压低"]


def test_counters_pushed(link: TcpMcuLink, server):
    counters: list[LinkEvent] = []
    link.on_event(lambda e: counters.append(e) if e.kind == "counters" else None)
    link.run()
    link.send_eject([EjectItem(10**6, 1, 1)])
    assert wait_for(lambda: link.status.counters.get("eject_rx") == 1, 1.2)
    st = link.status
    assert st.counters["period_ms"] == 1000 and "eject_done" in st.counters
    assert counters and counters[-1].data["eject_rx"] == 1


# ---- 并发 / 关闭 ----


def test_concurrent_get_params_matches_acks(link: TcpMcuLink, server):
    link.set_params({P.ParamId.DELAY: 321, P.ParamId.VALVE_HOLD: 9})
    errors: list[BaseException] = []
    results: list[dict[int, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            for _ in range(20):
                vals = link.get_params([P.ParamId.DELAY, P.ParamId.VALVE_HOLD])
                with lock:
                    results.append(vals)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, name=f"t{i}", daemon=True) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert len(results) == 200
    assert all(r == {P.ParamId.DELAY: 321, P.ParamId.VALVE_HOLD: 9} for r in results)
    assert link.status.ack_timeouts == 0 and link.debug_counters()["late_acks"] == 0


def test_close_is_idempotent_and_joins_threads(make_link, server):
    n0 = threading.active_count()
    lk = make_link()
    lk.connect()
    assert wait_for(lambda: lk.connected and server.model.reset_link_count >= 1, 2.0)
    link_threads = [t.name for t in threading.enumerate() if t.name.startswith("mcu-")]
    assert sorted(link_threads) == ["mcu-conn", "mcu-heartbeat", "mcu-reader"]
    # 服务端为这条连接开了 1 个处理线程（按线程名判断，不依赖精确总数）
    assert any(t.name.startswith("mock-mcu-client-") for t in threading.enumerate())
    assert threading.active_count() >= n0 + 4
    events: list[str] = []
    lk.on_event(lambda e: events.append(e.kind))
    t0 = time.monotonic()
    lk.close()
    assert time.monotonic() - t0 < 2.0
    assert not [t for t in threading.enumerate() if t.name.startswith("mcu-")], "close 后已 join"
    assert wait_for(lambda: threading.active_count() == n0, 2.0), "服务端连接线程随断开退出"
    assert events == ["disconnected"]
    assert not lk.connected
    with pytest.raises(McuDisconnected):
        lk.run()
    lk.send_eject([EjectItem(1, 1, 1)])
    assert lk.eject_dropped_offline == 1
    lk.close()
    assert events == ["disconnected"], "重复 close 不再发事件"
    # 关闭后可再次 connect
    lk.connect()
    assert wait_for(lambda: lk.connected, 2.0)
    lk.close()
    assert not [t for t in threading.enumerate() if t.name.startswith("mcu-")]
    assert wait_for(lambda: threading.active_count() == n0, 2.0)


def test_close_wakes_pending_waiter(link: TcpMcuLink, server):
    server.drop_acks(10)
    result: list[BaseException] = []

    def call() -> None:
        try:
            link.run()
        except BaseException as exc:
            result.append(exc)

    t = threading.Thread(target=call, daemon=True)
    t.start()
    time.sleep(0.15)  # 处于等待 ACK 中
    link.close()
    t.join(2.0)
    assert not t.is_alive()
    assert len(result) == 1 and isinstance(result[0], McuDisconnected)


def test_close_inside_event_callback_does_not_join_self(make_link, server):
    """在事件回调（运行于链路读线程）里调 close()：不抛 RuntimeError，其余线程 join 完，当前线程随后退出。"""
    lk = make_link()
    lk.connect()
    assert wait_for(lambda: lk.connected and server.model.reset_link_count >= 1, 2.0)
    kinds: list[str] = []
    errors: list[BaseException] = []
    threads_in_cb: list[str] = []

    def cb(e: LinkEvent) -> None:
        kinds.append(e.kind)
        if e.kind == "event":
            threads_in_cb.append(threading.current_thread().name)
            try:
                lk.close()
            except BaseException as exc:
                errors.append(exc)

    lk.on_event(cb)
    server.send_event(P.EventCode.LOW_PRESSURE, 2, 0)
    assert wait_for(lambda: "disconnected" in kinds, 2.0)
    assert threads_in_cb == ["mcu-reader"]
    assert errors == []
    assert not lk.connected
    assert wait_for(
        lambda: not [t for t in threading.enumerate() if t.name.startswith("mcu-")], 2.0
    ), "读线程在回调返回后自行退出"
    with pytest.raises(McuDisconnected):
        lk.run()
    # 关闭后仍可重新 connect，旧线程不会复活抢连接
    lk.connect()
    assert wait_for(lambda: lk.connected and lk.status.state == "stopped", 2.0)
    assert sorted(t.name for t in threading.enumerate() if t.name.startswith("mcu-")) == [
        "mcu-conn",
        "mcu-heartbeat",
        "mcu-reader",
    ]
    lk.close()


def test_close_racing_with_connect_leaves_no_socket(server, tmp_path):
    """close() 与连接线程竞争：连接线程已拿到连上的 socket、但在 close() 开始之后才装配，
    close 之后不能残留已连接的 socket（连接线程必须在锁内看到 stop 并把新 socket 关掉）。"""
    gate = threading.Event()

    class SlowOpen(TcpMcuLink):
        def _open_socket(self) -> socket.socket:
            s = super()._open_socket()
            gate.wait(2.0)  # 持有已连上的 socket，等 close() 开始后再返回装配
            return s

    lk = SlowOpen(make_cfg(tmp_path), host="127.0.0.1", port=server.port, **FAST)
    try:
        lk.connect()
        assert wait_for(lambda: server.accepted_total >= 1, 2.0), "连接线程已拿到 socket，卡在 gate"
        assert not lk.connected

        def release_gate_late() -> None:
            time.sleep(0.05)
            gate.set()

        threading.Thread(target=release_gate_late, daemon=True).start()
        lk.close()
        assert not lk.connected
        assert not [t for t in threading.enumerate() if t.name.startswith("mcu-")]
        assert wait_for(lambda: server.clients == 0, 2.0), "服务端看到连接被关闭，没有泄漏 socket"
    finally:
        gate.set()
        lk.close()


# ---- _request 收尾判定 / 心跳 NACK / 退避 ----


def test_retransmit_send_failure_raises_disconnected_not_nack(tmp_path, baseline):
    """重发时 sendall 失败、等待者又没被 _drop_conn 标记（竞争窗口）→ 必须抛 McuDisconnected，
    不能用等待者初值 result=-1 误报成 McuNack / 伪造 nack 事件。"""
    from oresorter.mcu.link import _Conn

    lk = TcpMcuLink(make_cfg(tmp_path), host="127.0.0.1", port=1, ack_timeout=0.05, ack_retries=3)
    events: list[LinkEvent] = []
    lk.on_event(events.append)

    class FakeSock:
        n = 0

        def sendall(self, data: bytes) -> None:
            self.n += 1
            if self.n == 1:
                # 模拟另一线程在首发后立刻 _drop_conn：句柄换掉、等待者表清空但未标记 disconnected
                lk._conn = None
                lk._waiters.clear()
                return
            raise OSError(32, "Broken pipe")

        def shutdown(self, *_a: object) -> None:
            pass

        def close(self) -> None:
            pass

    lk._conn = _Conn(FakeSock())  # type: ignore[arg-type]
    with pytest.raises(McuDisconnected):
        lk.run()
    assert not [e for e in events if e.kind == "nack"], "不得伪造 nack 事件"
    assert lk.status.ack_timeouts == 0
    assert lk._waiters == {}


def test_send_offline_raises_and_leaves_no_waiter(tmp_path, baseline):
    """未连接时 send 抛 McuDisconnected 且不留下等待者（连接句柄检查与登记在同一临界区）。"""
    lk = TcpMcuLink(make_cfg(tmp_path), host="127.0.0.1", port=1, **FAST)
    with pytest.raises(McuDisconnected):
        lk.send(P.Cmd.RUN)
    assert lk._waiters == {}


def test_heartbeat_nack_keeps_link_alive(link: TcpMcuLink, server):
    """下位机对 HEARTBEAT 只回 1 字节 NACK（如 DEVICE_FAULT）：对端活着，看门狗刷新，不判心跳超时。"""
    events: list[LinkEvent] = []
    link.on_event(events.append)
    r0 = link.status.reconnects
    server.heartbeat_result = int(P.AckResult.DEVICE_FAULT)
    time.sleep(FAST["heartbeat_timeout"] * 2.5)  # 远超 heartbeat_timeout
    assert link.connected and link.status.reconnects == r0
    assert not any(e.kind == "fault" for e in events), "没有 MCU_HEARTBEAT_TIMEOUT"
    assert not any(e.kind == "disconnected" for e in events)
    st = link.status
    assert st.last_heartbeat_ms is not None and st.last_heartbeat_ms < 500
    assert "心跳被拒绝: 设备故障" in st.faults
    assert link.debug_counters()["heartbeat_nack_result"] == 5
    # 恢复正常应答后故障文字消失，状态快照继续更新
    server.heartbeat_result = 0
    assert wait_for(lambda: "心跳被拒绝: 设备故障" not in link.status.faults, 1.0)
    assert link.status.state == "stopped" and link.connected
    link.run()
    assert wait_for(lambda: link.status.state == "running", 1.0)


def test_unresponsive_server_backs_off_instead_of_storming(make_link, server):
    """下位机 TCP 能连上但不应答：每次"连上→参数下发超时→断开"都要翻倍退避，而不是固定节奏刷屏；
    稳定连接（收到过 STATUS 且存活 ≥ 2×heartbeat_timeout）断开后退避重置为 backoff_min。"""
    server.silent = True
    lk = make_link(ack_timeout=0.05, ack_retries=1, backoff_min=0.05, backoff_max=1.0)
    events: list[LinkEvent] = []
    lk.on_event(events.append)
    lk.connect()
    # 退避按 0.05 → 0.1 → 0.2 → 0.4 增长（debug_counters["backoff_s"] 为最近一次断开后采用的值）
    assert wait_for(lambda: lk.debug_counters()["backoff_s"] >= 0.4, 4.0), lk.debug_counters()
    conn_ts = [e.ts for e in events if e.kind == "connected"]
    assert 3 <= len(conn_ts) <= 6, f"重连次数应被退避压住: {len(conn_ts)}"
    gaps = [b - a for a, b in itertools.pairwise(conn_ts)]
    assert gaps[-1] > gaps[0] * 1.5, f"相邻重连间隔应增长: {gaps}"
    assert all(e.data["code"] == "MCU_ACK_TIMEOUT" for e in events if e.kind == "fault")
    # 恢复应答 → 连上并完成参数下发（最多等一个 backoff_max）
    server.silent = False
    assert wait_for(lambda: lk.connected and server.model.reset_link_count >= 1, 3.0)
    # 存活 ≥ 2×heartbeat_timeout（0.6 s）后被服务端断开 → 稳定连接，退避重置为 backoff_min
    time.sleep(FAST["heartbeat_timeout"] * 2 + 0.15)
    r0 = lk.status.reconnects
    server.disconnect_clients()
    assert wait_for(lambda: lk.status.reconnects > r0 and lk.connected, 2.0)
    assert lk.debug_counters()["backoff_s"] == pytest.approx(0.05)
