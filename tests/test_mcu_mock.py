"""进程内 Mock 下位机：状态迁移、参数越界、喷阀队列、故障注入。"""

from __future__ import annotations

import time

import pytest

from oresorter.mcu import EjectItem, McuDisconnected, McuNack, McuTimeout, build_mcu_link
from oresorter.mcu import protocol as P
from oresorter.mcu.mock import MockMcuLink
from tests.conftest import make_cfg


@pytest.fixture
def link(tmp_path) -> MockMcuLink:
    cfg = make_cfg(tmp_path)
    lk = build_mcu_link(cfg)
    assert isinstance(lk, MockMcuLink)
    lk.connect()
    return lk


def test_disconnected_raises(tmp_path):
    lk = MockMcuLink(make_cfg(tmp_path))
    with pytest.raises(McuDisconnected):
        lk.run()
    lk.send_eject([EjectItem(1, 1, 2)])  # 不抛，计数
    assert lk.eject_dropped_offline == 1


def test_push_all_params_and_get(link: MockMcuLink):
    ack = link.push_all_params()
    assert ack.ok
    vals = link.get_params([P.ParamId.VALVE_OPEN_TIME, P.ParamId.DELAY, P.ParamId.FEED_SPEED_9])
    assert vals == {
        P.ParamId.VALVE_OPEN_TIME: 255,
        P.ParamId.DELAY: 376,
        P.ParamId.FEED_SPEED_9: 50,
    }


def test_set_param_out_of_range_nack(link: MockMcuLink):
    with pytest.raises(McuNack) as ei:
        link.set_params({P.ParamId.VALVE_OPEN_TIME: 300})
    assert ei.value.result == P.AckResult.OUT_OF_RANGE
    assert link.model.params[P.ParamId.VALVE_OPEN_TIME] == 0, "整批不生效"


def test_state_transitions_and_encoder(link: MockMcuLink):
    link.model.pulses_per_s = 100_000
    link.sync()
    link.run()
    assert link.status.state == "running"
    time.sleep(0.05)
    e1 = link.status.encoder_count
    assert e1 > 0
    link.pause()
    e2 = link.status.encoder_count
    time.sleep(0.02)
    assert link.status.encoder_count == e2, "暂停时编码器冻结"
    link.stop()
    assert link.status.state == "stopped"
    link.sync()
    assert link.status.encoder_count == 0


def test_heartbeat_status(link: MockMcuLink):
    sp = link.heartbeat(host_ms=4242)
    assert sp.host_ms_echo == 4242
    assert link.status.last_heartbeat_ms is not None
    assert link.status.rtt_ms is not None


def test_eject_queue_and_settle(link: MockMcuLink):
    link.model.pulses_per_s = 0  # 手动控制编码器
    link.set_params({P.ParamId.DELAY: 10})
    link.sync()
    link.run()
    link.send_eject(
        [
            EjectItem(line=100, nozzle_first=3, nozzle_last=5),
            EjectItem(line=50, nozzle_first=1, nozzle_last=1),
        ]
    )
    st = link.status
    assert st.counters["eject_rx"] == 2 and st.queue_len == 2
    link.model._enc_base = 61  # 50+10 已到
    st = link.status
    assert st.counters["eject_done"] == 1 and st.queue_len == 1
    assert link.model.fired == [(50, 1, 1, 0)]
    # 过期条目计 dropped
    link.send_eject([EjectItem(line=1, nozzle_first=1, nozzle_last=1)])
    assert link.status.counters["eject_dropped"] == 1


def test_eject_splits_over_100(link: MockMcuLink):
    link.run()
    link.send_eject([EjectItem(line=10**6 + i, nozzle_first=1, nozzle_last=1) for i in range(250)])
    ejects = [c for c, _ in link.sent_log if c == P.Cmd.EJECT]
    assert len(ejects) == 3
    assert link.status.counters["eject_rx"] == 250


def test_valve_test_rules(link: MockMcuLink):
    link.valve_test_single(10)
    with pytest.raises(McuNack):
        link.valve_test_single(0)
    link.run()
    with pytest.raises(McuNack) as ei:
        link.valve_test_single(10)
    assert ei.value.result == P.AckResult.BAD_STATE
    link.stop()
    link.valve_test_cycle(1, 64, 200)
    assert link.status.state == "valve_test"
    link.valve_test_stop()
    assert link.status.state == "stopped"


def test_fault_injection(link: MockMcuLink):
    link.inject_ack_timeouts(2)
    with pytest.raises(McuTimeout):
        link.run()
    with pytest.raises(McuTimeout):
        link.run()
    assert link.run().ok
    link.inject_nack(P.AckResult.DEVICE_FAULT)
    with pytest.raises(McuNack):
        link.stop()
    events = []
    link.on_event(events.append)
    link.set_offline(True)
    assert not link.connected
    link.set_offline(False)
    assert [e.kind for e in events] == ["disconnected", "connected"]
    link.inject_fault(P.EventCode.LOW_PRESSURE)
    assert "气压低" in link.status.faults
