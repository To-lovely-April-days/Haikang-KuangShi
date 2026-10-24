"""协议负载层打包/解包往返（帧层测试在 test_protocol.py）。"""

from __future__ import annotations

import pytest

from oresorter.mcu import protocol as P
from tests.conftest import make_cfg


def test_set_param_roundtrip():
    params = {P.ParamId.VALVE_OPEN_TIME: 255, P.ParamId.DELAY: -5, P.ParamId.FEED_SPEED_3: 77}
    b = P.pack_set_param(params)
    assert b[0] == 3 and len(b) == 1 + 6 * 3
    assert P.unpack_set_param(b) == {int(k): v for k, v in params.items()}


def test_get_param_roundtrip():
    ids = [P.ParamId.CLEAN_TIME, P.ParamId.NOZZLE_COUNT]
    assert P.unpack_get_param(P.pack_get_param(ids)) == [int(i) for i in ids]
    vals = {1: 10, 2: -3}
    assert P.unpack_param_values(P.pack_param_values(vals)) == vals


def test_eject_roundtrip_and_limits():
    items = [(123456, 1, 64, 0), (2**32 - 1, 700, 768, 1)]
    assert P.unpack_eject(P.pack_eject(items)) == items
    with pytest.raises(P.ProtocolError):
        P.pack_eject([])
    with pytest.raises(P.ProtocolError):
        P.pack_eject([(1, 1, 1, 0)] * 101)
    with pytest.raises(P.ProtocolError):
        P.unpack_eject(b"\x02" + b"\x00" * 9)


def test_status_event_counters_roundtrip():
    sp = P.StatusPayload(
        state=1,
        encoder_count=99999,
        belt_speed=9000,
        faults=0b101,
        queue_len=7,
        pressure=0xFFFF,
        host_ms_echo=5,
    )
    sp2 = P.StatusPayload.unpack(sp.pack())
    assert sp2 == sp and sp2.state_text == "running" and sp2.fault_codes() == [0, 2]
    ev = P.EventPayload(code=P.EventCode.LOW_PRESSURE, level=2, detail=3)
    assert P.EventPayload.unpack(ev.pack()) == ev and ev.text == "气压低"
    c = P.CountersPayload(1, 2, 3, 1000)
    assert P.CountersPayload.unpack(c.pack()).to_dict() == {
        "eject_rx": 1,
        "eject_done": 2,
        "eject_dropped": 3,
        "period_ms": 1000,
    }


def test_params_from_config_covers_sections(tmp_path):
    cfg = make_cfg(tmp_path)
    p = P.params_from_config(cfg)
    assert p[P.ParamId.VALVE_OPEN_TIME] == 255 and p[P.ParamId.CLEAN_ENABLE] == 0
    assert all(p[pid] == 50 for pid in P.FEED_SPEED_IDS)
    assert p[P.ParamId.NOZZLE_COUNT] == 64
    assert set(P.params_for_section(cfg, "valve")) == set(P.SECTION_PARAM_IDS["valve"])
    assert P.check_param_range(p) == []
    assert P.check_param_range({P.ParamId.VALVE_REBLOW: 51}) == [
        (P.ParamId.VALVE_REBLOW, 51, (0, 50))
    ]
