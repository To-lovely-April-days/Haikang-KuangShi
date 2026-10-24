"""协议帧层：CRC16、帧编码字节布局、解析器（粘包/半包/垃圾/CRC 错/LEN 超限/缓冲上限）。"""

from __future__ import annotations

import random
import struct

import pytest

from oresorter.mcu import protocol as P


def _crc16_bitwise(data: bytes, init: int = 0xFFFF) -> int:
    """独立的逐位参考实现（多项式 0x8005 反射 0xA001），用来校对查表实现。"""
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# ---- CRC ----


def test_crc16_known_vector():
    assert P.crc16_modbus(b"123456789") == 0x4B37
    assert P.crc16_modbus(b"") == 0xFFFF
    assert _crc16_bitwise(b"123456789") == 0x4B37


def test_crc16_table_matches_bitwise_random():
    rng = random.Random(1234)
    for n in (1, 2, 7, 64, 1024):
        data = bytes(rng.randrange(256) for _ in range(n))
        assert P.crc16_modbus(data) == _crc16_bitwise(data)
    # 自定义初值、memoryview / bytearray 输入
    data = bytes(range(256))
    assert P.crc16_modbus(data, init=0x1234) == _crc16_bitwise(data, init=0x1234)
    assert P.crc16_modbus(memoryview(bytearray(data))) == P.crc16_modbus(data)
    assert len(P.CRC16_TABLE) == 256 and P.CRC16_TABLE[1] == 0xC0C1


# ---- 编码 ----


def test_encode_byte_layout():
    payload = b"\x01\x02\x03"
    frame = P.encode(P.Cmd.SET_PARAM, payload, seq=0x1234, addr=P.ADDR_MAIN)
    assert frame[0:2] == b"\xaa\x55", "SOF"
    assert frame[2] == P.VERSION, "VER"
    assert struct.unpack_from("<H", frame, 3)[0] == 0x1234, "SEQ 小端"
    assert frame[3:5] == b"\x34\x12"
    assert frame[5] == P.ADDR_MAIN, "ADDR"
    assert frame[6] == P.Cmd.SET_PARAM, "CMD"
    assert struct.unpack_from("<H", frame, 7)[0] == len(payload), "LEN"
    assert frame[9 : 9 + len(payload)] == payload
    crc = struct.unpack_from("<H", frame, 9 + len(payload))[0]
    assert crc == P.crc16_modbus(frame[2 : 9 + len(payload)]), "CRC 覆盖 VER~PAYLOAD"
    assert len(frame) == P.HEADER_LEN + len(payload) + P.CRC_LEN


def test_encode_empty_payload_and_fields():
    frame = P.encode(P.Cmd.RUN, seq=1)
    assert len(frame) == P.HEADER_LEN + P.CRC_LEN
    assert struct.unpack_from("<H", frame, 7)[0] == 0
    (fr,) = P.decode(frame)
    assert fr == P.McuFrame(seq=1, addr=P.ADDR_MAIN, cmd=P.Cmd.RUN, payload=b"", ver=P.VERSION)
    bc = P.encode(P.Cmd.STOP, seq=65535, addr=P.ADDR_BROADCAST, ver=2)
    # 默认解析器对 VER≠当前版本做头部预检会丢弃；关掉预检可以解出字段
    (fr,) = P.FrameParser(check_header=False).feed(bc)
    assert fr.seq == 65535 and fr.addr == 0xFF and fr.ver == 2 and fr.is_broadcast
    assert P.decode(bc) == []


def test_encode_rejects_out_of_range_fields():
    with pytest.raises(P.ProtocolError):
        P.encode(P.Cmd.RUN, seq=65536)
    with pytest.raises(P.ProtocolError):
        P.encode(0x100, seq=1)
    with pytest.raises(P.ProtocolError):
        P.encode(P.Cmd.RUN, seq=1, addr=256)


def test_max_payload_boundary():
    ok = P.encode(P.Cmd.EJECT, bytes(P.MAX_PAYLOAD), seq=5)
    (fr,) = P.decode(ok)
    assert len(fr.payload) == 1024
    with pytest.raises(P.ProtocolError):
        P.encode(P.Cmd.EJECT, bytes(P.MAX_PAYLOAD + 1), seq=5)


def test_encode_ack():
    req = P.McuFrame(seq=77, addr=1, cmd=P.Cmd.GET_PARAM, payload=b"\x01\x01\x00")
    data = P.pack_param_values({1: 42})
    (ack,) = P.decode(P.encode_ack(req, P.AckResult.OK, data))
    assert ack.cmd == P.Cmd.GET_PARAM | 0x80 and ack.is_ack and not ack.is_report
    assert ack.base_cmd == P.Cmd.GET_PARAM and ack.seq == 77 and ack.addr == 1
    assert ack.payload[0] == 0 and P.unpack_param_values(ack.payload[1:]) == {1: 42}
    (nack,) = P.decode(P.encode_ack(req, P.AckResult.OUT_OF_RANGE))
    assert nack.payload == b"\x03"
    hb = P.McuFrame(seq=9, addr=1, cmd=P.Cmd.HEARTBEAT, payload=P.pack_heartbeat(1))
    (st,) = P.decode(P.encode_ack(hb, 0, P.StatusPayload(state=1).pack()))
    assert st.cmd == P.Report.STATUS and st.is_ack
    assert P.StatusPayload.unpack(st.payload[1:]).state_text == "running"


def test_frame_properties():
    assert P.McuFrame(1, 1, 0x81, b"").is_ack
    assert P.McuFrame(1, 1, 0x81, b"").base_cmd == 0x01
    assert P.McuFrame(1, 1, 0xA0, b"").base_cmd == 0x20
    rep = P.McuFrame(1, 1, 0xC0, b"")
    assert rep.is_report and not rep.is_ack and rep.base_cmd == 0x40
    assert P.McuFrame(1, 1, 0xCF, b"").is_report
    assert not P.McuFrame(1, 1, 0xD0, b"").is_report and P.McuFrame(1, 1, 0xD0, b"").is_ack
    req = P.McuFrame(1, 1, P.Cmd.RUN, b"")
    assert not req.is_ack and not req.is_report and not req.is_broadcast


# ---- 解析器 ----


def _frames():
    a = P.encode(P.Cmd.SYNC, P.pack_sync(True), seq=1)
    b = P.encode(P.Cmd.SET_PARAM, P.pack_set_param({P.ParamId.DELAY: 300}), seq=2)
    return a, b


def test_roundtrip_single_frame():
    a, _ = _frames()
    p = P.FrameParser()
    out = p.feed(a)
    assert len(out) == 1 and out[0].cmd == P.Cmd.SYNC and out[0].payload == b"\x01"
    assert p.frames_ok == 1 and p.crc_errors == 0 and p.resyncs == 0 and p.bytes_dropped == 0
    assert p.pending == 0


def test_two_frames_in_one_feed():
    a, b = _frames()
    out = P.FrameParser().feed(a + b)
    assert [f.cmd for f in out] == [P.Cmd.SYNC, P.Cmd.SET_PARAM]
    assert [f.seq for f in out] == [1, 2]
    assert P.unpack_set_param(out[1].payload) == {P.ParamId.DELAY: 300}
    assert P.decode(a + b) == out


def test_byte_by_byte_feed():
    a, b = _frames()
    p = P.FrameParser()
    out = []
    stream = a + b
    for i, byte in enumerate(stream):
        got = p.feed(bytes([byte]))
        if got:
            assert i in (len(a) - 1, len(stream) - 1), "只在最后一个字节到达时出帧"
        out += got
    assert [f.seq for f in out] == [1, 2] and p.frames_ok == 2 and p.pending == 0


def test_garbage_before_sof():
    a, b = _frames()
    garbage = b"\x00\x11\x22\xaa\x33\x55"  # 含孤立的 0xAA / 0x55，不能误判为 SOF
    p = P.FrameParser()
    out = p.feed(garbage + a)
    assert len(out) == 1 and out[0].seq == 1
    assert p.bytes_dropped == len(garbage) and p.resyncs == 1
    # 失步后恢复同步，再来垃圾算第二次失步
    out = p.feed(b"\xff" * 3 + b)
    assert len(out) == 1 and out[0].seq == 2
    assert p.resyncs == 2 and p.bytes_dropped == len(garbage) + 3


def test_trailing_half_sof_is_kept():
    a, _ = _frames()
    p = P.FrameParser()
    assert p.feed(b"\x01\x02" + a[:1]) == []  # 以 0xAA 结尾：可能是 SOF 前半
    assert p.pending == 1 and p.bytes_dropped == 2
    assert len(p.feed(a[1:])) == 1


def test_crc_error_dropped_and_following_frame_ok():
    a, b = _frames()
    bad = bytearray(a)
    bad[9] ^= 0x01  # 翻转负载一位
    p = P.FrameParser()
    out = p.feed(bytes(bad) + b)
    assert [f.seq for f in out] == [2]
    assert p.crc_errors == 1 and p.frames_ok == 1 and p.resyncs == 1
    assert p.bytes_dropped == len(a), "坏帧整体被丢弃"
    # CRC 字段本身错
    bad2 = a[:-1] + bytes([a[-1] ^ 0xFF])
    out = p.feed(bad2 + a)
    assert len(out) == 1 and p.crc_errors == 2 and p.frames_ok == 2


def test_len_overflow_resyncs():
    a, _ = _frames()
    hdr = P.SOF + struct.pack("<BHBBH", P.VERSION, 1, 1, P.Cmd.RUN, P.MAX_PAYLOAD + 1)
    p = P.FrameParser()
    out = p.feed(hdr + b"\x00" * 10 + a)
    assert len(out) == 1 and out[0].seq == 1
    assert p.resyncs == 1 and p.bytes_dropped == len(hdr) + 10 and p.crc_errors == 0
    assert p.header_errors == 1


def test_header_plausible():
    assert P.header_plausible(P.VERSION, P.ADDR_MAIN, P.Cmd.RUN)
    assert P.header_plausible(P.VERSION, P.ADDR_BROADCAST, P.Report.EVENT)
    assert not P.header_plausible(0x02, P.ADDR_MAIN, P.Cmd.RUN), "VER 不对"
    assert not P.header_plausible(P.VERSION, 0x02, P.Cmd.RUN), "ADDR 非法"
    assert not P.header_plausible(P.VERSION, P.ADDR_MAIN, 0x00), "CMD=0"


def test_header_precheck_resyncs_without_waiting_for_len():
    """头部（VER/ADDR/CMD）被破坏的 SOF 立刻按垃圾丢弃，不等 LEN 字节到齐，后续帧不被吞。"""
    a, b = _frames()
    # 一帧 STATUS（LEN=20）的 VER 字节被破坏：若无预检，解析器会等 31 B 到齐再因 CRC 失败才重同步
    hb = P.encode_ack(P.McuFrame(1, 1, P.Cmd.HEARTBEAT, b""), 0, P.StatusPayload().pack())
    bad_ver = bytearray(hb)
    bad_ver[2] ^= 0x10
    p = P.FrameParser()
    out = p.feed(bytes(bad_ver))
    assert out == [] and p.header_errors == 1 and p.resyncs == 1
    assert p.pending <= 1, "坏头部整帧被当作垃圾丢掉（最多保留一个可能是 SOF 前半的 0xAA）"
    out = p.feed(a)
    assert [f.seq for f in out] == [1], "紧随其后的第一帧立刻解出"
    # ADDR 非法 / CMD=0 同样被丢
    bad_addr = P.encode(P.Cmd.RUN, seq=3, addr=0x02)
    bad_cmd = P.encode(0x00, seq=4)
    out = p.feed(bad_addr + bad_cmd + b)
    assert [f.seq for f in out] == [2] and p.header_errors == 3 and p.crc_errors == 0
    # 关闭预检则放行（例如将来接多板 / 新版本时诊断用）
    loose = P.FrameParser(check_header=False)
    out = loose.feed(bytes(bad_ver) + bad_addr + bad_cmd)
    assert [f.seq for f in out] == [
        3,
        4,
    ] and loose.header_errors == 0, "坏 VER 帧 CRC 不过，其余两帧解出"
    assert loose.crc_errors == 1
    # reset 清零 header_errors
    p.reset()
    assert p.header_errors == 0


def test_len_corruption_still_recovers_by_crc():
    """只有 LEN 被破坏（头部其余合理）时预检帮不上，仍靠 CRC 失败重同步并最终恢复——记录现状。"""
    hb = P.encode_ack(P.McuFrame(1, 1, P.Cmd.HEARTBEAT, b""), 0, P.StatusPayload().pack())
    bad = bytearray(hb)
    bad[8] ^= 0x02  # LEN 20 -> 532
    p = P.FrameParser()
    got = 0
    for i in range(40):
        got += len(p.feed(bytes(bad) if i == 0 else hb))
        if got:
            break
    assert got >= 1 and p.crc_errors == 1 and p.header_errors == 0
    assert p.frames_ok >= 1


def test_half_frame_waits_for_rest():
    a, _ = _frames()
    p = P.FrameParser()
    assert p.feed(a[:5]) == [] and p.pending == 5
    assert p.feed(a[5:-1]) == [] and p.pending == len(a) - 1
    assert len(p.feed(a[-1:])) == 1 and p.pending == 0 and p.bytes_dropped == 0


def test_buffer_cap_default_and_garbage_flood():
    p = P.FrameParser()
    assert p.max_buffer == 64 * 1024
    p.feed(b"\x00" * (100 * 1024))
    assert p.pending == 0 and p.bytes_dropped == 100 * 1024
    p.feed(b"\xaa" * 5000)  # 全是 0xAA：只保留最后一个
    assert p.pending == 1


def test_buffer_cap_trims_oversized_partial_frame():
    a, _ = _frames()
    p = P.FrameParser(max_buffer=32)
    hdr = P.SOF + struct.pack("<BHBBH", P.VERSION, 1, 1, P.Cmd.EJECT, 1000)
    assert p.feed(hdr + b"\x00" * 20) == []
    assert p.pending == 29
    p.feed(b"\x00" * 20)
    assert p.pending <= 32 and p.bytes_dropped >= 17
    # 上限之后仍能解析后续正常帧
    assert len(p.feed(a)) == 1
    with pytest.raises(P.ProtocolError):
        P.FrameParser(max_buffer=4)


def test_reset_clears_buffer_and_counters():
    a, _ = _frames()
    p = P.FrameParser()
    p.feed(b"\x01" + a[:4])
    assert p.pending > 0 and p.bytes_dropped == 1
    p.reset()
    assert p.pending == 0 and p.bytes_dropped == 0 and p.resyncs == 0
    assert len(p.feed(a)) == 1


def test_decode_ignores_incomplete_tail():
    a, b = _frames()
    out = P.decode(a + b[:7])
    assert len(out) == 1 and out[0].seq == 1
