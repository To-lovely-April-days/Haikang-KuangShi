"""下位机通讯协议（docs/04 草案 v0.1）：常量、参数表、负载打包/解包。

本文件分两层：
- **负载层**：指令码、参数 ID、ACK 结果码、各指令 PAYLOAD 的 ``pack_*`` / ``unpack_*``；
- **帧层**：``crc16_modbus`` / ``McuFrame`` / ``encode`` / ``encode_ack`` / ``decode`` / ``FrameParser``，
  处理 SOF 同步、粘包/半包、CRC 校验与重新同步（docs/04 §2）。

所有多字节整数小端。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oresorter.config import AppConfig

SOF = b"\xaa\x55"
VERSION = 0x01
ADDR_MAIN = 0x01
ADDR_BROADCAST = 0xFF
ACK_FLAG = 0x80
MAX_PAYLOAD = 1024
HEADER_LEN = 9  # SOF(2) VER(1) SEQ(2) ADDR(1) CMD(1) LEN(2)
CRC_LEN = 2
EJECT_MAX_ITEMS = 100
EJECT_ITEM_FMT = "<IHHB"  # line u32, nozzle_first u16, nozzle_last u16, flags u8
EJECT_ITEM_LEN = struct.calcsize(EJECT_ITEM_FMT)


class Cmd(IntEnum):
    """主机 → 下位机 指令码。应答 = CMD | 0x80。"""

    HEARTBEAT = 0x01
    SYNC = 0x02
    RUN = 0x10
    PAUSE = 0x11
    STOP = 0x12
    SET_PARAM = 0x20
    GET_PARAM = 0x21
    VALVE_TEST_SINGLE = 0x30
    VALVE_TEST_CYCLE = 0x31
    VALVE_TEST_STOP = 0x32
    VALVE_RESET = 0x33
    CLEAN_NOW = 0x34
    EJECT = 0x40
    RESET_LINK = 0x7F


class Report(IntEnum):
    """下位机 → 主机 上报类指令码。"""

    STATUS = 0x81  # 对 HEARTBEAT 的应答（= HEARTBEAT | 0x80）
    EVENT = 0xC0
    COUNTERS = 0xC1


NEED_ACK: frozenset[int] = frozenset(c for c in Cmd if c != Cmd.EJECT)


class AckResult(IntEnum):
    OK = 0
    CRC_ERROR = 1
    UNKNOWN_CMD = 2
    OUT_OF_RANGE = 3
    BAD_STATE = 4
    DEVICE_FAULT = 5
    QUEUE_FULL = 6


ACK_RESULT_TEXT = {
    AckResult.OK: "成功",
    AckResult.CRC_ERROR: "CRC 错误",
    AckResult.UNKNOWN_CMD: "未知指令",
    AckResult.OUT_OF_RANGE: "参数越界",
    AckResult.BAD_STATE: "当前状态不允许",
    AckResult.DEVICE_FAULT: "设备故障",
    AckResult.QUEUE_FULL: "队列满",
}


class McuState(IntEnum):
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2
    FAULT = 3
    VALVE_TEST = 4
    CLEANING = 5


MCU_STATE_TEXT = {
    McuState.STOPPED: "stopped",
    McuState.RUNNING: "running",
    McuState.PAUSED: "paused",
    McuState.FAULT: "fault",
    McuState.VALVE_TEST: "valve_test",
    McuState.CLEANING: "cleaning",
}


class ParamId(IntEnum):
    """SET_PARAM / GET_PARAM 参数 ID（docs/04 §4）。"""

    VALVE_OPEN_TIME = 0x0001
    VALVE_HOLD = 0x0002
    VALVE_REBLOW = 0x0003
    DELAY = 0x0004
    CLEAN_TIME = 0x0010
    CLEAN_PERIOD = 0x0011
    CLEAN_COUNT = 0x0012
    CLEAN_ENABLE = 0x0013
    FEED_SPEED_1 = 0x0020
    FEED_SPEED_2 = 0x0021
    FEED_SPEED_3 = 0x0022
    FEED_SPEED_4 = 0x0023
    FEED_SPEED_5 = 0x0024
    FEED_SPEED_6 = 0x0025
    FEED_SPEED_7 = 0x0026
    FEED_SPEED_8 = 0x0027
    FEED_SPEED_9 = 0x0028
    CONVEYOR_ON = 0x0030
    VALVE_SUPPLY_ON = 0x0031
    FEEDER_ON = 0x0032
    LINES_PER_PULSE = 0x0040
    NOZZLE_COUNT = 0x0041


FEED_SPEED_IDS: tuple[ParamId, ...] = tuple(ParamId(0x0020 + i) for i in range(9))

PARAM_LIMITS: dict[int, tuple[int, int]] = {
    ParamId.VALVE_OPEN_TIME: (0, 255),
    ParamId.VALVE_HOLD: (0, 255),
    ParamId.VALVE_REBLOW: (0, 50),
    ParamId.DELAY: (0, 800),
    ParamId.CLEAN_TIME: (0, 200),
    ParamId.CLEAN_PERIOD: (0, 200),
    ParamId.CLEAN_COUNT: (0, 5),
    ParamId.CLEAN_ENABLE: (0, 1),
    **dict.fromkeys(FEED_SPEED_IDS, (0, 100)),
    ParamId.CONVEYOR_ON: (0, 1),
    ParamId.VALVE_SUPPLY_ON: (0, 1),
    ParamId.FEEDER_ON: (0, 1),
    ParamId.LINES_PER_PULSE: (1, 64),
    ParamId.NOZZLE_COUNT: (64, 768),
}


class EventCode(IntEnum):
    LOW_PRESSURE = 0x0001
    VALVE_OVERCURRENT = 0x0002
    ENCODER_NO_SIGNAL = 0x0003
    ENCODER_REVERSED = 0x0004
    FEEDER_FAULT = 0x0005
    CONVEYOR_FAULT = 0x0006
    HEARTBEAT_LOST_STOPPED = 0x0007
    EJECT_QUEUE_OVERFLOW = 0x0008
    EMERGENCY_STOP = 0x0010


EVENT_TEXT = {
    EventCode.LOW_PRESSURE: "气压低",
    EventCode.VALVE_OVERCURRENT: "喷阀驱动过流/短路",
    EventCode.ENCODER_NO_SIGNAL: "编码器无信号",
    EventCode.ENCODER_REVERSED: "编码器反转",
    EventCode.FEEDER_FAULT: "给料机故障",
    EventCode.CONVEYOR_FAULT: "输送带故障",
    EventCode.HEARTBEAT_LOST_STOPPED: "心跳超时已自动停阀",
    EventCode.EJECT_QUEUE_OVERFLOW: "喷阀队列溢出",
    EventCode.EMERGENCY_STOP: "急停触发",
}


class ProtocolError(ValueError):
    """负载/帧格式错误。"""


# --------------------------------------------------------------------------------------
# 配置 → 参数表
# --------------------------------------------------------------------------------------


def params_from_config(cfg: AppConfig) -> dict[int, int]:
    """把 AppConfig 中需要下发下位机的分组（valve/clean/feed/outputs）转成 ``{ParamId: value}``。"""
    out: dict[int, int] = {
        ParamId.VALVE_OPEN_TIME: cfg.valve.open_time,
        ParamId.VALVE_HOLD: cfg.valve.hold,
        ParamId.VALVE_REBLOW: cfg.valve.reblow,
        ParamId.DELAY: cfg.valve.delay,
        ParamId.CLEAN_TIME: cfg.clean.time,
        ParamId.CLEAN_PERIOD: cfg.clean.period,
        ParamId.CLEAN_COUNT: cfg.clean.count,
        ParamId.CLEAN_ENABLE: int(cfg.clean.enabled),
        ParamId.CONVEYOR_ON: int(cfg.outputs.conveyor),
        ParamId.VALVE_SUPPLY_ON: int(cfg.outputs.valve),
        ParamId.FEEDER_ON: int(cfg.outputs.feeder),
        ParamId.NOZZLE_COUNT: max(64, min(768, cfg.total_nozzles)),
    }
    for pid, v in zip(FEED_SPEED_IDS, cfg.feed.speeds, strict=False):
        out[pid] = int(v)
    return out


SECTION_PARAM_IDS: dict[str, tuple[int, ...]] = {
    "valve": (ParamId.VALVE_OPEN_TIME, ParamId.VALVE_HOLD, ParamId.VALVE_REBLOW, ParamId.DELAY),
    "clean": (ParamId.CLEAN_TIME, ParamId.CLEAN_PERIOD, ParamId.CLEAN_COUNT, ParamId.CLEAN_ENABLE),
    "feed": FEED_SPEED_IDS,
    "outputs": (ParamId.CONVEYOR_ON, ParamId.VALVE_SUPPLY_ON, ParamId.FEEDER_ON),
}


def params_for_section(cfg: AppConfig, section: str) -> dict[int, int]:
    """某个分组对应的参数子集（PUT /params/{section} 后只下发这一组）。"""
    ids = SECTION_PARAM_IDS.get(section)
    if ids is None:
        return {}
    allp = params_from_config(cfg)
    return {pid: allp[pid] for pid in ids if pid in allp}


# --------------------------------------------------------------------------------------
# 负载打包 / 解包
# --------------------------------------------------------------------------------------


def pack_heartbeat(host_ms: int) -> bytes:
    return struct.pack("<I", host_ms & 0xFFFFFFFF)


def pack_sync(clear_encoder: bool = True) -> bytes:
    return struct.pack("<B", 1 if clear_encoder else 0)


def pack_set_param(params: dict[int, int]) -> bytes:
    if not 0 < len(params) <= 255:
        raise ProtocolError(f"SET_PARAM 参数个数 {len(params)} 超出 1~255")
    out = bytearray(struct.pack("<B", len(params)))
    for pid, val in params.items():
        out += struct.pack("<Hi", int(pid), int(val))
    return bytes(out)


def unpack_set_param(payload: bytes) -> dict[int, int]:
    if len(payload) < 1:
        raise ProtocolError("SET_PARAM 负载为空")
    n = payload[0]
    if len(payload) != 1 + 6 * n:
        raise ProtocolError(f"SET_PARAM 负载长度 {len(payload)} 与 n={n} 不符")
    out: dict[int, int] = {}
    for i in range(n):
        pid, val = struct.unpack_from("<Hi", payload, 1 + 6 * i)
        out[pid] = val
    return out


def pack_get_param(ids: list[int]) -> bytes:
    if not 0 < len(ids) <= 255:
        raise ProtocolError(f"GET_PARAM 参数个数 {len(ids)} 超出 1~255")
    return struct.pack("<B", len(ids)) + b"".join(struct.pack("<H", int(i)) for i in ids)


def unpack_get_param(payload: bytes) -> list[int]:
    if len(payload) < 1:
        raise ProtocolError("GET_PARAM 负载为空")
    n = payload[0]
    if len(payload) != 1 + 2 * n:
        raise ProtocolError(f"GET_PARAM 负载长度 {len(payload)} 与 n={n} 不符")
    return [struct.unpack_from("<H", payload, 1 + 2 * i)[0] for i in range(n)]


def pack_param_values(params: dict[int, int]) -> bytes:
    """GET_PARAM 应答数据部分：n × {id u16, value i32}。"""
    return b"".join(struct.pack("<Hi", int(pid), int(v)) for pid, v in params.items())


def unpack_param_values(data: bytes) -> dict[int, int]:
    if len(data) % 6:
        raise ProtocolError(f"参数值数据长度 {len(data)} 不是 6 的倍数")
    return {
        struct.unpack_from("<Hi", data, 6 * i)[0]: struct.unpack_from("<Hi", data, 6 * i)[1]
        for i in range(len(data) // 6)
    }


def pack_valve_test_single(hole: int) -> bytes:
    return struct.pack("<H", hole)


def pack_valve_test_cycle(frm: int, to: int, interval_ms: int) -> bytes:
    return struct.pack("<HHH", frm, to, interval_ms)


def pack_eject(items: list[tuple[int, int, int, int]]) -> bytes:
    """items: ``[(line, nozzle_first, nozzle_last, flags), ...]``，长度 ≤ 100。"""
    if not 0 < len(items) <= EJECT_MAX_ITEMS:
        raise ProtocolError(f"EJECT 条目数 {len(items)} 超出 1~{EJECT_MAX_ITEMS}")
    out = bytearray(struct.pack("<B", len(items)))
    for line, nf, nl, flags in items:
        out += struct.pack(EJECT_ITEM_FMT, line & 0xFFFFFFFF, nf, nl, flags & 0xFF)
    return bytes(out)


def unpack_eject(payload: bytes) -> list[tuple[int, int, int, int]]:
    if len(payload) < 1:
        raise ProtocolError("EJECT 负载为空")
    n = payload[0]
    if len(payload) != 1 + EJECT_ITEM_LEN * n:
        raise ProtocolError(f"EJECT 负载长度 {len(payload)} 与 n={n} 不符")
    return [struct.unpack_from(EJECT_ITEM_FMT, payload, 1 + EJECT_ITEM_LEN * i) for i in range(n)]


@dataclass(slots=True)
class StatusPayload:
    """STATUS 负载（docs/04 §6）。"""

    state: int = 0
    encoder_count: int = 0
    belt_speed: int = 0
    faults: int = 0
    queue_len: int = 0
    pressure: int = 0xFFFF
    host_ms_echo: int = 0

    FMT = "<BIHIHHI"

    def pack(self) -> bytes:
        return struct.pack(
            self.FMT,
            self.state & 0xFF,
            self.encoder_count & 0xFFFFFFFF,
            self.belt_speed & 0xFFFF,
            self.faults & 0xFFFFFFFF,
            self.queue_len & 0xFFFF,
            self.pressure & 0xFFFF,
            self.host_ms_echo & 0xFFFFFFFF,
        )

    @classmethod
    def unpack(cls, data: bytes) -> StatusPayload:
        size = struct.calcsize(cls.FMT)
        if len(data) < size:
            raise ProtocolError(f"STATUS 负载长度 {len(data)} < {size}")
        return cls(*struct.unpack_from(cls.FMT, data, 0))

    @property
    def state_text(self) -> str:
        try:
            return MCU_STATE_TEXT[McuState(self.state)]
        except ValueError:
            return f"unknown({self.state})"

    def fault_codes(self) -> list[int]:
        return [bit for bit in range(32) if self.faults & (1 << bit)]


@dataclass(slots=True)
class EventPayload:
    code: int
    level: int  # 0 信息 / 1 警告 / 2 故障
    detail: int = 0

    FMT = "<HBI"

    def pack(self) -> bytes:
        return struct.pack(self.FMT, self.code, self.level, self.detail)

    @classmethod
    def unpack(cls, data: bytes) -> EventPayload:
        size = struct.calcsize(cls.FMT)
        if len(data) < size:
            raise ProtocolError(f"EVENT 负载长度 {len(data)} < {size}")
        return cls(*struct.unpack_from(cls.FMT, data, 0))

    @property
    def text(self) -> str:
        try:
            return EVENT_TEXT[EventCode(self.code)]
        except ValueError:
            return f"未知事件 0x{self.code:04X}"


@dataclass(slots=True)
class CountersPayload:
    eject_rx: int = 0
    eject_done: int = 0
    eject_dropped: int = 0
    period_ms: int = 1000

    FMT = "<IIIH"

    def pack(self) -> bytes:
        return struct.pack(
            self.FMT, self.eject_rx, self.eject_done, self.eject_dropped, self.period_ms
        )

    @classmethod
    def unpack(cls, data: bytes) -> CountersPayload:
        size = struct.calcsize(cls.FMT)
        if len(data) < size:
            raise ProtocolError(f"COUNTERS 负载长度 {len(data)} < {size}")
        return cls(*struct.unpack_from(cls.FMT, data, 0))

    def to_dict(self) -> dict[str, int]:
        return {
            "eject_rx": self.eject_rx,
            "eject_done": self.eject_done,
            "eject_dropped": self.eject_dropped,
            "period_ms": self.period_ms,
        }


def check_param_range(params: dict[int, int]) -> list[tuple[int, int, tuple[int, int]]]:
    """返回越界项 ``[(id, value, (lo, hi))]``；未知 id 以 (0, 0) 范围报告。"""
    bad = []
    for pid, v in params.items():
        lim = PARAM_LIMITS.get(pid)
        if lim is None:
            bad.append((pid, v, (0, 0)))
        elif not (lim[0] <= v <= lim[1]):
            bad.append((pid, v, lim))
    return bad


# --------------------------------------------------------------------------------------
# 帧层：CRC16 / 帧编码 / 帧解析（docs/04 §2）
# --------------------------------------------------------------------------------------

CRC16_POLY_REFLECTED = 0xA001  # 多项式 0x8005 的反射形式
CRC16_INIT = 0xFFFF
REPORT_CMD_MIN = 0xC0
REPORT_CMD_MAX = 0xCF
HEADER_BODY_FMT = "<BHBBH"  # VER u8, SEQ u16, ADDR u8, CMD u8, LEN u16（紧跟 SOF 之后）
DEFAULT_PARSER_BUFFER = 64 * 1024


def _build_crc16_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ CRC16_POLY_REFLECTED
            else:
                crc >>= 1
        table.append(crc)
    return tuple(table)


CRC16_TABLE: tuple[int, ...] = _build_crc16_table()


def crc16_modbus(data: bytes, init: int = CRC16_INIT) -> int:
    """CRC16-MODBUS（多项式 0x8005 反射 0xA001，初值 0xFFFF，不取反），查表实现。

    已知向量：``crc16_modbus(b"123456789") == 0x4B37``。``data`` 可为 bytes / bytearray / memoryview。
    """
    crc = init & 0xFFFF
    table = CRC16_TABLE
    for b in data:
        crc = (crc >> 8) ^ table[(crc ^ b) & 0xFF]
    return crc


@dataclass(slots=True)
class McuFrame:
    """一帧已解析的协议帧（不含 SOF / LEN / CRC，这些在编解码时自动处理）。"""

    seq: int
    addr: int
    cmd: int
    payload: bytes
    ver: int = VERSION

    @property
    def is_report(self) -> bool:
        """是否为下位机主动上报（EVENT / COUNTERS 等，0xC0~0xCF）。"""
        return REPORT_CMD_MIN <= self.cmd <= REPORT_CMD_MAX

    @property
    def is_ack(self) -> bool:
        """是否为应答帧（CMD | 0x80，且不是上报类）。STATUS(0x81) 也是 ACK。"""
        return bool(self.cmd & ACK_FLAG) and not self.is_report

    @property
    def base_cmd(self) -> int:
        """去掉 ACK 标志后的请求指令码。"""
        return self.cmd & 0x7F

    @property
    def is_broadcast(self) -> bool:
        return self.addr == ADDR_BROADCAST


def _check_u8(name: str, value: int) -> None:
    if not 0 <= value <= 0xFF:
        raise ProtocolError(f"{name}={value} 超出 u8 范围")


def encode(
    cmd: int,
    payload: bytes = b"",
    *,
    seq: int,
    addr: int = ADDR_MAIN,
    ver: int = VERSION,
) -> bytes:
    """编码一帧：SOF(2) VER(1) SEQ(2 LE) ADDR(1) CMD(1) LEN(2 LE) PAYLOAD CRC16(2 LE)。

    CRC 覆盖 VER~PAYLOAD（偏移 2 到 9+LEN−1）。``LEN > MAX_PAYLOAD`` 或字段越界抛 ``ProtocolError``。
    """
    payload = bytes(payload)
    n = len(payload)
    if n > MAX_PAYLOAD:
        raise ProtocolError(f"负载长度 {n} 超过上限 {MAX_PAYLOAD}")
    if not 0 <= seq <= 0xFFFF:
        raise ProtocolError(f"seq={seq} 超出 u16 范围")
    _check_u8("cmd", cmd)
    _check_u8("addr", addr)
    _check_u8("ver", ver)
    body = struct.pack(HEADER_BODY_FMT, ver, seq, addr, cmd, n) + payload
    return SOF + body + struct.pack("<H", crc16_modbus(body))


def encode_ack(req: McuFrame, result: int, data: bytes = b"") -> bytes:
    """按请求帧构造应答：CMD | 0x80，SEQ / ADDR / VER 原样返回，PAYLOAD = result u8 + data。"""
    _check_u8("result", result)
    return encode(
        req.cmd | ACK_FLAG,
        bytes([result]) + bytes(data),
        seq=req.seq,
        addr=req.addr,
        ver=req.ver,
    )


def header_plausible(ver: int, addr: int, cmd: int) -> bool:
    """帧头合理性预检：VER 必须等于当前版本、ADDR ∈ {主控, 广播}、CMD ≠ 0。

    协议 §2 没有规定头部校验，只靠 CRC 的话，一个 SOF 后面跟着被破坏的头部要等 ``LEN`` 字节到齐
    才能发现错误，期间会吞掉后续有效帧；通过预检把明显不合理的 SOF 立刻当垃圾丢弃，可以更快止损。
    """
    return ver == VERSION and addr in (ADDR_MAIN, ADDR_BROADCAST) and cmd != 0


class FrameParser:
    """流式帧解析器：按 SOF 同步、按 LEN 取帧、CRC 校验；处理粘包/半包与垃圾字节。

    - ``check_header``（默认开）：SOF 后的 VER / ADDR / CMD 不满足 ``header_plausible`` 视为垃圾，
      丢掉这个 SOF 立刻重新同步（计入 ``header_errors``），不等 LEN 字节到齐；LEN 超限同样计入；

    - 可以一个字节一个字节地 ``feed``，内部用 bytearray 缓冲；
    - CRC 错误 / LEN 超限：丢弃当前 SOF，从下一个 SOF 重新同步；
    - 缓冲区超过 ``max_buffer`` 时丢弃最旧字节（防止无 SOF 的垃圾流撑爆内存）；
    - 计数器：``frames_ok`` / ``crc_errors`` / ``resyncs``（失步次数）/ ``bytes_dropped``（丢弃字节数）。
    """

    def __init__(self, max_buffer: int = DEFAULT_PARSER_BUFFER, *, check_header: bool = True):
        if max_buffer < HEADER_LEN + CRC_LEN:
            raise ProtocolError(f"max_buffer={max_buffer} 过小")
        self.max_buffer = max_buffer
        self.check_header = bool(check_header)
        self._buf = bytearray()
        self._in_sync = True
        self.frames_ok = 0
        self.crc_errors = 0
        self.header_errors = 0
        self.resyncs = 0
        self.bytes_dropped = 0

    def reset(self) -> None:
        """清空缓冲与计数器（重连后调用）。"""
        self._buf.clear()
        self._in_sync = True
        self.frames_ok = 0
        self.crc_errors = 0
        self.header_errors = 0
        self.resyncs = 0
        self.bytes_dropped = 0

    @property
    def pending(self) -> int:
        """缓冲区中尚未成帧的字节数。"""
        return len(self._buf)

    def _drop(self, n: int) -> None:
        if n <= 0:
            return
        del self._buf[:n]
        self.bytes_dropped += n
        if self._in_sync:
            self._in_sync = False
            self.resyncs += 1

    def feed(self, data: bytes) -> list[McuFrame]:
        """喂入任意长度的字节，返回这次新解析出的完整帧（可能为空列表）。"""
        buf = self._buf
        buf += data
        out: list[McuFrame] = []
        while True:
            idx = buf.find(SOF)
            if idx < 0:
                # 没有完整 SOF：若尾字节是 SOF 的前半个，保留它等待后续字节
                keep = 1 if buf and buf[-1] == SOF[0] else 0
                self._drop(len(buf) - keep)
                break
            if idx > 0:
                self._drop(idx)
            if len(buf) < HEADER_LEN:
                break
            ver, seq, addr, cmd, length = struct.unpack_from(HEADER_BODY_FMT, buf, len(SOF))
            if length > MAX_PAYLOAD or (self.check_header and not header_plausible(ver, addr, cmd)):
                self.header_errors += 1
                self._drop(len(SOF))  # 当作垃圾，从下一个 SOF 重新同步，不等 LEN 字节到齐
                continue
            total = HEADER_LEN + length + CRC_LEN
            if len(buf) < total:
                break
            crc_calc = crc16_modbus(memoryview(buf)[len(SOF) : HEADER_LEN + length])
            (crc_rx,) = struct.unpack_from("<H", buf, HEADER_LEN + length)
            if crc_calc != crc_rx:
                self.crc_errors += 1
                self._drop(len(SOF))
                continue
            payload = bytes(buf[HEADER_LEN : HEADER_LEN + length])
            del buf[:total]
            self.frames_ok += 1
            self._in_sync = True
            out.append(McuFrame(seq=seq, addr=addr, cmd=cmd, payload=payload, ver=ver))
        if len(buf) > self.max_buffer:
            self._drop(len(buf) - self.max_buffer)
        return out


def decode(buf: bytes) -> list[McuFrame]:
    """一次性解析一段字节中的所有完整帧（便捷函数；残缺尾部被忽略）。"""
    return FrameParser().feed(buf)
