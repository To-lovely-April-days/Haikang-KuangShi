"""海康相机驱动：用假的 ``MvCameraControl_class`` 模块覆盖全部逻辑（Mac 上无 SDK 也能跑）。

假 SDK 以 ctypes 结构体镜像真实 MvImport 中用到的字段（设备列表 / 帧输出 / 节点值结构），
``FakeWorld`` 持有一台假相机的节点表、帧源与故障注入开关，并记录全部 SDK 调用。
"""

# ruff: noqa: N801, N802

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time
import types
from ctypes import (
    POINTER,
    Structure,
    Union,
    c_char,
    c_float,
    c_int,
    c_int64,
    c_ubyte,
    c_uint,
    c_ushort,
)

import numpy as np
import pytest

from oresorter.camera import CameraDisconnected, CameraError, CameraTimeout, build_camera, hik
from oresorter.camera.base import CameraParams
from oresorter.camera.hik import HikCamera
from oresorter.config import CameraConfig

MV_OK = 0
MV_E_HANDLE = 0x80000000
MV_E_NODATA = 0x80000007
MV_E_ABNORMAL_IMAGE = 0x8000000B
MV_E_GC_ARGUMENT = 0x80000101
MV_E_GC_RANGE = 0x80000102
MV_E_GC_ACCESS = 0x80000106
MV_E_ACCESS_DENIED = 0x80000203
MV_E_NETER = 0x80000206
PIX_RGB8 = 0x02180014
PIX_BGR8 = 0x02180015
PIX_MONO8 = 0x01080001
PIX_BAYER_RG8 = 0x01080009
WIDTH = 4096


# ------------------------------------------------------------------------- ctypes 结构体
class _GigEInfo(Structure):
    _fields_ = [
        ("nIpCfgOption", c_uint),
        ("nIpCfgCurrent", c_uint),
        ("nCurrentIp", c_uint),
        ("nCurrentSubNetMask", c_uint),
        ("nDefultGateWay", c_uint),
        ("chManufacturerName", c_ubyte * 32),
        ("chModelName", c_ubyte * 32),
        ("chDeviceVersion", c_ubyte * 32),
        ("chManufacturerSpecificInfo", c_ubyte * 48),
        ("chSerialNumber", c_ubyte * 16),
        ("chUserDefinedName", c_ubyte * 16),
        ("nNetExport", c_uint),
        ("nReserved", c_uint * 4),
    ]


class _SpecialInfo(Union):
    _fields_ = [("stGigEInfo", _GigEInfo), ("nReserved", c_uint * 128)]


class MV_CC_DEVICE_INFO(Structure):
    _fields_ = [
        ("nMajorVer", c_ushort),
        ("nMinorVer", c_ushort),
        ("nMacAddrHigh", c_uint),
        ("nMacAddrLow", c_uint),
        ("nTLayerType", c_uint),
        ("nReserved", c_uint * 4),
        ("SpecialInfo", _SpecialInfo),
    ]


class MV_CC_DEVICE_INFO_LIST(Structure):
    _fields_ = [("nDeviceNum", c_uint), ("pDeviceInfo", POINTER(MV_CC_DEVICE_INFO) * 256)]


class MV_FRAME_OUT_INFO_EX(Structure):
    _fields_ = [
        ("nWidth", c_ushort),
        ("nHeight", c_ushort),
        ("enPixelType", c_int),
        ("nFrameNum", c_uint),
        ("nDevTimeStampHigh", c_uint),
        ("nDevTimeStampLow", c_uint),
        ("nReserved0", c_uint),
        ("nHostTimeStamp", c_int64),
        ("nFrameLen", c_uint),
        ("nLostPacket", c_uint),
        ("nReserved", c_uint * 16),
    ]


class MV_FRAME_OUT(Structure):
    _fields_ = [
        ("pBufAddr", POINTER(c_ubyte)),
        ("stFrameInfo", MV_FRAME_OUT_INFO_EX),
        ("nRes", c_uint * 16),
    ]


class MVCC_INTVALUE_EX(Structure):
    _fields_ = [
        ("nCurValue", c_int64),
        ("nMax", c_int64),
        ("nMin", c_int64),
        ("nInc", c_int64),
        ("nReserved", c_uint * 16),
    ]


class MVCC_FLOATVALUE(Structure):
    _fields_ = [
        ("fCurValue", c_float),
        ("fMax", c_float),
        ("fMin", c_float),
        ("nReserved", c_uint * 4),
    ]


class MVCC_ENUMVALUE(Structure):
    _fields_ = [
        ("nCurValue", c_uint),
        ("nSupportedNum", c_uint),
        ("nSupportValue", c_uint * 64),
        ("nReserved", c_uint * 4),
    ]


class MVCC_ENUMENTRY(Structure):
    _fields_ = [("nValue", c_uint), ("chSymbolic", c_char * 64), ("nReserved", c_uint * 4)]


def _fill(arr, text: str) -> None:
    raw = text.encode()
    for i in range(len(arr)):
        arr[i] = raw[i] if i < len(raw) else 0


def _ip_int(ip: str) -> int:
    a, b, c, d = (int(x) for x in ip.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


# ------------------------------------------------------------------------- 假相机世界
class _IntNode:
    def __init__(self, cur, lo, hi, inc=1):
        self.cur, self.lo, self.hi, self.inc = int(cur), int(lo), int(hi), int(inc)


class _FloatNode:
    def __init__(self, cur, lo, hi):
        self.cur, self.lo, self.hi = float(cur), float(lo), float(hi)


class _EnumNode:
    def __init__(self, entries: dict[str, int], cur: str):
        self.entries = dict(entries)
        self.cur = self.entries[cur]

    def symbolic(self) -> str | None:
        for k, v in self.entries.items():
            if v == self.cur:
                return k
        return None


class _BoolNode:
    def __init__(self, cur: bool):
        self.cur = bool(cur)


def default_nodes() -> dict:
    """接近 MV-CL043-A1GC 的节点表（步进为测试取值，真机以 MVS 实测为准）。"""
    return {
        "Width": _IntNode(4096, 64, 4096, 16),
        "OffsetX": _IntNode(0, 0, 4032, 16),
        "Height": _IntNode(4096, 1, 4096, 1),
        "ExposureTime": _FloatNode(50.0, 3.0, 10000.0),
        "AcquisitionLineRate": _IntNode(9000, 100, 9000, 1),
        "AcquisitionLineRateEnable": _BoolNode(True),
        "GevSCPSPacketSize": _IntNode(1500, 500, 9000, 1),
        "PixelFormat": _EnumNode(
            {
                "Mono8": PIX_MONO8,
                "BayerRG8": PIX_BAYER_RG8,
                "RGB8Packed": PIX_RGB8,
                "BGR8Packed": PIX_BGR8,
            },
            "BayerRG8",
        ),
        "BalanceWhiteAuto": _EnumNode({"Off": 0, "Continuous": 1, "Once": 2}, "Off"),
        "ExposureAuto": _EnumNode({"Off": 0, "Once": 1, "Continuous": 2}, "Off"),
        "TriggerSelector": _EnumNode({"FrameBurstStart": 6, "LineStart": 9}, "LineStart"),
        "TriggerMode": _EnumNode({"Off": 0, "On": 1}, "Off"),
        "TriggerSource": _EnumNode(
            {
                "Line0": 0,
                "Line1": 1,
                "Line2": 2,
                "Line3": 3,
                "Counter0": 4,
                "EncoderModuleOut": 6,
                "Software": 7,
                "FrequencyConverter": 8,
            },
            "Line0",
        ),
        "EncoderSelector": _EnumNode({"Encoder0": 0}, "Encoder0"),
        "EncoderSourceA": _EnumNode({"Line0": 0, "Line1": 1}, "Line0"),
        "EncoderSourceB": _EnumNode({"Line0": 0, "Line1": 1}, "Line1"),
        "EncoderTriggerMode": _EnumNode({"AnyDirection": 0, "ForwardOnly": 1}, "AnyDirection"),
        # 故意不提供 EncoderCounterMode：验证可选节点缺失只告警
    }


class FakeWorld:
    """一台假相机：设备列表、节点表、帧源、故障注入与调用记录。"""

    def __init__(self):
        self.devices = [
            {
                "ip": "192.168.10.10",
                "serial": "DA1234567",
                "model": "MV-CL043-A1GC",
                "name": "cam1",
            },
            {
                "ip": "192.168.10.11",
                "serial": "DA7654321",
                "model": "MV-CL043-A1GC",
                "name": "cam2",
            },
        ]
        self.nodes = default_nodes()
        self.online = True
        self.enum_fail_code = MV_OK
        self.open_fail_code = MV_E_ACCESS_DENIED
        self.open_fail_times = 0
        self.optimal_packet = 8164
        self.auto_emit = False
        self.pending: queue.Queue = queue.Queue()
        self.grab_errors: list[int] = []
        self.next_frame_num = 1
        self.grabbing = False
        self.open_count = 0
        self.live_handles = 0
        self.exception_cb = None
        self.calls: list[tuple] = []
        self.emitted: dict[int, np.ndarray] = {}
        self.outstanding_buffers = 0
        self._infos: list = []
        self.lock = threading.Lock()

    # ---- 测试侧控制 ----
    def emit(
        self, n: int = 1, *, skip: int = 0, height: int | None = None, pixel: int = PIX_RGB8
    ) -> None:
        """排队 n 帧；``skip`` 先跳过若干帧号（模拟 SDK 丢块）。"""
        with self.lock:
            self.next_frame_num += skip
            for _ in range(n):
                fnum = self.next_frame_num
                self.next_frame_num += 1
                h = height or self.nodes["Height"].cur
                w = self.nodes["Width"].cur
                img = self._make_image(fnum, h, w)
                self.emitted[fnum] = img
                self.pending.put((fnum, h, w, pixel, img))

    @staticmethod
    def _make_image(fnum: int, h: int, w: int) -> np.ndarray:
        img = np.empty((h, w, 3), np.uint8)
        img[:, :, 0] = (np.arange(h) % 256)[:, None]
        img[:, :, 1] = fnum % 256
        img[:, :, 2] = 7
        return img

    def fire_exception(self, msg_type: int) -> None:
        assert self.exception_cb is not None, "驱动未注册异常回调"
        self.exception_cb(msg_type, None)

    def set_calls(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]

    def reset_nodes(self) -> None:
        """模拟相机断电重启：节点恢复出厂值。"""
        self.nodes = default_nodes()


class FakeMvCamera:
    """``MvCameraControl_class.MvCamera`` 的假实现（只实现驱动用到的方法）。"""

    world: FakeWorld

    def __init__(self):
        self._open = False
        self._handle = None
        self._buffers: dict[int, object] = {}

    # ---- 静态 ----
    @classmethod
    def MV_CC_Initialize(cls):
        cls.world.calls.append(("Initialize",))
        return MV_OK

    @classmethod
    def MV_CC_EnumDevices(cls, layer, lst):
        w = cls.world
        w.calls.append(("EnumDevices", layer))
        if w.enum_fail_code:
            return w.enum_fail_code
        w._infos = []
        lst.nDeviceNum = len(w.devices)
        for i, d in enumerate(w.devices):
            info = MV_CC_DEVICE_INFO()
            info.nTLayerType = 1
            g = info.SpecialInfo.stGigEInfo
            g.nCurrentIp = _ip_int(d["ip"])
            _fill(g.chSerialNumber, d["serial"])
            _fill(g.chModelName, d["model"])
            _fill(g.chUserDefinedName, d["name"])
            w._infos.append(info)
            lst.pDeviceInfo[i] = ctypes.pointer(info)
        return MV_OK

    # ---- 句柄与连接 ----
    def MV_CC_CreateHandle(self, info):
        self.world.calls.append(("CreateHandle", hik._describe_device(info)["ip"]))
        self._handle = info
        self.world.live_handles += 1
        return MV_OK

    def MV_CC_DestroyHandle(self):
        self.world.calls.append(("DestroyHandle",))
        self._handle = None
        self.world.live_handles -= 1
        return MV_OK

    def MV_CC_OpenDevice(self, mode=1, key=0):
        w = self.world
        w.calls.append(("OpenDevice", mode))
        if w.open_fail_times > 0:
            w.open_fail_times -= 1
            return w.open_fail_code
        if not w.online:
            return MV_E_NETER
        self._open = True
        w.open_count += 1
        return MV_OK

    def MV_CC_CloseDevice(self):
        self.world.calls.append(("CloseDevice",))
        self._open = False
        return MV_OK

    def MV_CC_IsDeviceConnected(self):
        return self._open and self.world.online

    def MV_CC_GetOptimalPacketSize(self):
        return self.world.optimal_packet

    def MV_GIGE_SetResend(self, enable, pct=10, timeout=50):
        self.world.calls.append(("SetResend", enable, pct, timeout))
        return MV_OK

    def MV_CC_SetImageNodeNum(self, n):
        self.world.calls.append(("SetImageNodeNum", n))
        return MV_OK

    def MV_CC_RegisterExceptionCallBack(self, cb, user):
        self.world.exception_cb = cb
        self.world.calls.append(("RegisterExceptionCallBack",))
        return MV_OK

    # ---- 节点 ----
    def _node(self, key, kind):
        if not self._open:
            return MV_E_HANDLE, None
        if not self.world.online:
            return MV_E_NETER, None
        n = self.world.nodes.get(key)
        if n is None or not isinstance(n, kind):
            return MV_E_GC_ACCESS, None
        return MV_OK, n

    def MV_CC_GetIntValueEx(self, key, st):
        ret, n = self._node(key, _IntNode)
        if ret:
            return ret
        st.nCurValue, st.nMin, st.nMax, st.nInc = n.cur, n.lo, n.hi, n.inc
        return MV_OK

    def MV_CC_SetIntValueEx(self, key, value):
        ret, n = self._node(key, _IntNode)
        self.world.calls.append(("SetInt", key, int(value)))
        if ret:
            return ret
        v = int(value)
        if v < n.lo or v > n.hi or (v - n.lo) % n.inc:
            return MV_E_GC_RANGE
        if key == "Width" and v + self.world.nodes["OffsetX"].cur > 4096:
            return MV_E_GC_RANGE
        if key == "OffsetX" and v + self.world.nodes["Width"].cur > 4096:
            return MV_E_GC_RANGE
        n.cur = v
        return MV_OK

    def MV_CC_GetFloatValue(self, key, st):
        ret, n = self._node(key, _FloatNode)
        if ret:
            return ret
        st.fCurValue, st.fMin, st.fMax = n.cur, n.lo, n.hi
        return MV_OK

    def MV_CC_SetFloatValue(self, key, value):
        ret, n = self._node(key, _FloatNode)
        self.world.calls.append(("SetFloat", key, float(value)))
        if ret:
            return ret
        if value < n.lo or value > n.hi:
            return MV_E_GC_RANGE
        n.cur = float(value)
        return MV_OK

    def MV_CC_GetEnumValue(self, key, st):
        ret, n = self._node(key, _EnumNode)
        if ret:
            return ret
        st.nCurValue = n.cur
        return MV_OK

    def MV_CC_SetEnumValueByString(self, key, sym):
        ret, n = self._node(key, _EnumNode)
        self.world.calls.append(("SetEnum", key, sym))
        if ret:
            return ret
        if sym not in n.entries:
            return MV_E_GC_ARGUMENT
        n.cur = n.entries[sym]
        return MV_OK

    def MV_CC_GetEnumEntrySymbolic(self, key, entry):
        ret, n = self._node(key, _EnumNode)
        if ret:
            return ret
        for k, v in n.entries.items():
            if v == int(entry.nValue):
                entry.chSymbolic = k.encode()
                return MV_OK
        return MV_E_GC_ARGUMENT

    def MV_CC_GetBoolValue(self, key, b):
        ret, n = self._node(key, _BoolNode)
        if ret:
            return ret
        b.value = n.cur
        return MV_OK

    def MV_CC_SetBoolValue(self, key, flag):
        ret, n = self._node(key, _BoolNode)
        self.world.calls.append(("SetBool", key, bool(flag)))
        if ret:
            return ret
        n.cur = bool(flag)
        return MV_OK

    # ---- 取流 ----
    def MV_CC_StartGrabbing(self):
        w = self.world
        w.calls.append(("StartGrabbing", w.nodes["Height"].cur))
        if not self._open:
            return MV_E_HANDLE
        if not w.online:
            return MV_E_NETER
        with w.lock:
            w.grabbing = True
            w.next_frame_num = 1
        return MV_OK

    def MV_CC_StopGrabbing(self):
        self.world.calls.append(("StopGrabbing",))
        self.world.grabbing = False
        return MV_OK

    def MV_CC_GetImageBuffer(self, st, ms):
        w = self.world
        wait = min(ms / 1000.0, 0.02)
        if not self._open or not w.online or not w.grabbing:
            time.sleep(wait)
            return MV_E_NODATA
        if w.grab_errors:
            return w.grab_errors.pop(0)
        try:
            fnum, h, width, pixel, img = w.pending.get(timeout=wait)
        except queue.Empty:
            if not w.auto_emit:
                return MV_E_NODATA
            with w.lock:
                fnum = w.next_frame_num
                w.next_frame_num += 1
                h, width = w.nodes["Height"].cur, w.nodes["Width"].cur
                img = w._make_image(fnum, h, width)
                w.emitted[fnum] = img
                pixel = PIX_RGB8
        if pixel == PIX_BGR8:
            payload = np.ascontiguousarray(img[:, :, ::-1]).tobytes()
        elif pixel == PIX_MONO8:
            payload = np.ascontiguousarray(img[:, :, 0]).tobytes()
        else:
            payload = img.tobytes()
        buf = (c_ubyte * len(payload)).from_buffer_copy(payload)
        self._buffers[ctypes.addressof(buf)] = buf
        w.outstanding_buffers += 1
        st.pBufAddr = ctypes.cast(buf, POINTER(c_ubyte))
        info = st.stFrameInfo
        info.nWidth, info.nHeight = width, h
        info.enPixelType = pixel
        info.nFrameNum = fnum
        info.nFrameLen = len(payload)
        return MV_OK

    def MV_CC_FreeImageBuffer(self, st):
        addr = ctypes.addressof(st.pBufAddr.contents)
        if self._buffers.pop(addr, None) is None:
            return MV_E_GC_ARGUMENT
        self.world.outstanding_buffers -= 1
        return MV_OK


def make_fake_sdk(world: FakeWorld) -> types.ModuleType:
    mod = types.ModuleType("MvCameraControl_class")
    cam_cls = type("MvCamera", (FakeMvCamera,), {"world": world})
    mod.MvCamera = cam_cls
    for name, obj in {
        "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
        "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
        "MV_FRAME_OUT": MV_FRAME_OUT,
        "MV_FRAME_OUT_INFO_EX": MV_FRAME_OUT_INFO_EX,
        "MVCC_INTVALUE_EX": MVCC_INTVALUE_EX,
        "MVCC_FLOATVALUE": MVCC_FLOATVALUE,
        "MVCC_ENUMVALUE": MVCC_ENUMVALUE,
        "MVCC_ENUMENTRY": MVCC_ENUMENTRY,
        "MV_GIGE_DEVICE": 1,
        "MV_ACCESS_Exclusive": 1,
        "MV_OK": 0,
    }.items():
        setattr(mod, name, obj)
    return mod


# ------------------------------------------------------------------------- fixtures
@pytest.fixture
def world(monkeypatch) -> FakeWorld:
    w = FakeWorld()
    monkeypatch.setattr(hik, "_import_sdk", lambda: make_fake_sdk(w))
    return w


def hik_cfg(**over) -> CameraConfig:
    base = {"id": 1, "kind": "hik", "ip": "192.168.10.10", "runtime_chunk_lines": 512}
    base.update(over)
    return CameraConfig(**base)


def new_cam(world: FakeWorld, **over) -> HikCamera:
    cam = HikCamera(hik_cfg(**over))
    cam.grab_timeout_ms = 20
    cam.reconnect_backoff_s = (0.01, 0.02, 0.05)
    cam.awb_settle_s = 0.0
    cam.offline_after_s = 0.3
    return cam


@pytest.fixture
def cam(world: FakeWorld):
    c = new_cam(world)
    yield c
    c.close()


def wait_until(pred, timeout: float = 3.0, what: str = "条件") -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError(f"{timeout}s 内未满足: {what}")


def hik_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("hikcam")]


# ------------------------------------------------------------------------- SDK 导入
def test_import_sdk_missing_gives_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("MVS_PYTHON_DIR", str(tmp_path / "nope"))
    monkeypatch.setitem(sys.modules, "MvCameraControl_class", None)
    with pytest.raises(CameraError, match="MVS SDK"):
        hik._import_sdk()
    with pytest.raises(CameraError, match="MVS SDK"):
        build_camera(hik_cfg())


def test_import_sdk_via_sys_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("MVS_PYTHON_DIR", str(tmp_path / "nope"))
    w = FakeWorld()
    mod = make_fake_sdk(w)
    monkeypatch.setitem(sys.modules, "MvCameraControl_class", mod)
    assert hik._import_sdk() is mod
    cam = build_camera(hik_cfg())
    assert isinstance(cam, HikCamera)
    cam.open()
    assert cam.connected
    cam.close()


def test_build_camera_mock_path_unaffected(tmp_path):
    from oresorter.camera.mock import MockCamera

    assert isinstance(build_camera(CameraConfig(id=1, kind="mock")), MockCamera)


# ------------------------------------------------------------------------- 枚举 / 打开
def test_open_by_ip_sets_up_camera(world, cam):
    cam.open()
    assert cam.connected and not cam.streaming
    assert cam.device_info["serial"] == "DA1234567"
    assert world.set_calls("CreateHandle") == [("CreateHandle", "192.168.10.10")]
    assert ("SetEnum", "PixelFormat", "RGB8Packed") in world.calls
    assert world.nodes["PixelFormat"].cur == PIX_RGB8
    assert world.nodes["GevSCPSPacketSize"].cur == 8164
    assert world.set_calls("SetResend")[0][1] == 1
    assert world.set_calls("SetImageNodeNum")[0][1] >= 8
    assert ("Initialize",) in world.calls
    assert world.exception_cb is not None
    # 内触发：TriggerMode Off、固定行频开启
    assert world.nodes["TriggerMode"].symbolic() == "Off"
    assert world.nodes["AcquisitionLineRateEnable"].cur is True
    assert world.nodes["AcquisitionLineRate"].cur == 9000
    assert world.nodes["ExposureTime"].cur == 50.0


def test_open_by_serial(world):
    cam = new_cam(world, ip=None, serial="DA7654321")
    cam.open()
    assert cam.device_info["ip"] == "192.168.10.11"
    cam.close()


def test_open_no_match_lists_found_devices(world):
    cam = new_cam(world, ip="192.168.10.99")
    with pytest.raises(CameraError, match=r"192\.168\.10\.10") as ei:
        cam.open()
    assert "未找到" in str(ei.value)
    assert not cam.connected
    assert world.live_handles == 0


def test_open_device_failure_maps_error_code(world, cam):
    world.open_fail_times = 1
    world.open_fail_code = MV_E_ACCESS_DENIED
    with pytest.raises(CameraError) as ei:
        cam.open()
    assert "0x80000203" in str(ei.value) and "访问权限" in str(ei.value)
    assert world.live_handles == 0, "打开失败后必须销毁句柄"
    assert not cam.connected
    # 恢复后可再次打开
    cam.open()
    assert cam.connected


def test_enum_failure_is_camera_error(world, cam):
    world.enum_fail_code = MV_E_NETER
    with pytest.raises(CameraError, match="0x80000206"):
        cam.open()


def test_open_awb_once_locks_to_off(world):
    world.auto_emit = True
    cam = new_cam(world, awb_mode="once")
    cam.open()
    awb = [c for c in world.calls if c[0] == "SetEnum" and c[1] == "BalanceWhiteAuto"]
    assert [c[2] for c in awb] == ["Once", "Off"]
    assert cam.get_params().awb_mode == "off"
    assert not world.grabbing, "Once 时的临时抓帧结束后必须停流"
    cam.close()


def test_open_encoder_trigger(world, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="oresorter.camera.hik"):
        cam = new_cam(world, trigger="encoder")
        cam.open()
    assert world.nodes["TriggerMode"].symbolic() == "On"
    assert world.nodes["TriggerSource"].symbolic() == "EncoderModuleOut"
    assert world.nodes["TriggerSelector"].symbolic() == "LineStart"
    assert world.nodes["AcquisitionLineRateEnable"].cur is False
    assert cam.get_params().trigger == "encoder"
    # 缺失的 EncoderCounterMode 只告警
    assert any(
        "EncoderCounterMode" in r.getMessage() or r.__dict__.get("node") == "EncoderCounterMode"
        for r in caplog.records
    )
    cam.close()


# ------------------------------------------------------------------------- 参数
def test_params_roundtrip_and_step_alignment(world, cam):
    cam.open()
    p = cam.get_params()
    assert p == CameraParams(
        awb_mode="off",
        exposure_us=50.0,
        width=4096,
        offset_x=0,
        height=4096,
        line_rate_hz=9000,
        trigger="internal",
    )
    p.width = 4070  # inc 16 → 4064
    p.offset_x = 24  # inc 16 → 16
    p.exposure_us = 77.0
    p.height = 1024
    p.line_rate_hz = 6000
    applied = cam.set_params(p)
    assert (applied.width, applied.offset_x) == (4064, 16)
    assert applied.exposure_us == 77.0 and applied.height == 1024 and applied.line_rate_hz == 6000
    assert world.nodes["Width"].cur == 4064 and world.nodes["OffsetX"].cur == 16
    assert world.nodes["Height"].cur == 1024
    assert cam.get_params() == applied


def test_float_param_clamped_to_camera_range(world, cam):
    world.nodes["ExposureTime"].hi = 5000.0
    cam.open()
    p = cam.get_params()
    p.exposure_us = 8000.0
    assert cam.set_params(p).exposure_us == 5000.0


def test_params_require_connection(world, cam):
    with pytest.raises(CameraDisconnected):
        cam.get_params()
    with pytest.raises(CameraDisconnected):
        cam.set_params(CameraParams())
    with pytest.raises(CameraDisconnected):
        cam.snapshot(64)
    with pytest.raises(CameraDisconnected):
        cam.start_stream(512)


def test_awb_once_endpoint(world, cam):
    world.auto_emit = True
    cam.open()
    world.calls.clear()
    cam.awb_once()
    awb = [c[2] for c in world.calls if c[0] == "SetEnum" and c[1] == "BalanceWhiteAuto"]
    assert awb == ["Once", "Off"]
    assert cam.get_params().awb_mode == "off"


def test_enum_symbolic_fallback_table(world, monkeypatch, cam):
    monkeypatch.delattr(FakeMvCamera, "MV_CC_GetEnumEntrySymbolic")
    cam.open()
    p = cam.get_params()
    assert p.awb_mode == "off" and p.trigger == "internal"
    p.awb_mode = "continuous"
    assert cam.set_params(p).awb_mode == "continuous"
    assert cam.get_params().awb_mode == "continuous"


# ------------------------------------------------------------------------- 流式采集
def test_stream_frames_contiguous(world, cam):
    cam.open()
    cam.start_stream(512)
    assert cam.streaming
    assert world.nodes["Height"].cur == 512
    assert world.set_calls("StartGrabbing")[-1] == ("StartGrabbing", 512)
    world.emit(5)
    frames = [cam.get_frame(2.0) for _ in range(5)]
    assert all(f is not None for f in frames)
    for i, f in enumerate(frames):
        assert f.image.shape == (512, WIDTH, 3) and f.image.dtype == np.uint8
        assert f.seq == i and f.line_start == i * 512 and f.camera_id == 1
        np.testing.assert_array_equal(f.image, world.emitted[i + 1])
        assert f.image.flags["C_CONTIGUOUS"] and f.image.flags["WRITEABLE"]
    assert cam.stats.chunks_total == 5 and cam.stats.dropped_chunks == 0
    wait_until(lambda: world.outstanding_buffers == 0, what="全部帧缓冲归还")
    cam.stop_stream()
    assert not cam.streaming and not world.grabbing
    wait_until(lambda: not [t for t in hik_threads() if t.endswith("-grab")], what="取帧线程退出")
    assert cam.connected


def test_frame_gap_compensates_line_start(world, cam):
    cam.open()
    cam.start_stream(512)
    world.emit(2)
    world.emit(1, skip=3)
    world.emit(1)
    frames = [cam.get_frame(2.0) for _ in range(4)]
    assert [f.seq for f in frames] == [0, 1, 5, 6]
    assert [f.line_start for f in frames] == [0, 512, 5 * 512, 6 * 512]
    assert cam.stats.dropped_chunks == 3
    assert cam.get_frame(0.05) is None


def test_get_frame_timeout_returns_none(world, cam):
    cam.open()
    cam.start_stream(256)
    t0 = time.monotonic()
    assert cam.get_frame(0.1) is None
    assert time.monotonic() - t0 < 1.0
    assert cam.connected and cam.streaming


def test_reset_line_counter(world, cam):
    cam.open()
    cam.start_stream(128)
    world.emit(3)
    for _ in range(3):
        assert cam.get_frame(2.0) is not None
    cam.reset_line_counter()
    world.emit(1)
    f = cam.get_frame(2.0)
    assert f is not None and f.line_start == 0 and f.seq == 3


def test_queue_full_counts_dropped(world, monkeypatch):
    monkeypatch.setattr(hik, "QUEUE_SIZE", 2)
    cam = new_cam(world)
    cam.open()
    cam.start_stream(128)
    world.emit(5)
    wait_until(lambda: cam.stats.dropped_chunks >= 3, what="队列满丢块计数")
    f = cam.get_frame(1.0)
    assert f is not None and f.seq == 0
    cam.close()


def test_pixel_format_fallbacks(world, cam):
    cam.open()
    cam.start_stream(64)
    world.emit(1, pixel=PIX_BGR8)
    world.emit(1, pixel=PIX_MONO8)
    f1, f2 = cam.get_frame(2.0), cam.get_frame(2.0)
    np.testing.assert_array_equal(f1.image, world.emitted[1])
    assert f2.image.shape == (64, WIDTH, 3)
    np.testing.assert_array_equal(f2.image[:, :, 1], world.emitted[2][:, :, 0])


def test_frames_iterator_and_timeout(world, cam):
    cam.open()
    cam.start_stream(64)
    world.emit(2)
    it = cam.frames(timeout=0.2)
    assert next(it).seq == 0 and next(it).seq == 1
    with pytest.raises(CameraTimeout):
        next(it)


# ------------------------------------------------------------------------- 快照
def test_snapshot_idle_restores_height(world, cam):
    world.auto_emit = True
    cam.open()
    img = cam.snapshot(300)
    assert img.shape == (300, WIDTH, 3) and img.dtype == np.uint8
    assert world.nodes["Height"].cur == 4096, "快照后应恢复采样帧高"
    assert not world.grabbing and world.outstanding_buffers == 0
    assert ("StartGrabbing", 300) in world.calls


def test_snapshot_idle_timeout(world, cam):
    cam.open()
    cam._params.line_rate_hz = 9000
    with pytest.raises(CameraTimeout):
        cam.snapshot(64)
    assert cam.connected and not world.grabbing


def test_snapshot_while_streaming_tees_frames(world, cam):
    cam.open()
    cam.start_stream(128)
    result: dict = {}

    def run():
        try:
            result["img"] = cam.snapshot(300)
        except Exception as exc:
            result["err"] = exc

    t = threading.Thread(target=run)
    t.start()
    wait_until(lambda: cam._snap_req is not None, what="快照登记")
    world.emit(3)
    t.join(3.0)
    assert "err" not in result, result.get("err")
    assert result["img"].shape == (300, WIDTH, 3)
    np.testing.assert_array_equal(result["img"][:128], world.emitted[1])
    # 主消费者仍能拿到全部 3 块
    frames = [cam.get_frame(1.0) for _ in range(3)]
    assert [f.seq for f in frames] == [0, 1, 2]


def test_snapshot_while_streaming_timeout(world, cam):
    cam.open()
    cam.start_stream(128)
    cam._params.line_rate_hz = 9000
    # 把等待上限压到很短：直接 monkeypatch 实例方法
    cam._collect_timeout = lambda h: 0.1  # type: ignore[method-assign]
    with pytest.raises(CameraTimeout):
        cam.snapshot(256)
    assert cam._snap_req is None


# ------------------------------------------------------------------------- 取流中改参数
def test_set_params_while_streaming(world, cam):
    cam.open()
    cam.start_stream(512)
    world.calls.clear()
    p = cam.get_params()
    p.exposure_us = 120.0
    out = cam.set_params(p)
    assert out.exposure_us == 120.0 and world.nodes["ExposureTime"].cur == 120.0
    assert not world.set_calls("StopGrabbing"), "只改曝光不应重启取流"
    assert cam.get_params().height == 4096, "取流中 get_params 的 height 应为采样帧高而非块高"
    # 改几何 → 停流重开，块高仍为 512
    p.width = 2048
    out = cam.set_params(p)
    assert out.width == 2048 and world.nodes["Width"].cur == 2048
    assert world.set_calls("StopGrabbing") and world.set_calls("StartGrabbing")[-1] == (
        "StartGrabbing",
        512,
    )
    assert cam.streaming and world.grabbing
    world.emit(1)
    f = cam.get_frame(2.0)
    assert f is not None and f.image.shape == (512, 2048, 3)


# ------------------------------------------------------------------------- 掉线与重连
def test_offline_reconnect_restores_params_and_stream(world, cam):
    cam.open()
    p = cam.get_params()
    p.exposure_us = 88.0
    cam.set_params(p)
    cam.start_stream(512)
    world.emit(2)
    assert cam.get_frame(2.0).seq == 0 and cam.get_frame(2.0).seq == 1
    # 拔网线
    world.online = False
    wait_until(lambda: not cam.connected, what="检测到掉线")
    assert cam.streaming, "掉线期间保持开流意图"
    with pytest.raises(CameraDisconnected):
        cam.get_frame(0.05)
    with pytest.raises(CameraDisconnected):
        cam.get_params()
    assert cam.stats.errors >= 1
    # 相机断电重启：节点恢复出厂；前 2 次打开仍失败，验证退避重试
    world.reset_nodes()
    world.open_fail_times = 2
    world.online = True
    wait_until(lambda: cam.connected, what="重连成功")
    assert cam.stats.reconnects == 1
    assert world.open_count == 2
    assert world.nodes["ExposureTime"].cur == 88.0, "重连后恢复参数"
    assert world.nodes["PixelFormat"].cur == PIX_RGB8
    assert world.nodes["Height"].cur == 512 and world.grabbing, "重连后重新开流"
    world.emit(1)
    f = cam.get_frame(2.0)
    assert f is not None
    assert f.seq == 2 and f.line_start == 1024, "重连后帧号重新计数不应被当作丢块，行号接着走"
    assert cam.stats.dropped_chunks == 0
    assert len([t for t in hik_threads() if t.endswith("-grab")]) == 1


def test_offline_detected_by_node_write(world, cam):
    cam.open()
    world.online = False
    with pytest.raises(CameraDisconnected):
        cam.set_params(CameraParams())
    assert not cam.connected
    world.online = True
    wait_until(lambda: cam.connected, what="未开流状态下的后台重连")
    assert cam.stats.reconnects == 1 and not cam.streaming and not world.grabbing
    assert cam.get_params().width == 4096


def test_exception_callback_marks_offline_and_reconnects(world, cam):
    cam.open()
    cam.start_stream(256)
    world.fire_exception(hik.MV_EXCEPTION_DEV_DISCONNECT)
    assert not cam.connected
    wait_until(lambda: cam.connected and world.grabbing, what="回调触发的重连")
    assert cam.stats.reconnects == 1
    # 非断开类型的回调只记日志
    world.fire_exception(0x1234)
    assert cam.connected


def test_consecutive_grab_errors_trigger_reconnect(world, cam):
    cam.open()
    cam.start_stream(256)
    world.grab_errors = [MV_E_ABNORMAL_IMAGE] * cam.max_consecutive_errors
    wait_until(lambda: cam.stats.reconnects == 1 and cam.connected, what="连续取帧失败后重连")
    assert cam.stats.errors >= cam.max_consecutive_errors + 1
    world.emit(1)
    assert cam.get_frame(2.0) is not None


def test_internal_trigger_silence_is_offline(world, cam):
    """内触发下长时间无帧而 SDK 仍称在线（相机卡死）→ 按掉线处理并重连。"""
    cam.offline_after_s = 0.1
    cam._params.line_rate_hz = 9000
    cam.open()
    cam.start_stream(128)  # 块周期 14 ms，4 倍 < 0.1 s
    wait_until(lambda: cam.stats.reconnects >= 1, timeout=3.0, what="静默触发重连")


def test_close_during_reconnect(world, cam):
    cam.open()
    cam.start_stream(256)
    world.online = False
    wait_until(lambda: not cam.connected, what="掉线")
    t0 = time.monotonic()
    cam.close()
    assert time.monotonic() - t0 < 2.5
    wait_until(lambda: not hik_threads(), what="全部线程退出")
    assert not cam.connected and not cam.streaming


# ------------------------------------------------------------------------- 关闭
def test_close_idempotent_and_threads_joined(world, cam):
    cam.open()
    cam.start_stream(512)
    world.emit(1)
    assert cam.get_frame(1.0) is not None
    t0 = time.monotonic()
    cam.close()
    cam.close()
    assert time.monotonic() - t0 < 2.5
    assert not cam.connected and not cam.streaming
    assert not hik_threads()
    assert world.set_calls("CloseDevice") == [("CloseDevice",)]
    assert world.set_calls("DestroyHandle") == [("DestroyHandle",)]
    assert world.live_handles == 0 and not world.grabbing
    with pytest.raises(CameraDisconnected):
        cam.get_params()
    with pytest.raises(CameraDisconnected):
        cam.get_frame(0.01)
    # 关闭后可重新打开
    cam.open()
    assert cam.connected
    cam.close()


def test_close_without_open(world, cam):
    cam.close()
    assert not cam.connected


# ------------------------------------------------------------------------- 返回码
def test_error_code_mapping():
    assert hik.norm_code(-2147483641) == 0x80000007
    assert hik.norm_code(0x80000007) == 0x80000007
    assert hik.format_error(0x80000007) == "0x80000007 无数据（取帧超时）"
    assert hik.format_error(-2147483641).startswith("0x80000007")
    assert "未知错误" in hik.format_error(0x8FFFFFFF)
    assert "访问权限" in hik.format_error(MV_E_ACCESS_DENIED)


def test_check_raises_camera_error_with_code(world, cam):
    cam.open()
    with pytest.raises(CameraError, match="0x80000102"):
        cam._check(MV_E_GC_RANGE, "设置 Width")
    assert cam.connected, "普通参数错误不应判为掉线"
    world.online = False  # 让后台重连一直失败，便于断言离线状态
    with pytest.raises(CameraDisconnected, match="0x80000206"):
        cam._check(MV_E_NETER, "设置 Width")
    assert not cam.connected


def test_align_helper():
    assert hik._align(4070, 64, 4096, 16) == 4064
    assert hik._align(10, 64, 4096, 16) == 64
    assert hik._align(9999, 64, 4096, 16) == 4096
    assert hik._align(24, 0, 4032, 16) == 16
    assert hik._align(513, 1, 4096, 1) == 513


def test_to_rgb_rejects_short_buffer():
    buf = (c_ubyte * 10)()
    with pytest.raises(CameraError, match="帧长度不足"):
        hik._to_rgb(buf, 10, 4, 4, PIX_RGB8)
    with pytest.raises(CameraError, match="不支持的像素格式"):
        hik._to_rgb(buf, 10, 2, 2, 0x12345678)
