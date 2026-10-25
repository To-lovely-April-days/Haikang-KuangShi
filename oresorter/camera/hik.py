"""海康 MVS SDK 线阵相机封装（MV-CL043-A1GC：GigE，4096 px 三线 CMOS，RGB8 下最高 9 kHz 行频）。

依赖 MVS SDK（Linux aarch64）官方 Python 封装 MvImport 的 ``MvCameraControl_class`` 模块
（``MvCamera`` 类 + ctypes 结构体/常量）。开发机无 SDK 时 ``_import_sdk`` 抛出带安装提示的 ``CameraError``，
不影响 Mock 路径；单元测试（tests/test_camera_hik.py）用假的 ``MvCameraControl_class`` 模块覆盖全部逻辑。

实现要点（docs/02 §3.2）：

- 枚举 GigE 设备，按 ``cfg.ip`` / ``cfg.serial`` 匹配打开；``GevSCPSPacketSize`` 取
  ``MV_CC_GetOptimalPacketSize``；开启 GVSP 重传并加大图像缓存节点数。
- 流式采集：把相机 ``Height`` 设为 ``chunk_lines``，每个 GVSP 帧即一个块；独立线程
  ``MV_CC_GetImageBuffer`` 拉取后 memmove 到 numpy ``(h, w, 3) uint8 RGB``，``MV_CC_FreeImageBuffer`` 归还；
  ``nFrameNum`` 断号按丢失帧数补偿 ``line_start`` 并计 ``stats.dropped_chunks``。
- 掉线（SDK 返回网络错误码 / ``MV_CC_IsDeviceConnected`` 为假 / 取帧连续失败 / 异常回调上报断开）→
  ``connected=False``，后台线程指数退避重连，成功后恢复参数并重新开流，``stats.reconnects`` +1。
- 所有 SDK 调用检查返回码（非 0 → ``CameraError``，十六进制码 + 中文说明），取帧带超时；
  ``close()`` 幂等并在 2 s 内回收线程。

文件顶部的 GenICam 节点名 / 枚举项按 SFNC 与海康线阵相机惯例预置，**以 MVS 客户端属性树实测为准**，
现场核对清单见 ``deploy/jetson_checklist.md``。
"""

from __future__ import annotations

import ctypes
import importlib
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from oresorter.camera.base import (
    Camera,
    CameraDisconnected,
    CameraError,
    CameraParams,
    CameraTimeout,
    Frame,
)
from oresorter.config import CameraConfig
from oresorter.log import get_logger

log = get_logger(__name__)

# ======================================================================================
# GenICam 节点名与枚举项（以 MVS 客户端实测为准，核对清单见 deploy/jetson_checklist.md）
# ======================================================================================
NODE_PIXEL_FORMAT = "PixelFormat"
PIXEL_FORMAT_RGB8 = "RGB8Packed"  # MVS 属性树显示 "RGB 8"，符号名 RGB8Packed（旧固件可能为 RGB8）
NODE_WIDTH = "Width"
NODE_OFFSET_X = "OffsetX"
NODE_HEIGHT = "Height"  # 线阵相机：一帧包含的行数（流式时 = chunk_lines）
NODE_EXPOSURE_AUTO = "ExposureAuto"
NODE_EXPOSURE = "ExposureTime"  # 单位 µs
NODE_AWB = "BalanceWhiteAuto"  # Off / Once / Continuous
NODE_LINE_RATE_ENABLE = "AcquisitionLineRateEnable"
NODE_LINE_RATE = "AcquisitionLineRate"  # Hz
NODE_TRIGGER_SELECTOR = "TriggerSelector"
NODE_TRIGGER_MODE = "TriggerMode"
NODE_TRIGGER_SOURCE = "TriggerSource"
TRIGGER_SELECTOR_LINE = "LineStart"  # 行触发；帧触发为 FrameBurstStart
TRIGGER_SOURCE_ENCODER = "EncoderModuleOut"  # 编码器模块输出；需分频/倍频时改为 FrequencyConverter
NODE_PACKET_SIZE = "GevSCPSPacketSize"
NODE_DEVICE_SERIAL = "DeviceSerialNumber"

# 编码器行触发时顺带下发的编码器模块设置（按现场接线调整：A/B 相接的 Line 编号、计数方向）。
# 这些节点缺失或拒绝时只告警不中断（可在 MVS 客户端设置后保存到 UserSet 代替）。
ENCODER_SETUP: tuple[tuple[str, str], ...] = (
    ("EncoderSelector", "Encoder0"),
    ("EncoderSourceA", "Line0"),
    ("EncoderSourceB", "Line1"),
    ("EncoderTriggerMode", "AnyDirection"),
    ("EncoderCounterMode", "IgnoreDirection"),
)

# 枚举数值 → 符号名（SDK 缺少 MV_CC_GetEnumEntrySymbolic 时的兜底表，数值为海康 SDK 文档约定）
ENUM_SYMBOLS: dict[str, dict[int, str]] = {
    NODE_AWB: {0: "Off", 1: "Continuous", 2: "Once"},
    NODE_TRIGGER_MODE: {0: "Off", 1: "On"},
    NODE_TRIGGER_SOURCE: {
        0: "Line0",
        1: "Line1",
        2: "Line2",
        3: "Line3",
        4: "Counter0",
        6: "EncoderModuleOut",
        7: "Software",
        8: "FrequencyConverter",
    },
}
AWB_SYMBOLS: dict[str, str] = {"off": "Off", "once": "Once", "continuous": "Continuous"}

# ======================================================================================
# SDK 常量（GigE Vision / MVS 固定值，不依赖 SDK 模块属性）
# ======================================================================================
MV_OK = 0x00000000
MV_GIGE_DEVICE = 0x00000001
MV_ACCESS_EXCLUSIVE = 1
MV_EXCEPTION_DEV_DISCONNECT = 0x00008001  # 异常回调：设备断开
_CODE_MASK = 0xFFFFFFFF

MV_E_HANDLE = 0x80000000
MV_E_NODATA = 0x80000007
MV_E_GC_ACCESS = 0x80000106
MV_E_GC_TIMEOUT = 0x80000107
MV_E_BUSY = 0x80000204
MV_E_NETER = 0x80000206

ERROR_TEXT: dict[int, str] = {
    0x80000000: "句柄无效",
    0x80000001: "不支持的功能",
    0x80000002: "缓存已满",
    0x80000003: "调用顺序错误",
    0x80000004: "参数错误",
    0x80000006: "资源申请失败",
    0x80000007: "无数据（取帧超时）",
    0x80000008: "前置条件错误或运行环境已改变",
    0x80000009: "版本不匹配",
    0x8000000A: "缓存不足",
    0x8000000B: "异常图像（丢包导致图像不完整）",
    0x8000000C: "动态库加载失败",
    0x8000000D: "没有可输出的缓存",
    0x800000FF: "未知错误",
    0x80000100: "GenICam 通用错误",
    0x80000101: "GenICam 参数非法",
    0x80000102: "GenICam 值超出范围",
    0x80000103: "GenICam 属性错误",
    0x80000104: "GenICam 运行环境错误",
    0x80000105: "GenICam 逻辑错误",
    0x80000106: "GenICam 节点访问条件错误（节点不存在 / 不可写 / 设备已断开）",
    0x80000107: "GenICam 超时",
    0x80000108: "GenICam 类型转换异常",
    0x800001FF: "GenICam 未知错误",
    0x80000200: "命令不被设备支持",
    0x80000201: "访问的目标地址不存在",
    0x80000202: "目标地址不可写",
    0x80000203: "设备无访问权限（已被其他客户端独占）",
    0x80000204: "设备忙或网络断开",
    0x80000205: "网络包数据错误",
    0x80000206: "网络相关错误（设备可能已掉线）",
    0x80000221: "设备 IP 冲突",
}
NODATA_CODES = frozenset({MV_E_NODATA, MV_E_GC_TIMEOUT})
OFFLINE_CODES = frozenset({MV_E_NETER, MV_E_BUSY})

# GVSP 像素格式（PixelType_Gvsp_*）
PIXEL_RGB8 = 0x02180014
PIXEL_BGR8 = 0x02180015
PIXEL_MONO8 = 0x01080001
PIXEL_BAYER_GR8 = 0x01080008
PIXEL_BAYER_RG8 = 0x01080009
PIXEL_BAYER_GB8 = 0x0108000A
PIXEL_BAYER_BG8 = 0x0108000B
# OpenCV 的 Bayer 常量按第二行第 2/3 列命名，与 GVSP 命名相反
_BAYER_CVT = {
    PIXEL_BAYER_RG8: cv2.COLOR_BayerBG2RGB,
    PIXEL_BAYER_GB8: cv2.COLOR_BayerGR2RGB,
    PIXEL_BAYER_GR8: cv2.COLOR_BayerGB2RGB,
    PIXEL_BAYER_BG8: cv2.COLOR_BayerRG2RGB,
}

# ======================================================================================
# 运行参数（实例属性可覆盖，测试缩短）
# ======================================================================================
QUEUE_SIZE = 64
"""内部块队列长度，满则丢块并计 dropped_chunks。"""
IMAGE_NODE_NUM = 16
"""SDK 图像缓存节点数（512 行 × 4096 × RGB ≈ 6.3 MB/节点，16 个 ≈ 100 MB，见 docs/01 §6）。"""
RESEND_MAX_PERCENT = 10
RESEND_TIMEOUT_MS = 50
GRAB_TIMEOUT_MS = 1000
"""单次 MV_CC_GetImageBuffer 等待。"""
RECONNECT_BACKOFF_S: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
AWB_SETTLE_S = 0.5
"""流式采集中执行 Once 白平衡后等待相机完成的时间。"""
OFFLINE_AFTER_S = 5.0
"""内触发下连续无帧超过该时间（且不少于 4 个块周期）视为掉线。"""
MAX_CONSECUTIVE_ERRORS = 3
"""取帧连续硬错误（非超时）达到该次数视为掉线。"""
THREAD_JOIN_S = 2.0

_SDK_HINT = (
    "未找到海康 MVS SDK 的 Python 封装（MvCameraControl_class）。"
    "请安装 MVS_STD_GML_V2.x_aarch64.deb 到 /opt/MVS，并设置 "
    "MVCAM_COMMON_RUNENV=/opt/MVS/lib、LD_LIBRARY_PATH 追加 /opt/MVS/lib/aarch64，"
    "或把 /opt/MVS/Samples/aarch64/Python/MvImport 加入 PYTHONPATH / 复制到 oresorter/camera/hik_sdk/。"
)
_SDK_INIT_FLAG = "_oresorter_sdk_initialized"


def _import_sdk() -> ModuleType:
    """导入 MvImport 的 ``MvCameraControl_class`` 模块；失败抛 CameraError 并给出安装提示。

    返回模块本身（含 ``MvCamera`` 类与 ctypes 结构体），便于测试注入假模块。
    """
    candidates = [
        Path(__file__).parent / "hik_sdk",
        Path(os.environ.get("MVS_PYTHON_DIR", "/opt/MVS/Samples/aarch64/Python/MvImport")),
    ]
    for c in candidates:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.append(str(c))
    try:
        sdk = importlib.import_module("MvCameraControl_class")
    except Exception as exc:
        raise CameraError(_SDK_HINT) from exc
    if not hasattr(sdk, "MvCamera"):
        raise CameraError(_SDK_HINT)
    return sdk


def norm_code(ret: Any) -> int:
    """把 SDK 返回码规范为无符号 32 位（ctypes 以 c_int 返回时 0x8000xxxx 会变成负数）。"""
    return int(ret) & _CODE_MASK


def format_error(ret: Any) -> str:
    """``0x8000000X 中文说明``。"""
    code = norm_code(ret)
    return f"0x{code:08X} {ERROR_TEXT.get(code, '未知错误')}"


def _cstr(arr: Any) -> str:
    """ctypes 字节数组（c_ubyte/c_char 数组）→ 去 NUL 的字符串。"""
    raw = bytes(arr)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def _ip_str(ip: int) -> str:
    ip = int(ip) & _CODE_MASK
    return f"{(ip >> 24) & 255}.{(ip >> 16) & 255}.{(ip >> 8) & 255}.{ip & 255}"


def _describe_device(info: Any) -> dict[str, str]:
    """从 ``MV_CC_DEVICE_INFO``（GigE）提取 ip / serial / model / name。"""
    g = info.SpecialInfo.stGigEInfo
    return {
        "ip": _ip_str(g.nCurrentIp),
        "serial": _cstr(g.chSerialNumber),
        "model": _cstr(g.chModelName),
        "name": _cstr(g.chUserDefinedName),
    }


def _align(value: int, lo: int, hi: int, inc: int) -> int:
    """把整数夹到 ``[lo, hi]`` 并向下对齐到 ``lo + k·inc``。"""
    v = min(max(int(value), lo), hi)
    if inc > 1:
        v = lo + ((v - lo) // inc) * inc
    return v


def _to_rgb(buf_ptr: Any, nbytes: int, width: int, height: int, pixel_type: int) -> np.ndarray:
    """SDK 帧缓冲 → ``(h, w, 3) uint8`` RGB（一次 memmove 拷贝，缓冲随后即可归还 SDK）。"""
    if width <= 0 or height <= 0 or nbytes <= 0:
        raise CameraError(f"帧信息非法: width={width} height={height} len={nbytes}")
    data = np.empty(int(nbytes), np.uint8)
    ctypes.memmove(data.ctypes.data, buf_ptr, int(nbytes))
    npix = width * height
    if pixel_type in (PIXEL_RGB8, PIXEL_BGR8):
        need = npix * 3
        if nbytes < need:
            raise CameraError(f"帧长度不足: {nbytes} < {need}（{width}x{height} RGB）")
        img = data[:need].reshape(height, width, 3)
        if pixel_type == PIXEL_BGR8:
            img = np.ascontiguousarray(img[:, :, ::-1])
        return img
    if nbytes < npix:
        raise CameraError(f"帧长度不足: {nbytes} < {npix}（{width}x{height} 单通道）")
    mono = data[:npix].reshape(height, width)
    if pixel_type == PIXEL_MONO8:
        return np.repeat(mono[:, :, None], 3, axis=2)
    if pixel_type in _BAYER_CVT:
        return cv2.cvtColor(mono, _BAYER_CVT[pixel_type])
    raise CameraError(f"不支持的像素格式 0x{pixel_type:08X}（请在相机侧设为 RGB8Packed）")


class _SnapshotRequest:
    """流式采集中的快照请求：取帧线程把块复制一份喂进来，凑够行数即完成。"""

    def __init__(self, height: int):
        self.height = int(height)
        self.parts: list[np.ndarray] = []
        self.lines = 0
        self.done = threading.Event()

    def feed(self, frame: Frame) -> None:
        if self.done.is_set():
            return
        self.parts.append(frame.image)
        self.lines += frame.lines
        if self.lines >= self.height:
            self.done.set()

    def image(self) -> np.ndarray:
        if not self.parts:
            return np.zeros((0, 0, 3), np.uint8)
        img = self.parts[0] if len(self.parts) == 1 else np.concatenate(self.parts, axis=0)
        return np.ascontiguousarray(img[: self.height])


class HikCamera(Camera):
    """海康 MV-CL043-A1GC（MVS SDK）实现，接口见 ``Camera``。

    线程模型：调用方线程（open/set_params/snapshot/...）、取帧线程（``_grab_loop``）、
    重连线程（``_reconnect_loop``）、SDK 异常回调线程。``_lock`` 保护状态与非取帧 SDK 调用；
    任何需要 join 线程的地方都在锁外进行。
    """

    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self._sdk = _import_sdk()
        self._cam: Any = None  # MvCamera 实例（持有句柄）
        self._dev_list: Any = None  # 枚举结果，CreateHandle 期间须存活
        self.device_info: dict[str, str] = {}
        self._params = CameraParams.from_config(cfg)
        self._connected = False
        self._streaming = False  # 调用方意图：start_stream 后为 True，直到 stop_stream
        self._grabbing = False  # SDK 实际在取流
        self._closed = False
        self._chunk_lines = int(cfg.runtime_chunk_lines)
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=QUEUE_SIZE)
        self._lock = threading.RLock()
        self._stream_stop = threading.Event()
        self._closing = threading.Event()
        self._grab_thread: threading.Thread | None = None
        self._reconnect_thread: threading.Thread | None = None
        self._exc_cb: Any = None  # ctypes 回调对象，必须持有引用防止被回收
        self._seq = 0
        self._line = 0
        self._last_frame_num: int | None = None
        self._snap_req: _SnapshotRequest | None = None
        # 可调参数（测试缩短）
        self.grab_timeout_ms = GRAB_TIMEOUT_MS
        self.reconnect_backoff_s: tuple[float, ...] = RECONNECT_BACKOFF_S
        self.awb_settle_s = AWB_SETTLE_S
        self.offline_after_s = OFFLINE_AFTER_S
        self.max_consecutive_errors = MAX_CONSECUTIVE_ERRORS

    # ------------------------------------------------------------------ 连接
    def open(self) -> None:
        with self._lock:
            self._closed = False
            self._closing.clear()
            if self._connected:
                return
            self._open_device()
            try:
                self._apply_params(self._params)
            except CameraError:
                self._release_device()
                raise
            self._connected = True
            log.info(
                "海康相机已打开",
                extra={
                    "camera": self.id,
                    "device": self.device_info,
                    "params": self._params.to_dict(),
                },
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._closing.set()
        self.stop_stream()
        t = self._reconnect_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=THREAD_JOIN_S)
            if t.is_alive():
                log.warning("重连线程未能及时退出", extra={"camera": self.id})
        with self._lock:
            self._reconnect_thread = None
            self._release_device()
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def streaming(self) -> bool:
        return self._streaming

    # ------------------------------------------------------------------ 参数
    def get_params(self) -> CameraParams:
        with self._lock:
            self._require_connected()
            p = CameraParams(
                awb_mode=self._read_awb(),
                exposure_us=self._get_float(NODE_EXPOSURE),
                width=self._get_int(NODE_WIDTH),
                offset_x=self._get_int(NODE_OFFSET_X),
                height=self._params.height if self._grabbing else self._get_int(NODE_HEIGHT),
                line_rate_hz=self._get_int(NODE_LINE_RATE),
                trigger=self._read_trigger(),
            )
            self._params = CameraParams(**p.to_dict())
            return p

    def set_params(self, p: CameraParams) -> CameraParams:
        with self._lock:
            self._require_connected()
            restart = self._grabbing and self._needs_restart(p)
        if restart:
            self._stop_grabbing()
        try:
            with self._lock:
                self._require_connected()
                if self._grabbing:
                    return self._apply_live(p)
                return self._apply_params(p, height=self._chunk_lines if self._streaming else None)
        finally:
            if restart:
                with self._lock:
                    if self._connected and self._streaming and not self._grabbing:
                        self._start_grabbing(self._chunk_lines)

    def awb_once(self) -> None:
        with self._lock:
            self._require_connected()
            self._awb_once_locked()
            self._params.awb_mode = "off"

    # ------------------------------------------------------------------ 采集
    def snapshot(self, height: int) -> np.ndarray:
        height = int(height)
        with self._lock:
            self._require_connected()
            grabbing = self._grabbing
            if grabbing:
                req = _SnapshotRequest(height)
                self._snap_req = req
            else:
                img = self._grab_single(height, tolerate_timeout=False)
                assert img is not None
                return img
        try:
            ok = req.done.wait(self._collect_timeout(height))
        finally:
            with self._lock:
                if self._snap_req is req:
                    self._snap_req = None
        if not ok or req.lines < height:
            if not self._connected:
                raise CameraDisconnected(f"相机 {self.id} 掉线，快照中断")
            if not self._grabbing:
                raise CameraError(f"相机 {self.id} 流已停止，快照中断（拼到 {req.lines} 行）")
            raise CameraTimeout(
                f"相机 {self.id} 在时限内只拼到 {req.lines} 行（请求 {height}），"
                "编码器触发下请确认皮带在运行"
            )
        return req.image()

    def start_stream(self, chunk_lines: int) -> None:
        with self._lock:
            self._require_connected()
            if self._streaming:
                return
            self._seq = 0
            self._drain_queue()
            self._start_grabbing(int(chunk_lines))
            self._streaming = True

    def stop_stream(self) -> None:
        with self._lock:
            if not self._streaming and not self._grabbing:
                return
            self._streaming = False
        self._stop_grabbing()

    def get_frame(self, timeout: float) -> Frame | None:
        if not self._connected:
            raise CameraDisconnected(f"相机 {self.id} 已掉线")
        try:
            f = self._queue.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None
        self.stats.on_frame(f.ts)
        return f

    def reset_line_counter(self) -> None:
        with self._lock:
            self._line = 0
            self._drain_queue()

    # ------------------------------------------------------------------ 设备打开/释放
    def _open_device(self) -> None:
        """枚举 → 匹配 → 创建句柄 → 独占打开 → GigE 传输层设置。失败抛 CameraError。"""
        sdk = self._sdk
        self._sdk_initialize()
        info, dev_list, desc = self._find_device()
        cam = sdk.MvCamera()
        self._check(cam.MV_CC_CreateHandle(info), "创建句柄")
        try:
            self._check(cam.MV_CC_OpenDevice(MV_ACCESS_EXCLUSIVE, 0), f"打开设备 {desc['ip']}")
        except CameraError:
            self._ignore(cam.MV_CC_DestroyHandle(), "销毁句柄")
            raise
        self._cam = cam
        self._dev_list = dev_list
        self.device_info = desc
        try:
            self._setup_transport()
        except CameraError:
            self._release_device()
            raise

    def _sdk_initialize(self) -> None:
        """SDK 4.x 需要先 ``MV_CC_Initialize``；旧版没有该方法。每个 SDK 模块只做一次。"""
        sdk = self._sdk
        if getattr(sdk, _SDK_INIT_FLAG, False):
            return
        init = getattr(sdk.MvCamera, "MV_CC_Initialize", None)
        if init is not None:
            self._check(init(), "初始化 SDK")
        setattr(sdk, _SDK_INIT_FLAG, True)

    def _find_device(self) -> tuple[Any, Any, dict[str, str]]:
        sdk = self._sdk
        dev_list = sdk.MV_CC_DEVICE_INFO_LIST()
        self._check(sdk.MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list), "枚举 GigE 设备")
        found: list[tuple[dict[str, str], Any]] = []
        for i in range(int(dev_list.nDeviceNum)):
            info = ctypes.cast(
                dev_list.pDeviceInfo[i], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)
            ).contents
            found.append((_describe_device(info), info))
        want_ip, want_sn = self.cfg.ip, self.cfg.serial
        for desc, info in found:
            if (want_ip and desc["ip"] == want_ip) or (want_sn and desc["serial"] == want_sn):
                if want_ip and want_sn and (desc["ip"] != want_ip or desc["serial"] != want_sn):
                    log.warning(
                        "相机 ip 与 serial 只有一个匹配",
                        extra={
                            "camera": self.id,
                            "want_ip": want_ip,
                            "want_sn": want_sn,
                            "device": desc,
                        },
                    )
                return info, dev_list, desc
        seen = ", ".join(f"{d['model']}@{d['ip']}(sn={d['serial']})" for d, _ in found) or "无"
        raise CameraError(
            f"相机 {self.id} 未找到匹配设备（ip={want_ip} serial={want_sn}），当前枚举到: {seen}"
        )

    def _setup_transport(self) -> None:
        """GigE 包长取最优值、开启 GVSP 重传、加大缓存节点、注册断线回调（后三者失败只告警）。"""
        cam = self._cam
        n = int(cam.MV_CC_GetOptimalPacketSize())
        if 0 < n < 0x80000000:
            try:
                self._set_int(NODE_PACKET_SIZE, n)
            except CameraError as exc:
                log.warning(
                    "设置 GevSCPSPacketSize 失败，沿用相机当前值",
                    extra={"camera": self.id, "packet": n, "err": str(exc)},
                )
        else:
            log.warning(
                "获取最优包长失败，沿用相机当前值",
                extra={"camera": self.id, "err": format_error(n)},
            )
        self._ignore(
            cam.MV_GIGE_SetResend(1, RESEND_MAX_PERCENT, RESEND_TIMEOUT_MS), "开启 GVSP 重传"
        )
        self._ignore(cam.MV_CC_SetImageNodeNum(IMAGE_NODE_NUM), "设置图像缓存节点数")
        register = getattr(cam, "MV_CC_RegisterExceptionCallBack", None)
        if register is not None:
            cb_type = ctypes.CFUNCTYPE(None, ctypes.c_uint, ctypes.c_void_p)
            self._exc_cb = cb_type(self._on_sdk_exception)
            self._ignore(register(self._exc_cb, None), "注册异常回调")

    def _release_device(self) -> None:
        """尽力停流、关闭并销毁句柄；可重复调用（调用方持锁）。"""
        cam = self._cam
        self._cam = None
        self._dev_list = None
        if cam is None:
            return
        if self._grabbing:
            self._grabbing = False
            self._ignore(cam.MV_CC_StopGrabbing(), "停止取流")
        self._ignore(cam.MV_CC_CloseDevice(), "关闭设备")
        self._ignore(cam.MV_CC_DestroyHandle(), "销毁句柄")

    def _on_sdk_exception(self, msg_type: int, _user: Any) -> None:
        """SDK 异常回调（SDK 线程）：只记状态，不抛异常。"""
        try:
            if int(msg_type) == MV_EXCEPTION_DEV_DISCONNECT:
                self._mark_offline("SDK 异常回调上报设备断开")
            else:
                log.warning(
                    "SDK 异常回调", extra={"camera": self.id, "msg_type": hex(int(msg_type))}
                )
        except Exception:  # 回调里绝不能抛
            log.exception("处理 SDK 异常回调失败", extra={"camera": self.id})

    # ------------------------------------------------------------------ 掉线与重连
    def _mark_offline(self, reason: str) -> None:
        with self._lock:
            if not self._connected or self._closed:
                return
            self._connected = False
            self.stats.errors += 1
            log.error("相机掉线，进入后台重连", extra={"camera": self.id, "reason": reason})
            self._start_reconnect_thread()

    def _start_reconnect_thread(self) -> None:
        t = self._reconnect_thread
        if t is not None and t.is_alive():
            return
        t = threading.Thread(
            target=self._reconnect_loop, name=f"hikcam{self.id}-reconnect", daemon=True
        )
        self._reconnect_thread = t
        t.start()

    def _reconnect_loop(self) -> None:
        old = self._grab_thread
        if old is not None and old is not threading.current_thread():
            old.join(timeout=THREAD_JOIN_S)
        backoff = self.reconnect_backoff_s or (1.0,)
        attempt = 0
        while not self._closing.is_set():
            delay = backoff[min(attempt, len(backoff) - 1)]
            if self._closing.wait(delay):
                return
            attempt += 1
            try:
                with self._lock:
                    if self._closed:
                        return
                    self._grab_thread = None
                    self._release_device()
                    self._open_device()
                    self._apply_params(
                        self._params, height=self._chunk_lines if self._streaming else None
                    )
                    self._connected = True
                    self.stats.reconnects += 1
                    if self._streaming:
                        self._start_grabbing(self._chunk_lines)
                log.info(
                    "相机重连成功",
                    extra={"camera": self.id, "attempt": attempt, "streaming": self._streaming},
                )
                return
            except CameraError as exc:
                log.warning(
                    "相机重连失败，稍后重试",
                    extra={"camera": self.id, "attempt": attempt, "err": str(exc)},
                )
            except Exception:  # 重连线程不能死
                log.exception("相机重连异常", extra={"camera": self.id, "attempt": attempt})

    def _sdk_connected(self) -> bool:
        cam = self._cam
        if cam is None:
            return False
        probe = getattr(cam, "MV_CC_IsDeviceConnected", None)
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:
            return False

    # ------------------------------------------------------------------ 取流
    def _start_grabbing(self, chunk_lines: int) -> None:
        """Height = chunk_lines → StartGrabbing → 启动取帧线程（调用方持锁）。"""
        cam = self._cam_or_raise()
        actual = self._set_int(NODE_HEIGHT, chunk_lines)
        if actual != chunk_lines:
            log.warning(
                "块高按相机范围/步进对齐",
                extra={"camera": self.id, "requested": chunk_lines, "actual": actual},
            )
        self._chunk_lines = actual
        self._last_frame_num = None
        self._check(cam.MV_CC_StartGrabbing(), "开始取流")
        self._grabbing = True
        self._stream_stop.clear()
        t = threading.Thread(
            target=self._grab_loop, args=(cam,), name=f"hikcam{self.id}-grab", daemon=True
        )
        self._grab_thread = t
        t.start()

    def _stop_grabbing(self) -> None:
        """StopGrabbing 并回收取帧线程；可重复调用。调用方不得持锁（要 join）。"""
        with self._lock:
            self._stream_stop.set()
            t = self._grab_thread
            self._grab_thread = None
            cam = self._cam
            was = self._grabbing
            self._grabbing = False
            if was and cam is not None:
                self._ignore(cam.MV_CC_StopGrabbing(), "停止取流")
            req = self._snap_req
            self._snap_req = None
        if req is not None:
            req.done.set()
        if t is not None and t is not threading.current_thread():
            t.join(timeout=THREAD_JOIN_S)
            if t.is_alive():
                log.warning("取帧线程未能在 2 s 内退出", extra={"camera": self.id})

    def _grab_loop(self, cam: Any) -> None:
        sdk = self._sdk
        frame = sdk.MV_FRAME_OUT()
        consecutive_errors = 0
        last_ok = time.monotonic()
        with self._lock:
            trigger = self._params.trigger
            period = self._chunk_lines / max(1, self._params.line_rate_hz)
        offline_after = max(self.offline_after_s, 4.0 * period)
        while not self._stream_stop.is_set() and self._connected:
            ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
            ret = norm_code(cam.MV_CC_GetImageBuffer(frame, int(self.grab_timeout_ms)))
            if self._stream_stop.is_set():
                if ret == MV_OK:
                    self._ignore(cam.MV_CC_FreeImageBuffer(frame), "归还帧缓冲")
                break
            if ret == MV_OK:
                try:
                    self._handle_frame(frame)
                except Exception as exc:  # 单帧异常不杀线程
                    self.stats.errors += 1
                    log.warning("帧处理失败", extra={"camera": self.id, "err": str(exc)})
                finally:
                    self._ignore(cam.MV_CC_FreeImageBuffer(frame), "归还帧缓冲")
                consecutive_errors = 0
                last_ok = time.monotonic()
                continue
            if ret in NODATA_CODES:
                if not self._sdk_connected():
                    self._mark_offline("取帧超时且 SDK 报告设备已断开")
                    break
                if trigger == "internal" and time.monotonic() - last_ok > offline_after:
                    self._mark_offline(f"内触发下连续 {offline_after:.1f}s 无帧")
                    break
                continue
            consecutive_errors += 1
            self.stats.errors += 1
            log.warning(
                "取帧失败",
                extra={
                    "camera": self.id,
                    "err": format_error(ret),
                    "consecutive": consecutive_errors,
                },
            )
            if (
                ret in OFFLINE_CODES
                or consecutive_errors >= self.max_consecutive_errors
                or not self._sdk_connected()
            ):
                self._mark_offline(f"取帧连续失败 {consecutive_errors} 次: {format_error(ret)}")
                break
            self._stream_stop.wait(0.05)

    def _handle_frame(self, frame: Any) -> None:
        info = frame.stFrameInfo
        width, height = int(info.nWidth), int(info.nHeight)
        img = _to_rgb(frame.pBufAddr, int(info.nFrameLen), width, height, int(info.enPixelType))
        fnum = int(info.nFrameNum)
        ts = time.monotonic()
        with self._lock:
            missed = 0
            if self._last_frame_num is not None:
                gap = fnum - self._last_frame_num - 1
                if gap > 0:
                    missed = gap
            self._last_frame_num = fnum
            if missed:
                self._seq += missed
                self._line += missed * self._chunk_lines
                self.stats.dropped_chunks += missed
                log.warning(
                    "SDK 丢块，按块高补偿行号",
                    extra={"camera": self.id, "missed": missed, "frame_num": fnum},
                )
            out = Frame(seq=self._seq, line_start=self._line, image=img, ts=ts, camera_id=self.id)
            self._seq += 1
            self._line += height
            snap = self._snap_req
        if snap is not None:
            snap.feed(out)
        try:
            self._queue.put_nowait(out)
        except queue.Full:
            self.stats.dropped_chunks += 1
            log.warning("内部块队列已满，丢块", extra={"camera": self.id, "seq": out.seq})

    def _grab_single(self, height: int, *, tolerate_timeout: bool) -> np.ndarray | None:
        """未开流时临时以 ``height`` 行抓一帧（调用方持锁）；结束后恢复 Height 为采样帧高。"""
        cam = self._cam_or_raise()
        actual = self._set_int(NODE_HEIGHT, height)
        timeout_ms = int(self._collect_timeout(actual) * 1000)
        self._last_frame_num = None
        self._check(cam.MV_CC_StartGrabbing(), "开始取流（单帧）")
        try:
            frame = self._sdk.MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
            ret = norm_code(cam.MV_CC_GetImageBuffer(frame, timeout_ms))
            if ret in NODATA_CODES:
                if tolerate_timeout:
                    return None
                raise CameraTimeout(
                    f"相机 {self.id} 在 {timeout_ms} ms 内未收到 {actual} 行图像"
                    "（编码器触发下请确认皮带在运行）"
                )
            self._check(ret, "单帧取图")
            try:
                info = frame.stFrameInfo
                return _to_rgb(
                    frame.pBufAddr,
                    int(info.nFrameLen),
                    int(info.nWidth),
                    int(info.nHeight),
                    int(info.enPixelType),
                )
            finally:
                self._ignore(cam.MV_CC_FreeImageBuffer(frame), "归还帧缓冲")
        finally:
            self._ignore(cam.MV_CC_StopGrabbing(), "停止取流（单帧）")
            if actual != self._params.height:
                try:
                    self._set_int(NODE_HEIGHT, self._params.height)
                except CameraError as exc:
                    log.warning("恢复采样帧高失败", extra={"camera": self.id, "err": str(exc)})

    def _collect_timeout(self, height: int) -> float:
        """等待 ``height`` 行的上限：3 倍按行频走完的时间，且不少于 2 s。"""
        rate = max(1, int(self._params.line_rate_hz))
        return max(2.0, 3.0 * height / rate)

    # ------------------------------------------------------------------ 参数下发
    def _needs_restart(self, p: CameraParams) -> bool:
        cur = self._params
        return p.width != cur.width or p.offset_x != cur.offset_x or p.trigger != cur.trigger

    def _apply_params(self, p: CameraParams, *, height: int | None = None) -> CameraParams:
        """按顺序下发全部参数（调用方持锁，且 SDK 未在取流），返回相机实际接受的值。

        ``height`` 覆盖 Height 节点（开流时为 chunk_lines，此时返回值中的 height 仍为采样帧高）。
        """
        out = CameraParams(**p.to_dict())
        self._set_enum(NODE_PIXEL_FORMAT, PIXEL_FORMAT_RGB8)
        # 几何：先把 OffsetX 归零再设 Width，避免 OffsetX+Width 越界被拒
        self._set_int(NODE_OFFSET_X, 0)
        out.width = self._set_int(NODE_WIDTH, p.width)
        out.offset_x = self._set_int(NODE_OFFSET_X, p.offset_x)
        h = self._set_int(NODE_HEIGHT, p.height if height is None else height)
        if height is None:
            out.height = h
        # 触发与行频（先于曝光：行频决定曝光上限）
        self._apply_trigger(p.trigger)
        out.line_rate_hz = self._apply_line_rate(p.trigger, p.line_rate_hz)
        # 曝光
        self._set_enum_optional(NODE_EXPOSURE_AUTO, "Off")
        out.exposure_us = self._set_float(NODE_EXPOSURE, p.exposure_us)
        # 先缓存（白平衡 Once 的临时抓帧结束后要把 Height 恢复成新的采样帧高）
        self._params = CameraParams(**out.to_dict())
        # 白平衡（once → 执行一次并锁定为 off）
        out.awb_mode = self._apply_awb(p.awb_mode)
        self._params.awb_mode = out.awb_mode
        return CameraParams(**out.to_dict())

    def _apply_live(self, p: CameraParams) -> CameraParams:
        """取流中只改可热改的参数（曝光 / 行频 / 白平衡），几何与触发保持当前值。"""
        out = CameraParams(**self._params.to_dict())
        out.height = p.height  # 采样帧高仅缓存，下次 snapshot / 开流时生效
        out.line_rate_hz = self._apply_line_rate(out.trigger, p.line_rate_hz)
        out.exposure_us = self._set_float(NODE_EXPOSURE, p.exposure_us)
        out.awb_mode = self._apply_awb(p.awb_mode)
        self._params = CameraParams(**out.to_dict())
        return CameraParams(**out.to_dict())

    def _apply_trigger(self, mode: str) -> None:
        self._set_enum_optional(NODE_TRIGGER_SELECTOR, TRIGGER_SELECTOR_LINE)
        if mode == "internal":
            self._set_enum(NODE_TRIGGER_MODE, "Off")
            return
        for key, val in ENCODER_SETUP:
            self._set_enum_optional(key, val)
        self._set_enum(NODE_TRIGGER_SOURCE, TRIGGER_SOURCE_ENCODER)
        self._set_enum(NODE_TRIGGER_MODE, "On")

    def _apply_line_rate(self, trigger: str, line_rate_hz: int) -> int:
        if trigger == "internal":
            self._set_bool_optional(NODE_LINE_RATE_ENABLE, True)
            return self._set_int(NODE_LINE_RATE, line_rate_hz)
        # 编码器触发：行频由编码器决定，关闭固定行频；节点仍尝试写入作为上限参考，失败不致命
        self._set_bool_optional(NODE_LINE_RATE_ENABLE, False)
        try:
            return self._set_int(NODE_LINE_RATE, line_rate_hz)
        except CameraDisconnected:
            raise
        except CameraError as exc:
            log.info("编码器触发下行频节点不可写，忽略", extra={"camera": self.id, "err": str(exc)})
            return int(line_rate_hz)

    def _apply_awb(self, mode: str) -> str:
        if mode == "once":
            self._awb_once_locked()
            return "off"
        self._set_enum(NODE_AWB, AWB_SYMBOLS.get(mode, "Off"))
        return mode

    def _awb_once_locked(self) -> None:
        """BalanceWhiteAuto=Once → 让相机采到图完成计算 → 锁定为 Off（调用方持锁）。"""
        self._set_enum(NODE_AWB, "Once")
        if self._grabbing:
            if self.awb_settle_s > 0:
                time.sleep(self.awb_settle_s)
        else:
            # Once 需要图像数据才会执行：临时抓一帧（编码器触发无脉冲时允许超时）
            self._grab_single(self._get_int(NODE_HEIGHT), tolerate_timeout=True)
        self._set_enum(NODE_AWB, "Off")

    def _read_awb(self) -> str:
        sym = (self._get_enum_symbolic(NODE_AWB) or "").lower()
        return sym if sym in AWB_SYMBOLS else self._params.awb_mode

    def _read_trigger(self) -> str:
        self._set_enum_optional(NODE_TRIGGER_SELECTOR, TRIGGER_SELECTOR_LINE, quiet=True)
        sym = (self._get_enum_symbolic(NODE_TRIGGER_MODE) or "").lower()
        if sym == "on":
            return "encoder"
        if sym == "off":
            return "internal"
        return self._params.trigger

    # ------------------------------------------------------------------ 节点读写
    def _cam_or_raise(self) -> Any:
        cam = self._cam
        if cam is None:
            raise CameraDisconnected(f"相机 {self.id} 未打开")
        return cam

    def _require_connected(self) -> None:
        if self._closed or not self._connected or self._cam is None:
            raise CameraDisconnected(f"相机 {self.id} 未连接")

    def _int_api(self) -> tuple[Any, Any, Any]:
        """返回 (getter, setter, 结构体类)：优先 64 位 Ex 接口。"""
        cam = self._cam_or_raise()
        sdk = self._sdk
        if hasattr(cam, "MV_CC_GetIntValueEx") and hasattr(sdk, "MVCC_INTVALUE_EX"):
            return cam.MV_CC_GetIntValueEx, cam.MV_CC_SetIntValueEx, sdk.MVCC_INTVALUE_EX
        return cam.MV_CC_GetIntValue, cam.MV_CC_SetIntValue, sdk.MVCC_INTVALUE

    def _get_int_range(self, key: str) -> tuple[int, int, int, int]:
        getter, _, struct = self._int_api()
        st = struct()
        self._check(getter(key, st), f"读取 {key}")
        return int(st.nCurValue), int(st.nMin), int(st.nMax), int(st.nInc)

    def _get_int(self, key: str) -> int:
        return self._get_int_range(key)[0]

    def _set_int(self, key: str, value: int) -> int:
        """按相机范围/步进对齐后写入，返回读回的实际值。"""
        _, setter, _ = self._int_api()
        _, lo, hi, inc = self._get_int_range(key)
        v = _align(int(value), lo, hi, max(1, inc))
        if v != int(value):
            log.info(
                "整数参数按相机范围/步进对齐",
                extra={"camera": self.id, "node": key, "requested": int(value), "actual": v},
            )
        self._check(setter(key, v), f"设置 {key}={v}")
        return self._get_int(key)

    def _get_float_range(self, key: str) -> tuple[float, float, float]:
        cam = self._cam_or_raise()
        st = self._sdk.MVCC_FLOATVALUE()
        self._check(cam.MV_CC_GetFloatValue(key, st), f"读取 {key}")
        return float(st.fCurValue), float(st.fMin), float(st.fMax)

    def _get_float(self, key: str) -> float:
        return self._get_float_range(key)[0]

    def _set_float(self, key: str, value: float) -> float:
        cam = self._cam_or_raise()
        _, lo, hi = self._get_float_range(key)
        v = min(max(float(value), lo), hi)
        if v != float(value):
            log.info(
                "浮点参数按相机范围夹取",
                extra={"camera": self.id, "node": key, "requested": float(value), "actual": v},
            )
        self._check(cam.MV_CC_SetFloatValue(key, v), f"设置 {key}={v}")
        return self._get_float(key)

    def _set_enum(self, key: str, symbolic: str) -> None:
        cam = self._cam_or_raise()
        self._check(cam.MV_CC_SetEnumValueByString(key, symbolic), f"设置 {key}={symbolic}")

    def _set_enum_optional(self, key: str, symbolic: str, *, quiet: bool = False) -> bool:
        try:
            self._set_enum(key, symbolic)
            return True
        except CameraDisconnected:
            raise
        except CameraError as exc:
            if not quiet:
                log.warning(
                    "可选枚举节点设置失败，忽略（请在 MVS 客户端核对节点名）",
                    extra={"camera": self.id, "node": key, "value": symbolic, "err": str(exc)},
                )
            return False

    def _set_bool_optional(self, key: str, flag: bool) -> bool:
        cam = self._cam_or_raise()
        try:
            self._check(cam.MV_CC_SetBoolValue(key, bool(flag)), f"设置 {key}={flag}")
            return True
        except CameraDisconnected:
            raise
        except CameraError as exc:
            log.warning(
                "可选布尔节点设置失败，忽略（请在 MVS 客户端核对节点名）",
                extra={"camera": self.id, "node": key, "value": flag, "err": str(exc)},
            )
            return False

    def _get_enum_symbolic(self, key: str) -> str | None:
        """读枚举当前值的符号名：优先 SDK 的 GetEnumEntrySymbolic，否则查 ``ENUM_SYMBOLS``。"""
        cam = self._cam_or_raise()
        sdk = self._sdk
        st = sdk.MVCC_ENUMVALUE()
        self._check(cam.MV_CC_GetEnumValue(key, st), f"读取 {key}")
        n = int(st.nCurValue)
        getter = getattr(cam, "MV_CC_GetEnumEntrySymbolic", None)
        entry_cls = getattr(sdk, "MVCC_ENUMENTRY", None)
        if getter is not None and entry_cls is not None:
            entry = entry_cls()
            entry.nValue = n
            if norm_code(getter(key, entry)) == MV_OK:
                sym = _cstr(entry.chSymbolic)
                if sym:
                    return sym
        return ENUM_SYMBOLS.get(key, {}).get(n)

    # ------------------------------------------------------------------ 返回码
    def _check(self, ret: Any, what: str) -> None:
        """非 0 返回码 → CameraError；网络类错误或 SDK 报告断开 → 标记掉线并抛 CameraDisconnected。"""
        code = norm_code(ret)
        if code == MV_OK:
            return
        msg = f"{what}失败: {format_error(code)}"
        if self._connected and (code in OFFLINE_CODES or not self._sdk_connected()):
            self._mark_offline(msg)
            raise CameraDisconnected(f"相机 {self.id} 掉线：{msg}")
        raise CameraError(f"相机 {self.id} {msg}")

    def _ignore(self, ret: Any, what: str) -> None:
        """尽力而为的调用：失败只记 debug。"""
        code = norm_code(ret)
        if code != MV_OK:
            log.debug(
                "SDK 调用失败（忽略）",
                extra={"camera": self.id, "what": what, "err": format_error(code)},
            )

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
