"""WebSocket：``/ws/events`` 事件流与 ``/ws/preview/{camera_id}`` 预览流（docs/03 §7、§11；docs/01 §7.1）。

/ws/events
    连接后立即发一条 ``state`` 全量事件（``ctx.runtime.snapshot()``），随后转发 ``EventBus`` 的全部事件。
    ``EventBus`` 回调来自后台线程 → ``loop.call_soon_threadsafe`` 入 ``asyncio.Queue``（满时丢最旧）。
    先订阅再取快照，期间发生的变化不会丢；但订阅与快照之间已发布的 **全量型** 事件
    （:data:`SNAPSHOT_EVENT_TYPES`：state / stats / mcu）内容比快照旧，若排在快照之后送达会让 HMI 把状态
    回退到旧值，因此发送阶段丢弃发布时刻（事件 ``ts``）早于快照观察时刻的全量型事件；增量型事件
    （job / capture_sample / param_sync / fault）一律保留。
    服务端每 :data:`PING_INTERVAL_S` 发 ``{"type": "ping"}``，客户端回 ``{"type": "pong"}``（任何应用层消息
    都算存活）；:data:`CLIENT_TIMEOUT_S` 内无客户端消息则以 1001 断开。WebSocket 协议层 ping/pong 由
    uvicorn 处理、不会到达应用层，因此 HMI 必须回应用层 pong。客户端断开时取消订阅。

/ws/preview/{camera_id}?mode=full|edges&fps=5&quality=70&max_width=1000&edge_px=500
    ``acquire_preview`` 取句柄，在线程池里取帧；``full`` 整幅等比缩放到 ``max_width``；``edges`` 把
    ``[0, edge_px)`` 与 ``[w-edge_px, w)`` 两段横向拼接（宽 ``2*edge_px``，不再缩放），并在 ``start_px`` /
    ``end_px``（传感器坐标，减去 ``offset_x`` 后映射到拼接图）落在段内时画 2 px 白竖线。每帧先发文本
    ``{"type":"frame","seq","w","h","start_px","end_px","ts"}`` 再发二进制 JPEG。查询参数越界一律拒绝、
    不静默截断：``fps`` 限 1~``system.preview_max_fps``（上限可从 ``GET /system/info.preview_max_fps`` 取）、
    ``quality`` 10~100、``max_width`` 16~16384、``edge_px`` 1~相机宽、``mode`` ∈ full|edges；拒绝时发
    ``{"type":"error","code":"PARAM_OUT_OF_RANGE"|"VALIDATION","detail":{"field","input","range"}}``
    并以 1008 关闭。每个发送时隙只发 **最新** 一帧（之间到达的帧丢弃）。相机离线 →
    ``{"type":"error","code":"DEVICE_OFFLINE"}`` 并以 1011 关闭；客户端关闭 → ``release()``（无流水线在跑
    时相机停流；停流会 join 采集线程、可达数百 ms，因此在线程池里执行，不阻塞事件循环）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from oresorter.camera.base import Frame
from oresorter.config import CameraConfig, ConfigError
from oresorter.log import get_logger
from oresorter.runtime.context import AppContext
from oresorter.runtime.service import PreviewHandle

log = get_logger(__name__)

router = APIRouter(tags=["ws"])

PING_INTERVAL_S = 5.0
"""/ws/events 服务端 ping 周期。"""
CLIENT_TIMEOUT_S = 15.0
"""/ws/events 客户端静默上限：超过即断开。"""
EVENT_QUEUE_MAX = 200
"""/ws/events 每连接发送队列长度，满时丢最旧。"""
SNAPSHOT_EVENT_TYPES = frozenset({"state", "stats", "mcu"})
"""全量型事件：每条都是当时的完整快照，比首条 state 快照更旧的会被丢弃（见模块说明）。"""
PREVIEW_FRAME_WAIT_S = 1.0
"""预览单次取帧等待。"""
PREVIEW_DRAIN_MAX = 64
"""预览"只发最新"时一次最多跳过的旧帧数。"""
PREVIEW_MODES = ("full", "edges")
QUALITY_RANGE = (10, 100)
MAX_WIDTH_RANGE = (16, 16384)
LINE_PX = 2
"""edges 模式起止像素白线宽度。"""

WS_NORMAL = 1000
WS_GOING_AWAY = 1001
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _frame_iso(frame: Frame) -> str:
    """把帧的单调时钟换算成墙上时间。"""
    wall = time.time() - max(0.0, time.monotonic() - frame.ts)
    return (
        datetime.fromtimestamp(wall, tz=timezone.utc)
        .astimezone()
        .isoformat(timespec="milliseconds")
    )


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


async def _send_json(websocket: WebSocket, obj: Any) -> None:
    await websocket.send_text(_dumps(obj))


async def _close(websocket: WebSocket, code: int = WS_NORMAL, reason: str | None = None) -> None:
    """尽力关闭：客户端已断开 / 已关闭时静默。"""
    if (
        websocket.client_state is WebSocketState.DISCONNECTED
        or websocket.application_state is WebSocketState.DISCONNECTED
    ):
        return
    with contextlib.suppress(Exception):
        await websocket.close(code=code, reason=reason)


async def _send_error(websocket: WebSocket, code: str, message: str, **detail: Any) -> None:
    with contextlib.suppress(Exception):
        await _send_json(
            websocket,
            {"type": "error", "code": code, "message": message, "detail": detail, "ts": _now_iso()},
        )


async def _cancel(task: asyncio.Task[Any] | asyncio.Future[Any]) -> None:
    if not task.done():
        task.cancel()
    with contextlib.suppress(BaseException):
        await task


# --------------------------------------------------------------------------------------
# /ws/events
# --------------------------------------------------------------------------------------


class _EventFeed:
    """一个连接的事件入队器：后台线程 → 事件循环（满时丢最旧）。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int):
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.closed = False
        self.dropped = 0
        self.stale = 0
        """被丢弃的比首条快照更旧的全量型事件数。"""

    def _enqueue(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(event)

    def on_event(self, event: dict[str, Any]) -> None:
        """EventBus 回调（任意线程）。"""
        if self.closed:
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:
            # 事件循环已关闭（进程退出）
            self.closed = True


class _Liveness:
    """客户端最近一次应用层消息的时间。"""

    def __init__(self) -> None:
        self.last_seen = time.monotonic()

    def touch(self) -> None:
        self.last_seen = time.monotonic()

    def idle_s(self) -> float:
        return time.monotonic() - self.last_seen


def _parse_ts(ts: Any) -> datetime | None:
    """解析 EventBus 的 ISO 时间戳；无法解析返回 None（事件照常转发）。"""
    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def is_stale_snapshot_event(event: dict[str, Any], since: datetime) -> bool:
    """全量型事件（:data:`SNAPSHOT_EVENT_TYPES`）且发布时刻早于 ``since``（首条 state 快照的观察时刻）→ True。

    增量型事件（job / capture_sample / param_sync / fault）永不视为过期；``ts`` 缺失或无法解析的事件也保留。
    """
    if event.get("type") not in SNAPSHOT_EVENT_TYPES:
        return False
    ts = _parse_ts(event.get("ts"))
    return ts is not None and ts < since


def _snapshot_with_time(ctx: AppContext) -> tuple[datetime, dict[str, Any]]:
    """在工作线程里取快照，并记下取快照前一刻的墙上时间（与 EventBus 的 ``ts`` 同一时钟）。

    观察时刻取在 ``snapshot()`` 之前：``ts`` 早于它的事件一定比快照旧，可以安全丢弃；``snapshot()``
    执行期间（毫秒级）发布的事件内容新旧不确定，保留（至多多一条略旧的 state，不会丢新状态）。
    """
    since = datetime.fromtimestamp(time.time(), tz=timezone.utc)
    return since, ctx.runtime.snapshot()


async def _events_sender(
    websocket: WebSocket, feed: _EventFeed, live: _Liveness, since: datetime
) -> str:
    """转发事件（丢弃比首条快照更旧的全量型事件）+ 周期 ping + 静默检测。返回结束原因。"""
    next_ping = time.monotonic() + PING_INTERVAL_S
    while True:
        if live.idle_s() > CLIENT_TIMEOUT_S:
            return "client_timeout"
        timeout = next_ping - time.monotonic()
        if timeout <= 0:
            await _send_json(websocket, {"type": "ping", "ts": _now_iso()})
            next_ping = time.monotonic() + PING_INTERVAL_S
            continue
        try:
            event = await asyncio.wait_for(feed.queue.get(), timeout)
        except asyncio.TimeoutError:
            continue
        if is_stale_snapshot_event(event, since):
            feed.stale += 1
            continue
        await websocket.send_text(_dumps(event))


async def _events_receiver(websocket: WebSocket, live: _Liveness) -> str:
    """收客户端消息：任何消息（含 pong）刷新存活时间；断开则返回。"""
    while True:
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return "disconnect"
        if message["type"] == "websocket.disconnect":
            return "disconnect"
        live.touch()


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """事件流（见模块说明）。"""
    ctx: AppContext = websocket.app.state.ctx
    await websocket.accept()
    feed = _EventFeed(asyncio.get_running_loop(), EVENT_QUEUE_MAX)
    live = _Liveness()
    # 先订阅再取快照：期间发生的变化不会丢；比快照旧的全量型事件由 sender 按 ts 丢弃（见模块说明）
    unsubscribe = ctx.events.subscribe(feed.on_event)
    reason = "error"
    code = WS_NORMAL
    tasks: list[asyncio.Task[str]] = []
    try:
        since, snap = await asyncio.to_thread(_snapshot_with_time, ctx)
        await _send_json(websocket, {"type": "state", "ts": _now_iso(), "data": snap})
        tasks = [
            asyncio.create_task(
                _events_sender(websocket, feed, live, since), name="ws-events-send"
            ),
            asyncio.create_task(_events_receiver(websocket, live), name="ws-events-recv"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            await _cancel(t)
        for t in done:
            exc = t.exception()
            if exc is None:
                reason = t.result()
            elif not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                raise exc
            else:
                reason = "disconnect"
        if reason == "client_timeout":
            code = WS_GOING_AWAY
    except (WebSocketDisconnect, RuntimeError):
        reason = "disconnect"
    except Exception:
        log.exception("事件流异常")
        code = WS_INTERNAL_ERROR
    finally:
        feed.closed = True
        unsubscribe()
        for t in tasks:
            await _cancel(t)
        await _close(websocket, code, reason)
        log.info(
            "事件流结束", extra={"reason": reason, "dropped": feed.dropped, "stale": feed.stale}
        )


# --------------------------------------------------------------------------------------
# /ws/preview/{camera_id}
# --------------------------------------------------------------------------------------


class PreviewOptions:
    """校验后的预览参数。"""

    def __init__(self, mode: str, fps: int, quality: int, max_width: int, edge_px: int):
        self.mode = mode
        self.fps = fps
        self.quality = quality
        self.max_width = max_width
        self.edge_px = edge_px

    @property
    def period_s(self) -> float:
        return 1.0 / float(self.fps)


def _check_range(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ConfigError(
            "PARAM_OUT_OF_RANGE",
            f"{name} = {value} 超出范围 {lo}~{hi}",
            {"field": name, "input": value, "range": [lo, hi]},
        )
    return value


def preview_options(
    ctx: AppContext,
    camera_id: int,
    mode: str,
    fps: int,
    quality: int,
    max_width: int,
    edge_px: int,
) -> PreviewOptions:
    """校验预览查询参数；相机不存在 → ``NOT_FOUND``，任何越界 → ``PARAM_OUT_OF_RANGE``（不夹断）。

    ``fps`` 的上限是 ``system.preview_max_fps``（HMI 可从 ``GET /system/info.preview_max_fps`` 取）。
    """
    cam_cfg = ctx.cfg.camera(camera_id)
    if mode not in PREVIEW_MODES:
        raise ConfigError(
            "VALIDATION", f"mode 必须是 {'|'.join(PREVIEW_MODES)}", {"field": "mode", "input": mode}
        )
    _check_range("fps", fps, 1, int(ctx.cfg.system.preview_max_fps))
    _check_range("quality", quality, *QUALITY_RANGE)
    _check_range("max_width", max_width, *MAX_WIDTH_RANGE)
    _check_range("edge_px", edge_px, 1, int(cam_cfg.width))
    return PreviewOptions(mode, fps, quality, max_width, edge_px)


def _latest_frame(handle: PreviewHandle, timeout: float) -> Frame | None:
    """取下一帧并跳过已排队的旧帧（只发最新）。"""
    frame = handle.get_frame(timeout)
    if frame is None:
        return None
    for _ in range(PREVIEW_DRAIN_MAX):
        newer = handle.get_frame(0.0)
        if newer is None:
            break
        frame = newer
    return frame


def _edge_column(col: int, width: int, edge: int) -> int | None:
    """帧局部列号 → 拼接图列号；不在两段内返回 None。"""
    if 0 <= col < edge:
        return col
    if width - edge <= col < width:
        return edge + (col - (width - edge))
    return None


def compose_edges(image: np.ndarray, cam_cfg: CameraConfig, edge_px: int) -> np.ndarray:
    """docs/03 §7 两段式：``[0, edge)`` 与 ``[w-edge, w)`` 横向拼接，并在起止像素处画白线。

    ``start_px`` / ``end_px`` 是传感器坐标（``end_px`` 为开区间），减 ``offset_x`` 得到帧局部列号；
    起始线占有效区首 2 列，结束线占有效区末 2 列。
    """
    w = int(image.shape[1])
    edge = max(1, min(int(edge_px), w))
    out = np.concatenate([image[:, :edge], image[:, w - edge :]], axis=1).copy()
    start_local = int(cam_cfg.start_px) - int(cam_cfg.offset_x)
    end_local = int(cam_cfg.end_px) - int(cam_cfg.offset_x)
    columns = [start_local + i for i in range(LINE_PX)] + [
        end_local - 1 - i for i in range(LINE_PX)
    ]
    for col in columns:
        x = _edge_column(col, w, edge)
        if x is not None:
            out[:, x] = 255
    return out


def encode_preview(
    frame: Frame, cam_cfg: CameraConfig, opts: PreviewOptions
) -> tuple[dict[str, Any], bytes]:
    """按模式合成 + JPEG 编码，返回 ``(帧头消息, jpeg 字节)``。"""
    image = frame.image
    if opts.mode == "edges":
        out = compose_edges(image, cam_cfg, opts.edge_px)
    else:
        out = image
        h, w = out.shape[:2]
        if w > opts.max_width:
            scale = opts.max_width / float(w)
            out = cv2.resize(
                out, (opts.max_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA
            )
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(opts.quality)],
    )
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    header = {
        "type": "frame",
        "seq": int(frame.seq),
        "w": int(out.shape[1]),
        "h": int(out.shape[0]),
        "start_px": int(cam_cfg.start_px),
        "end_px": int(cam_cfg.end_px),
        "ts": _frame_iso(frame),
    }
    return header, buf.tobytes()


async def _release_off_loop(handle: PreviewHandle) -> None:
    """在线程池里 ``release()``。

    最后一个持有者释放时会 ``camera.stop_stream()``（SDK 停流 + join 采集线程，Mock 上限 2 s、真机常见
    数百 ms），不能在事件循环线程里同步执行，否则期间所有 HTTP / WS 请求（含 /health、/run/stop）停顿。
    处理协程被取消（服务关闭）时线程里的 release 仍会执行到底，这里只是不再等它。
    """
    fut = asyncio.ensure_future(asyncio.to_thread(handle.release))
    try:
        await asyncio.shield(fut)
    except asyncio.CancelledError:
        pass
    except Exception:
        log.exception("预览句柄释放异常", extra={"camera": handle.camera_id})


async def _wait_disconnect(websocket: WebSocket) -> None:
    """预览通道不处理客户端消息，只等断开。"""
    while True:
        try:
            message = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return
        if message["type"] == "websocket.disconnect":
            return


@router.websocket("/ws/preview/{camera_id}")
async def ws_preview(
    websocket: WebSocket,
    camera_id: int,
    mode: str = "full",
    fps: int = 5,
    quality: int = 70,
    max_width: int = 1000,
    edge_px: int = 500,
) -> None:
    """连续采集预览（见模块说明）。"""
    ctx: AppContext = websocket.app.state.ctx
    svc = ctx.runtime
    await websocket.accept()
    try:
        opts = preview_options(ctx, camera_id, mode, fps, quality, max_width, edge_px)
        handle = svc.acquire_preview(camera_id)
    except ConfigError as exc:
        await _send_error(websocket, exc.code, exc.message, **exc.detail)
        await _close(websocket, WS_POLICY_VIOLATION, exc.code)
        return

    receiver = asyncio.create_task(_wait_disconnect(websocket), name=f"ws-preview-recv-{camera_id}")
    frame_fut: asyncio.Future[Frame | None] | None = None
    code = WS_NORMAL
    reason = "client_closed"
    sent = 0
    next_send = 0.0
    try:
        while True:
            wait = next_send - time.monotonic()
            if wait > 0:
                done, _ = await asyncio.wait({receiver}, timeout=wait)
                if done:
                    break
            frame_fut = asyncio.ensure_future(
                asyncio.to_thread(_latest_frame, handle, PREVIEW_FRAME_WAIT_S)
            )
            done, _ = await asyncio.wait({frame_fut, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                break
            frame = frame_fut.result()
            frame_fut = None
            if frame is None:
                cam = ctx.cameras.get(camera_id)
                if cam is None or not cam.connected:
                    await _send_error(
                        websocket, "DEVICE_OFFLINE", f"相机 {camera_id} 离线", camera_id=camera_id
                    )
                    code, reason = WS_INTERNAL_ERROR, "DEVICE_OFFLINE"
                    break
                continue
            cam_cfg = ctx.cfg.camera(camera_id)
            header, jpeg = await asyncio.to_thread(encode_preview, frame, cam_cfg, opts)
            await websocket.send_text(_dumps(header))
            await websocket.send_bytes(jpeg)
            sent += 1
            next_send = time.monotonic() + opts.period_s
    except (WebSocketDisconnect, RuntimeError):
        reason = "client_closed"
    except Exception:
        log.exception("预览流异常", extra={"camera": camera_id})
        code, reason = WS_INTERNAL_ERROR, "INTERNAL"
    finally:
        if frame_fut is not None:
            # 等在途取帧结束再释放，避免它在 release 之后重新开流
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(frame_fut, PREVIEW_FRAME_WAIT_S + 1.0)
        await _cancel(receiver)
        await _release_off_loop(handle)
        await _close(websocket, code, reason)
        log.info(
            "预览流结束",
            extra={
                "camera": camera_id,
                "mode": opts.mode,
                "fps": opts.fps,
                "sent": sent,
                "reason": reason,
            },
        )
