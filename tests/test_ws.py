"""WebSocket 集成测试：/ws/events（首条 state、事件转发、过期全量事件过滤、ping/pong、静默断开、取消订阅、
生产 INFO 级别下的终态 job 事件）与 /ws/preview/{id}（full / edges 两种模式、白线、限频与 fps 越界拒绝、
参数拒绝、离线、关闭后在线程池释放）。全部用例按生产默认日志级别运行（见 test_api._production_log_level）。"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from oresorter.api import ws as WS
from oresorter.camera.base import Frame
from oresorter.config import AppConfig, ConfigService
from oresorter.runtime.context import AppContext
from tests.test_api import (  # noqa: F401  （_production_log_level 作为 autouse fixture 被 pytest 收集）
    CAM_WIDTH,
    CHUNK,
    FINISHED,
    JOBS_LOGGER,
    _production_log_level,
    api_cfg,
    wait_job,
    wait_until,
)

API = "/api/v1"
PING_S = 0.3
TIMEOUT_S = 1.0


@pytest.fixture(autouse=True)
def _fast_ping(monkeypatch: pytest.MonkeyPatch):
    """缩短 ping 周期与静默上限，测试才跑得快。"""
    monkeypatch.setattr(WS, "PING_INTERVAL_S", PING_S)
    monkeypatch.setattr(WS, "CLIENT_TIMEOUT_S", TIMEOUT_S)


@pytest.fixture
def client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from oresorter.api.app import create_app

    cfg: AppConfig = api_cfg(tmp_path)
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    svc.save_atomic()
    ctx = AppContext(svc)
    with TestClient(create_app(ctx)) as tc:
        tc.ctx = ctx  # type: ignore[attr-defined]
        yield tc


def recv_until(ws, pred: Callable[[dict], bool], max_messages: int = 60) -> dict:
    """读消息直到满足条件（ping 保证了消息流不会停）。"""
    for _ in range(max_messages):
        m = ws.receive_json()
        if pred(m):
            return m
    raise AssertionError("未在限定消息数内收到期望事件")


def recv_frame(ws) -> tuple[dict, np.ndarray]:
    header = ws.receive_json()
    assert header["type"] == "frame", header
    jpeg = ws.receive_bytes()
    assert jpeg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return header, img


def is_white_column(img: np.ndarray, x: int) -> bool:
    """整列每一行都很亮（JPEG 色度抽样会给白线掺入邻近蓝色，故按三通道均值判断；蓝底均值约 85）。"""
    return bool(img[:, x].mean(axis=1).min() >= 170)


# --------------------------------------------------------------------------------------
# /ws/events
# --------------------------------------------------------------------------------------


def test_events_first_state_then_param_sync(client):
    ctx: AppContext = client.ctx
    subs = ctx.events.subscriber_count
    with client.websocket_connect(f"{API}/ws/events") as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["ts"]
        snap = first["data"]
        assert snap["run_state"] == "stopped" and snap["cameras"][0]["id"] == 1
        assert snap["mcu"]["connected"] is True and "stats" in snap
        assert ctx.events.subscriber_count == subs + 1
        r = client.put(f"{API}/params/valve", json={"open_time": 77})
        assert r.status_code == 200
        ev = recv_until(ws, lambda m: m["type"] == "param_sync")
        assert ev["data"] == {"section": "valve", "mcu_ack": True, "error": None}
        # 配置类别变化 → state 事件
        client.put(f"{API}/config/classes", json={"class_names": ["X", "Y"]})
        ev = recv_until(
            ws, lambda m: m["type"] == "state" and "X" in m["data"]["stats"]["per_class_total"]
        )
        assert ev["data"]["stats"]["per_class_total"] == {"X": 0, "Y": 0}
    deadline = time.monotonic() + 2.0
    while ctx.events.subscriber_count != subs and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ctx.events.subscriber_count == subs, "断开后取消订阅"


def test_events_job_and_capture_sample_events(client):
    assert client.post(f"{API}/models/1").status_code == 201
    with client.websocket_connect(f"{API}/ws/events") as ws:
        assert ws.receive_json()["type"] == "state"
        r = client.post(
            f"{API}/models/1/classes/A/capture",
            json={"camera_id": 1, "max_samples": 4, "max_frames": 500, "timeout_s": 30},
        )
        assert r.status_code == 200
        jid = r.json()["job_id"]
        statuses: list[str] = []
        samples = 0
        states: list[str] = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            m = ws.receive_json()
            if m["type"] == "job" and m["data"]["job_id"] == jid:
                statuses.append(m["data"]["status"])
                assert m["data"]["type"] == "capture" and m["data"]["model_id"] == 1
                if m["data"]["status"] == "succeeded":
                    assert m["data"]["result"]["count"] == 4 and m["data"]["progress"] == 1.0
                    break
            elif m["type"] == "capture_sample":
                samples += 1
                d = m["data"]
                assert d["model_id"] == 1 and d["class"] == "A" and d["count"] == samples
                assert d["sample"]["sample_id"].startswith("1/A/cam1_")
            elif m["type"] == "state":
                states.append(m["data"]["run_state"])
        assert statuses[0] == "queued" and "running" in statuses and statuses[-1] == "succeeded"
        assert samples == 4 and "sampling" in states
        wait_job(client, jid, 5)


def test_events_job_finish_at_full_info_level(client):
    """docs/03 §11：任务完成 / 失败 / 取消的 ``job`` 事件必须推到 /ws/events —— 在 **完整** 的生产 INFO
    级别下验证（其余用例把 jobs logger 压到 WARNING 绕开了该 bug，见 test_api._production_log_level）。"""
    logging.getLogger(JOBS_LOGGER).setLevel(logging.NOTSET)  # 恢复继承根 logger 的 INFO
    assert client.post(f"{API}/models/1").status_code == 201
    with client.websocket_connect(f"{API}/ws/events") as ws:
        assert ws.receive_json()["type"] == "state"
        r = client.post(f"{API}/models/1/backup", json={"target_id": 2})
        assert r.status_code == 200
        jid = r.json()["job_id"]
        # 任务本身是完成的（GET /jobs 可见），缺的只是终态事件
        assert wait_job(client, jid, 10)["status"] == "succeeded"
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            m = ws.receive_json()  # ping 保证消息流不停
            if (
                m["type"] == "job"
                and m["data"]["job_id"] == jid
                and m["data"]["status"] in FINISHED
            ):
                assert m["data"]["status"] == "succeeded"
                break
        else:
            raise AssertionError("INFO 级别下未收到终态 job 事件")


def test_events_drop_stale_snapshot_events_after_first_state(client, monkeypatch):
    """订阅后、快照前已发布的全量型事件（state / stats / mcu）内容比首条快照旧，不得排在快照之后送达；
    增量型事件（job 等）即使 ts 更早也照常转发。"""
    ctx: AppContext = client.ctx
    old_ts = (
        (datetime.now(timezone.utc) - timedelta(seconds=1))
        .astimezone()
        .isoformat(timespec="milliseconds")
    )
    stale_state = {"type": "state", "ts": old_ts, "data": {**ctx.snapshot(), "run_state": "OLD"}}
    stale_stats = {"type": "stats", "ts": old_ts, "data": {"marker": "OLD"}}
    stale_mcu = {"type": "mcu", "ts": old_ts, "data": {"marker": "OLD"}}
    early_job = {"type": "job", "ts": old_ts, "data": {"job_id": "j-early", "status": "queued"}}
    real_subscribe = ctx.events.subscribe

    def hooked_subscribe(cb):
        unsub = real_subscribe(cb)
        # 模拟"订阅之后、快照观察之前"总线已经分发到本连接的事件
        for ev in (stale_state, stale_stats, stale_mcu, early_job):
            cb(ev)
        return unsub

    monkeypatch.setattr(ctx.events, "subscribe", hooked_subscribe)
    with client.websocket_connect(f"{API}/ws/events") as ws:
        first = ws.receive_json()
        assert first["type"] == "state" and first["data"]["run_state"] == "stopped"
        got: list[dict] = []
        while True:
            m = ws.receive_json()
            if m["type"] == "ping":
                break
            got.append(m)
        kinds = [
            (
                m["type"],
                m["data"].get("run_state") or m["data"].get("marker") or m["data"].get("job_id"),
            )
            for m in got
        ]
        assert ("job", "j-early") in kinds, "增量型事件不丢"
        # 真实的 state 心跳（内容新、ts 新）可以出现，但带 OLD 标记的过期全量事件一条都不能有
        assert not [k for k in kinds if k[1] == "OLD"], kinds
        # 之后发生的真实变化照常到达且内容是新的
        client.put(f"{API}/config/classes", json={"class_names": ["X", "Y"]})
        ev = recv_until(ws, lambda m: m["type"] == "state")
        assert "X" in ev["data"]["stats"]["per_class_total"]


def test_is_stale_snapshot_event_unit():
    since = datetime(2026, 10, 24, 8, 0, 0, tzinfo=timezone.utc)
    older = "2026-10-24T15:59:59.999+08:00"  # == 07:59:59.999Z
    newer = "2026-10-24T16:00:00.001+08:00"
    assert WS.is_stale_snapshot_event({"type": "state", "ts": older}, since)
    assert WS.is_stale_snapshot_event({"type": "stats", "ts": older}, since)
    assert WS.is_stale_snapshot_event({"type": "mcu", "ts": older}, since)
    assert not WS.is_stale_snapshot_event({"type": "state", "ts": newer}, since)
    for t in ("job", "capture_sample", "param_sync", "fault"):
        assert not WS.is_stale_snapshot_event({"type": t, "ts": older}, since), t
    # ts 缺失 / 无法解析 → 保留
    assert not WS.is_stale_snapshot_event({"type": "state"}, since)
    assert not WS.is_stale_snapshot_event({"type": "state", "ts": "yesterday"}, since)
    assert not WS.is_stale_snapshot_event({"type": "state", "ts": 12345}, since)


def test_events_ping_pong_and_silent_client_disconnect(client):
    with client.websocket_connect(f"{API}/ws/events") as ws:
        assert ws.receive_json()["type"] == "state"
        t0 = time.monotonic()
        ping = recv_until(ws, lambda m: m["type"] == "ping")
        assert ping["ts"] and time.monotonic() - t0 < PING_S * 3
        # 持续回 pong：超过静默上限仍保持连接
        until = time.monotonic() + TIMEOUT_S * 2
        pings = 1
        while time.monotonic() < until:
            ws.send_json({"type": "pong"})
            m = ws.receive_json()
            if m["type"] == "ping":
                pings += 1
        assert pings >= 4
        assert client.get(f"{API}/health").status_code == 200
        # 停止回应 → 服务端在 CLIENT_TIMEOUT_S 后以 1001 断开
        t0 = time.monotonic()
        while True:
            m = ws.receive()
            if m["type"] == "websocket.close":
                break
            assert time.monotonic() - t0 < TIMEOUT_S + PING_S * 3
        assert m["code"] == WS.WS_GOING_AWAY
        assert TIMEOUT_S * 0.8 <= time.monotonic() - t0 <= TIMEOUT_S + PING_S * 3


def test_event_feed_drops_oldest_when_full():
    async def scenario() -> None:
        feed = WS._EventFeed(asyncio.get_running_loop(), maxsize=3)
        for i in range(5):
            feed.on_event({"type": "stats", "ts": "", "data": {"i": i}})
        await asyncio.sleep(0.01)
        got = [feed.queue.get_nowait()["data"]["i"] for _ in range(feed.queue.qsize())]
        assert got == [2, 3, 4] and feed.dropped == 2
        feed.closed = True
        feed.on_event({"type": "stats", "ts": "", "data": {"i": 9}})
        await asyncio.sleep(0.01)
        assert feed.queue.empty()

    asyncio.run(scenario())


# --------------------------------------------------------------------------------------
# /ws/preview/{camera_id}
# --------------------------------------------------------------------------------------


def test_preview_full_mode_and_release(client):
    ctx: AppContext = client.ctx
    cam = ctx.cameras[1]
    assert cam.streaming is False
    with client.websocket_connect(
        f"{API}/ws/preview/1?mode=full&fps=10&max_width=256&quality=60"
    ) as ws:
        header, img = recv_frame(ws)
        assert set(header) == {"type", "seq", "w", "h", "start_px", "end_px", "ts"}
        assert header["w"] == 256 and header["h"] == round(CHUNK * 256 / CAM_WIDTH)
        assert header["start_px"] == 0 and header["end_px"] == CAM_WIDTH
        assert img.shape[1] == header["w"] and img.shape[0] == header["h"]
        assert cam.streaming is True and ctx.runtime.preview_holders(1) == 1
        header2, _ = recv_frame(ws)
        assert header2["seq"] > header["seq"]
    deadline = time.monotonic() + 3.0
    while ctx.runtime.preview_holders(1) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ctx.runtime.preview_holders(1) == 0
    assert cam.streaming is False, "关闭连接 → release → 无流水线时停流"


def test_preview_release_runs_off_event_loop(client, monkeypatch):
    """最后一个持有者 release → camera.stop_stream()（SDK 停流 + join 采集线程，可达数百 ms）必须在
    线程池里执行，不能阻塞事件循环（否则期间 /health、/run/stop 等全部停顿）。"""
    ctx: AppContext = client.ctx
    cam = ctx.cameras[1]
    loop_thread = client.portal.call(threading.get_ident)
    seen: dict[str, object] = {}
    real_stop = cam.stop_stream

    def slow_stop() -> None:
        seen["thread"] = threading.get_ident()
        seen["name"] = threading.current_thread().name
        time.sleep(0.3)
        real_stop()

    monkeypatch.setattr(cam, "stop_stream", slow_stop)
    with client.websocket_connect(f"{API}/ws/preview/1?fps=10&max_width=128") as ws:
        recv_frame(ws)
        assert cam.streaming is True
    t0 = time.monotonic()
    # 停流进行期间（0.3 s）事件循环仍能服务 HTTP 请求
    assert client.get(f"{API}/health").status_code == 200
    first_health_latency = time.monotonic() - t0
    wait_until(lambda: "thread" in seen and not cam.streaming, 3)
    assert seen["thread"] != loop_thread, f"stop_stream 跑在事件循环线程 {seen['name']}"
    assert first_health_latency < 0.25, first_health_latency
    assert ctx.runtime.preview_holders(1) == 0


def test_preview_edges_mode_white_lines(client):
    edge = 300
    with client.websocket_connect(f"{API}/ws/preview/1?mode=edges&fps=10&edge_px={edge}") as ws:
        header, img = recv_frame(ws)
        assert header["w"] == 2 * edge and img.shape[1] == 2 * edge and img.shape[0] == CHUNK
        assert header["start_px"] == 0 and header["end_px"] == CAM_WIDTH
        # start_px=0 → 拼接图 0,1 列；end_px=1024 → 局部 1022,1023 → 拼接图 598,599 列
        assert is_white_column(img, 0) and is_white_column(img, 1)
        assert is_white_column(img, 2 * edge - 2) and is_white_column(img, 2 * edge - 1)
        assert not is_white_column(img, 2) and not is_white_column(img, 2 * edge - 3)
    # 起止像素改变后线位置跟随（start_px=64 → 64,65；end_px=960 → 局部 958,959 → 300+(958-724)=534,535）
    r = client.put(f"{API}/cameras/1/params", json={"start_px": 64, "end_px": 960})
    assert r.status_code == 200
    with client.websocket_connect(f"{API}/ws/preview/1?mode=edges&fps=10&edge_px={edge}") as ws:
        header, img = recv_frame(ws)
        assert header["start_px"] == 64 and header["end_px"] == 960
        assert is_white_column(img, 64) and is_white_column(img, 65)
        assert is_white_column(img, 534) and is_white_column(img, 535)
        assert not is_white_column(img, 0) and not is_white_column(img, 2 * edge - 1)


def test_compose_edges_unit():
    from oresorter.config import CameraConfig

    img = np.zeros((8, 1000, 3), np.uint8)
    img[:] = (0, 0, 200)
    cfg = CameraConfig(id=1, width=1000, offset_x=100, start_px=120, end_px=1090, nozzles=64)
    out = WS.compose_edges(img, cfg, 300)
    assert out.shape == (8, 600, 3)
    # start 局部 20 → 20,21；end 局部 990 → 988,989 → 300+(988-700)=588,589
    for x in (20, 21, 588, 589):
        assert (out[:, x] == 255).all(), x
    for x in (19, 22, 587, 590):
        assert (out[:, x] == (0, 0, 200)).all(), x
    # 起止不在两段内时不画线
    cfg2 = CameraConfig(id=1, width=1000, offset_x=0, start_px=400, end_px=600, nozzles=64)
    out2 = WS.compose_edges(img, cfg2, 300)
    assert (out2 == (0, 0, 200)).all()
    # edge_px 超过帧宽时夹到帧宽
    assert WS.compose_edges(img, cfg2, 5000).shape[1] == 2000
    frame = Frame(seq=3, line_start=0, image=img, ts=time.monotonic(), camera_id=1)
    header, jpeg = WS.encode_preview(frame, cfg2, WS.PreviewOptions("full", 5, 70, 250, 300))
    assert header["w"] == 250 and header["h"] == 2 and header["seq"] == 3
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (2, 250, 3)


def test_preview_fps_limit(client):
    with client.websocket_connect(f"{API}/ws/preview/1?mode=full&fps=2&max_width=128") as ws:
        stamps = []
        for _ in range(3):
            recv_frame(ws)
            stamps.append(time.monotonic())
        gaps = [b - a for a, b in itertools.pairwise(stamps)]
        assert all(g >= 0.45 for g in gaps), gaps
    max_fps = client.ctx.cfg.system.preview_max_fps
    # 恰好等于上限（10）可用：4 帧至少跨 3 个 0.1 s 周期
    with client.websocket_connect(
        f"{API}/ws/preview/1?mode=full&fps={max_fps}&max_width=128"
    ) as ws:
        stamps = []
        for _ in range(4):
            recv_frame(ws)
            stamps.append(time.monotonic())
        assert stamps[-1] - stamps[0] >= 0.27, stamps
    # 超过上限不夹断：拒绝并说明范围（HMI 从 GET /system/info.preview_max_fps 取上限）
    with client.websocket_connect(f"{API}/ws/preview/1?mode=full&fps={max_fps + 1}") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and m["code"] == "PARAM_OUT_OF_RANGE"
        assert m["detail"] == {"field": "fps", "input": max_fps + 1, "range": [1, max_fps]}
        close = ws.receive()
        assert close["type"] == "websocket.close" and close["code"] == WS.WS_POLICY_VIOLATION
    assert client.get(f"{API}/system/info").json()["preview_max_fps"] == max_fps
    assert client.ctx.runtime.preview_holders(1) == 0


def test_preview_rejects_bad_params(client):
    cases = {
        "9": ("NOT_FOUND", None),
        "1?quality=5": ("PARAM_OUT_OF_RANGE", "quality"),
        "1?quality=101": ("PARAM_OUT_OF_RANGE", "quality"),
        "1?mode=half": ("VALIDATION", "mode"),
        "1?fps=0": ("PARAM_OUT_OF_RANGE", "fps"),
        "1?fps=100": ("PARAM_OUT_OF_RANGE", "fps"),
        "1?max_width=8": ("PARAM_OUT_OF_RANGE", "max_width"),
        f"1?mode=edges&edge_px={CAM_WIDTH + 1}": ("PARAM_OUT_OF_RANGE", "edge_px"),
    }
    for suffix, (code, field) in cases.items():
        with client.websocket_connect(f"{API}/ws/preview/{suffix}") as ws:
            m = ws.receive_json()
            assert m["type"] == "error" and m["code"] == code, (suffix, m)
            if field:
                assert m["detail"]["field"] == field
            close = ws.receive()
            assert close["type"] == "websocket.close" and close["code"] == WS.WS_POLICY_VIOLATION
    assert client.ctx.runtime.preview_holders(1) == 0
    with (
        pytest.raises(WebSocketDisconnect) as ei,
        client.websocket_connect(f"{API}/ws/preview/1?fps=abc"),
    ):
        pass
    assert ei.value.code == WS.WS_POLICY_VIOLATION


def test_preview_camera_offline(client):
    ctx: AppContext = client.ctx
    cam = ctx.cameras[1]
    cam.set_offline(True)  # type: ignore[attr-defined]
    with client.websocket_connect(f"{API}/ws/preview/1?fps=5") as ws:
        m = ws.receive_json()
        assert m["type"] == "error" and m["code"] == "DEVICE_OFFLINE"
        close = ws.receive()
        assert close["type"] == "websocket.close" and close["code"] == WS.WS_INTERNAL_ERROR
    cam.set_offline(False)  # type: ignore[attr-defined]
    # 流中掉线
    with client.websocket_connect(f"{API}/ws/preview/1?fps=10") as ws:
        recv_frame(ws)
        cam.set_offline(True)  # type: ignore[attr-defined]
        while True:
            m = ws.receive()
            if m["type"] == "websocket.close":
                raise AssertionError("应先收到 error 再关闭")
            if "text" in m:
                doc = json.loads(m["text"])
                if doc["type"] == "error":
                    assert doc["code"] == "DEVICE_OFFLINE"
                    break
        close = ws.receive()
        assert close["type"] == "websocket.close" and close["code"] == WS.WS_INTERNAL_ERROR
    cam.set_offline(False)  # type: ignore[attr-defined]
    assert ctx.runtime.preview_holders(1) == 0


def test_preview_shares_pipeline_frames_while_sampling(client):
    """采样期间预览复用流水线的帧，不打断采集，连接关闭不停流。"""
    ctx: AppContext = client.ctx
    assert client.post(f"{API}/models/1").status_code == 201
    r = client.post(
        f"{API}/models/1/classes/A/capture",
        json={"camera_id": 1, "max_samples": 100_000, "max_frames": 100_000, "timeout_s": 60},
    )
    jid = r.json()["job_id"]
    deadline = time.monotonic() + 5
    while client.get(f"{API}/state").json()["run_state"] != "sampling":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    with client.websocket_connect(f"{API}/ws/preview/1?fps=10&max_width=128") as ws:
        h1, _ = recv_frame(ws)
        h2, _ = recv_frame(ws)
        assert h2["seq"] > h1["seq"]
        assert ctx.pipeline is not None and ctx.pipeline.running
    assert ctx.cameras[1].streaming is True, "流水线仍在跑，预览关闭不停流"
    assert client.post(f"{API}/jobs/{jid}/cancel").json()["cancelled"] is True
    assert wait_job(client, jid, 10)["status"] == "cancelled"
    assert client.get(f"{API}/state").json()["run_state"] == "stopped"
