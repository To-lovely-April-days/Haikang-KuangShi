"""docs/03 REST 接口逐条集成测试（TestClient + 真实 AppContext + Mock 相机/下位机）。

每个接口至少一条正常路径 + 一条错误路径；关键链路（创建 → 采集 → 样本管理 → 真训练 → 部署 → 运行/暂停/停止）
在 :func:`test_full_chain_capture_train_deploy_run` 一次跑完（训练只做一次以控制时长）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import pytest

from oresorter.api.idempotency import IdempotencyCache, extract_request_id
from oresorter.config import AppConfig, ConfigService, SystemConfig
from oresorter.mcu import protocol as P
from oresorter.runtime import state as S
from oresorter.runtime.context import AppContext
from tests.conftest import make_cfg

API = "/api/v1"
CAM_WIDTH = 1024
CHUNK = 256
TRAIN_SIZE = 128
MIN_SAMPLES = 5
FINISHED = {"succeeded", "failed", "cancelled"}
PRODUCTION_LOG_LEVEL = logging.getLevelName(
    str(SystemConfig.model_fields["log_level"].default).upper()
)
"""生产默认日志级别（``system.log_level``，INFO）。"""
JOBS_LOGGER = "oresorter.runtime.jobs"


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _production_log_level():
    """按生产默认级别（``system.log_level`` = INFO）跑全部接口 / WS 集成测试。

    pytest 默认根日志级别是 WARNING，只在 INFO 下触发的问题（典型：``extra`` 用了 LogRecord 保留键 →
    ``makeRecord`` 抛 KeyError）会被盖住；这里是唯一按 HMI 视角在真实 AppContext 下验证契约的门禁，
    必须在生产级别运行。

    """
    root = logging.getLogger()
    saved = root.level
    root.setLevel(PRODUCTION_LOG_LEVEL)
    try:
        yield
    finally:
        root.setLevel(saved)


def api_cfg(tmp_path: Path, **over) -> AppConfig:
    """color 背景（无需 learn-bg）、小模型快训、Mock 相机 1024 宽 / 256 行块、按行频节奏出块。"""
    base = {
        "capture.bg_mode": "color",
        "capture.bg_color": [0, 0, 255],
        "capture.min_area_px": 100,
        "blow.classes": ["C"],
        "system.auto_run_on_boot": False,
        "ml.image_size": TRAIN_SIZE,
        "ml.max_epochs": 1,
        "ml.head_epochs": 0,
        "ml.pretrained": False,
        "ml.device": "cpu",
        "ml.min_samples_per_class": MIN_SAMPLES,
        "ml.batch_size": 8,
        "ml.num_workers": 0,
    }
    base.update(over)
    cfg = make_cfg(tmp_path, **base)
    data = cfg.model_dump(mode="json")
    cam = data["cameras"][0]
    cam.update(width=CAM_WIDTH, offset_x=0, start_px=0, end_px=CAM_WIDTH, runtime_chunk_lines=CHUNK)
    cam["mock"].update(speed=1.0, particles_per_chunk=4, particle_px=[24, 80], seed=7)
    data["pipeline"].update(overlap_lines=128, queue_size=8)
    return AppConfig.model_validate(data)


@pytest.fixture
def client(tmp_path: Path):
    """覆盖 conftest 的 client：用 api_cfg 构造 ConfigService + AppContext + TestClient。"""
    from fastapi.testclient import TestClient

    from oresorter.api.app import create_app

    cfg = api_cfg(tmp_path)
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    svc.save_atomic()
    ctx = AppContext(svc)
    app = create_app(ctx)
    with TestClient(app) as tc:
        tc.ctx = ctx  # type: ignore[attr-defined]
        yield tc


def wait_until(pred: Callable[[], bool], timeout: float = 10.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("条件在超时内未满足")
        time.sleep(step)


def wait_job(client, job_id: str, timeout: float = 30.0) -> dict:
    """轮询 ``GET /jobs/{id}`` 直到结束。"""
    deadline = time.monotonic() + timeout
    while True:
        r = client.get(f"{API}/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in FINISHED:
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"任务 {job_id} 超时未结束: {job}")
        time.sleep(0.05)


def err(r) -> dict:
    body = r.json()
    assert "error" in body, body
    e = body["error"]
    assert set(e) == {"code", "message", "detail"}
    return e


def capture(client, model_id: int, cls: str, n: int) -> list[dict]:
    r = client.post(
        f"{API}/models/{model_id}/classes/{cls}/capture",
        json={"camera_id": 1, "max_samples": n, "max_frames": 500, "timeout_s": 60},
    )
    assert r.status_code == 200, r.text
    job = wait_job(client, r.json()["job_id"], 40)
    assert job["status"] == "succeeded", job
    assert job["type"] == "capture" and job["model_id"] == model_id
    assert job["result"]["count"] == n and job["progress"] == 1.0
    page = client.get(f"{API}/models/{model_id}/classes/{cls}/samples").json()
    assert page["total"] == n
    return page["items"]


# --------------------------------------------------------------------------------------
# §3 系统与状态
# --------------------------------------------------------------------------------------


def test_health(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["version"]


def test_state_shape(client):
    r = client.get(f"{API}/state")
    assert r.status_code == 200
    s = r.json()
    for key in (
        "time",
        "uptime_total_s",
        "run_state",
        "deployed_model_id",
        "cameras",
        "mcu",
        "faults",
        "jobs",
        "stats",
        "version",
    ):
        assert key in s, key
    assert s["run_state"] == "stopped" and s["deployed_model_id"] is None
    assert s["cameras"][0]["id"] == 1 and s["cameras"][0]["connected"] is True
    assert s["mcu"]["connected"] is True
    assert s["faults"] == [] and s["jobs"] == []
    stats = s["stats"]
    assert set(stats["per_class_per_min"]) == set(client.ctx.cfg.system.class_names)
    assert isinstance(stats["particles_per_min"], int)


def test_system_info_and_save(client):
    r = client.get(f"{API}/system/info")
    assert r.status_code == 200
    info = r.json()
    assert info["disk_free_mb"] > 0 and info["version"] and "ips" in info
    assert info["preview_max_fps"] == client.ctx.cfg.system.preview_max_fps
    r = client.post(f"{API}/system/save")
    assert r.status_code == 200 and r.json()["saved"] is True
    assert (client.ctx.data_dir / "config.backup.json").exists()
    r = client.get(f"{API}/system/save")
    assert r.status_code == 405 and err(r)["code"] == "VALIDATION"


def test_system_shutdown_and_reboot_guarded(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ORESORTER_ALLOW_POWER", raising=False)
    r = client.post(f"{API}/system/shutdown", json={"request_id": "pwr-1"})
    assert r.status_code == 200 and r.json() == {"accepted": True, "action": "poweroff"}
    r = client.post(f"{API}/system/reboot")
    assert r.status_code == 200 and r.json() == {"accepted": True, "action": "reboot"}
    assert client.get(f"{API}/state").json()["run_state"] == "stopped"
    # 环境变量未设置：只记日志，不执行 systemctl；进程仍活着
    assert client.get(f"{API}/health").status_code == 200


def test_404_is_structured(client):
    r = client.get(f"{API}/nope")
    assert r.status_code == 404
    assert err(r)["code"] == "NOT_FOUND"


def test_static_mount(client):
    data: Path = client.ctx.data_dir
    (data / "snapshots").mkdir(exist_ok=True)
    (data / "snapshots" / "x.txt").write_text("hi")
    r = client.get("/static/snapshots/x.txt")
    assert r.status_code == 200 and r.text == "hi"
    r = client.get("/static/snapshots/missing.jpg")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"


def test_static_only_exposes_contract_paths(client):
    """docs/03 §12 只承诺 models/{id}/thumbs|samples、snapshots、previews 四类只读路径；
    数据目录里的配置 / 状态 / 权重 / 背景 / 日志不得经 /static 暴露。"""
    data: Path = client.ctx.data_dir
    allowed = {
        "models/1/thumbs/A/s.jpg": b"\xff\xd8thumb",
        "models/1/thumbs/.trash/A/s.jpg": b"\xff\xd8trash",
        "models/1/samples/A/s.png": b"\x89PNGsample",
        "snapshots/cam1_x.jpg": b"\xff\xd8snap",
        "previews/cam1_x.jpg": b"\xff\xd8prev",
    }
    denied = {
        "config.json": b"{}",
        "config.backup.json": b"{}",
        "state.json": b"{}",
        "models/1/best.pt": b"pt",
        "models/1/model.plan": b"plan",
        "models/1/meta.json": b"{}",
        "models/1/train/progress.json": b"{}",
        "background/cam1.npz": b"npz",
        "logs/oresorter.log": b"log",
    }
    for rel, content in {**allowed, **denied}.items():
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    for rel, content in allowed.items():
        r = client.get(f"/static/{rel}")
        assert r.status_code == 200 and r.content == content, rel
    for rel in denied:
        assert (data / rel).exists()
        r = client.get(f"/static/{rel}")
        assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND", rel
    # 目录本身、前缀相似的路径、路径穿越
    for rel in (
        "snapshots",
        "snapshots/",
        "models/1/thumbs",
        "models/x/thumbs/a.jpg",
        "snapshotsx/a.jpg",
    ):
        assert client.get(f"/static/{rel}").status_code == 404, rel
    assert client.get("/static/snapshots/../config.json").status_code == 404


def test_validation_error_body(client):
    r = client.post(f"{API}/valve/test/single", json={"hole": "x"})
    assert r.status_code == 400
    e = err(r)
    assert e["code"] == "VALIDATION" and e["detail"]["errors"][0]["field"] == "body.hole"
    r = client.post(f"{API}/models/abc", json={"name": "x"})
    assert r.status_code == 400 and err(r)["detail"]["errors"][0]["field"] == "path.model_id"


# --------------------------------------------------------------------------------------
# §4 运行控制（无模型时）
# --------------------------------------------------------------------------------------


def test_run_start_without_model_409_and_stats(client):
    r = client.post(f"{API}/run/start")
    assert r.status_code == 409
    e = err(r)
    assert e["code"] == "CONFLICT" and S.NO_MODEL_DEPLOYED in e["detail"]["reasons"]
    assert e["detail"]["state"] == "stopped"
    r = client.post(f"{API}/run/pause")
    assert r.status_code == 409 and S.NOT_RUNNING in err(r)["detail"]["reasons"]
    # 设备前置条件也在 reasons 里一并给出
    client.ctx.cameras[1].set_offline(True)
    r = client.post(f"{API}/run/start")
    assert r.status_code == 409
    reasons = err(r)["detail"]["reasons"]
    assert S.NO_MODEL_DEPLOYED in reasons and S.CAMERA_DISCONNECTED in reasons
    client.ctx.cameras[1].set_offline(False)
    client.ctx.mcu.set_offline(True)  # type: ignore[attr-defined]
    wait_until(lambda: not client.ctx.mcu.connected, 3)
    r = client.post(f"{API}/run/start")
    assert r.status_code == 409 and S.MCU_DISCONNECTED in err(r)["detail"]["reasons"]
    client.ctx.mcu.set_offline(False)  # type: ignore[attr-defined]
    wait_until(lambda: client.get(f"{API}/state").json()["run_state"] == "stopped", 5)
    r = client.post(f"{API}/run/stop", json={"request_id": "stop-1"})
    assert r.status_code == 200 and r.json() == {"run_state": "stopped", "mcu_ack": True}
    r = client.get(f"{API}/run/stats")
    assert r.status_code == 200
    stats = r.json()
    for key in ("particles_per_min", "chunk_latency_ms_p99", "per_class_total", "since_start"):
        assert key in stats, key
    assert stats["since_start"]["chunks_total"] == 0 and stats["pipelines"] == {}


# --------------------------------------------------------------------------------------
# §5 参数
# --------------------------------------------------------------------------------------


def test_params_get_and_put(client):
    ctx: AppContext = client.ctx
    model = ctx.mcu.model  # type: ignore[attr-defined]
    r = client.get(f"{API}/params")
    assert r.status_code == 200
    p = r.json()
    assert set(p) == {"valve", "clean", "feed", "outputs", "blow", "capture", "limits"}
    lim = p["limits"]
    assert lim["valve"]["open_time"] == [0, 255] and lim["valve"]["delay"] == [0, 800]
    assert lim["clean"]["count"] == [0, 5] and lim["feed"]["speed"] == [0, 100]
    # 孔位范围 1 ~ 相机数×每相机喷头数（配置项 hole=0 仅表示“未选”，HMI 输入用这里的 1~N）
    assert lim["valve_test"]["hole"] == [1, ctx.cfg.total_nozzles] == [1, 64]
    assert p["feed"]["speeds"] == [50] * 9 and p["blow"]["classes"] == ["C"]

    r = client.put(f"{API}/params/valve", json={"open_time": 100, "delay": 300})
    assert r.status_code == 200
    out = r.json()
    assert out["applied"]["open_time"] == 100 and out["applied"]["delay"] == 300
    assert out["mcu_ack"] is True and out["warnings"] == []
    assert model.params[P.ParamId.VALVE_OPEN_TIME] == 100
    assert client.get(f"{API}/params").json()["valve"]["open_time"] == 100

    # 越界 → 400，不做部分应用
    r = client.put(f"{API}/params/valve", json={"hold": 10, "reblow": 999})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    assert ctx.cfg.valve.hold == 255 and model.params[P.ParamId.VALVE_HOLD] == 255
    r = client.put(f"{API}/params/valve", json={"nope": 1})
    assert r.status_code == 400
    r = client.put(f"{API}/params/valve", json=[1, 2])
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.put(f"{API}/params/system", json={"max_models": 3})
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"

    # 不下发下位机的分组 mcu_ack = null
    r = client.put(f"{API}/params/blow", json={"min_confidence": 200, "classes": ["A", "C"]})
    assert r.status_code == 200 and r.json()["mcu_ack"] is None
    assert r.json()["applied"]["classes"] == ["A", "C"]
    r = client.put(f"{API}/params/capture", json={"bg_threshold": 30})
    assert r.status_code == 200 and r.json()["mcu_ack"] is None
    assert r.json()["applied"]["bg_threshold"] == 30
    r = client.put(f"{API}/params/blow", json={"classes": ["Z"]})
    assert r.status_code == 400
    # 其余下位机分组
    r = client.put(f"{API}/params/clean", json={"time": 50, "enabled": True})
    assert r.status_code == 200 and r.json()["applied"]["enabled"] is True
    assert model.params[P.ParamId.CLEAN_TIME] == 50
    r = client.put(f"{API}/params/feed", json={"speeds": [10] * 9})
    assert r.status_code == 200 and r.json()["applied"]["speeds"] == [10] * 9
    r = client.put(f"{API}/params/feed", json={"speeds": [10] * 3})
    assert r.status_code == 400
    r = client.put(f"{API}/params/outputs", json={"conveyor": True})
    assert r.status_code == 200 and r.json()["applied"]["conveyor"] is True
    assert r.json()["mcu_ack"] is True


def test_params_mcu_offline_and_nack(client):
    ctx: AppContext = client.ctx
    mcu = ctx.mcu
    mcu.inject_nack(int(P.AckResult.OUT_OF_RANGE))  # type: ignore[attr-defined]
    r = client.put(f"{API}/params/clean", json={"time": 50})
    assert r.status_code == 502 and err(r)["code"] == "MCU_NACK"
    assert ctx.cfg.clean.time == 20, "NACK 回滚"
    mcu.set_offline(True)  # type: ignore[attr-defined]
    wait_until(lambda: client.get(f"{API}/state").json()["run_state"] == "fault", 3)
    r = client.put(f"{API}/params/feed", json={"speeds": [10] * 9})
    assert r.status_code == 200
    assert r.json()["mcu_ack"] is False and r.json()["warnings"]
    assert ctx.cfg.feed.speeds == [10] * 9
    r = client.post(f"{API}/run/stop")
    assert r.status_code == 200 and r.json()["run_state"] == "fault"
    mcu.set_offline(False)  # type: ignore[attr-defined]
    wait_until(lambda: client.get(f"{API}/state").json()["run_state"] == "stopped", 5)


def test_request_id_idempotency(client):
    hdr = {"X-Request-Id": "rid-valve-1"}
    r1 = client.put(f"{API}/params/valve", json={"open_time": 90}, headers=hdr)
    assert r1.status_code == 200 and r1.json()["applied"]["open_time"] == 90
    assert "x-idempotent-replay" not in r1.headers
    r2 = client.put(f"{API}/params/valve", json={"open_time": 91}, headers=hdr)
    assert r2.status_code == 200 and r2.json() == r1.json(), "重复 request_id 返回首次结果"
    assert r2.headers.get("x-idempotent-replay") == "1"
    assert client.ctx.cfg.valve.open_time == 90, "第二次没有执行"
    # 体内 request_id：同 request_id 两次 POST /models/{id} 第二次仍是 201 而非 409
    body = {"name": "m1", "request_id": "rid-create-1"}
    r1 = client.post(f"{API}/models/1", json=body)
    assert r1.status_code == 201 and r1.json()["name"] == "m1"
    r2 = client.post(f"{API}/models/1", json=body)
    assert r2.status_code == 201 and r2.json() == r1.json()
    r3 = client.post(f"{API}/models/1", json={"name": "m1", "request_id": "rid-create-2"})
    assert r3.status_code == 409 and err(r3)["code"] == "CONFLICT"
    # 不同 path 不共用键；无 request_id 不缓存
    r = client.post(f"{API}/models/2", json=body)
    assert r.status_code == 201
    assert client.post(f"{API}/models/2", json={"name": "m2"}).status_code == 409
    assert len(client.app.state.idempotency) == 4, "valve + 2×create(1) + create(2)，409 也缓存"
    # 错误响应同样缓存
    r = client.put(f"{API}/params/valve", json={"reblow": 999, "request_id": "rid-bad"})
    assert r.status_code == 400
    r = client.put(f"{API}/params/valve", json={"reblow": 5, "request_id": "rid-bad"})
    assert r.status_code == 400 and r.headers.get("x-idempotent-replay") == "1"
    assert client.ctx.cfg.valve.reblow == 20


def test_idempotency_cache_unit():
    now = [1000.0]
    cache: IdempotencyCache[str] = IdempotencyCache(ttl_s=60, max_entries=2, clock=lambda: now[0])
    key = ("POST", "/x", "r1")
    assert cache.get(key) is None
    cache.put(key, "a")
    assert cache.get(key) == "a" and len(cache) == 1
    now[0] += 59.9
    assert cache.get(key) == "a"
    now[0] += 0.2
    assert cache.get(key) is None and len(cache) == 0, "60 s 过期"
    cache.put(("POST", "/x", "r2"), "b")
    cache.put(("POST", "/x", "r3"), "c")
    cache.put(("POST", "/x", "r4"), "d")
    assert len(cache) == 2 and cache.get(("POST", "/x", "r2")) is None, "超出上限淘汰最早的"
    with pytest.raises(ValueError):
        IdempotencyCache(ttl_s=0)

    calls = {"n": 0}

    async def scenario() -> None:
        c2: IdempotencyCache[str] = IdempotencyCache(ttl_s=60)

        async def slow() -> str:
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return "first"

        results = await asyncio.gather(c2.run("k", slow), c2.run("k", slow), c2.run("k", slow))
        assert [r[0] for r in results] == ["first"] * 3
        assert sorted(r[1] for r in results) == [False, True, True], "后到者等待并复用"
        assert calls["n"] == 1 and c2.hits == 2 and c2.misses == 1

        async def boom() -> str:
            raise RuntimeError("x")

        async def ok() -> str:
            await asyncio.sleep(0.01)
            return "ok"

        first = asyncio.ensure_future(c2.run("k2", boom))
        await asyncio.sleep(0)
        second = asyncio.ensure_future(c2.run("k2", ok))
        with pytest.raises(RuntimeError):
            await first
        assert await second == ("ok", False), "先到者失败，等待者自己执行"
        assert not c2.in_flight("k2")

    asyncio.run(scenario())

    assert extract_request_id([(b"x-request-id", b" abc ")], b"{}") == "abc"
    assert extract_request_id([], b'{"request_id": "r9", "a": 1}') == "r9"
    assert extract_request_id([], b'{"request_id": ""}') is None
    assert extract_request_id([], b"not json") is None
    assert extract_request_id([], b"[1,2]") is None


def test_config_classes(client):
    r = client.get(f"{API}/config/classes")
    assert r.status_code == 200
    assert r.json() == {"class_names": ["A", "B", "C", "D", "E", "F", "G"], "max_models": 7}
    r = client.put(f"{API}/config/classes", json={"class_names": ["A", "A"]})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.put(f"{API}/config/classes", json={"class_names": []})
    assert r.status_code == 400
    r = client.put(f"{API}/config/classes", json={"class_names": ["X", "Y", "Z"]})
    assert r.status_code == 200 and r.json()["class_names"] == ["X", "Y", "Z"]
    assert client.get(f"{API}/params").json()["blow"]["classes"] == [], "blow.classes 同步过滤"
    assert client.get(f"{API}/models").json()[0]["classes"][0]["name"] == "X"
    assert client.get(f"{API}/state").json()["stats"]["per_class_per_min"] == {
        "X": 0,
        "Y": 0,
        "Z": 0,
    }


# --------------------------------------------------------------------------------------
# §6 喷阀检测 / §10 任务
# --------------------------------------------------------------------------------------


def test_valve_test_endpoints_and_jobs(client):
    ctx: AppContext = client.ctx
    model = ctx.mcu.model  # type: ignore[attr-defined]
    r = client.post(f"{API}/valve/test/single", json={"hole": 0})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.post(f"{API}/valve/test/single", json={"hole": 5, "request_id": "vt-1"})
    assert r.status_code == 200 and r.json() == {"accepted": True}
    assert ("single", (5,)) in model.valve_tests and ctx.cfg.valve_test.hole == 5

    r = client.post(f"{API}/valve/test/cycle", json={"from": 10, "to": 5, "interval_ms": 100})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.post(f"{API}/valve/test/cycle", json={"from": 1, "to": 8, "interval_ms": 0})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.post(f"{API}/valve/test/cycle", json={"from": 1, "to": 8, "interval_ms": 100})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    wait_until(lambda: model.state == P.McuState.VALVE_TEST, 5)
    job = client.get(f"{API}/jobs/{jid}").json()
    assert job["status"] == "running" and job["type"] == "valve_test"
    assert job["meta"] == {"from": 1, "to": 8, "interval_ms": 100} and job["created_at"]
    r = client.post(f"{API}/valve/test/cycle", json={"from": 1, "to": 8, "interval_ms": 100})
    assert r.status_code == 409 and err(r)["code"] == "CONFLICT"
    active = client.get(f"{API}/jobs", params={"active": "true"}).json()
    assert [j["job_id"] for j in active] == [jid]
    assert any(j["job_id"] == jid for j in client.get(f"{API}/state").json()["jobs"])
    r = client.post(f"{API}/valve/test/stop")
    assert r.status_code == 200 and r.json() == {"stopped": True}
    job = wait_job(client, jid, 5)
    assert job["status"] == "cancelled" and model.state == P.McuState.STOPPED
    assert client.get(f"{API}/jobs", params={"active": "true"}).json() == []
    assert len(client.get(f"{API}/jobs").json()) == 1
    r = client.post(f"{API}/valve/reset")
    assert r.status_code == 200 and r.json() == {"accepted": True, "mcu_ack": True}
    assert ("reset", ()) in model.valve_tests
    r = client.post(f"{API}/jobs/{jid}/cancel")
    assert r.status_code == 200 and r.json() == {"cancelled": False}
    r = client.get(f"{API}/jobs/j-00000000-9999")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    # 取消运行中的任务
    r = client.post(f"{API}/valve/test/cycle", json={"from": 1, "to": 8, "interval_ms": 100})
    jid2 = r.json()["job_id"]
    r = client.post(f"{API}/jobs/{jid2}/cancel", json={"request_id": "cancel-1"})
    assert r.status_code == 200 and r.json() == {"cancelled": True}
    assert wait_job(client, jid2, 5)["status"] == "cancelled"


# --------------------------------------------------------------------------------------
# §7 相机
# --------------------------------------------------------------------------------------


def test_cameras_params_awb_nozzle_map(client):
    ctx: AppContext = client.ctx
    r = client.get(f"{API}/cameras")
    assert r.status_code == 200
    cams = r.json()
    assert len(cams) == 1 and cams[0]["id"] == 1 and cams[0]["connected"] is True
    assert set(cams[0]) >= {"id", "name", "model", "serial", "ip", "connected", "nozzles"}
    assert cams[0]["nozzles"] == 64

    r = client.get(f"{API}/cameras/1/params")
    assert r.status_code == 200
    p = r.json()
    assert p["width"] == CAM_WIDTH and p["start_px"] == 0 and p["end_px"] == CAM_WIDTH
    assert p["runtime_chunk_lines"] == CHUNK and p["awb_mode"] == "once"
    assert p["limits"]["px"] == [0, 4096] and p["limits"]["exposure_us"] == [3, 10000]
    r = client.get(f"{API}/cameras/9/params")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"

    r = client.put(
        f"{API}/cameras/1/params",
        json={"exposure_us": 80, "start_px": 64, "end_px": 960, "request_id": "cam-1"},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["applied"]["exposure_us"] == 80 and out["applied"]["start_px"] == 64
    assert out["warnings"] == []
    assert ctx.cameras[1].get_params().exposure_us == 80 and ctx.cfg.camera(1).end_px == 960
    r = client.get(f"{API}/cameras/1/nozzle-map")
    assert r.status_code == 200
    nm = r.json()
    assert nm["start_px"] == 64 and nm["end_px"] == 960 and nm["nozzles"] == 64
    assert nm["quotient"] == 14 and nm["remainder"] == 0
    assert nm["ranges"][0] == {"nozzle": 1, "px_from": 64, "px_to": 77}
    assert len(nm["ranges"]) == 64
    r = client.get(f"{API}/cameras/9/nozzle-map")
    assert r.status_code == 404
    r = client.put(f"{API}/cameras/1/params", json={"exposure_us": 99999})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.put(f"{API}/cameras/1/params", json={"ip": "1.2.3.4"})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.put(f"{API}/cameras/1/params", json={"start_px": 900, "end_px": 100})
    assert r.status_code == 400
    r = client.put(f"{API}/cameras/9/params", json={"exposure_us": 60})
    assert r.status_code == 404

    r = client.post(f"{API}/cameras/1/awb-once")
    assert r.status_code == 200 and r.json() == {"applied": True}
    assert ctx.cfg.camera(1).awb_mode == "off" and ctx.cameras[1].get_params().awb_mode == "off"
    r = client.post(f"{API}/cameras/9/awb-once")
    assert r.status_code == 404

    # 相机离线：查询 / 白平衡 503
    ctx.cameras[1].set_offline(True)  # type: ignore[attr-defined]
    r = client.get(f"{API}/cameras/1/params")
    assert r.status_code == 503 and err(r)["code"] == "DEVICE_OFFLINE"
    assert err(r)["detail"]["device"] == "camera"
    r = client.post(f"{API}/cameras/1/awb-once")
    assert r.status_code == 503
    assert client.get(f"{API}/cameras").json()[0]["connected"] is False
    ctx.cameras[1].set_offline(False)  # type: ignore[attr-defined]


def test_camera_params_put_while_offline_saves_and_warns(client):
    """相机离线时 PUT /cameras/{id}/params：算力盒侧字段（start_px / end_px / runtime_chunk_lines）立即
    生效，相机侧字段持久化、重连后下发，响应 200 且 ``warnings`` 说明"已保存、连接后生效"。

    注意：docs/03 §7 目前写的是"相机离线 → 503 DEVICE_OFFLINE"，与实现不一致。这里按实现断言，因为
    503 会让起止像素这类不依赖相机的字段在相机离线时也无法修改、且 HMI 无从得知哪些字段已保存；
    文档需同步改成本行为（已列入 interface_requests，service / docs 均不在本模块文件内）。
    """
    ctx: AppContext = client.ctx
    device_exposure = ctx.cameras[1].get_params().exposure_us
    ctx.cameras[1].set_offline(True)  # type: ignore[attr-defined]
    r = client.put(f"{API}/cameras/1/params", json={"exposure_us": 60, "start_px": 32})
    assert r.status_code == 200
    out = r.json()
    assert out["applied"]["exposure_us"] == 60 and out["applied"]["start_px"] == 32
    assert any("离线" in w and "保存" in w for w in out["warnings"]), out["warnings"]
    assert ctx.cfg.camera(1).exposure_us == 60 and ctx.cfg.camera(1).start_px == 32, "已持久化"
    assert (
        client.get(f"{API}/cameras/1/nozzle-map").json()["start_px"] == 32
    ), "算力盒侧字段离线也生效"
    # 越界 / 不存在的相机：离线时同样先校验，不保存
    r = client.put(f"{API}/cameras/1/params", json={"exposure_us": 99999})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    assert ctx.cfg.camera(1).exposure_us == 60
    ctx.cameras[1].set_offline(False)  # type: ignore[attr-defined]
    # 相机侧参数在离线期间未能下发（Mock 重连不会自动重放，重连补发由相机链路负责，见 docs/01 §7.3）
    assert ctx.cameras[1].get_params().exposure_us == device_exposure
    r = client.put(f"{API}/cameras/1/params", json={"exposure_us": 60})
    assert r.status_code == 200 and r.json()["warnings"] == []
    assert ctx.cameras[1].get_params().exposure_us == 60


def test_snapshot_and_capture_preview(client):
    ctx: AppContext = client.ctx
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 256, "max_width": 300})
    assert r.status_code == 200
    shot = r.json()
    assert shot["width"] == CAM_WIDTH and shot["height"] == 256 and "count" not in shot
    assert shot["image_url"].startswith("/static/snapshots/") and shot["image_url"].endswith(".jpg")
    img = client.get(shot["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert img.content[:2] == b"\xff\xd8"
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 256, "overlay": True})
    assert r.status_code == 200 and r.json()["count"] >= 0
    r = client.post(f"{API}/cameras/1/snapshot")
    assert r.status_code == 200 and r.json()["height"] == ctx.cfg.camera(1).height
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 0})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 256, "max_width": 1})
    assert r.status_code == 400
    r = client.post(f"{API}/cameras/9/snapshot", json={"height": 256})
    assert r.status_code == 404

    r = client.post(
        f"{API}/capture/preview", json={"camera_id": 1, "settings": {"min_area_px": 100}}
    )
    assert r.status_code == 200
    prev = r.json()
    assert prev["count"] == len(prev["particles"]) > 0
    assert all(set(x) == {"bbox", "area_px"} for x in prev["particles"])
    assert client.get(prev["image_url"]).status_code == 200
    assert prev["image_url"].startswith("/static/previews/")
    r = client.post(
        f"{API}/capture/preview", json={"camera_id": 1, "settings": {"min_area_px": 1_500_000}}
    )
    assert r.status_code == 200 and r.json()["count"] == 0
    assert ctx.cfg.capture.min_area_px == 100, "preview 覆盖是临时的"
    r = client.post(f"{API}/capture/preview", json={"camera_id": 1, "settings": {"erode_px": -1}})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.post(f"{API}/capture/preview", json={"camera_id": 9})
    assert r.status_code == 404
    r = client.post(f"{API}/capture/preview", json={})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    ctx.cameras[1].set_offline(True)  # type: ignore[attr-defined]
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 256})
    assert r.status_code == 503 and err(r)["code"] == "DEVICE_OFFLINE"
    ctx.cameras[1].set_offline(False)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# §8 模型
# --------------------------------------------------------------------------------------


def test_models_crud_and_backup(client):
    r = client.get(f"{API}/models")
    assert r.status_code == 200
    models = r.json()
    assert [m["id"] for m in models] == list(range(1, 8))
    assert all(m["created_at"] is None and m["deployed"] is False for m in models)
    assert [c["name"] for c in models[0]["classes"]] == ["A", "B", "C", "D", "E", "F", "G"]
    assert client.get(f"{API}/models/current").json() is None

    r = client.post(f"{API}/models/1", json={"name": "模型1", "description": "d"})
    assert r.status_code == 201
    m = r.json()
    assert m["id"] == 1 and m["name"] == "模型1" and m["created_at"] and m["trained_at"] is None
    assert m["classes"][0] == {"name": "A", "sample_count": 0, "enabled": False}
    assert client.post(f"{API}/models/1", json={"name": "x"}).status_code == 409
    r = client.post(f"{API}/models/99", json={"name": "x"})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    assert client.post(f"{API}/models/2").status_code == 201, "空体可创建"

    r = client.patch(f"{API}/models/1", json={"description": "新说明"})
    assert r.status_code == 200 and r.json()["description"] == "新说明"
    assert r.json()["name"] == "模型1"
    r = client.patch(f"{API}/models/1", json={"nope": 1})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.patch(f"{API}/models/3", json={"description": "x"})
    assert r.status_code == 404
    r = client.get(f"{API}/models/1/metrics")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    r = client.post(f"{API}/models/1/deploy")
    assert r.status_code == 409 and err(r)["code"] == "CONFLICT", "未训练"
    r = client.post(f"{API}/models/3/deploy")
    assert r.status_code == 404

    r = client.post(f"{API}/models/1/backup", json={"target_id": 2})
    assert r.status_code == 409, "目标存在需 overwrite"
    r = client.post(f"{API}/models/1/backup", json={"target_id": 3})
    assert r.status_code == 200
    job = wait_job(client, r.json()["job_id"], 20)
    assert job["status"] == "succeeded" and job["type"] == "backup"
    assert job["meta"] == {"target_id": 3, "overwrite": False}
    assert client.get(f"{API}/models").json()[2]["created_at"] is not None
    r = client.post(f"{API}/models/1/backup", json={"target_id": 2, "overwrite": True})
    assert (
        r.status_code == 200 and wait_job(client, r.json()["job_id"], 20)["status"] == "succeeded"
    )
    r = client.post(f"{API}/models/1/backup", json={})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.post(f"{API}/models/5/backup", json={"target_id": 6})
    assert r.status_code == 404

    r = client.delete(f"{API}/models/3")
    assert r.status_code == 200 and r.json() == {"deleted": True, "id": 3}
    assert client.get(f"{API}/models").json()[2]["created_at"] is None
    assert client.delete(f"{API}/models/3").status_code == 404
    r = client.delete(f"{API}/models/2", headers={"X-Request-Id": "del-2"})
    assert r.status_code == 200
    r = client.delete(f"{API}/models/2", headers={"X-Request-Id": "del-2"})
    assert r.status_code == 200 and r.headers.get("x-idempotent-replay") == "1"


# --------------------------------------------------------------------------------------
# §9 样本（错误路径）
# --------------------------------------------------------------------------------------


def test_capture_and_sample_errors(client):
    r = client.post(f"{API}/models/2/classes/A/capture", json={"camera_id": 1})
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    assert client.post(f"{API}/models/2").status_code == 201
    r = client.post(f"{API}/models/2/classes/Z/capture", json={"camera_id": 1})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.post(f"{API}/models/2/classes/A/capture", json={"camera_id": 9})
    assert r.status_code == 404
    r = client.post(f"{API}/models/2/classes/A/capture", json={"camera_id": 1, "max_samples": 0})
    assert r.status_code == 400 and err(r)["code"] == "PARAM_OUT_OF_RANGE"
    r = client.post(f"{API}/models/2/classes/A/capture", json={})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.get(f"{API}/models/2/classes/Z/samples")
    assert r.status_code == 400
    r = client.get(f"{API}/models/3/classes/A/samples")
    assert r.status_code == 404
    r = client.get(f"{API}/models/2/classes/A/samples", params={"limit": 0})
    assert r.status_code == 400
    r = client.get(f"{API}/models/2/classes/A/samples")
    assert r.status_code == 200 and r.json() == {"total": 0, "items": []}
    r = client.delete(f"{API}/samples/not-an-id")
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.delete(f"{API}/samples/{quote('2/A/nope', safe='')}")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    r = client.post(f"{API}/samples/{quote('2/A/nope', safe='')}/restore")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    r = client.delete(f"{API}/samples/{quote('9/A/nope', safe='')}")
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.post(f"{API}/samples/delete", json={"ids": []})
    assert r.status_code == 400
    r = client.post(f"{API}/samples/delete", json={"ids": ["2/A/nope", "9/A/x"]})
    assert r.status_code == 404
    r = client.post(f"{API}/models/2/classes/A/undo-delete")
    assert r.status_code == 404 and err(r)["code"] == "NOT_FOUND"
    r = client.delete(f"{API}/models/2/classes/A/samples")
    assert r.status_code == 200 and r.json() == {"deleted": 0, "undo_batch": None}
    r = client.post(f"{API}/models/2/train", json={})
    assert r.status_code == 400 and err(r)["code"] == "VALIDATION"
    r = client.post(f"{API}/models/2/train", json={"classes": ["A", "Q"]})
    assert r.status_code == 400 and err(r)["detail"]["unknown"] == ["Q"]
    r = client.post(f"{API}/models/3/train")
    assert r.status_code == 404


def test_run_stop_during_sampling_cancels_capture(client):
    """docs/03 §4：sampling 状态下 POST /run/stop 取消采样任务并回到 stopped（不只是 /jobs/{id}/cancel）。"""
    ctx: AppContext = client.ctx
    assert client.post(f"{API}/models/1").status_code == 201
    r = client.post(
        f"{API}/models/1/classes/A/capture",
        json={"camera_id": 1, "max_samples": 100_000, "max_frames": 100_000, "timeout_s": 60},
    )
    assert r.status_code == 200
    jid = r.json()["job_id"]
    wait_until(lambda: client.get(f"{API}/state").json()["run_state"] == "sampling", 5)
    assert ctx.cameras[1].streaming is True
    # 采样中不能再采 / 不能 start（需先停止）
    r = client.post(f"{API}/models/1/classes/B/capture", json={"camera_id": 1})
    assert r.status_code == 409 and S.SAMPLING_IN_PROGRESS in err(r)["detail"]["reasons"]
    r = client.post(f"{API}/run/stop", json={"request_id": "stop-sampling"})
    assert r.status_code == 200 and r.json() == {"run_state": "stopped", "mcu_ack": True}
    job = client.get(f"{API}/jobs/{jid}").json()
    assert job["status"] == "cancelled" and job["finished_at"], job
    assert client.get(f"{API}/state").json()["run_state"] == "stopped"
    assert ctx.pipeline is None and ctx.cameras[1].streaming is False
    assert client.get(f"{API}/jobs", params={"active": "true"}).json() == []
    # 已经采到的样本保留
    total = client.get(f"{API}/models/1/classes/A/samples").json()["total"]
    assert total == job["result"]["count"] >= 0


# --------------------------------------------------------------------------------------
# 关键链路：创建 → 采集 → 样本管理 → 训练 → 部署 → 运行 / 暂停 / 停止
# --------------------------------------------------------------------------------------


@pytest.mark.timeout(240)
def test_full_chain_capture_train_deploy_run(client):
    ctx: AppContext = client.ctx
    mcu_model = ctx.mcu.model  # type: ignore[attr-defined]
    assert client.post(f"{API}/models/1", json={"name": "模型1"}).status_code == 201

    # ---- 采集（期间状态 sampling）----
    r = client.post(
        f"{API}/models/1/classes/A/capture",
        json={"camera_id": 1, "max_samples": 8, "max_frames": 500, "timeout_s": 60},
    )
    assert r.status_code == 200
    jid = r.json()["job_id"]
    wait_until(
        lambda: client.get(f"{API}/state").json()["run_state"] == "sampling"
        or client.get(f"{API}/jobs/{jid}").json()["status"] in FINISHED,
        5,
    )
    job = wait_job(client, jid, 40)
    assert job["status"] == "succeeded" and job["result"]["count"] == 8, job
    assert job["meta"]["class"] == "A" and job["started_at"] and job["finished_at"]
    assert client.get(f"{API}/state").json()["run_state"] == "stopped"
    items = client.get(f"{API}/models/1/classes/A/samples").json()["items"]
    assert len(items) == 8
    s0 = items[0]
    assert s0["sample_id"].startswith("1/A/cam1_") and s0["deleted"] is False
    assert set(s0) >= {"thumb_url", "image_url", "created_at", "camera_id", "bbox", "area_px"}
    assert client.get(s0["thumb_url"]).status_code == 200
    assert client.get(s0["image_url"]).headers["content-type"] == "image/png"
    page = client.get(f"{API}/models/1/classes/A/samples", params={"offset": 6, "limit": 5}).json()
    assert page["total"] == 8 and len(page["items"]) == 2
    assert client.get(f"{API}/models").json()[0]["classes"][0]["sample_count"] == 8

    # ---- 样本管理：%2F 路径、删除 / 恢复 / 批量 / 撤销 / 清空 ----
    sid0, sid1, sid2 = (it["sample_id"] for it in items[:3])
    enc = quote(sid0, safe="")
    assert "%2F" in enc
    r = client.delete(f"{API}/samples/{enc}")
    assert r.status_code == 200 and r.json() == {"deleted": True, "undo_batch": "b-1"}
    assert client.get(f"{API}/models/1/classes/A/samples").json()["total"] == 7
    r = client.get(f"{API}/models/1/classes/A/samples", params={"include_deleted": "true"}).json()
    assert r["total"] == 8
    gone = next(it for it in r["items"] if it["sample_id"] == sid0)
    assert gone["deleted"] is True and "/.trash/" in gone["thumb_url"]
    assert client.get(gone["thumb_url"]).status_code == 200
    r = client.post(f"{API}/samples/{enc}/restore")
    assert r.status_code == 200 and r.json() == {"restored": True}
    assert client.post(f"{API}/samples/{enc}/restore").json() == {"restored": False}
    r = client.delete(f"{API}/samples/{sid1}")  # 未编码的 / 也接受
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert client.post(f"{API}/models/1/classes/A/undo-delete").json() == {"restored": 1}
    r = client.post(f"{API}/samples/delete", json={"ids": [sid1, sid2, sid1]})
    assert r.status_code == 200 and r.json() == {"deleted": 2, "undo_batches": ["b-3"]}
    assert client.get(f"{API}/models/1/classes/A/samples").json()["total"] == 6
    r = client.post(f"{API}/models/1/classes/A/undo-delete")
    assert r.status_code == 200 and r.json() == {"restored": 2}
    r = client.delete(f"{API}/models/1/classes/A/samples")
    assert r.status_code == 200 and r.json() == {"deleted": 8, "undo_batch": "b-4"}
    assert client.get(f"{API}/models/1/classes/A/samples").json()["total"] == 0
    assert client.post(f"{API}/models/1/classes/A/undo-delete").json() == {"restored": 8}
    r = client.post(f"{API}/models/1/classes/A/undo-delete")
    assert r.status_code == 404

    # ---- 训练：样本不足 → 400 detail.insufficient；足够 → 202 真训练 ----
    r = client.post(f"{API}/models/1/train", json={"classes": ["A", "B"]})
    assert r.status_code == 400
    e = err(r)
    assert e["code"] == "VALIDATION" and e["detail"]["insufficient"] == {"B": 0}
    capture(client, 1, "B", 6)
    r = client.post(f"{API}/models/1/train", json={"request_id": "train-1"})
    assert r.status_code == 202, r.text
    tjid = r.json()["job_id"]
    r = client.post(f"{API}/models/1/train", json={"request_id": "train-1"})
    assert r.status_code == 202 and r.json()["job_id"] == tjid, "幂等"
    r = client.post(f"{API}/models/1/train")
    assert r.status_code == 409 and err(r)["code"] == "CONFLICT", "已有训练任务"
    r = client.delete(f"{API}/models/1")
    assert r.status_code == 409, "训练中不能删除"
    job = wait_job(client, tjid, 150)
    assert job["status"] == "succeeded", job
    assert job["type"] == "train" and job["meta"]["classes"] == ["A", "B"]
    res = job["result"]
    assert res["classes"] == ["A", "B"] and res["artifacts"]["onnx"] is True
    assert set(res["metrics_summary"]) >= {"val_accuracy", "macro_f1"}
    r = client.get(f"{API}/models/1/metrics")
    assert r.status_code == 200
    metrics = r.json()
    assert metrics["labels"] == ["A", "B"] and len(metrics["matrix"]) == 2
    m1 = client.get(f"{API}/models").json()[0]
    assert m1["trained_at"] and m1["image_size"] == TRAIN_SIZE and m1["metrics_summary"]

    # ---- 部署 ----
    r = client.post(f"{API}/models/1/deploy")
    assert r.status_code == 200
    job = wait_job(client, r.json()["job_id"], 60)
    assert job["status"] == "succeeded", job
    assert client.get(f"{API}/state").json()["deployed_model_id"] == 1
    cur = client.get(f"{API}/models/current").json()
    assert cur["id"] == 1 and cur["deployed"] is True
    assert client.get(f"{API}/models").json()[0]["deployed"] is True
    r = client.delete(f"{API}/models/1")
    assert r.status_code == 409 and err(r)["code"] == "CONFLICT", "当前部署模型不能删除"

    # ---- 运行 / 暂停 / 停止 ----
    r = client.post(f"{API}/run/start", json={"request_id": "start-1"})
    assert r.status_code == 200 and r.json() == {"run_state": "running", "mcu_ack": True}
    assert client.get(f"{API}/state").json()["run_state"] == "running"
    assert mcu_model.state == P.McuState.RUNNING and mcu_model.sync_count >= 1
    assert client.post(f"{API}/run/start").json()["run_state"] == "running", "幂等"
    r = client.post(f"{API}/valve/test/single", json={"hole": 3})
    assert r.status_code == 409 and err(r)["detail"]["reasons"] == [S.ALREADY_RUNNING]
    r = client.post(f"{API}/valve/test/cycle", json={"from": 1, "to": 4, "interval_ms": 100})
    assert r.status_code == 409
    r = client.post(f"{API}/valve/reset")
    assert r.status_code == 409
    r = client.post(f"{API}/models/1/classes/A/capture", json={"camera_id": 1})
    assert r.status_code == 409 and S.ALREADY_RUNNING in err(r)["detail"]["reasons"]
    r = client.put(f"{API}/config/classes", json={"class_names": ["X", "Y"]})
    assert r.status_code == 409
    wait_until(lambda: client.get(f"{API}/run/stats").json()["chunks_total"] >= 2, 15)
    stats = client.get(f"{API}/run/stats").json()
    assert stats["since_start"]["chunks_total"] >= 2 and "1" in stats["pipelines"]
    # 运行中快照取自流水线
    r = client.post(f"{API}/cameras/1/snapshot", json={"height": 512, "max_width": 200})
    assert r.status_code == 200 and r.json()["height"] == 512
    r = client.post(f"{API}/run/pause")
    assert r.status_code == 200 and r.json() == {"run_state": "paused", "mcu_ack": True}
    assert mcu_model.state == P.McuState.PAUSED
    assert client.post(f"{API}/run/pause").json()["run_state"] == "paused"
    r = client.post(f"{API}/run/start")
    assert r.status_code == 200 and r.json()["run_state"] == "running"
    r = client.post(f"{API}/run/stop")
    assert r.status_code == 200 and r.json() == {"run_state": "stopped", "mcu_ack": True}
    assert mcu_model.state == P.McuState.STOPPED
    assert client.get(f"{API}/state").json()["run_state"] == "stopped"
    assert ctx.pipeline is None
    # 训练后类别名不可改
    r = client.put(f"{API}/config/classes", json={"class_names": ["X", "Y"]})
    assert r.status_code == 409 and err(r)["detail"]["trained_models"] == [1]


# --------------------------------------------------------------------------------------
# 生命周期 / 契约
# --------------------------------------------------------------------------------------


def test_client_lifecycle_returns_threads_to_baseline(tmp_path: Path):
    from fastapi.testclient import TestClient

    from oresorter.api.app import create_app

    base = threading.active_count()
    cfg = api_cfg(tmp_path)
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    ctx = AppContext(svc)
    with TestClient(create_app(ctx)) as tc:
        assert tc.get(f"{API}/health").status_code == 200
        assert tc.post(f"{API}/models/1").status_code == 201
    assert ctx.cameras[1].connected is False, "lifespan 结束 → ctx.stop()"
    deadline = time.monotonic() + 3.0
    while threading.active_count() > base and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() == base


def test_openapi_export_is_in_sync():
    """docs/openapi.json 必须与代码一致（scripts/export_openapi.py --check）。"""
    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "export_openapi.py"), "--check"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    spec = (repo / "docs" / "openapi.json").read_text(encoding="utf-8")
    for path in (
        "/api/v1/run/start",
        "/api/v1/params/{section}",
        "/api/v1/cameras/{camera_id}/nozzle-map",
        "/api/v1/models/{model_id}/train",
        "/api/v1/samples/{sample_id}",
        "/api/v1/jobs/{job_id}/cancel",
    ):
        assert f'"{path}"' in spec, path
    # PUT /params/{section} 的合法分组写进 OpenAPI（HMI 生成客户端用），但仍按 str 接收以保持 404 语义
    from oresorter.api.routes_params import PARAM_SECTIONS

    doc = json.loads(spec)
    (section,) = [
        p
        for p in doc["paths"]["/api/v1/params/{section}"]["put"]["parameters"]
        if p["in"] == "path"
    ]
    assert section["name"] == "section" and section["schema"]["enum"] == list(PARAM_SECTIONS)
    assert set(PARAM_SECTIONS) == {"valve", "clean", "feed", "outputs", "blow", "capture"}
