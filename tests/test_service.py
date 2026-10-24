"""RuntimeService 门面：部署 → 运行/暂停/停止 与 MCU 同步、参数下发与回滚、喷阀检测、采样链路、
训练（预检 + 子进程 + 自动重部署）、备份、故障流、快照、关机动作。全部 Mock 相机 + Mock 下位机。

训练出的小模型在模块级 fixture 里只训一次，各测试按需复制到自己的数据目录。
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from oresorter.api.errors import ApiError
from oresorter.camera.base import CameraError, Frame
from oresorter.config import DEFAULT_CLASS_NAMES, AppConfig, ConfigError, ConfigService
from oresorter.data.samples import StoreError
from oresorter.data.store import ModelStore
from oresorter.fsutil import atomic_write_json, read_json
from oresorter.mcu import protocol as P
from oresorter.mcu.link import McuNack
from oresorter.mcu.mock import MockMcuServer
from oresorter.ml import train as T
from oresorter.ml.train import TrainError
from oresorter.runtime import service as SVC
from oresorter.runtime import state as S
from oresorter.runtime.context import AppContext
from oresorter.runtime.jobs import JobConflict, JobStatus, JobType
from oresorter.runtime.state import IllegalTransition, RunState
from oresorter.vision.synthetic import make_sample_set
from tests.conftest import make_cfg

CLASSES3 = ["A", "B", "C"]
TRAIN_SIZE = 64
RETRAIN_SIZE = 128
SAMPLES_PER_CLASS = 10
CAM_WIDTH = 1024
CHUNK = 256


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_root_logger():
    """其它测试文件可能把根 logger 调到 INFO 并泄漏到后续测试；本文件固定按 WARNING 运行，结束后恢复。"""
    root = logging.getLogger()
    saved = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(saved)


def wait_until(pred: Callable[[], bool], timeout: float = 10.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("条件在超时内未满足")
        time.sleep(step)


def read_rgba(path: Path) -> np.ndarray:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert bgra is not None and bgra.shape[2] == 4
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """模型 1：3 类 × 10 张合成样本，64 px / 1 epoch / 无预训练，产物含 best.pt + model.onnx。"""
    root = tmp_path_factory.mktemp("trained")
    store = ModelStore(root / "models", 7, list(DEFAULT_CLASS_NAMES), min_free_mb=0)
    store.ensure_root()
    store.create(1, "模型1", "测试模型")
    repo = store.samples(1)
    rng = np.random.default_rng(1)
    paths = make_sample_set(
        root / "synth", CLASSES3, SAMPLES_PER_CLASS, rng=rng, size_range=(24, 60)
    )
    for cls, pngs in paths.items():
        for p in pngs:
            rgba = read_rgba(p)
            repo.add(cls, rgba, {"camera_id": 1, "bbox": [0, 0, rgba.shape[1], rgba.shape[0]]})
    cfg = T.TrainConfig(
        backbone="resnet18",
        image_size=TRAIN_SIZE,
        max_epochs=1,
        head_epochs=0,
        batch_size=8,
        pretrained=False,
        device="cpu",
        min_samples_per_class=5,
        num_workers=0,
        seed=3,
    )
    result = T.run(store.model_dir(1), None, cfg, class_names=list(DEFAULT_CLASS_NAMES))
    assert result["artifacts"]["onnx"] is True
    return store.model_dir(1)


def svc_cfg(tmp_path: Path, **over) -> AppConfig:
    base = {
        "capture.bg_mode": "color",
        "capture.bg_color": [0, 0, 255],
        "capture.min_area_px": 100,
        "blow.classes": ["C"],
        "system.auto_run_on_boot": False,
        "ml.image_size": RETRAIN_SIZE,
        "ml.max_epochs": 1,
        "ml.head_epochs": 0,
        "ml.pretrained": False,
        "ml.device": "cpu",
        "ml.min_samples_per_class": 5,
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


def build_ctx(cfg: AppConfig, trained: Path | None = None, *, deployed: bool = False) -> AppContext:
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    svc.save_atomic()
    data_dir = cfg.system.data_dir
    if trained is not None:
        shutil.copytree(trained, data_dir / "models" / "1")
    if deployed:
        atomic_write_json(data_dir / "state.json", {"deployed_model_id": 1})
    return AppContext(svc)


@pytest.fixture
def ctx(tmp_path: Path):
    c = build_ctx(svc_cfg(tmp_path))
    c.start()
    try:
        yield c
    finally:
        c.stop()


@pytest.fixture
def deployed(tmp_path: Path, trained_model_dir: Path):
    """开机即加载已部署模型 1 的上下文。"""
    c = build_ctx(svc_cfg(tmp_path), trained_model_dir, deployed=True)
    c.start()
    try:
        yield c
    finally:
        c.stop()


@pytest.fixture
def events(ctx: AppContext) -> list[dict]:
    got: list[dict] = []
    ctx.events.subscribe(got.append)
    return got


def types_of(events: list[dict]) -> set[str]:
    return {e["type"] for e in events}


def wait_job(c: AppContext, job_id: str, timeout: float = 60.0):
    job = c.jobs.wait(job_id, timeout)
    return job


# --------------------------------------------------------------------------------------
# 运行控制
# --------------------------------------------------------------------------------------


def test_start_requires_deployed_model(ctx: AppContext):
    svc = ctx.runtime
    with pytest.raises(IllegalTransition) as ei:
        svc.start()
    err = ei.value
    assert err.code == "CONFLICT" and S.NO_MODEL_DEPLOYED in err.reasons
    assert err.detail["reasons"] == [S.NO_MODEL_DEPLOYED] and err.detail["state"] == "stopped"
    assert ctx.run_state == "stopped" and ctx.mcu.model.sync_count == 0  # type: ignore[attr-defined]
    # pause 在 stopped 非法；stop 幂等
    with pytest.raises(IllegalTransition):
        svc.pause()
    assert svc.stop() == {"run_state": "stopped", "mcu_ack": True}


def test_deploy_then_run_pause_stop(ctx: AppContext, trained_model_dir: Path, events: list[dict]):
    svc = ctx.runtime
    shutil.copytree(trained_model_dir, ctx.data_dir / "models" / "1")
    assert ctx.store.trained(1)
    # 未创建 / 未训练
    with pytest.raises(StoreError) as ei:
        svc.deploy_model(2)
    assert ei.value.code == "NOT_FOUND"
    svc.create_model(2, "m2")
    with pytest.raises(StoreError) as ei:
        svc.deploy_model(2)
    assert ei.value.code == "CONFLICT"

    r = svc.deploy_model(1)
    job = wait_job(ctx, r["job_id"])
    assert job.status is JobStatus.succeeded, job.message
    assert job.result["backend"] in ("onnx", "torch") and job.result["engine"] is False
    assert job.result["consistency"]["checked"] is True
    assert job.result["consistency"]["top1_agree_rate"] == 1.0
    assert ctx.classifier is not None and ctx.classifier.image_size == TRAIN_SIZE
    assert ctx.deployed_model_id == 1
    assert read_json(ctx.data_dir / "state.json")["deployed_model_id"] == 1
    assert "state" in types_of(events) and "job" in types_of(events)
    assert svc.current_model()["id"] == 1 and svc.current_model()["deployed"] is True
    models = svc.list_models()
    assert len(models) == 7 and models[0]["deployed"] is True and models[2]["created_at"] is None
    assert [c["sample_count"] for c in models[0]["classes"]][:3] == [10, 10, 10]
    metrics = svc.model_metrics(1)
    assert metrics["labels"] == CLASSES3 and "matrix" in metrics
    with pytest.raises(StoreError) as ei:
        svc.model_metrics(2)
    assert ei.value.code == "NOT_FOUND"

    mcu = ctx.mcu
    model = mcu.model  # type: ignore[attr-defined]
    out = svc.start()
    assert out == {"run_state": "running", "mcu_ack": True}
    assert ctx.run_state == "running" and model.state == P.McuState.RUNNING
    assert model.sync_count >= 1, "RUN 前先 SYNC"
    assert ctx.pipeline is not None and ctx.pipeline.running and ctx.pipeline.blow_enabled
    wait_until(lambda: ctx.pipeline.stats()["chunks"] >= 2, timeout=15)
    assert svc.start() == {"run_state": "running", "mcu_ack": True}, "幂等"
    with pytest.raises(IllegalTransition) as ei:
        svc.start_capture(1, "A", 1)
    assert S.ALREADY_RUNNING in ei.value.reasons
    with pytest.raises(ApiError) as ai:
        svc.valve_test_single(3)
    assert ai.value.code == "CONFLICT" and ai.value.detail["reasons"] == [S.ALREADY_RUNNING]
    with pytest.raises(ApiError):
        svc.valve_reset()
    # 运行中快照：按请求高度拼接流水线连续块，而不是只给一块
    shot = svc.snapshot_image(1, height=1000, max_width=300)
    assert shot["height"] == 1000 and shot["width"] == CAM_WIDTH and shot["warnings"] == []
    img = cv2.imread(str(ctx.data_dir / shot["image_url"].removeprefix("/static/")))
    assert img.shape[1] == 300 and img.shape[0] == round(
        1000 * 300 / CAM_WIDTH
    ), "等比缩放到 max_width"

    pipe = ctx.pipeline
    assert svc.pause() == {"run_state": "paused", "mcu_ack": True}
    assert model.state == P.McuState.PAUSED and ctx.pipeline is pipe and pipe.running
    assert pipe.blow_enabled is False, "暂停：继续分类不吹"
    assert svc.pause()["run_state"] == "paused"
    # paused 下喷阀检测也拒绝（否则下位机被带离 PAUSED 而后端仍是 paused）
    for call in (
        lambda: svc.valve_test_single(3),
        lambda: svc.valve_test_cycle(1, 4, 50),
        lambda: svc.valve_reset(),
    ):
        with pytest.raises(ApiError) as ai:
            call()
        assert ai.value.status == 409 and ai.value.detail == {
            "state": "paused",
            "reasons": [S.ALREADY_RUNNING],
        }
    assert model.state == P.McuState.PAUSED and ctx.jobs.running(JobType.valve_test) is None
    sync_before = model.sync_count
    assert svc.start() == {"run_state": "running", "mcu_ack": True}
    assert model.state == P.McuState.RUNNING and pipe.blow_enabled is True
    assert model.sync_count == sync_before, "paused → running 不再 SYNC"
    stats = svc.run_stats()
    assert "since_start" in stats and stats["since_start"]["chunks_total"] >= 2
    assert set(stats["since_start"]) >= {"particles_total", "blown_total", "per_class_total"}
    assert "1" in stats["pipelines"]

    base_threads = threading.active_count()
    t0 = time.monotonic()
    assert svc.stop() == {"run_state": "stopped", "mcu_ack": True}
    assert time.monotonic() - t0 < 3.0
    assert model.state == P.McuState.STOPPED and ctx.pipeline is None and not pipe.running
    assert threading.active_count() == base_threads - 4, "3 条流水线线程 + Mock 相机生产线程"
    states = [e["data"]["run_state"] for e in events if e["type"] == "state"]
    assert {"running", "paused", "stopped"} <= set(states)
    # 删除部署中的模型 → 409；删除其它模型可以
    with pytest.raises(StoreError) as ei:
        svc.delete_model(1)
    assert ei.value.code == "CONFLICT"
    svc.delete_model(2)
    assert not ctx.store.exists(2)


def test_boot_loads_deployed_model_and_can_start(deployed: AppContext):
    assert deployed.classifier is not None and deployed.deployed_model_id == 1
    assert deployed.faults.active() == []
    assert deployed.runtime.start()["run_state"] == "running"
    wait_until(lambda: deployed.pipeline.stats()["chunks"] >= 1, timeout=15)
    assert deployed.runtime.stop()["run_state"] == "stopped"


def test_boot_with_missing_deployed_model_clears_it(tmp_path: Path):
    cfg = svc_cfg(tmp_path)
    c = build_ctx(cfg, deployed=True)  # state.json 指向不存在的模型 1
    c.start()
    try:
        assert c.classifier is None and c.deployed_model_id is None
        assert c.run_state == "stopped"
    finally:
        c.stop()


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------


def test_update_params_mcu_sync_and_offline(ctx: AppContext, events: list[dict]):
    svc = ctx.runtime
    mcu = ctx.mcu
    model = mcu.model  # type: ignore[attr-defined]
    params = svc.get_params()
    assert set(params) == {"valve", "clean", "feed", "outputs", "blow", "capture", "limits"}
    assert params["limits"]["valve"]["open_time"] == [0, 255]
    assert params["limits"]["valve_test"]["hole"] == [1, 64]

    out = svc.update_params("valve", {"open_time": 100, "delay": 300})
    assert out["applied"]["open_time"] == 100 and out["applied"]["delay"] == 300
    assert out["mcu_ack"] is True and out["warnings"] == []
    assert model.params[P.ParamId.VALVE_OPEN_TIME] == 100 and model.params[P.ParamId.DELAY] == 300
    sync = [e["data"] for e in events if e["type"] == "param_sync"]
    assert sync[-1] == {"section": "valve", "mcu_ack": True, "error": None}
    assert read_json(ctx.data_dir / "config.json")["valve"]["open_time"] == 100

    # 越界 → 400，不做部分应用
    with pytest.raises(ConfigError) as ei:
        svc.update_params("valve", {"hold": 10, "reblow": 999})
    assert ei.value.code == "PARAM_OUT_OF_RANGE"
    assert ctx.cfg.valve.hold == 255 and model.params[P.ParamId.VALVE_HOLD] != 10
    with pytest.raises(ConfigError) as ei:
        svc.update_params("system", {"max_models": 3})
    assert ei.value.code == "NOT_FOUND"
    with pytest.raises(ConfigError):
        svc.update_params("valve", {"nope": 1})

    # 不下发下位机的分组
    assert svc.update_params("blow", {"min_confidence": 200})["mcu_ack"] is None
    cap = svc.update_params("capture", {"bg_threshold": 30})
    assert cap["mcu_ack"] is None and cap["applied"]["bg_threshold"] == 30

    # NACK → 回滚配置并抛 McuNack
    mcu.inject_nack(int(P.AckResult.OUT_OF_RANGE))  # type: ignore[attr-defined]
    with pytest.raises(McuNack):
        svc.update_params("clean", {"time": 50})
    assert ctx.cfg.clean.time == 20 and model.params[P.ParamId.CLEAN_TIME] != 50

    # 下位机离线：持久化、mcu_ack False、param_sync 事件；恢复后补发使参数一致
    mcu.set_offline(True)  # type: ignore[attr-defined]
    wait_until(lambda: ctx.state_machine.state is RunState.fault, timeout=3)
    n_sync = len([e for e in events if e["type"] == "param_sync"])
    out = svc.update_params("feed", {"speeds": [10] * 9})
    assert out["mcu_ack"] is False and out["warnings"]
    assert ctx.cfg.feed.speeds == [10] * 9 and model.params[P.ParamId.FEED_SPEED_1] != 10
    sync = [e["data"] for e in events if e["type"] == "param_sync"][n_sync:]
    assert sync[-1]["section"] == "feed" and sync[-1]["mcu_ack"] is False and sync[-1]["error"]
    assert ctx.pending_param_sections() == ["feed"]
    mcu.set_offline(False)  # type: ignore[attr-defined]
    wait_until(lambda: model.params[P.ParamId.FEED_SPEED_1] == 10, timeout=5)
    wait_until(
        lambda: any(
            e["type"] == "param_sync"
            and e["data"] == {"section": "feed", "mcu_ack": True, "error": None}
            for e in events[n_sync:]
        ),
        timeout=5,
    )
    assert ctx.pending_param_sections() == []
    assert model.params[P.ParamId.VALVE_OPEN_TIME] == 100, "全量补发不改其它值"
    wait_until(lambda: ctx.state_machine.state is RunState.stopped, timeout=3)

    # 不经门面的配置更新也会后台同步并发 param_sync
    ctx.config.update("valve", {"reblow": 7})
    wait_until(lambda: model.params[P.ParamId.VALVE_REBLOW] == 7, timeout=5)


def test_valve_test_and_cycle_job(ctx: AppContext):
    svc = ctx.runtime
    model = ctx.mcu.model  # type: ignore[attr-defined]
    with pytest.raises(ConfigError) as ei:
        svc.valve_test_single(0)
    assert ei.value.code == "PARAM_OUT_OF_RANGE"
    with pytest.raises(ConfigError):
        svc.valve_test_single(65)
    assert svc.valve_test_single(5) == {"accepted": True}
    assert ("single", (5,)) in model.valve_tests
    assert ctx.cfg.valve_test.hole == 5

    with pytest.raises(ConfigError):
        svc.valve_test_cycle(10, 5, 100)
    with pytest.raises(ConfigError):
        svc.valve_test_cycle(1, 8, 0)
    r = svc.valve_test_cycle(1, 8, 100)
    jid = r["job_id"]
    wait_until(lambda: model.state == P.McuState.VALVE_TEST, timeout=5)
    assert svc.get_job(jid)["status"] == "running" and svc.get_job(jid)["type"] == "valve_test"
    with pytest.raises(JobConflict):
        svc.valve_test_cycle(1, 8, 100)
    assert svc.list_jobs(active=True)[0]["job_id"] == jid
    assert svc.valve_test_stop() == {"stopped": True}
    job = wait_job(ctx, jid, 5)
    assert job.status is JobStatus.cancelled
    assert model.state == P.McuState.STOPPED and ("stop", ()) in model.valve_tests
    assert svc.valve_reset() == {"accepted": True, "mcu_ack": True}
    assert ("reset", ()) in model.valve_tests
    assert svc.cancel_job(jid) == {"cancelled": False}
    with pytest.raises(ApiError) as ai:
        svc.get_job("j-00000000-9999")
    assert ai.value.code == "NOT_FOUND" and ai.value.status == 404


# --------------------------------------------------------------------------------------
# 采样 / 样本管理
# --------------------------------------------------------------------------------------


def test_capture_job_and_sample_chain(ctx: AppContext, events: list[dict]):
    svc = ctx.runtime
    with pytest.raises(StoreError) as ei:
        svc.start_capture(2, "A", 1)
    assert ei.value.code == "NOT_FOUND"
    svc.create_model(2, "m2")
    with pytest.raises(StoreError) as ei:
        svc.start_capture(2, "Z", 1)
    assert ei.value.code == "VALIDATION"
    with pytest.raises(ConfigError):
        svc.start_capture(2, "A", 9)
    with pytest.raises(ConfigError):
        svc.start_capture(2, "A", 1, max_samples=0)

    r = svc.start_capture(2, "A", 1, max_samples=8, max_frames=500, timeout_s=30)
    wait_until(
        lambda: ctx.run_state in ("sampling",) or ctx.jobs.get(r["job_id"]).finished, timeout=5
    )
    job = wait_job(ctx, r["job_id"], 40)
    assert job.status is JobStatus.succeeded, job.message
    assert job.result["count"] == 8 and job.result["stop_reason"] == "max_samples"
    assert job.result["frames"] >= 1 and job.progress == 1.0
    assert ctx.run_state == "stopped" and ctx.pipeline is None
    cap = [e["data"] for e in events if e["type"] == "capture_sample"]
    assert len(cap) == 8 and [c["count"] for c in cap] == list(range(1, 9))
    assert cap[0]["model_id"] == 2 and cap[0]["class"] == "A"
    assert cap[0]["sample"]["sample_id"].startswith("2/A/cam1_")
    states = [e["data"]["run_state"] for e in events if e["type"] == "state"]
    assert "sampling" in states and states[-1] == "stopped"

    page = svc.list_samples(2, "A", offset=0, limit=100)
    assert page["total"] == 8 and len(page["items"]) == 8
    sid0, sid1, sid2 = (it["sample_id"] for it in page["items"][:3])
    assert svc.list_samples(2, "A", offset=6, limit=5)["items"].__len__() == 2

    d = svc.delete_sample(sid0)
    assert d == {"deleted": True, "undo_batch": "b-1"}
    assert svc.list_samples(2, "A")["total"] == 7
    assert svc.list_samples(2, "A", include_deleted=True)["total"] == 8
    assert svc.restore_sample(sid0) == {"restored": True}
    assert svc.restore_sample(sid0) == {"restored": False}
    assert svc.list_samples(2, "A")["total"] == 8

    d2 = svc.delete_samples([sid1, sid2, sid1])
    assert d2["deleted"] == 2 and d2["undo_batches"] == ["b-2"]
    assert svc.list_samples(2, "A")["total"] == 6
    assert svc.undo_delete(2, "A") == {"restored": 2}
    assert svc.list_samples(2, "A")["total"] == 8

    c = svc.clear_samples(2, "A")
    assert c["deleted"] == 8 and c["undo_batch"] == "b-3"
    assert svc.list_samples(2, "A")["total"] == 0
    assert svc.undo_delete(2, "A") == {"restored": 8}
    with pytest.raises(StoreError) as ei:
        svc.undo_delete(2, "A")
    assert ei.value.code == "NOT_FOUND"
    assert svc.clear_samples(2, "B") == {"deleted": 0, "undo_batch": None}
    with pytest.raises(StoreError):
        svc.delete_sample("not-an-id")

    # 取消采样：job cancelled、状态恢复
    r2 = svc.start_capture(2, "B", 1, max_samples=100_000, max_frames=100_000, timeout_s=60)
    wait_until(lambda: ctx.run_state == "sampling", timeout=5)
    with pytest.raises(IllegalTransition) as ei:
        svc.start_capture(2, "B", 1)
    assert ei.value.reasons == [S.SAMPLING_IN_PROGRESS]
    with pytest.raises(IllegalTransition):
        svc.start()
    assert svc.cancel_job(r2["job_id"]) == {"cancelled": True}
    job2 = wait_job(ctx, r2["job_id"], 10)
    assert job2.status is JobStatus.cancelled and ctx.run_state == "stopped"
    # 采样中 stop → 取消采样并 stopped
    r3 = svc.start_capture(2, "C", 1, max_samples=100_000, max_frames=100_000, timeout_s=60)
    wait_until(lambda: ctx.run_state == "sampling", timeout=5)
    assert svc.stop()["run_state"] == "stopped"
    assert wait_job(ctx, r3["job_id"], 10).status is JobStatus.cancelled
    assert ctx.pipeline is None


def test_capture_from_paused_resumes_sort_pipeline_on_same_line_axis(
    deployed: AppContext, monkeypatch: pytest.MonkeyPatch
):
    """paused → 采样 → paused → running：恢复的分选流水线不得重置相机行号、不得再 SYNC（docs/01 §5.5）。"""
    svc = deployed.runtime
    cam = deployed.cameras[1]
    model = deployed.mcu.model  # type: ignore[attr-defined]
    svc.create_model(2)
    svc.update_params(
        "blow", {"classes": CLASSES3, "min_confidence": 0}
    )  # 所有颗粒都吹，便于观察 EJECT
    svc.start()
    wait_until(lambda: deployed.pipeline.stats()["chunks"] >= 2, timeout=15)
    svc.pause()
    sort_pipe = deployed.pipeline
    assert sort_pipe is not None and sort_pipe.running
    line_before = sort_pipe.latest_frame.line_start
    assert line_before > 0 and model.sync_count == 1
    resets: list[int] = []
    orig_reset = cam.reset_line_counter
    monkeypatch.setattr(cam, "reset_line_counter", lambda: (resets.append(1), orig_reset()))

    r = svc.start_capture(2, "A", 1, max_samples=5, max_frames=300, timeout_s=30)
    job = wait_job(deployed, r["job_id"], 40)
    assert job.status is JobStatus.succeeded, job.message
    assert not sort_pipe.running, "采样独占相机：旧分选流水线已停"
    assert deployed.run_state == "paused"
    new_pipe = deployed.pipeline
    assert new_pipe is not None and new_pipe is not sort_pipe and new_pipe.mode == "sort"
    assert new_pipe.running and new_pipe.blow_enabled is False
    assert resets == [] and model.sync_count == 1, "恢复流水线：不重置行号、不 SYNC"
    wait_until(lambda: new_pipe.latest_frame is not None, timeout=5)
    assert new_pipe.latest_frame.line_start > line_before, "行号续用同一时间轴"

    rx_before = model.eject_rx
    assert svc.start()["run_state"] == "running" and new_pipe.blow_enabled is True
    assert resets == [] and model.sync_count == 1, "paused → running 也不 SYNC / 不重置"
    wait_until(lambda: model.eject_rx > rx_before, timeout=15)
    assert new_pipe.latest_frame.line_start > line_before
    svc.stop()
    assert resets == []


# --------------------------------------------------------------------------------------
# 训练 / 部署 / 备份
# --------------------------------------------------------------------------------------


def test_train_precheck_insufficient_samples(ctx: AppContext, trained_model_dir: Path):
    svc = ctx.runtime
    svc.create_model(3, "m3")
    repo = ctx.store.samples(3)
    rng = np.random.default_rng(2)
    paths = make_sample_set(ctx.data_dir / "synth", ["A", "B"], 2, rng=rng, size_range=(24, 40))
    for cls, pngs in paths.items():
        for p in pngs:
            repo.add(cls, read_rgba(p), {"camera_id": 1})
    with pytest.raises(TrainError) as ei:
        svc.train_model(3)
    assert ei.value.code == "VALIDATION"
    assert ei.value.detail["insufficient"] == {"A": 2, "B": 2}
    assert ei.value.detail["min_samples_per_class"] == 5
    with pytest.raises(TrainError) as ei:
        svc.train_model(3, ["A", "Z"])
    assert ei.value.code == "VALIDATION" and ei.value.detail["unknown"] == ["Z"]
    with pytest.raises(TrainError):
        svc.train_model(3, [])
    svc.create_model(4)
    with pytest.raises(TrainError) as ei:
        svc.train_model(4)
    assert ei.value.code == "VALIDATION" and "2 个" in ei.value.message
    with pytest.raises(StoreError):
        svc.train_model(5)
    assert ctx.jobs.running(JobType.train) is None


@pytest.mark.slow
def test_train_subprocess_then_auto_redeploy(deployed: AppContext):
    """部署中的模型 1 重训（子进程、--config 严格 image_size=128）→ 成功 → 自动重新部署 → 新分类器。"""
    svc = deployed.runtime
    assert deployed.classifier is not None and deployed.classifier.image_size == TRAIN_SIZE
    events: list[dict] = []
    deployed.events.subscribe(events.append)
    r = svc.train_model(1)
    with pytest.raises(JobConflict):
        svc.train_model(1)
    with pytest.raises(StoreError) as ei:
        svc.delete_model(1)
    assert ei.value.code == "CONFLICT"
    job = wait_job(deployed, r["job_id"], 150)
    assert (
        job.status is JobStatus.succeeded
    ), f"{job.message} / {(deployed.store.train_dir(1) / 'subprocess.log').read_text()[-1500:]}"
    res = job.result
    assert set(res["metrics_summary"]) == {"val_accuracy", "macro_f1"}
    assert res["artifacts"] == {"onnx": True, "engine": False}
    assert res["classes"] == CLASSES3 and res["epochs"] == 1
    assert "redeploy_job_id" in res
    assert (deployed.data_dir / "runtime_config.yaml").is_file()
    cfg_doc = read_json(deployed.store.train_dir(1) / "metrics.json")["config"]
    assert cfg_doc["image_size"] == RETRAIN_SIZE and cfg_doc["pretrained"] is False
    redeploy = wait_job(deployed, res["redeploy_job_id"], 60)
    assert redeploy.status is JobStatus.succeeded, redeploy.message
    assert deployed.classifier is not None and deployed.classifier.image_size == RETRAIN_SIZE
    assert deployed.deployed_model_id == 1
    progress = [
        e["data"]["progress"]
        for e in events
        if e["type"] == "job" and e["data"]["job_id"] == r["job_id"]
    ]
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert svc.list_models()[0]["image_size"] == RETRAIN_SIZE
    assert svc.start()["run_state"] == "running"
    wait_until(lambda: deployed.pipeline.stats()["chunks"] >= 1, timeout=15)
    svc.stop()


def test_train_cancel_terminates_subprocess(deployed: AppContext):
    svc = deployed.runtime
    with deployed.suppress_param_sync():
        deployed.config.update("ml", {"max_epochs": 200, "patience": 100})
    r = svc.train_model(1)
    wait_until(lambda: deployed.jobs.get(r["job_id"]).status is JobStatus.running, timeout=5)
    time.sleep(1.5)
    assert svc.cancel_job(r["job_id"]) == {"cancelled": True}
    job = wait_job(deployed, r["job_id"], 30)
    assert job.status is JobStatus.cancelled
    assert deployed.classifier is not None and deployed.classifier.image_size == TRAIN_SIZE


def test_backup_model_job(ctx: AppContext, trained_model_dir: Path):
    svc = ctx.runtime
    shutil.copytree(trained_model_dir, ctx.data_dir / "models" / "1")
    with pytest.raises(StoreError) as ei:
        svc.backup_model(5, 2)
    assert ei.value.code == "NOT_FOUND"
    r = svc.backup_model(1, 3)
    job = wait_job(ctx, r["job_id"], 30)
    assert job.status is JobStatus.succeeded and job.result["target_id"] == 3
    assert ctx.store.exists(3) and ctx.store.trained(3)
    assert svc.list_models()[2]["classes"][0]["sample_count"] == 10
    with pytest.raises(StoreError) as ei:
        svc.backup_model(1, 3)
    assert ei.value.code == "CONFLICT"
    r2 = svc.backup_model(1, 3, overwrite=True)
    assert wait_job(ctx, r2["job_id"], 30).status is JobStatus.succeeded
    d = svc.deploy_model(3)
    assert wait_job(ctx, d["job_id"], 60).status is JobStatus.succeeded
    with pytest.raises(StoreError) as ei:
        svc.backup_model(1, 3, overwrite=True)
    assert ei.value.code == "CONFLICT", "不能覆盖部署中的模型"
    assert svc.patch_model(3, {"name": "备份", "description": "d"})["name"] == "备份"
    with pytest.raises(StoreError):
        svc.patch_model(3, {"id": 9})
    with pytest.raises(StoreError):
        svc.create_model(3)


# --------------------------------------------------------------------------------------
# 故障流 / 类别 / 快照 / 相机 / 关机
# --------------------------------------------------------------------------------------


def test_fault_flow_mcu_and_camera(ctx: AppContext, events: list[dict]):
    mcu = ctx.mcu
    mcu.inject_fault(int(P.EventCode.LOW_PRESSURE))  # type: ignore[attr-defined]
    assert ctx.run_state == "fault" and ctx.faults.has("MCU_FAULT_0001")
    snap = ctx.snapshot()
    assert snap["faults"][0]["code"] == "MCU_FAULT_0001" and snap["faults"][0]["detail"] == "气压低"
    faults = [e["data"] for e in events if e["type"] == "fault"]
    assert faults[-1] == {"code": "MCU_FAULT_0001", "active": True, "detail": "气压低"}
    with pytest.raises(IllegalTransition) as ei:
        ctx.runtime.start()
    assert S.IN_FAULT in ei.value.reasons
    assert ctx.runtime.stop()["run_state"] == "fault", "故障态停止：幂等、仍下发 STOP"
    mcu.inject_fault(int(P.EventCode.LOW_PRESSURE), active=False)  # type: ignore[attr-defined]
    assert ctx.run_state == "stopped" and ctx.faults.active() == []
    assert [e["data"] for e in events if e["type"] == "fault"][-1]["active"] is False

    cam = ctx.cameras[1]
    cam.set_offline(True)  # type: ignore[attr-defined]
    wait_until(lambda: ctx.faults.has("CAMERA_1_DISCONNECTED"), timeout=4)
    wait_until(lambda: ctx.run_state == "fault", timeout=3)
    assert ctx.snapshot()["cameras"][0]["connected"] is False
    cam.set_offline(False)  # type: ignore[attr-defined]
    wait_until(lambda: ctx.run_state == "stopped", timeout=4)
    assert ctx.faults.active() == []

    # MCU 事件 level 2 → 故障；level 1 只发 mcu 事件
    n_mcu = len([e for e in events if e["type"] == "mcu"])
    mcu._emit("event", code=0x0003, level=2, detail=0, text="编码器无信号")  # type: ignore[attr-defined]
    assert ctx.faults.has("MCU_EVENT_0003") and ctx.run_state == "fault"
    ctx.faults.clear("MCU_EVENT_0003")
    assert ctx.run_state == "stopped"
    mcu._emit("event", code=0x0001, level=1, detail=0, text="气压低")  # type: ignore[attr-defined]
    assert ctx.run_state == "stopped"
    assert len([e for e in events if e["type"] == "mcu"]) > n_mcu


def test_fault_during_running_stops_pipeline(deployed: AppContext):
    svc = deployed.runtime
    svc.start()
    pipe = deployed.pipeline
    wait_until(lambda: pipe.stats()["chunks"] >= 1, timeout=15)
    deployed.mcu.inject_fault(int(P.EventCode.EMERGENCY_STOP))  # type: ignore[attr-defined]
    assert deployed.run_state == "fault"
    wait_until(lambda: not pipe.running and deployed.pipeline is None, timeout=5)
    wait_until(lambda: deployed.mcu.model.state == P.McuState.STOPPED, timeout=5)  # type: ignore[attr-defined]
    deployed.mcu.inject_fault(int(P.EventCode.EMERGENCY_STOP), active=False)  # type: ignore[attr-defined]
    assert deployed.run_state == "stopped"
    assert svc.start()["run_state"] == "running"
    svc.stop()


def test_late_fault_reaction_does_not_kill_restarted_run(
    deployed: AppContext, monkeypatch: pytest.MonkeyPatch
):
    """故障置位后立刻清除（抖动），操作员已重新 start；迟到的故障反应线程必须复查状态、不得再 STOP。"""
    svc = deployed.runtime
    mcu = deployed.mcu
    model = mcu.model  # type: ignore[attr-defined]
    captured: list[tuple[Callable, tuple]] = []
    real_spawn = deployed.spawn

    def spawn(name: str, target: Callable, *args):
        if name.startswith("fault-react"):
            captured.append((target, args))
            return None
        return real_spawn(name, target, *args)

    monkeypatch.setattr(deployed, "spawn", spawn)
    svc.start()
    old_pipe = deployed.pipeline
    wait_until(lambda: old_pipe.stats()["chunks"] >= 1, timeout=15)
    mcu.inject_fault(int(P.EventCode.LOW_PRESSURE))  # type: ignore[attr-defined]
    assert deployed.run_state == "fault" and len(captured) == 1
    mcu.inject_fault(int(P.EventCode.LOW_PRESSURE), active=False)  # type: ignore[attr-defined]
    assert deployed.run_state == "stopped"
    assert old_pipe.running, "反应线程尚未运行：旧流水线还在"

    assert svc.start()["run_state"] == "running"
    assert not old_pipe.running, "重新 start 先回收遗留流水线（不能两条流水线共用一台相机）"
    new_pipe = deployed.pipeline
    assert new_pipe is not None and new_pipe is not old_pipe and new_pipe.running
    assert model.state == P.McuState.RUNNING

    target, args = captured[0]
    target(*args)  # 迟到的反应线程现在才跑
    assert deployed.run_state == "running" and model.state == P.McuState.RUNNING
    assert deployed.pipeline is new_pipe and new_pipe.running
    assert deployed.snapshot()["run_state"] == "running"
    svc.stop()
    assert model.state == P.McuState.STOPPED and not new_pipe.running


def test_start_failure_reasons_are_specific(deployed: AppContext, monkeypatch: pytest.MonkeyPatch):
    """背景未学习 → BACKGROUND_NOT_LEARNED 且在 SYNC 之前拒绝；相机开流失败 → CAMERA_DISCONNECTED。"""
    svc = deployed.runtime
    cam = deployed.cameras[1]
    model = deployed.mcu.model  # type: ignore[attr-defined]
    with deployed.suppress_param_sync():
        deployed.config.update("capture", {"bg_mode": "column"})
    assert deployed.backgrounds == {}
    with pytest.raises(ApiError) as ai:
        svc.start()
    assert ai.value.status == 409 and ai.value.detail["reasons"] == [SVC.BACKGROUND_NOT_LEARNED]
    assert ai.value.detail["camera_id"] == 1 and "背景" in ai.value.message
    assert model.sync_count == 0, "背景预检在 SYNC 之前"
    assert deployed.run_state == "stopped" and deployed.pipeline is None
    with deployed.suppress_param_sync():
        deployed.config.update("capture", {"bg_mode": "color"})

    monkeypatch.setattr(
        cam, "start_stream", lambda n: (_ for _ in ()).throw(CameraError("SDK MV_E_RESOURCE"))
    )
    with pytest.raises(ApiError) as ai:
        svc.start()
    assert ai.value.status == 409 and ai.value.detail["reasons"] == [S.CAMERA_DISCONNECTED]
    assert ai.value.detail["camera_id"] == 1 and "MV_E_RESOURCE" in ai.value.message
    assert SVC.BACKGROUND_NOT_LEARNED not in ai.value.detail["reasons"]
    assert deployed.faults.has("CAMERA_1_DISCONNECTED") and deployed.pipeline is None
    monkeypatch.undo()
    wait_until(lambda: deployed.run_state == "stopped", timeout=5)
    assert deployed.faults.active() == []
    assert svc.start()["run_state"] == "running"
    svc.stop()


def test_deploy_and_train_are_mutually_exclusive_per_model(
    ctx: AppContext, trained_model_dir: Path
):
    svc = ctx.runtime
    shutil.copytree(trained_model_dir, ctx.data_dir / "models" / "1")
    hold = lambda h: h.cancel_event.wait(30) or {}  # noqa: E731
    train_job = ctx.jobs.submit(JobType.train, hold, model_id=1)
    try:
        with pytest.raises(StoreError) as ei:
            svc.deploy_model(1)
        assert ei.value.code == "CONFLICT" and ei.value.detail["job_id"] == train_job.job_id
        # 训练任务收尾提交自动重部署：排除自己后可提交
        r = svc.deploy_model(1, exclude_job_id=train_job.job_id)
        assert wait_job(ctx, r["job_id"], 60).status is JobStatus.succeeded
        assert ctx.deployed_model_id == 1
    finally:
        ctx.jobs.cancel(train_job.job_id)
        wait_job(ctx, train_job.job_id, 5)

    deploy_job = ctx.jobs.submit(JobType.deploy, hold, model_id=1)
    try:
        with pytest.raises(StoreError) as ei:
            svc.train_model(1)
        assert ei.value.code == "CONFLICT" and ei.value.detail["job_id"] == deploy_job.job_id
    finally:
        ctx.jobs.cancel(deploy_job.job_id)
        wait_job(ctx, deploy_job.job_id, 5)
    # 备份到别的目标不影响部署源模型
    backup_job = ctx.jobs.submit(JobType.backup, hold, model_id=1, meta={"target_id": 3})
    try:
        r = svc.deploy_model(1)
        assert wait_job(ctx, r["job_id"], 60).status is JobStatus.succeeded
    finally:
        ctx.jobs.cancel(backup_job.job_id)
        wait_job(ctx, backup_job.job_id, 5)


def test_auto_run_on_boot_waits_for_tcp_mcu(tmp_path: Path, trained_model_dir: Path):
    """真机形态：TCP 下位机开机后几百毫秒才连上 → 先 fault（MCU_DISCONNECTED），恢复后仍要自动运行。"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cfg = svc_cfg(tmp_path, **{"system.auto_run_on_boot": True})
    data = cfg.model_dump(mode="json")
    data["system"].update(mcu_kind="tcp", mcu_host="127.0.0.1", mcu_port=port)
    cfg = AppConfig.model_validate(data)
    c = build_ctx(cfg, trained_model_dir, deployed=True)
    server = MockMcuServer(host="127.0.0.1", port=port)
    c.start()
    try:
        assert c.classifier is not None
        assert c.run_state == "fault" and c.faults.has("MCU_DISCONNECTED")
        assert c._auto_run_deadline is not None, "fault 期间保留 auto-run 窗口"
        time.sleep(0.5)
        server.start()
        wait_until(lambda: c.mcu.connected, timeout=10)
        wait_until(lambda: c.run_state == "running", timeout=10)
        assert server.model.sync_count >= 1 and server.model.state == P.McuState.RUNNING
        assert c.faults.active() == [] and c._auto_run_deadline is None
        wait_until(lambda: c.pipeline is not None and c.pipeline.stats()["chunks"] >= 1, timeout=10)
    finally:
        c.stop()
        server.stop()


def test_auto_run_gives_up_once_operator_took_over(tmp_path: Path, trained_model_dir: Path):
    c = build_ctx(
        svc_cfg(tmp_path, **{"system.auto_run_on_boot": True}), trained_model_dir, deployed=True
    )
    c.start()
    try:
        assert c.run_state == "running", "Mock 下位机即时连接：开机直接运行"
        assert c._auto_run_deadline is None
        c.runtime.stop()
        c._auto_run_deadline = time.monotonic() + 60
        c.runtime.create_model(2)
        r = c.runtime.start_capture(
            2, "A", 1, max_samples=100_000, max_frames=100_000, timeout_s=60
        )
        wait_until(lambda: c.run_state == "sampling", timeout=5)
        c._try_auto_run()
        assert c._auto_run_deadline is None, "人工操作后放弃自动运行"
        c.runtime.cancel_job(r["job_id"])
        wait_job(c, r["job_id"], 10)
        assert c.run_state == "stopped"
    finally:
        c.stop()


def test_classes_config(ctx: AppContext, trained_model_dir: Path):
    svc = ctx.runtime
    assert svc.get_classes() == {"class_names": list(DEFAULT_CLASS_NAMES), "max_models": 7}
    with pytest.raises(ConfigError):
        svc.set_classes([])
    with pytest.raises(ConfigError):
        svc.set_classes(["A", "A"])
    svc.create_model(2)
    out = svc.set_classes(["X", "Y", "C"])
    assert out["class_names"] == ["X", "Y", "C"]
    assert ctx.cfg.system.class_names == ["X", "Y", "C"] and ctx.cfg.blow.classes == ["C"]
    assert ctx.store.class_names == ["X", "Y", "C"]
    assert (ctx.store.model_dir(2) / "samples" / "X").is_dir()
    assert list(ctx.stats.snapshot()["per_class_total"]) == ["X", "Y", "C"]
    assert svc.set_classes(["X", "Y", "C"])["class_names"] == ["X", "Y", "C"], "无变化幂等"
    shutil.copytree(trained_model_dir, ctx.data_dir / "models" / "1")
    with pytest.raises(StoreError) as ei:
        svc.set_classes(["A", "B"])
    assert ei.value.code == "CONFLICT" and ei.value.detail["trained_models"] == [1]


def test_snapshot_cameras_and_images(ctx: AppContext):
    svc = ctx.runtime
    snap = svc.snapshot()
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
        "pipeline",
    ):
        assert key in snap, key
    json.dumps(snap)
    assert snap["run_state"] == "stopped" and snap["jobs"] == [] and snap["pipeline"] is None

    cams = svc.list_cameras()
    assert cams[0]["id"] == 1 and cams[0]["connected"] is True and cams[0]["nozzles"] == 64
    assert set(cams[0]) >= {"id", "name", "model", "serial", "ip", "connected", "nozzles"}
    p = svc.camera_params(1)
    assert p["width"] == CAM_WIDTH and p["start_px"] == 0 and p["end_px"] == CAM_WIDTH
    assert p["runtime_chunk_lines"] == CHUNK and p["limits"]["px"] == [0, 4096]
    with pytest.raises(ConfigError) as ei:
        svc.camera_params(9)
    assert ei.value.code == "NOT_FOUND"

    out = svc.update_camera_params(1, {"exposure_us": 80, "start_px": 64, "end_px": 960})
    assert out["applied"]["exposure_us"] == 80 and out["applied"]["start_px"] == 64
    assert ctx.cfg.camera(1).end_px == 960 and ctx.cameras[1].get_params().exposure_us == 80
    nm = svc.nozzle_map(1)
    assert nm["start_px"] == 64 and nm["end_px"] == 960 and nm["nozzles"] == 64
    assert nm["ranges"][0] == {"nozzle": 1, "px_from": 64, "px_to": 77}
    with pytest.raises(ConfigError):
        svc.update_camera_params(1, {"exposure_us": 99999})
    with pytest.raises(ConfigError):
        svc.update_camera_params(1, {"ip": "1.2.3.4"})
    assert svc.awb_once(1) == {"applied": True}
    assert ctx.cfg.camera(1).awb_mode == "off" and ctx.cameras[1].get_params().awb_mode == "off"

    shot = svc.snapshot_image(1, height=512, max_width=400, overlay=True)
    assert shot["width"] == CAM_WIDTH and shot["height"] == 512 and shot["count"] >= 0
    path = ctx.data_dir / shot["image_url"].removeprefix("/static/")
    assert path.is_file() and path.parent.name == "snapshots"
    img = cv2.imread(str(path))
    assert img is not None and img.shape[1] == 400
    with pytest.raises(ConfigError):
        svc.snapshot_image(1, height=0)
    with pytest.raises(ConfigError):
        svc.snapshot_image(1, height=512, max_width=1)

    prev = svc.capture_preview(1, {"min_area_px": 100})
    assert prev["count"] == len(prev["particles"]) > 0
    assert all(set(x) == {"bbox", "area_px"} for x in prev["particles"])
    assert (ctx.data_dir / prev["image_url"].removeprefix("/static/")).is_file()
    assert svc.capture_preview(1, {"min_area_px": 1_500_000})["count"] == 0
    with pytest.raises(ConfigError):
        svc.capture_preview(1, {"erode_px": -1})
    assert ctx.cfg.capture.min_area_px == 100, "preview 的覆盖是临时的"

    # 快照目录只保留最近 N 个
    with ctx.suppress_param_sync():
        ctx.config.update("pipeline", {"queue_size": 8})  # 无关变更，确认不影响
    keep = ctx.cfg.system.max_snapshots_kept
    for _ in range(3):
        svc.snapshot_image(1, height=256, max_width=200)
    assert len(list((ctx.data_dir / "snapshots").glob("*.jpg"))) <= keep

    # 相机离线 → 503
    ctx.cameras[1].set_offline(True)  # type: ignore[attr-defined]
    with pytest.raises(ApiError) as ai:
        svc.camera_params(1)
    assert ai.value.code == "DEVICE_OFFLINE" and ai.value.status == 503
    with pytest.raises(ApiError):
        svc.snapshot_image(1, height=256)
    out = svc.update_camera_params(1, {"exposure_us": 60})
    assert out["warnings"] and ctx.cfg.camera(1).exposure_us == 60
    ctx.cameras[1].set_offline(False)  # type: ignore[attr-defined]
    assert svc.save()["saved"] is True and (ctx.data_dir / "config.backup.json").is_file()


def test_preview_handle_refcount_and_pipeline_sharing(deployed: AppContext):
    svc = deployed.runtime
    cam = deployed.cameras[1]
    assert cam.streaming is False
    h1 = svc.acquire_preview(1)
    h2 = svc.acquire_preview(1)
    assert svc.preview_holders(1) == 2
    f1 = h1.get_frame(2.0)
    assert isinstance(f1, Frame) and cam.streaming is True
    f2 = h1.get_frame(2.0)
    assert f2 is not None and f2 is not f1
    h1.release()
    assert cam.streaming is True, "还有持有者"
    h2.release()
    h2.release()
    assert svc.preview_holders(1) == 0 and cam.streaming is False
    assert h2.get_frame(0.1) is None
    with pytest.raises(ConfigError):
        svc.acquire_preview(42)

    # 运行中：预览复用流水线的帧，不额外开/关流
    svc.start()
    with svc.acquire_preview(1) as h:
        a = h.get_frame(3.0)
        b = h.get_frame(3.0)
        assert a is not None and b is not None and b.seq > a.seq
        assert deployed.pipeline.latest_frame is not None
    assert cam.streaming is True
    svc.stop()
    assert cam.streaming is False


def test_capture_does_not_pollute_run_stats_and_body_details(ctx: AppContext, events: list[dict]):
    svc = ctx.runtime
    svc.create_model(2)
    r = svc.start_capture(2, "A", 1, max_samples=5, max_frames=300, timeout_s=30)
    assert wait_job(ctx, r["job_id"], 40).status is JobStatus.succeeded
    snap = ctx.stats.snapshot()
    assert snap["particles_total"] == 0 and snap["chunks_total"] == 0, "采样不计入分选累计"
    assert svc.run_stats()["since_start"]["particles_total"] == 0

    sid = svc.list_samples(2, "A")["items"][0]["sample_id"]
    assert svc.delete_sample(sid) == {"deleted": True, "undo_batch": "b-1"}
    assert svc.delete_sample(sid) == {"deleted": False, "undo_batch": None}, "重复删除不算删除"
    with pytest.raises(StoreError) as ei:
        svc.delete_sample("2/A/cam1_nope")
    assert ei.value.code == "NOT_FOUND"

    # 一次故障置位恰好一条 state 事件（fault 事件单独一条）
    n_state = len([e for e in events if e["type"] == "state"])
    ctx.mcu.inject_fault(int(P.EventCode.LOW_PRESSURE))  # type: ignore[attr-defined]
    assert ctx.run_state == "fault"
    assert len([e for e in events if e["type"] == "state"]) == n_state + 1
    assert [e for e in events if e["type"] == "state"][-1]["data"]["faults"][0][
        "code"
    ] == "MCU_FAULT_0001"
    with pytest.raises(ApiError) as ai:
        svc.set_classes(["A", "B"])
    assert ai.value.detail == {"state": "fault", "reasons": [S.IN_FAULT]}
    # fault 态下又来一个故障：没有迁移，但补一条 state 让 HMI 看到新的 faults 列表
    n_state = len([e for e in events if e["type"] == "state"])
    ctx.faults.set("MCU_EVENT_0003", "编码器无信号")
    assert len([e for e in events if e["type"] == "state"]) == n_state + 1
    ctx.faults.clear("MCU_EVENT_0003")
    ctx.mcu.inject_fault(int(P.EventCode.LOW_PRESSURE), active=False)  # type: ignore[attr-defined]
    assert ctx.run_state == "stopped"


def test_job_lifecycle_at_info_log_level(ctx: AppContext):
    """本文件其余用例以 WARNING 运行会掩盖该问题：这里显式按生产默认的 INFO 跑一次完整任务生命周期。"""
    logging.getLogger().setLevel(logging.INFO)
    job = ctx.jobs.submit(JobType.backup, lambda h: {"ok": True})
    done = ctx.jobs.wait(job.job_id, 3.0)
    assert done.status is JobStatus.succeeded and done.result == {"ok": True}
    events = [e for e in ctx.events.recent(type="job") if e["data"]["job_id"] == job.job_id]
    assert events and events[-1]["data"]["status"] == "succeeded"


def test_shutdown_power_action_guarded_by_env(ctx: AppContext, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(SVC, "POWER_ACTION_DELAY_S", 0.05)
    monkeypatch.setattr(SVC.subprocess, "run", lambda cmd, **kw: calls.append(list(cmd)))
    monkeypatch.delenv(SVC.POWER_ENV, raising=False)
    out = ctx.runtime.shutdown(reboot=True)
    assert out == {"accepted": True, "action": "reboot"}
    time.sleep(0.4)
    assert calls == [], "未设环境变量不执行电源命令"
    monkeypatch.setenv(SVC.POWER_ENV, "1")
    assert ctx.runtime.shutdown(reboot=False)["action"] == "poweroff"
    wait_until(lambda: calls == [["systemctl", "poweroff"]], timeout=3)
    assert ctx.run_state == "stopped"


def test_context_stop_returns_threads_to_baseline(tmp_path: Path, trained_model_dir: Path):
    baseline = threading.active_count()
    c = build_ctx(svc_cfg(tmp_path), trained_model_dir, deployed=True)
    c.start()
    c.runtime.start()
    wait_until(lambda: c.pipeline.stats()["chunks"] >= 1, timeout=15)
    c.stop()
    assert threading.active_count() == baseline
    assert c.classifier is None and c.pipeline is None
    assert (
        read_json(c.data_dir / "state.json")["last_run_state"] == "running"
    ), "进程退出不改状态（下次按 auto_run 决定）"
    c.stop()  # 幂等


def test_pipeline_derived_faults_are_clearable(ctx: AppContext):
    """CAMERA_{id}_TIMEOUT：流水线已停且相机在线时由设备巡检清除；PIPELINE_ERROR 与 TIMEOUT 在操作员 stop() 时清除，
    全部故障清除后状态机回到 stopped（不再需要重启服务）。"""
    from oresorter.runtime.pipeline import PIPELINE_ERROR_FAULT, camera_fault_code

    timeout_code = camera_fault_code(1, "TIMEOUT")
    ctx.faults.set(timeout_code, "测试注入")
    assert ctx.run_state == "fault"
    ctx._poll_devices()  # 相机在线、无流水线 → 清除
    assert not ctx.faults.has(timeout_code)
    assert ctx.run_state == "stopped"

    ctx.faults.set(PIPELINE_ERROR_FAULT, "测试注入")
    ctx.faults.set(timeout_code, "测试注入")
    assert ctx.run_state == "fault"
    out = ctx.runtime.stop()
    assert not ctx.faults.has(PIPELINE_ERROR_FAULT) and not ctx.faults.has(timeout_code)
    assert out["run_state"] in {"fault", "stopped"} and ctx.run_state == "stopped"
