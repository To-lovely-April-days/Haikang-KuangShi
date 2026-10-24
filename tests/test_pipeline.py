"""流水线：capture / sort 两种模式（Mock 相机 + Mock 下位机）、同步 process_frame、丢块、掉线/超时故障、
停止时限与线程回收、处理异常计数与 PIPELINE_ERROR。"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from oresorter.config import AppConfig, ConfigService
from oresorter.runtime import decision as D
from oresorter.runtime.context import AppContext
from oresorter.runtime.pipeline import (
    BackgroundUnavailable,
    CaptureSink,
    ClassifySink,
    Pipeline,
    PipelineError,
    camera_fault_code,
)
from oresorter.runtime.state import RunState
from oresorter.vision.nozzle import NozzleMap
from tests.conftest import make_cfg

CLASSES = ["A", "B", "C"]
CAM_WIDTH = 1024
CHUNK = 256


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


def pipe_cfg(tmp_path: Path, **over) -> AppConfig:
    """Mock 相机 1024 宽、256 行块、按真实行频节奏（≈35 块/s）、color 背景、吹 C 类。"""
    base = {
        "capture.bg_mode": "color",
        "capture.bg_color": [0, 0, 255],
        "capture.min_area_px": 100,
        "blow.classes": ["C"],
        "system.auto_run_on_boot": False,
    }
    base.update(over)
    cfg = make_cfg(tmp_path, **base)
    data = cfg.model_dump(mode="json")
    cam = data["cameras"][0]
    cam.update(width=CAM_WIDTH, offset_x=0, start_px=0, end_px=CAM_WIDTH, runtime_chunk_lines=CHUNK)
    cam["mock"].update(speed=1.0, particles_per_chunk=4, particle_px=[24, 80], seed=7)
    data["pipeline"].update(overlap_lines=128, queue_size=8)
    return AppConfig.model_validate(data)


def make_ctx(cfg: AppConfig) -> AppContext:
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    svc.save_atomic()
    return AppContext(svc)


@pytest.fixture
def ctx(tmp_path: Path):
    c = make_ctx(pipe_cfg(tmp_path))
    c.start()
    try:
        yield c
    finally:
        c.stop()


def wait_until(pred: Callable[[], bool], timeout: float = 10.0, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("条件在超时内未满足")
        time.sleep(step)


class FakeClassifier:
    """固定输出某一类的分类器替身。"""

    backend = "fake"

    def __init__(self, cls: str = "C", image_size: int = 32):
        self._cls = cls
        self._image_size = image_size
        self.calls = 0
        self.closed = False

    @property
    def classes(self) -> list[str]:
        return list(CLASSES)

    @property
    def image_size(self) -> int:
        return self._image_size

    def predict(self, batch: np.ndarray):
        assert batch.ndim == 4 and batch.shape[1:] == (3, self._image_size, self._image_size)
        self.calls += 1
        probs = np.full((batch.shape[0], len(CLASSES)), 0.05, np.float32)
        probs[:, CLASSES.index(self._cls)] = 0.9
        return probs, list(CLASSES)

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------------------
# capture 模式
# --------------------------------------------------------------------------------------


def test_capture_mode_writes_samples_and_reports(ctx: AppContext):
    ctx.store.create(1, "m1")
    repo = ctx.store.samples(1)
    seen: list[tuple[dict, int]] = []
    sink = CaptureSink(
        repo,
        "A",
        1,
        max_samples=24,
        max_frames=500,
        batch="job-x",
        on_sample=lambda s, n: seen.append((s, n)),
    )
    pipe = Pipeline(ctx, ctx.cameras[1], mode="capture", sink=sink)
    assert pipe.mode == "capture" and not pipe.running
    pipe.start()
    assert pipe.running
    assert sink.done.wait(20), "应在上限前完成采样"
    assert pipe.stop() is True
    stats = pipe.stats()
    assert stats["frames"] >= 3, "至少处理 3 块"
    assert stats["chunks"] >= 3 and stats["samples"] == 24
    assert sink.count == 24 and sink.stop_reason == "max_samples"
    assert repo.count("A") == 24
    assert [n for _, n in seen] == list(range(1, 25)), "count 单调递增"
    s0 = seen[0][0]
    assert s0["sample_id"].startswith("1/A/cam1_") and s0["camera_id"] == 1
    assert s0["thumb_url"].startswith("/static/models/1/thumbs/A/") and s0["deleted"] is False
    assert s0["area_px"] > 0 and len(s0["bbox"]) == 4
    first = repo.get(s0["sample_id"])
    assert first.batch == "job-x"
    snap = ctx.stats.snapshot()
    assert snap["chunks_total"] == 0 and snap["particles_total"] == 0, "采样不污染分选累计统计"
    assert pipe.latest_frame is not None and pipe.latest_frame.camera_id == 1


def test_capture_sink_max_frames_and_store_error(ctx: AppContext):
    ctx.store.create(1)
    repo = ctx.store.samples(1)
    sink = CaptureSink(repo, "B", 1, max_samples=10_000, max_frames=2)
    pipe = Pipeline(ctx, ctx.cameras[1], mode="capture", sink=sink)
    pipe.start()
    assert sink.done.wait(10)
    pipe.stop()
    assert sink.stop_reason == "max_frames" and sink.frames == 2
    assert sink.count == repo.count("B") > 0

    # 磁盘不足（min_free_mb 极大）→ CONFLICT → 采样结束并记录原因
    repo.min_free_mb = 10**9
    sink2 = CaptureSink(repo, "B", 1, max_samples=5)
    pipe2 = Pipeline(ctx, ctx.cameras[1], mode="capture", sink=sink2)
    pipe2.start()
    assert sink2.done.wait(10)
    pipe2.stop()
    assert sink2.stop_reason == "store_error" and sink2.error and "CONFLICT" in sink2.error
    assert sink2.count == 0
    repo.min_free_mb = 0
    with pytest.raises(ValueError):
        CaptureSink(repo, "B", 1, max_samples=0)
    with pytest.raises(ValueError):
        Pipeline(ctx, ctx.cameras[1], mode="capture")  # capture 必须给 sink
    with pytest.raises(ValueError):
        Pipeline(ctx, ctx.cameras[1], mode="weird")


# --------------------------------------------------------------------------------------
# sort 模式
# --------------------------------------------------------------------------------------


def test_sort_mode_sends_eject_to_mcu(ctx: AppContext):
    fake = FakeClassifier("C")
    ctx.classifier = fake  # type: ignore[assignment]
    pipe = Pipeline(ctx, ctx.cameras[1], mode="sort")
    assert isinstance(pipe.sink, ClassifySink) and pipe.sink.classifier is fake
    pipe.start()
    try:
        wait_until(lambda: ctx.mcu.model.eject_rx > 0, timeout=15)  # type: ignore[attr-defined]
        wait_until(lambda: pipe.stats()["chunks"] >= 3, timeout=10)
    finally:
        assert pipe.stop() is True
    st = pipe.stats()
    assert st["eject_sent"] > 0 and st["eject_failed"] == 0 and st["blown"] >= st["eject_sent"]
    assert fake.calls > 0
    snap = ctx.stats.snapshot()
    assert snap["blown_total"] > 0 and snap["per_class_total"]["C"] > 0
    assert snap["per_class_total"]["A"] == 0
    assert st["mode"] == "sort" and st["running"] is False and st["errors"] == 0


def test_process_frame_sync_path_nozzle_mapping_and_dropped(ctx: AppContext):
    cam = ctx.cameras[1]
    fake = FakeClassifier("C")
    ctx.classifier = fake  # type: ignore[assignment]
    pipe = Pipeline(ctx, cam, mode="sort")
    nm = NozzleMap.from_camera(ctx.cfg.camera(1), 0)
    assert pipe.nozzle_map == nm
    cam.start_stream(CHUNK)
    try:
        rx_before = ctx.mcu.model.eject_rx  # type: ignore[attr-defined]
        total_eject = 0
        total_particles = 0
        last_seq = -1
        for _ in range(6):
            frame = cam.get_frame(2.0)
            assert frame is not None
            res = pipe.process_frame(frame)
            assert res.seq == frame.seq and res.line_start == frame.line_start
            assert res.decisions is not None and len(res.decisions) == len(res.particles)
            total_particles += len(res.particles)
            blown = [(p, d) for p, d in zip(res.particles, res.decisions, strict=True) if d.blow]
            assert len(res.eject) == len(blown), "有效区覆盖全幅，不应有跳过"
            for item, (p, d) in zip(res.eject, blown, strict=True):
                assert d.cls == "C" and d.reason == D.REASON_BLOW
                x0, x1 = p.x_range
                assert nm.global_span(x0, x1, 0) == (item.nozzle_first, item.nozzle_last)
                assert item.line == p.trailing_line >= 0
                assert (
                    item.nozzle_first <= nm.pixel_to_nozzle(int(p.centroid_x)) <= item.nozzle_last
                )
                assert p.extra["cls"] == "C" and p.extra["blow"] is True
            total_eject += len(res.eject)
            last_seq = frame.seq
        assert total_particles > 0 and total_eject > 0
        assert ctx.mcu.model.eject_rx - rx_before == total_eject, "同步路径直接下发"  # type: ignore[attr-defined]
        assert pipe.stats()["dropped"] == 0

        # 丢块注入 → 拼接器 seq 断号计数
        cam.inject_drop(2)
        found = False
        for _ in range(200):
            frame = cam.get_frame(2.0)
            assert frame is not None
            pipe.process_frame(frame)
            if frame.seq > last_seq + 1:
                found = True
                break
            last_seq = frame.seq
        assert found, "应观察到 seq 跳号"
        assert pipe.stats()["dropped"] >= 2 and pipe.stats()["resets"] >= 1
        assert cam.stats.dropped_chunks >= 2
    finally:
        cam.stop_stream()


def test_blow_disabled_and_no_classifier(ctx: AppContext):
    cam = ctx.cameras[1]
    cam.start_stream(CHUNK)
    try:
        frames = [cam.get_frame(2.0) for _ in range(3)]
        assert all(f is not None for f in frames)
        # 无分类器：全部 no_classifier、不吹
        ctx.classifier = None
        pipe = Pipeline(ctx, cam, mode="sort")
        rx0 = ctx.mcu.model.eject_rx  # type: ignore[attr-defined]
        n = 0
        for f in frames:
            res = pipe.process_frame(f)  # type: ignore[arg-type]
            n += len(res.particles)
            assert res.eject == [] and res.result.blown == 0
            assert all(d.reason == D.REASON_NO_CLASSIFIER for d in res.decisions or [])
        assert n > 0 and ctx.mcu.model.eject_rx == rx0  # type: ignore[attr-defined]
        # 禁吹（paused）：分类统计照常，但不产生喷阀条目
        fake = FakeClassifier("C")
        pipe2 = Pipeline(ctx, cam, mode="sort")
        pipe2.replace_classifier(fake)  # type: ignore[arg-type]
        pipe2.set_blow_enabled(False)
        assert pipe2.blow_enabled is False
        per_class = 0
        for f in frames:
            res = pipe2.process_frame(f)  # type: ignore[arg-type]
            assert res.eject == [] and res.result.blown == 0
            per_class += res.result.per_class.get("C", 0)
            assert all(p.extra["blow"] is False for p in res.particles)
        assert per_class > 0 and fake.calls > 0
        pipe2.set_blow_enabled(True)
        pipe2.stitcher.reset()
        got = sum(len(pipe2.process_frame(f).eject) for f in frames)  # type: ignore[arg-type]
        assert got > 0
        # 替换分类器并关闭旧的
        other = FakeClassifier("A")
        old = pipe2.replace_classifier(other, close_old=True)  # type: ignore[arg-type]
        assert old is fake and fake.closed is True
        pipe2.stitcher.reset()
        res = pipe2.process_frame(frames[0])  # type: ignore[arg-type]
        assert res.eject == [] and all(
            d.reason == D.REASON_CLASS_NOT_SELECTED for d in res.decisions or []
        )
    finally:
        cam.stop_stream()


# --------------------------------------------------------------------------------------
# 故障：掉线 / 超时 / 处理异常
# --------------------------------------------------------------------------------------


def test_camera_offline_sets_fault_and_recovers(ctx: AppContext):
    cam = ctx.cameras[1]
    pipe = Pipeline(ctx, cam, mode="sort")
    code = camera_fault_code(1, "DISCONNECTED")
    pipe.start()
    try:
        wait_until(lambda: pipe.stats()["chunks"] >= 2)
        cam.set_offline(True)
        wait_until(lambda: ctx.faults.has(code), timeout=3)
        # 故障表先改字典再通知监听者：状态机迁移紧随其后，需等待而不是立即断言
        wait_until(lambda: ctx.state_machine.state is RunState.fault, timeout=3)
        assert pipe.running and pipe.healthy(), "掉线不算流水线不健康"
        time.sleep(0.2)  # 掉线前已入队的块处理完
        before = pipe.stats()["chunks"]
        time.sleep(0.3)
        assert pipe.stats()["chunks"] == before, "掉线期间不出块"
        cam.set_offline(False)
        wait_until(lambda: not ctx.faults.has(code), timeout=5)
        wait_until(lambda: pipe.stats()["chunks"] > before, timeout=5)
        wait_until(lambda: ctx.state_machine.state is RunState.stopped, timeout=3)
        assert pipe.stats()["frames"] >= 1
        assert ctx.faults.active() == []
    finally:
        assert pipe.stop() is True


def test_camera_timeout_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    c = make_ctx(pipe_cfg(tmp_path, **{"pipeline.frame_timeout_s": 0.5}))
    c.start()
    try:
        cam = c.cameras[1]
        pipe = Pipeline(c, cam, mode="sort")
        code = camera_fault_code(1, "TIMEOUT")
        pipe.start()
        try:
            wait_until(lambda: pipe.stats()["chunks"] >= 1)

            def silent(timeout: float):
                time.sleep(min(timeout, 0.05))
                return None

            monkeypatch.setattr(cam, "get_frame", silent)
            wait_until(lambda: c.faults.has(code), timeout=4)
            assert pipe.running
            monkeypatch.undo()
            wait_until(lambda: not c.faults.has(code), timeout=4)
            before = pipe.stats()["chunks"]
            wait_until(lambda: pipe.stats()["chunks"] > before, timeout=5)
        finally:
            assert pipe.stop() is True
    finally:
        c.stop()


def test_process_errors_counted_and_fault_after_threshold(ctx: AppContext):
    cam = ctx.cameras[1]
    pipe = Pipeline(ctx, cam, mode="sort")

    def boom(frame, particles):
        raise RuntimeError("sink exploded")

    pipe.sink.handle = boom  # type: ignore[method-assign]
    pipe.start()
    try:
        wait_until(lambda: ctx.faults.has("PIPELINE_ERROR"), timeout=15)
        st = pipe.stats()
        assert st["errors"] >= 20 and st["consecutive_errors"] >= 20 and st["chunks"] == 0
        assert pipe.running, "异常不退出线程"
        assert all(t.is_alive() for t in pipe._threads)
        del pipe.sink.handle  # type: ignore[attr-defined]  恢复类方法
        wait_until(lambda: not ctx.faults.has("PIPELINE_ERROR"), timeout=10)
        wait_until(lambda: pipe.stats()["chunks"] >= 1, timeout=10)
        assert pipe.stats()["consecutive_errors"] == 0
    finally:
        assert pipe.stop() is True


# --------------------------------------------------------------------------------------
# 停止 / 健康 / 热更新
# --------------------------------------------------------------------------------------


def test_stop_within_two_seconds_and_threads_back_to_baseline(ctx: AppContext):
    baseline = threading.active_count()
    pipe = Pipeline(ctx, ctx.cameras[1], mode="sort")
    assert pipe.healthy() is True, "未运行视为健康"
    pipe.start()
    assert threading.active_count() == baseline + 4, "acquire + process + send + Mock 相机生产线程"
    wait_until(lambda: pipe.stats()["chunks"] >= 2)
    assert pipe.healthy() is True
    assert ctx.cameras[1].streaming is True
    t0 = time.monotonic()
    assert pipe.stop() is True
    assert time.monotonic() - t0 < 2.0
    assert threading.active_count() == baseline
    assert ctx.cameras[1].streaming is False
    assert pipe.stats()["queue_len"] == 0 and pipe.stats()["send_queue_len"] == 0
    assert pipe.healthy() is True
    # 幂等
    assert pipe.stop() is True
    # 可再次启动
    pipe.start()
    wait_until(lambda: pipe.stats()["chunks"] >= 1)
    assert pipe.stop() is True
    assert threading.active_count() == baseline


def test_update_capture_settings_hot_and_column_without_background(tmp_path: Path):
    c = make_ctx(pipe_cfg(tmp_path))
    c.start()
    try:
        cam = c.cameras[1]
        cam.start_stream(CHUNK)
        try:
            frame = cam.get_frame(2.0)
            assert frame is not None
            pipe = Pipeline(c, cam, mode="sort")
            n_before = len(pipe.process_frame(frame).particles)
            assert n_before > 0
            huge = c.cfg.capture.model_copy(update={"min_area_px": 1_000_000})
            pipe.update_capture_settings(huge)
            pipe.stitcher.reset()
            assert len(pipe.process_frame(frame).particles) == 0
            # column 模式但没有背景 → 拒绝，旧参数保留
            column = c.cfg.capture.model_copy(update={"bg_mode": "column"})
            with pytest.raises(PipelineError):
                pipe.update_capture_settings(column)
            pipe.stitcher.reset()
            assert len(pipe.process_frame(frame).particles) == 0, "仍是 min_area 1e6 的参数"
            # 喷头映射热更新
            pipe.update_nozzle_map(NozzleMap(0, 512, 64), offset_x=0)
            assert pipe.nozzle_map.end_px == 512
        finally:
            cam.stop_stream()
    finally:
        c.stop()

    # column 模式、无背景文件：构造成功，start() 抛 PipelineError 并给出中文原因
    c2 = make_ctx(pipe_cfg(tmp_path / "col", **{"capture.bg_mode": "column"}))
    assert c2.backgrounds == {}
    pipe2 = Pipeline(c2, c2.cameras[1], mode="sort")
    with pytest.raises(PipelineError, match="背景"):
        pipe2.start()
    assert not pipe2.running
    c2.cameras[1].open()
    with pytest.raises(PipelineError):
        pipe2.process_frame(type("F", (), {})())  # 不会走到这里用它
    with pytest.raises(BackgroundUnavailable):
        pipe2.start()
    c2.cameras[1].close()


# --------------------------------------------------------------------------------------
# 行号时间轴 / 分类器故障 / 运行中拼块
# --------------------------------------------------------------------------------------


def test_start_keeps_line_counter_unless_reset_requested(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
):
    """docs/01 §5.5：行号零点只能与 SYNC 一起建立。Pipeline.start() 缺省不重置行号，reset_lines=True 才重置。"""
    cam = ctx.cameras[1]
    resets: list[int] = []
    orig_reset = cam.reset_line_counter
    monkeypatch.setattr(cam, "reset_line_counter", lambda: (resets.append(1), orig_reset()))
    pipe = Pipeline(ctx, cam, mode="sort")
    pipe.start()
    try:
        wait_until(lambda: pipe.stats()["chunks"] >= 3)
    finally:
        assert pipe.stop() is True
    assert resets == [], "缺省 start() 不得调用 reset_line_counter"
    line_before = pipe.latest_frame.line_start
    assert line_before >= 2 * CHUNK

    # 不重置地重新启动（采样后恢复分选流水线的路径）：行号续用
    pipe.start()
    try:
        first = pipe.wait_frame(None, 3.0)
        assert first is not None and first.line_start > line_before
        assert pipe.stats()["resets"] == 0, "新流首块 seq=0，但拼接器已 reset()，不算断块"
    finally:
        assert pipe.stop() is True
    assert resets == []

    # 显式要求重置：行号从 0 重新计
    pipe.start(reset_lines=True)
    try:
        first = pipe.wait_frame(None, 3.0)
        assert first is not None and first.line_start < line_before
    finally:
        assert pipe.stop() is True
    assert resets == [1]


class BrokenClassifier(FakeClassifier):
    """predict 永远抛异常（后端崩了 / 已关闭）。"""

    def predict(self, batch: np.ndarray):
        self.calls += 1
        raise RuntimeError("inference backend exploded")


def test_consecutive_predict_failures_raise_classifier_fault(ctx: AppContext):
    from oresorter.runtime.pipeline import CLASSIFIER_ERROR_FAULT, PREDICT_ERROR_FAULT_THRESHOLD

    cam = ctx.cameras[1]
    broken = BrokenClassifier()
    pipe = Pipeline(ctx, cam, mode="sort")
    pipe.replace_classifier(broken)  # type: ignore[arg-type]
    cam.start_stream(CHUNK)
    try:
        failed_blocks = 0
        for _ in range(4 * PREDICT_ERROR_FAULT_THRESHOLD):
            frame = cam.get_frame(2.0)
            assert frame is not None
            res = pipe.process_frame(frame)
            if res.particles:
                failed_blocks += 1
                assert res.eject == [] and all(
                    d.reason == D.REASON_NO_CLASSIFIER for d in res.decisions or []
                ), "推理失败的块按 no_classifier 不吹"
            if ctx.faults.has(CLASSIFIER_ERROR_FAULT):
                break
        assert ctx.faults.has(CLASSIFIER_ERROR_FAULT), "连续失败应置故障而不是静默不吹"
        assert failed_blocks >= PREDICT_ERROR_FAULT_THRESHOLD
        st = pipe.stats()
        assert st["consecutive_predict_errors"] >= PREDICT_ERROR_FAULT_THRESHOLD
        assert st["predict_errors"] == st["consecutive_predict_errors"]
        assert st["errors"] == 0, "推理失败不算流水线处理异常"
        wait_until(lambda: ctx.state_machine.state is RunState.fault, timeout=3)
        fault = next(f for f in ctx.faults.active() if f.code == CLASSIFIER_ERROR_FAULT)
        assert "分类器" in fault.detail and "相机 1" in fault.detail

        # 换上好的分类器：连续计数归零，下一块成功后故障清除、状态回到 stopped
        pipe.replace_classifier(FakeClassifier("C"))  # type: ignore[arg-type]
        assert pipe.stats()["consecutive_predict_errors"] == 0
        for _ in range(10):
            frame = cam.get_frame(2.0)
            assert frame is not None
            if pipe.process_frame(frame).particles:
                break
        assert not ctx.faults.has(CLASSIFIER_ERROR_FAULT)
        wait_until(lambda: ctx.state_machine.state is RunState.stopped, timeout=3)
    finally:
        cam.stop_stream()


def test_set_classifier_clears_classifier_fault(ctx: AppContext):
    """部署新模型（AppContext.set_classifier）是 CLASSIFIER_ERROR 的恢复路径。"""
    from oresorter.runtime.pipeline import CLASSIFIER_ERROR_FAULT

    ctx.faults.set(CLASSIFIER_ERROR_FAULT, "x")
    assert ctx.run_state == "fault"
    ctx.set_classifier(1, FakeClassifier("A"))  # type: ignore[arg-type]
    assert not ctx.faults.has(CLASSIFIER_ERROR_FAULT) and ctx.run_state == "stopped"
    assert ctx.deployed_model_id == 1
    ctx.state_store.update(deployed_model_id=None)
    ctx.classifier = None


def test_collect_lines_stacks_contiguous_blocks(ctx: AppContext):
    pipe = Pipeline(ctx, ctx.cameras[1], mode="sort")
    assert pipe.collect_lines(1000, 0.2) == [], "未运行：立即返回空"
    pipe.start()
    try:
        t0 = time.monotonic()
        frames = pipe.collect_lines(1000, 5.0)
        assert time.monotonic() - t0 < 3.0
        total = sum(f.lines for f in frames)
        assert total >= 1000 and len(frames) == 4, "256 行块 × 4 ≥ 1000 行"
        for a, b in itertools.pairwise(frames):
            assert b.line_start == a.line_end and b.seq == a.seq + 1, "块必须连续"
        img = np.concatenate([f.image for f in frames], axis=0)[:1000]
        assert img.shape == (1000, CAM_WIDTH, 3)
        # 不足一块：返回一块（调用方裁剪）
        one = pipe.collect_lines(10, 3.0)
        assert len(one) == 1 and one[0].lines == CHUNK
    finally:
        assert pipe.stop() is True
    assert pipe.collect_lines(1000, 0.2) == []


def test_pipeline_error_subclasses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from oresorter.camera.base import CameraError
    from oresorter.runtime.pipeline import CameraStreamError

    c = make_ctx(pipe_cfg(tmp_path))
    c.start()
    try:
        cam = c.cameras[1]
        pipe = Pipeline(c, cam, mode="sort")
        monkeypatch.setattr(
            cam, "start_stream", lambda n: (_ for _ in ()).throw(CameraError("SDK MV_E_RESOURCE"))
        )
        with pytest.raises(CameraStreamError) as ei:
            pipe.start()
        assert isinstance(ei.value, PipelineError) and ei.value.camera_id == 1
        assert "MV_E_RESOURCE" in str(ei.value) and not pipe.running
        assert c.faults.has(camera_fault_code(1, "DISCONNECTED"))
        monkeypatch.undo()
        wait_until(lambda: not c.faults.has(camera_fault_code(1, "DISCONNECTED")), timeout=4)
    finally:
        c.stop()
