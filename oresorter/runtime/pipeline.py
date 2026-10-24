"""运行时流水线（docs/01 §4.2、§7.3）：采集线程 → 环形队列 → 处理线程 → 发送线程。

- **acquire**：``camera.get_frame`` → 环形队列 ``deque(maxlen=queue_size)``（满则丢最旧并计 ``stats.on_dropped``）；
  连续 ``frame_timeout_s`` 无帧 → 置故障 ``CAMERA_{id}_TIMEOUT``（有帧后清除）；``CameraDisconnected`` →
  置故障 ``CAMERA_{id}_DISCONNECTED``，等待相机恢复后重新 ``start_stream`` 并让拼接器复位。
- **process**：``ChunkStitcher.process(frame, segment)`` → 颗粒 → ``sink.handle(frame, particles)``；
  sort 模式每块 ``stats.on_chunk``（capture 模式 **不写** ``ctx.stats``，采样不污染"自本次 start 以来"的
  分选累计）；处理异常只计数、记日志、继续下一块，连续 ≥ :data:`ERROR_FAULT_THRESHOLD` 次置故障
  ``PIPELINE_ERROR``（恢复后清除）；分类器连续 ≥ :data:`PREDICT_ERROR_FAULT_THRESHOLD` 块推理失败置故障
  ``CLASSIFIER_ERROR``（重新部署 / 替换分类器后清除），而不是静默不吹。
- **send**（sort 模式）：从发送队列取 ``EjectItem`` 批（一块一批）→ ``mcu.send_eject``，不等 ACK。

两种 ``ParticleSink``：

- :class:`ClassifySink`（分选）：预处理 → 分类器批量推理（按 ``infer_batch_max`` 分批）→ ``decide_batch`` →
  ``eject_items``；分类器可原子替换（``replace_classifier``），暂停时 ``set_blow_enabled(False)`` 只分类不吹；
- :class:`CaptureSink`（采样）：颗粒小图写入 ``SampleRepo``，每张回调 ``on_sample(sample_dict, count)``；
  达到上限（张数 / 帧数）或磁盘不足时置位 ``done``。

行号时间轴（docs/01 §5.5）：``start()`` **不会** 自行重置相机行号；SYNC + ``reset_line_counter`` 只在
``RuntimeService.start()``（stopped → running）一处完成，采样结束恢复分选流水线等场合以
``start(reset_lines=False)``（缺省）续用原行号，保证 ``EjectItem.line`` 与下位机编码器同一零点。

``process_frame(frame)`` 是不起线程的同步路径（push → segment → sink），供测试与 ``cli replay`` 复用。
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from oresorter.camera.base import Camera, CameraDisconnected, CameraError, Frame
from oresorter.config import BlowParams, CaptureSettings
from oresorter.data.samples import SampleRepo, StoreError
from oresorter.log import get_logger
from oresorter.mcu.link import EjectItem, McuError
from oresorter.runtime.decision import Decision, decide_batch, eject_items_counted
from oresorter.vision.background import ColumnBackground, background_path, make_background
from oresorter.vision.nozzle import NozzleMap
from oresorter.vision.preprocess import to_tensor_batch
from oresorter.vision.segment import Particle, foreground_mask, segment
from oresorter.vision.stitch import ChunkStitcher

if TYPE_CHECKING:
    from oresorter.ml.infer import Classifier
    from oresorter.runtime.context import AppContext

log = get_logger(__name__)

ERROR_FAULT_THRESHOLD = 20
"""处理线程连续异常多少次置 ``PIPELINE_ERROR`` 故障。"""
WARN_INTERVAL_S = 5.0
"""同类告警（延迟超限、发送失败、分类器不可用）的最小间隔。"""
BG_ADAPT_SAVE_EVERY_S = 60.0
"""背景自适应开启时，列背景多久原子保存一次。"""
SEND_QUEUE_MAX = 64
"""发送队列最多积压的批数（一块一批）；满则丢最旧。"""
ACQUIRE_POLL_S = 0.25
"""采集线程单次 ``get_frame`` 的最大等待，保证 stop() 及时。"""
STOP_JOIN_S = 2.0
PIPELINE_ERROR_FAULT = "PIPELINE_ERROR"
CLASSIFIER_ERROR_FAULT = "CLASSIFIER_ERROR"
"""分类器连续推理失败的故障码（重新部署 / 替换分类器后清除）。"""
PREDICT_ERROR_FAULT_THRESHOLD = 20
"""分类器连续多少块推理失败置 ``CLASSIFIER_ERROR`` 故障。"""


class PipelineError(RuntimeError):
    """流水线无法启动/处理的基类。"""


class BackgroundUnavailable(PipelineError):
    """column 模式列背景未学习（``start`` 的 409 原因 ``BACKGROUND_NOT_LEARNED``）。"""


class CameraStreamError(PipelineError):
    """相机无法开始采集（``start`` 的 409 原因 ``CAMERA_DISCONNECTED``）。"""

    def __init__(self, camera_id: int, message: str):
        super().__init__(message)
        self.camera_id = int(camera_id)


def camera_fault_code(camera_id: int, kind: str) -> str:
    """``CAMERA_{id}_DISCONNECTED`` / ``CAMERA_{id}_TIMEOUT``。"""
    return f"CAMERA_{int(camera_id)}_{kind}"


class _RateLimitedLog:
    """同一 key 的告警每 ``interval`` 秒至多一条。"""

    def __init__(self, interval: float = WARN_INTERVAL_S):
        self._interval = interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self._interval:
                return False
            self._last[key] = now
            return True


# --------------------------------------------------------------------------------------
# Sink
# --------------------------------------------------------------------------------------


@dataclass
class SinkResult:
    """一块的 sink 处理结果（统计用）。"""

    n_particles: int = 0
    blown: int = 0
    """决定吹气的颗粒数（暂停/禁吹时为 0）。"""
    per_class: dict[str, int] = field(default_factory=dict)
    eject: list[EjectItem] = field(default_factory=list)
    skipped: int = 0
    """决定吹但映射不到喷头（落在起止像素外）而跳过的数量。"""
    decisions: list[Decision] | None = None
    samples_added: int = 0


class ParticleSink(Protocol):
    """处理线程每块调用一次。"""

    def handle(self, frame: Frame, particles: list[Particle]) -> SinkResult: ...


@dataclass
class ChunkResult:
    """``process_frame`` 的返回：块元数据 + sink 结果 + 延迟。"""

    seq: int
    line_start: int
    particles: list[Particle]
    result: SinkResult
    latency_ms: float

    @property
    def eject(self) -> list[EjectItem]:
        return self.result.eject

    @property
    def decisions(self) -> list[Decision] | None:
        return self.result.decisions


class ClassifySink:
    """分选 sink：分类 → 决策 → 喷阀条目。全部可变状态由一把锁保护，可在运行中热更新。

    Args:
        classifier: 已部署的分类器；None 时全部颗粒 ``reason=no_classifier``、不吹。
        nozzle_map: 该相机的喷头映射。
        blow: 吹气参数。
        offset_x: 相机 ``offset_x``（图像列号 + offset_x = 传感器坐标）。
        expand: 喷头区间向两侧各扩展个数。
        max_area_px: 粒度归一化面积上限（``capture.max_area_px``）。
        infer_batch_max: 单次送入分类器的最大颗粒数（预处理张量按此分批，限制内存峰值）。
        blow_truncated: 截断颗粒是否允许吹。
    """

    def __init__(
        self,
        classifier: Classifier | None,
        nozzle_map: NozzleMap,
        *,
        blow: BlowParams,
        offset_x: int,
        expand: int = 0,
        max_area_px: int,
        infer_batch_max: int = 128,
        blow_truncated: bool = False,
    ):
        if infer_batch_max < 1:
            raise ValueError(f"infer_batch_max 必须 ≥ 1，得到 {infer_batch_max}")
        if max_area_px < 1:
            raise ValueError(f"max_area_px 必须 ≥ 1，得到 {max_area_px}")
        self._lock = threading.RLock()
        self._classifier = classifier
        self._nozzle_map = nozzle_map
        self._blow = blow
        self._offset_x = int(offset_x)
        self._expand = int(expand)
        self._max_area_px = int(max_area_px)
        self._infer_batch_max = int(infer_batch_max)
        self._blow_truncated = bool(blow_truncated)
        self._blow_enabled = True
        self._warn = _RateLimitedLog()
        self.predict_errors = 0
        """累计推理失败块数。"""
        self.consecutive_predict_errors = 0
        """连续推理失败块数（成功一块即归零；替换分类器时归零）。"""

    # ---- 热更新 ----
    def set_classifier(self, classifier: Classifier | None) -> Classifier | None:
        """原子替换分类器，返回旧实例（由调用方决定是否 close）；连续失败计数归零。"""
        with self._lock:
            old, self._classifier = self._classifier, classifier
            self.consecutive_predict_errors = 0
            return old

    @property
    def classifier(self) -> Classifier | None:
        with self._lock:
            return self._classifier

    def set_blow(self, blow: BlowParams) -> None:
        with self._lock:
            self._blow = blow

    def set_nozzle_map(self, nozzle_map: NozzleMap, *, offset_x: int, expand: int | None = None):
        with self._lock:
            self._nozzle_map = nozzle_map
            self._offset_x = int(offset_x)
            if expand is not None:
                self._expand = int(expand)

    def set_max_area_px(self, max_area_px: int) -> None:
        if max_area_px < 1:
            raise ValueError(f"max_area_px 必须 ≥ 1，得到 {max_area_px}")
        with self._lock:
            self._max_area_px = int(max_area_px)

    def set_blow_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._blow_enabled = bool(enabled)

    @property
    def blow_enabled(self) -> bool:
        with self._lock:
            return self._blow_enabled

    # ---- 处理 ----
    def _predict(self, clf: Classifier, particles: list[Particle]) -> tuple[np.ndarray, list[str]]:
        size = int(clf.image_size)
        chunks: list[np.ndarray] = []
        classes: list[str] = []
        for i in range(0, len(particles), self._infer_batch_max):
            crops = [p.crop_rgba for p in particles[i : i + self._infer_batch_max]]
            probs, classes = clf.predict(to_tensor_batch(crops, size))
            chunks.append(np.asarray(probs, np.float32))
        return (np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]), classes

    def handle(self, frame: Frame, particles: list[Particle]) -> SinkResult:
        """分类 + 决策 + 喷阀映射；分类器不可用（正在切换/已关闭）时按 ``no_classifier`` 处理本块。"""
        with self._lock:
            clf = self._classifier
            blow = self._blow
            nozzle_map = self._nozzle_map
            offset_x, expand = self._offset_x, self._expand
            max_area_px = self._max_area_px
            enabled = self._blow_enabled
            blow_truncated = self._blow_truncated
        n = len(particles)
        if n == 0:
            return SinkResult(decisions=[])
        decisions: list[Decision]
        if clf is None:
            decisions = [Decision.no_classifier() for _ in particles]
        else:
            try:
                probs, classes = self._predict(clf, particles)
                decisions = decide_batch(
                    particles,
                    probs,
                    classes,
                    blow,
                    max_area_px=max_area_px,
                    blow_truncated=blow_truncated,
                )
                self.consecutive_predict_errors = 0
            except Exception as exc:
                # 分类器正在切换/已关闭（ClassifierUnavailable）或后端异常：本块不吹，不让流水线失败；
                # 连续失败由 Pipeline 计入 CLASSIFIER_ERROR 故障
                self.predict_errors += 1
                self.consecutive_predict_errors += 1
                if self._warn.allow("predict"):
                    log.warning(
                        "分类器推理失败，本块按未部署处理",
                        extra={"camera": frame.camera_id, "seq": frame.seq, "err": str(exc)},
                    )
                decisions = [Decision.no_classifier() for _ in particles]
        per_class: Counter[str] = Counter()
        for p, d in zip(particles, decisions, strict=True):
            if d.cls:
                per_class[d.cls] += 1
            p.extra["cls"] = d.cls
            p.extra["conf"] = round(float(d.conf), 4)
            p.extra["blow"] = bool(d.blow and enabled)
            p.extra["reason"] = d.reason
        if not enabled:
            return SinkResult(n_particles=n, per_class=dict(per_class), decisions=decisions)
        items, skipped = eject_items_counted(
            particles, decisions, nozzle_map, offset_x=offset_x, expand=expand
        )
        blown = sum(1 for d in decisions if d.blow)
        return SinkResult(
            n_particles=n,
            blown=blown,
            per_class=dict(per_class),
            eject=items,
            skipped=skipped,
            decisions=decisions,
        )


OnSample = Callable[[dict[str, Any], int], None]


class CaptureSink:
    """采样 sink：把颗粒小图写入样本库。

    Args:
        repo: 目标模型的样本库。
        cls: 类别名。
        camera_id: 相机编号（写入样本元数据）。
        max_samples: 最多写入张数，达到后 ``done`` 置位（``stop_reason="max_samples"``）。
        max_frames: 最多处理块数（None 不限）。
        batch: 采集批次号（通常是任务 id）。
        on_sample: 每写入一张回调 ``(sample_dict, count)``。
        skip_truncated: 跳过被截断的颗粒（半颗矿石不进样本库）。
    """

    def __init__(
        self,
        repo: SampleRepo,
        cls: str,
        camera_id: int,
        *,
        max_samples: int,
        max_frames: int | None = None,
        batch: str | None = None,
        on_sample: OnSample | None = None,
        skip_truncated: bool = True,
    ):
        if max_samples < 1:
            raise ValueError(f"max_samples 必须 ≥ 1，得到 {max_samples}")
        if max_frames is not None and max_frames < 1:
            raise ValueError(f"max_frames 必须 ≥ 1，得到 {max_frames}")
        self.repo = repo
        self.cls = cls
        self.camera_id = int(camera_id)
        self.max_samples = int(max_samples)
        self.max_frames = int(max_frames) if max_frames is not None else None
        self.batch = batch
        self.on_sample = on_sample
        self.skip_truncated = bool(skip_truncated)
        self.done = threading.Event()
        self.stop_reason: str | None = None
        self.error: str | None = None
        self.count = 0
        self.frames = 0
        self.errors = 0
        self._lock = threading.Lock()

    def _finish(self, reason: str, error: str | None = None) -> None:
        if not self.done.is_set():
            self.stop_reason = reason
            self.error = error
            self.done.set()

    def handle(self, frame: Frame, particles: list[Particle]) -> SinkResult:
        """写入本块颗粒（直到达到上限）。磁盘不足 / 文件系统错误 → 结束采样并记录原因。"""
        with self._lock:
            if self.done.is_set():
                return SinkResult(n_particles=len(particles))
            self.frames += 1
            added = 0
            for p in particles:
                if self.count >= self.max_samples:
                    break
                if p.truncated and self.skip_truncated:
                    continue
                meta: dict[str, Any] = {
                    "camera_id": self.camera_id,
                    "bbox": [int(v) for v in p.bbox],
                    "area_px": int(p.area),
                    "trailing_line": int(p.trailing_line),
                    "seq": int(frame.seq),
                    "line_start": int(frame.line_start),
                    "batch": self.batch,
                }
                try:
                    sample = self.repo.add(self.cls, p.crop_rgba, meta)
                except StoreError as exc:
                    if exc.code in ("CONFLICT", "INTERNAL"):
                        log.error(
                            "采样终止：样本写入失败",
                            extra={"cls": self.cls, "code": exc.code, "err": exc.message},
                        )
                        self._finish("store_error", f"{exc.code}: {exc.message}")
                        break
                    self.errors += 1
                    log.warning("样本写入被拒绝，跳过", extra={"cls": self.cls, "err": exc.message})
                    continue
                self.count += 1
                added += 1
                if self.on_sample is not None:
                    try:
                        self.on_sample(sample.to_dict(), self.count)
                    except Exception:
                        log.exception("on_sample 回调异常")
            if self.count >= self.max_samples:
                self._finish("max_samples")
            elif self.max_frames is not None and self.frames >= self.max_frames:
                self._finish("max_frames")
            return SinkResult(n_particles=len(particles), samples_added=added)


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class Pipeline:
    """单台相机的三线程流水线。

    Args:
        ctx: 应用上下文（配置、故障表、统计、下位机、背景、分类器）。
        camera: 该流水线独占的相机。
        mode: ``"sort"``（分类 + 喷阀）或 ``"capture"``（采样，必须给 ``sink``）。
        sink: 自定义 sink；sort 模式缺省为 :class:`ClassifySink`。
        camera_index: 相机在配置列表中的序号（全局喷头编号用）；None 时自动查找。

    column 模式而列背景未学习时，构造不抛，``start()`` / ``process_frame()`` 抛 :class:`PipelineError`。
    """

    def __init__(
        self,
        ctx: AppContext,
        camera: Camera,
        *,
        mode: str = "sort",
        sink: ParticleSink | None = None,
        camera_index: int | None = None,
    ):
        if mode not in ("sort", "capture"):
            raise ValueError(f"mode 必须是 sort / capture，得到 {mode!r}")
        self.ctx = ctx
        self.camera = camera
        self.mode = mode
        cfg = ctx.cfg
        self.cam_cfg = cfg.camera(camera.id)
        ids = [c.id for c in cfg.cameras]
        self.camera_index = int(camera_index) if camera_index is not None else ids.index(camera.id)
        pcfg = cfg.pipeline
        self.frame_timeout_s = float(pcfg.frame_timeout_s)
        self.latency_warn_ms = float(pcfg.latency_warn_ms)
        self.queue_size = int(pcfg.queue_size)
        self.chunk_lines = int(self.cam_cfg.runtime_chunk_lines)
        self.stitcher = ChunkStitcher(pcfg.overlap_lines)

        self._seg_lock = threading.RLock()
        self._settings: CaptureSettings = cfg.capture
        self._bg: ColumnBackground | Any = None
        self._bg_error: str | None = None
        self._resolve_background(cfg.capture)
        self._last_bg_save = time.monotonic()

        self.nozzle_map = NozzleMap.from_camera(self.cam_cfg, self.camera_index)
        if sink is None:
            if mode == "capture":
                raise ValueError("capture 模式必须提供 CaptureSink")
            sink = ClassifySink(
                ctx.classifier,
                self.nozzle_map,
                blow=cfg.blow,
                offset_x=self.cam_cfg.offset_x,
                expand=pcfg.nozzle_expand,
                max_area_px=cfg.capture.max_area_px,
                infer_batch_max=pcfg.infer_batch_max,
            )
        self.sink: ParticleSink = sink

        # 线程与队列
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._q_cond = threading.Condition()
        self._queue: deque[Frame] = deque(maxlen=self.queue_size)
        self._send_cond = threading.Condition()
        self._send_queue: deque[list[EjectItem]] = deque(maxlen=SEND_QUEUE_MAX)
        self._latest_cond = threading.Condition()
        self._latest: Frame | None = None
        self._reset_pending = False
        self._warn = _RateLimitedLog()

        # 计数（原生 int 的读写在 GIL 下足够；只用于统计展示）
        self._queue_dropped = 0
        self._send_dropped = 0
        self._chunks = 0
        self._errors = 0
        self._consecutive_errors = 0
        self._eject_sent = 0
        self._eject_failed = 0
        self._eject_skipped = 0
        self._blown_total = 0
        self._last_latency_ms = 0.0
        self._last_received_t: float | None = None
        self._last_processed_t: float | None = None
        self._timeout_fault = False
        self._error_fault = False
        self._classifier_fault = False

    # ---- 背景 ----
    def _resolve_background(self, settings: CaptureSettings) -> None:
        try:
            self._bg = make_background(settings, self.ctx.backgrounds.get(self.camera.id))
            self._bg_error = None
        except ValueError as exc:
            self._bg = None
            self._bg_error = f"{exc}（请先用 learn-bg 学习相机 {self.camera.id} 的背景）"

    def _require_background(self) -> tuple[CaptureSettings, Any]:
        with self._seg_lock:
            if self._bg is None:
                raise BackgroundUnavailable(self._bg_error or "背景不可用")
            return self._settings, self._bg

    def update_capture_settings(self, settings: CaptureSettings) -> None:
        """热更新分割参数（PUT /params/capture）。column 模式背景缺失时抛 :class:`BackgroundUnavailable`，旧参数保留。"""
        with self._seg_lock:
            try:
                bg = make_background(settings, self.ctx.backgrounds.get(self.camera.id))
            except ValueError as exc:
                raise BackgroundUnavailable(str(exc)) from exc
            self._settings = settings
            self._bg = bg
            self._bg_error = None
        if isinstance(self.sink, ClassifySink):
            self.sink.set_max_area_px(settings.max_area_px)

    # ---- 热更新（sort）----
    def replace_classifier(
        self, classifier: Classifier | None, *, close_old: bool = False
    ) -> Classifier | None:
        """原子切换分类器；返回旧实例。``close_old=True`` 时顺手关闭旧实例（等在飞的推理结束）。"""
        if not isinstance(self.sink, ClassifySink):
            return None
        old = self.sink.set_classifier(classifier)
        if close_old and old is not None and old is not classifier:
            try:
                old.close()
            except Exception:
                log.exception("关闭旧分类器失败")
        return old

    def set_blow_enabled(self, enabled: bool) -> None:
        """暂停时关吹（继续分类与统计），恢复时开吹。capture 模式无操作。"""
        if isinstance(self.sink, ClassifySink):
            self.sink.set_blow_enabled(enabled)

    @property
    def blow_enabled(self) -> bool:
        return self.sink.blow_enabled if isinstance(self.sink, ClassifySink) else False

    def update_blow(self, blow: BlowParams) -> None:
        if isinstance(self.sink, ClassifySink):
            self.sink.set_blow(blow)

    def update_nozzle_map(
        self, nozzle_map: NozzleMap, *, offset_x: int, expand: int | None = None
    ) -> None:
        """起始/结束像素或 offset_x 变化后热更新喷头映射。"""
        self.nozzle_map = nozzle_map
        if isinstance(self.sink, ClassifySink):
            self.sink.set_nozzle_map(nozzle_map, offset_x=offset_x, expand=expand)

    # ---- 生命周期 ----
    @property
    def running(self) -> bool:
        return self._running

    def start(self, *, reset_lines: bool = False) -> None:
        """``start_stream`` → 启动线程。重复调用无操作。

        Args:
            reset_lines: True 时先 ``camera.reset_line_counter()``。**缺省不重置**：行号零点必须与下位机
                SYNC 同时建立（docs/01 §5.5），由 ``RuntimeService.start()`` 负责；采样结束恢复分选流水线
                等场合续用原行号。

        Raises:
            BackgroundUnavailable: column 模式列背景未学习。
            CameraStreamError: 相机无法开始采集（同时置故障 ``CAMERA_{id}_DISCONNECTED``）。
        """
        if self._running:
            return
        self._require_background()
        try:
            if reset_lines:
                self.camera.reset_line_counter()
            self.camera.start_stream(self.chunk_lines)
        except CameraError as exc:
            self.ctx.faults.set(camera_fault_code(self.camera.id, "DISCONNECTED"), str(exc))
            raise CameraStreamError(
                self.camera.id, f"相机 {self.camera.id} 无法开始采集: {exc}"
            ) from exc
        self._stop.clear()
        self._reset_pending = False
        self.stitcher.reset()
        with self._q_cond:
            self._queue.clear()
        with self._send_cond:
            self._send_queue.clear()
        self._last_received_t = None
        self._last_processed_t = None
        self._consecutive_errors = 0
        prefix = f"pipe{self.camera.id}"
        self._threads = [
            threading.Thread(target=self._acquire_loop, name=f"{prefix}-acquire", daemon=True),
            threading.Thread(target=self._process_loop, name=f"{prefix}-process", daemon=True),
        ]
        if self.mode == "sort":
            self._threads.append(
                threading.Thread(target=self._send_loop, name=f"{prefix}-send", daemon=True)
            )
        self._running = True
        for t in self._threads:
            t.start()
        log.info(
            "流水线启动",
            extra={"camera": self.camera.id, "mode": self.mode, "chunk_lines": self.chunk_lines},
        )

    def stop(self, timeout: float = STOP_JOIN_S) -> bool:
        """停止线程、停止相机流、清空队列；返回是否全部线程在 ``timeout`` 内退出。可重复调用。"""
        was_running = self._running
        self._running = False
        self._stop.set()
        with self._q_cond:
            self._q_cond.notify_all()
        with self._send_cond:
            self._send_cond.notify_all()
        with self._latest_cond:
            self._latest_cond.notify_all()
        try:
            self.camera.stop_stream()
        except Exception:
            log.exception("停止相机流异常", extra={"camera": self.camera.id})
        me = threading.current_thread()
        deadline = time.monotonic() + max(0.0, timeout)
        threads, self._threads = self._threads, []
        for t in threads:
            if t is me:
                continue
            t.join(max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive() and t is not me]
        with self._q_cond:
            self._queue.clear()
        with self._send_cond:
            self._send_queue.clear()
        if alive:
            log.error("流水线线程未能在时限内退出", extra={"threads": alive})
        if was_running:
            log.info("流水线停止", extra={"camera": self.camera.id, "mode": self.mode})
        return not alive

    # ---- 预览 ----
    @property
    def latest_frame(self) -> Frame | None:
        """最近收到的一块（采集线程写入），供预览/快照。"""
        with self._latest_cond:
            return self._latest

    def wait_frame(self, after: Frame | None, timeout: float) -> Frame | None:
        """等待一块与 ``after`` 不同的新帧；超时或流水线停止返回 None。"""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._latest_cond:
            while True:
                cur = self._latest
                if cur is not None and cur is not after:
                    return cur
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._latest_cond.wait(remaining)

    def _set_latest(self, frame: Frame) -> None:
        with self._latest_cond:
            self._latest = frame
            self._latest_cond.notify_all()

    def collect_lines(self, lines: int, timeout: float) -> list[Frame]:
        """从运行中的流水线收集 **行号连续** 的块，直到累计 ≥ ``lines`` 行或超时（运行中快照用）。

        从最近一块开始累计，之后等待新块；遇到断块（``line_start`` 不接 / 宽度变化）丢弃已累计的重新开始。
        返回已收集到的块（超时 / 流水线未运行时可能不足 ``lines`` 行，甚至为空），调用方自行拼接与裁剪。
        """
        lines = max(1, int(lines))
        deadline = time.monotonic() + max(0.0, float(timeout))
        frames: list[Frame] = []
        total = 0
        last: Frame | None = None
        while total < lines and self._running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            cur = self.latest_frame if last is None else None
            f = cur if cur is not None else self.wait_frame(last, remaining)
            if f is None:
                break
            last = f
            if frames and (f.line_start != frames[-1].line_end or f.width != frames[-1].width):
                frames, total = [], 0
            frames.append(f)
            total += f.lines
        return frames

    # ---- 同步处理 ----
    def process_frame(self, frame: Frame) -> ChunkResult:
        """同步处理一块：拼接 → 分割 → sink → 统计；sort 模式的喷阀条目在发送线程不在跑时直接下发。

        只有 sort 模式写 ``ctx.stats``（docs/03 §3 的"自本次 start 以来"累计）；capture 模式的计数见 ``stats()``。

        Raises:
            BackgroundUnavailable: 背景不可用。
            其它异常原样抛出（线程路径会捕获并计数）。
        """
        settings, bg = self._require_background()
        adapt = isinstance(bg, ColumnBackground) and settings.bg_adapt_rate > 0

        def seg(image: np.ndarray) -> list[Particle]:
            parts = segment(image, bg, settings)
            if adapt:
                self._adapt_background(image, bg, settings)
            return parts

        particles = self.stitcher.process(frame, seg)
        res = self.sink.handle(frame, particles)
        now = time.monotonic()
        latency_ms = max(0.0, (now - float(frame.ts)) * 1000.0)
        if self.mode == "sort":
            self.ctx.stats.on_chunk(latency_ms, res.n_particles, res.blown, res.per_class or None)
        if isinstance(self.sink, ClassifySink):
            self._update_classifier_fault(self.sink.consecutive_predict_errors)
        self._eject_skipped += res.skipped
        self._blown_total += res.blown
        if res.eject:
            self._dispatch_eject(res.eject)
        self._chunks += 1
        self._last_latency_ms = latency_ms
        self._last_processed_t = now
        if latency_ms > self.latency_warn_ms and self._warn.allow("latency"):
            log.warning(
                "块处理延迟超限",
                extra={
                    "camera": self.camera.id,
                    "seq": frame.seq,
                    "latency_ms": round(latency_ms, 1),
                    "warn_ms": self.latency_warn_ms,
                    "queue_len": len(self._queue),
                },
            )
        return ChunkResult(
            seq=int(frame.seq),
            line_start=int(frame.line_start),
            particles=particles,
            result=res,
            latency_ms=latency_ms,
        )

    def _update_classifier_fault(self, consecutive_errors: int) -> None:
        """分类器连续 ≥ 阈值块推理失败 → 置 ``CLASSIFIER_ERROR``（进入 fault 停机）；恢复成功一块即清除。"""
        if consecutive_errors >= PREDICT_ERROR_FAULT_THRESHOLD:
            if not self._classifier_fault:
                self._classifier_fault = True
                self.ctx.faults.set(
                    CLASSIFIER_ERROR_FAULT,
                    f"相机 {self.camera.id} 分类器连续 {consecutive_errors} 块推理失败，已停止分选",
                )
        elif consecutive_errors == 0 and self._classifier_fault:
            self._classifier_fault = False
            self.ctx.faults.clear(CLASSIFIER_ERROR_FAULT)

    def _adapt_background(
        self, image: np.ndarray, bg: ColumnBackground, settings: CaptureSettings
    ) -> None:
        try:
            mask = foreground_mask(image, bg, settings)
            bg.adapt(image, mask, float(settings.bg_adapt_rate))
        except ValueError as exc:
            if self._warn.allow("adapt"):
                log.warning("背景自适应失败", extra={"camera": self.camera.id, "err": str(exc)})
            return
        now = time.monotonic()
        if now - self._last_bg_save >= BG_ADAPT_SAVE_EVERY_S:
            self._last_bg_save = now
            try:
                bg.save(background_path(self.ctx.data_dir, self.camera.id))
                self.ctx.state_store.update(bg_version=self.ctx.state_store.state.bg_version + 1)
            except Exception:
                log.exception("保存自适应背景失败", extra={"camera": self.camera.id})

    # ---- 发送 ----
    def _dispatch_eject(self, items: list[EjectItem]) -> None:
        if self._running and self.mode == "sort":
            with self._send_cond:
                if len(self._send_queue) >= SEND_QUEUE_MAX:
                    self._send_queue.popleft()
                    self._send_dropped += 1
                self._send_queue.append(list(items))
                self._send_cond.notify()
        else:
            self._send_batch(items)

    def _send_batch(self, items: list[EjectItem]) -> None:
        try:
            self.ctx.mcu.send_eject(items)
            self._eject_sent += len(items)
        except (McuError, OSError) as exc:
            self._eject_failed += len(items)
            if self._warn.allow("send"):
                log.warning("喷阀批发送失败", extra={"n": len(items), "err": str(exc)})
        except Exception:
            self._eject_failed += len(items)
            log.exception("喷阀批发送异常")

    # ---- 线程 ----
    def _acquire_loop(self) -> None:
        cam = self.camera
        poll = min(ACQUIRE_POLL_S, self.frame_timeout_s)
        last_t = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = cam.get_frame(poll)
            except CameraDisconnected as exc:
                self._handle_disconnect(str(exc))
                last_t = time.monotonic()
                continue
            except Exception as exc:
                self._errors += 1
                if self._warn.allow("acquire"):
                    log.exception("采集异常", extra={"camera": cam.id, "err": str(exc)})
                self._stop.wait(0.1)
                continue
            now = time.monotonic()
            if frame is None:
                if (
                    not self._timeout_fault
                    and now - last_t > self.frame_timeout_s
                    and cam.connected
                ):
                    self._timeout_fault = True
                    self.ctx.faults.set(
                        camera_fault_code(cam.id, "TIMEOUT"),
                        f"相机 {cam.id} 超过 {self.frame_timeout_s:.1f}s 无帧",
                    )
                continue
            last_t = now
            self._last_received_t = now
            if self._timeout_fault:
                self._timeout_fault = False
                self.ctx.faults.clear(camera_fault_code(cam.id, "TIMEOUT"))
            self._set_latest(frame)
            with self._q_cond:
                if len(self._queue) >= self.queue_size:
                    self._queue.popleft()
                    self._queue_dropped += 1
                    if self.mode == "sort":
                        self.ctx.stats.on_dropped(1)
                self._queue.append(frame)
                self._q_cond.notify()

    def _handle_disconnect(self, reason: str) -> None:
        """相机掉线：置故障，等待恢复后重新开流并让拼接器复位。"""
        cam = self.camera
        code = camera_fault_code(cam.id, "DISCONNECTED")
        self.ctx.faults.set(code, reason)
        self._reset_pending = True
        with self._q_cond:
            self._queue.clear()
        while not self._stop.is_set():
            if cam.connected:
                try:
                    cam.start_stream(self.chunk_lines)
                except CameraError as exc:
                    log.warning(
                        "相机恢复后开流失败，重试", extra={"camera": cam.id, "err": str(exc)}
                    )
                    self._stop.wait(0.5)
                    continue
                self.ctx.faults.clear(code)
                log.info("相机恢复，重新开始采集", extra={"camera": cam.id})
                return
            self._stop.wait(0.1)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            with self._q_cond:
                while not self._queue and not self._stop.is_set():
                    self._q_cond.wait(0.2)
                if self._stop.is_set():
                    return
                frame = self._queue.popleft()
            if self._reset_pending:
                self._reset_pending = False
                self.stitcher.reset()
            try:
                self.process_frame(frame)
            except Exception as exc:
                self._errors += 1
                self._consecutive_errors += 1
                if self._warn.allow("process"):
                    log.exception(
                        "块处理异常，跳过该块",
                        extra={"camera": self.camera.id, "seq": frame.seq, "err": str(exc)},
                    )
                if self._consecutive_errors >= ERROR_FAULT_THRESHOLD and not self._error_fault:
                    self._error_fault = True
                    self.ctx.faults.set(
                        PIPELINE_ERROR_FAULT,
                        f"相机 {self.camera.id} 流水线连续 {self._consecutive_errors} 块处理失败: {exc}",
                    )
                continue
            self._consecutive_errors = 0
            if self._error_fault:
                self._error_fault = False
                self.ctx.faults.clear(PIPELINE_ERROR_FAULT)

    def _send_loop(self) -> None:
        while not self._stop.is_set():
            with self._send_cond:
                while not self._send_queue and not self._stop.is_set():
                    self._send_cond.wait(0.2)
                if self._stop.is_set():
                    return
                items = self._send_queue.popleft()
            self._send_batch(items)

    # ---- 健康与统计 ----
    def healthy(self) -> bool:
        """喂狗用的健康判定：未运行视为健康；运行中要求线程都活着，且处理线程没有"收到块却长时间不处理"。

        相机断线 / 无帧（设备问题，由重连与故障表反映）不算流水线不健康；判定窗口
        ``max(2 × frame_timeout_s, 5)`` 秒。
        """
        if not self._running:
            return True
        if any(not t.is_alive() for t in self._threads):
            return False
        recv, done = self._last_received_t, self._last_processed_t
        if recv is None:
            return True
        window = max(2.0 * self.frame_timeout_s, 5.0)
        if done is None:
            return (time.monotonic() - recv) < window
        # 收到块的时刻远晚于最近一次处理完成 → 处理线程卡住
        return (recv - done) < window

    def stats(self) -> dict[str, Any]:
        """拼接器统计 + 队列长度 + 处理/发送计数（全部原生类型）。"""
        d: dict[str, Any] = dict(self.stitcher.stats())
        with self._q_cond:
            qlen = len(self._queue)
        with self._send_cond:
            slen = len(self._send_queue)
        d.update(
            {
                "camera_id": self.camera.id,
                "mode": self.mode,
                "running": self._running,
                "queue_len": qlen,
                "queue_dropped": self._queue_dropped,
                "send_queue_len": slen,
                "send_dropped": self._send_dropped,
                "chunks": self._chunks,
                "errors": self._errors,
                "consecutive_errors": self._consecutive_errors,
                "blown": self._blown_total,
                "eject_sent": self._eject_sent,
                "eject_failed": self._eject_failed,
                "eject_skipped": self._eject_skipped,
                "last_latency_ms": round(self._last_latency_ms, 2),
                "blow_enabled": self.blow_enabled,
            }
        )
        if isinstance(self.sink, ClassifySink):
            d["predict_errors"] = self.sink.predict_errors
            d["consecutive_predict_errors"] = self.sink.consecutive_predict_errors
        if isinstance(self.sink, CaptureSink):
            d["samples"] = self.sink.count
        return d

    def __repr__(self) -> str:  # pragma: no cover
        return f"Pipeline(camera={self.camera.id}, mode={self.mode}, running={self._running})"
