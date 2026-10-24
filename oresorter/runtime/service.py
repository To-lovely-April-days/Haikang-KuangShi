"""``RuntimeService``：API 层唯一调用的业务门面（docs/03 全部接口的服务端实现）。

- 全部方法线程安全：运行控制（start/pause/stop/采样启停/故障反应）串行化在 ``_lock`` 下；
- 返回的 dict 即 docs/03 的响应体（字段名一致）；
- 错误用各层已有异常（``ConfigError`` / ``StoreError`` / ``TrainError`` / ``IllegalTransition`` /
  ``JobConflict`` / ``McuNack`` / ``McuTimeout`` / ``McuDisconnected``），设备离线等用
  ``oresorter.api.errors.ApiError``（API 层已注册 handler）。

流水线按相机各一条（多相机时 ``ctx.pipelines``），``ctx.pipeline`` 指向第一条；采样期间
``ctx.pipeline`` 是 capture 流水线（预览 / 快照共用其帧）。

行号时间轴（docs/01 §5.5、docs/04 §8）：SYNC（下位机编码器清零）与 ``camera.reset_line_counter()`` 只在
``start()`` 的 stopped → running 路径一处同时完成；恢复/重建流水线（采样后回到 paused、paused → running）
一律不重置行号也不再 SYNC，保证 ``EjectItem.line`` 与编码器同一零点。

``start()`` 的 409 原因：``check_start_preconditions`` 的枚举 + ``SAMPLING_IN_PROGRESS`` / ``IN_FAULT``，
以及本模块新增的 ``BACKGROUND_NOT_LEARNED``（column 模式列背景未学习，在 SYNC 之前预检）；相机开流失败
归入 ``CAMERA_DISCONNECTED``（同时置故障 ``CAMERA_{id}_DISCONNECTED``）。
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pydantic import ValidationError

from oresorter.api.errors import ApiError
from oresorter.camera.base import (
    Camera,
    CameraDisconnected,
    CameraError,
    CameraParams,
    Frame,
)
from oresorter.config import (
    MAX_SENSOR_PX,
    MCU_SECTIONS,
    AppConfig,
    CaptureSettings,
    ConfigError,
    camera_limits,
    deep_merge,
    parse_config,
    validation_error_to_config_error,
)
from oresorter.data.samples import StoreError, parse_sample_id
from oresorter.fsutil import atomic_write_bytes, disk_free_mb, ensure_dir, read_json
from oresorter.log import get_logger
from oresorter.mcu import protocol as P
from oresorter.mcu.link import McuDisconnected, McuError, McuNack, McuTimeout
from oresorter.ml.train import TrainError
from oresorter.runtime import state as S
from oresorter.runtime.jobs import JobConflict, JobType
from oresorter.runtime.pipeline import (
    BackgroundUnavailable,
    CameraStreamError,
    Pipeline,
    PipelineError,
)
from oresorter.runtime.state import Event, IllegalTransition, RunState
from oresorter.runtime.tasks import (
    backup_task,
    capture_task,
    deploy_task,
    train_task,
    valve_test_cycle_task,
)
from oresorter.vision.background import make_background
from oresorter.vision.nozzle import NozzleMap
from oresorter.vision.segment import draw_overlay, segment

if TYPE_CHECKING:
    from oresorter.runtime.context import AppContext

log = get_logger(__name__)

HMI_PARAM_SECTIONS: tuple[str, ...] = ("valve", "clean", "feed", "outputs", "blow", "capture")
"""PUT /params/{section} 可见的六个分组。"""
POWER_ENV = "ORESORTER_ALLOW_POWER"
"""置为 ``1`` 时 shutdown/reboot 真正执行 systemctl；缺省只记日志。"""
POWER_ACTION_DELAY_S = 2.0
SNAPSHOT_JPEG_QUALITY = 85
PREVIEW_MAX_WIDTH = 2000
CAPTURE_WAIT_CANCEL_S = 3.0
VALVE_TEST_WAIT_CANCEL_S = 2.0
SNAPSHOT_COLLECT_MIN_S = 1.0
SNAPSHOT_COLLECT_MAX_S = 5.0
BACKGROUND_NOT_LEARNED = "BACKGROUND_NOT_LEARNED"
"""``start`` 的 409 原因：column 模式列背景未学习（docs/03 §4 枚举之外的新增值，需同步文档）。"""
ALL_JOB_TYPES: tuple[JobType, ...] = (
    JobType.train,
    JobType.capture,
    JobType.deploy,
    JobType.backup,
)
NOZZLE_FIELDS = ("start_px", "end_px", "offset_x")
DEVICE_PARAM_FIELDS = (
    "awb_mode",
    "exposure_us",
    "width",
    "offset_x",
    "height",
    "line_rate_hz",
    "trigger",
)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class DeviceOffline(ApiError):
    """相机/下位机离线（HTTP 503 ``DEVICE_OFFLINE``）。"""

    def __init__(self, device: str, message: str, **detail: Any):
        super().__init__("DEVICE_OFFLINE", message, {"device": device, **detail})


# --------------------------------------------------------------------------------------
# 预览源
# --------------------------------------------------------------------------------------


class _PreviewSource:
    """一台相机的预览帧源（引用计数）：流水线在跑 → 复用其帧；否则自己开流，最后一个持有者释放时停流。"""

    def __init__(self, svc: RuntimeService, camera_id: int):
        self._svc = svc
        self.camera_id = int(camera_id)
        self._lock = threading.Lock()
        self.refcount = 0
        self._started_stream = False

    def acquire(self) -> None:
        with self._lock:
            self.refcount += 1

    def release(self) -> None:
        with self._lock:
            self.refcount = max(0, self.refcount - 1)
            last = self.refcount == 0
            started = self._started_stream
            if last:
                self._started_stream = False
        if last and started:
            pipe = self._svc._pipeline_for(self.camera_id)
            if pipe is None or not pipe.running:
                with contextlib.suppress(Exception):
                    self._svc.ctx.cameras[self.camera_id].stop_stream()

    def next_frame(self, after: Frame | None, timeout: float) -> Frame | None:
        timeout = max(0.0, float(timeout))
        pipe = self._svc._pipeline_for(self.camera_id)
        if pipe is not None and pipe.running:
            return pipe.wait_frame(after, timeout)
        cam = self._svc.ctx.cameras[self.camera_id]
        if not cam.connected:
            time.sleep(min(timeout, 0.2))
            return None
        try:
            if not cam.streaming:
                cam.start_stream(self._svc.ctx.cfg.camera(cam.id).runtime_chunk_lines)
                with self._lock:
                    self._started_stream = True
            return cam.get_frame(timeout)
        except CameraError as exc:
            log.debug("预览取帧失败", extra={"camera": cam.id, "err": str(exc)})
            time.sleep(min(timeout, 0.2))
            return None


class PreviewHandle:
    """``acquire_preview`` 返回的句柄：``get_frame(timeout)`` 取下一帧，``release()`` 归还。"""

    def __init__(self, source: _PreviewSource):
        self._src = source
        self._last: Frame | None = None
        self._released = False

    @property
    def camera_id(self) -> int:
        return self._src.camera_id

    def get_frame(self, timeout: float = 1.0) -> Frame | None:
        """下一帧（与上一帧不同）；超时 / 相机离线 / 已释放返回 None。"""
        if self._released:
            return None
        f = self._src.next_frame(self._last, timeout)
        if f is not None:
            self._last = f
        return f

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._src.release()

    def __enter__(self) -> PreviewHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# --------------------------------------------------------------------------------------
# RuntimeService
# --------------------------------------------------------------------------------------


class RuntimeService:
    """运行时门面。由 ``AppContext`` 构造并持有（``ctx.runtime``）。"""

    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._lock = threading.RLock()
        self._preview_lock = threading.Lock()
        self._previews: dict[int, _PreviewSource] = {}
        self._resume_after_capture = False

    # ---- 便捷 ----
    @property
    def _sm(self):
        return self.ctx.state_machine

    @property
    def lock(self) -> threading.RLock:
        """运行控制临界区（可重入）。``AppContext.set_classifier`` 用它与建流水线互斥。"""
        return self._lock

    @property
    def cfg(self) -> AppConfig:
        return self.ctx.cfg

    def _camera(self, camera_id: int) -> Camera:
        cam = self.ctx.cameras.get(int(camera_id)) if isinstance(camera_id, int) else None
        if cam is None:
            raise ConfigError("NOT_FOUND", f"相机 {camera_id} 不存在", {"camera_id": camera_id})
        return cam

    def _camera_index(self, camera_id: int) -> int:
        return [c.id for c in self.cfg.cameras].index(camera_id)

    def _pipeline_for(self, camera_id: int) -> Pipeline | None:
        return self.ctx.pipelines.get(int(camera_id))

    # ==================================================================================
    # 状态 / 统计
    # ==================================================================================

    def snapshot(self) -> dict[str, Any]:
        """``GET /state``。"""
        return self.ctx.snapshot()

    def run_stats(self) -> dict[str, Any]:
        """``GET /run/stats``：``state.stats`` + ``since_start`` 累计。"""
        snap = self.ctx.stats.snapshot()
        snap["since_start"] = {
            "since": snap.get("since"),
            "elapsed_s": snap.get("elapsed_s"),
            "chunks_total": snap.get("chunks_total"),
            "dropped_chunks": snap.get("dropped_chunks"),
            "particles_total": snap.get("particles_total"),
            "blown_total": snap.get("blown_total"),
            "per_class_total": dict(snap.get("per_class_total") or {}),
        }
        snap["pipelines"] = {str(cid): p.stats() for cid, p in self.ctx.pipelines.items()}
        return snap

    # ==================================================================================
    # 运行控制
    # ==================================================================================

    def _set_pipelines(self, pipes: dict[int, Pipeline]) -> None:
        self.ctx.pipelines = dict(pipes)
        self.ctx.pipeline = next(iter(pipes.values()), None)

    def _start_sort_pipelines(self, *, blow_enabled: bool) -> None:
        """为每台相机建一条 sort 流水线并启动；任一失败则全部回收后抛出。"""
        pipes: dict[int, Pipeline] = {}
        try:
            for idx, cam_cfg in enumerate(self.cfg.cameras):
                cam = self.ctx.cameras[cam_cfg.id]
                p = Pipeline(self.ctx, cam, mode="sort", camera_index=idx)
                p.set_blow_enabled(blow_enabled)
                p.start()
                pipes[cam.id] = p
        except BaseException:
            for p in pipes.values():
                p.stop()
            raise
        self._set_pipelines(pipes)

    def _stop_pipelines(self) -> None:
        pipes = list(self.ctx.pipelines.values())
        self._set_pipelines({})
        for p in pipes:
            try:
                p.stop()
            except Exception:
                log.exception("停止流水线异常", extra={"camera": p.camera.id})

    def _require_backgrounds(self) -> None:
        """每台相机的分割背景必须可用（column 模式需已学习），否则 409 ``BACKGROUND_NOT_LEARNED``。

        在 SYNC 之前预检：不要先清零下位机编码器再因背景缺失失败。
        """
        settings = self.cfg.capture
        for cam_cfg in self.cfg.cameras:
            try:
                make_background(settings, self.ctx.backgrounds.get(cam_cfg.id))
            except ValueError as exc:
                raise ApiError(
                    "CONFLICT",
                    f"{exc}（请先用 learn-bg 学习相机 {cam_cfg.id} 的背景）",
                    {
                        "state": self._sm.state.value,
                        "reasons": [BACKGROUND_NOT_LEARNED],
                        "camera_id": cam_cfg.id,
                    },
                ) from exc

    def _cancel_running_job(self, job_type: JobType, wait_s: float) -> None:
        """取消某类型正在跑的任务并最多等 ``wait_s`` 秒结束（等不到只记日志，不阻塞调用方）。"""
        job = self.ctx.jobs.running(job_type)
        if job is None:
            return
        self.ctx.jobs.cancel(job.job_id)
        try:
            self.ctx.jobs.wait(job.job_id, wait_s)
        except (TimeoutError, KeyError):
            log.warning("任务未在时限内结束", extra={"job_id": job.job_id, "type": job_type.value})

    def start(self) -> dict[str, Any]:
        """``POST /run/start``。

        - stopped → running：前置检查（含背景预检）→ 取消喷阀循环检测 → SYNC（编码器清零）+
          ``reset_line_counter`` → 启动流水线 → RUN；
        - paused → running：只发 RUN 并恢复吹气，**不** SYNC 也不重置行号（暂停期间编码器与行号同步冻结）；
          若分选流水线不在（采样后恢复失败），不重置行号地重建。

        409 原因：``NO_MODEL_DEPLOYED / CAMERA_DISCONNECTED / MCU_DISCONNECTED / SAMPLING_IN_PROGRESS / IN_FAULT``
        与 ``BACKGROUND_NOT_LEARNED``；相机开流失败 → ``CAMERA_DISCONNECTED``（已置相机故障）。
        """
        with self._lock:
            sm = self._sm
            state = sm.state
            reasons = S.check_start_preconditions(
                deployed_model_id=self.ctx.deployed_model_id,
                cameras_connected=all(c.connected for c in self.ctx.cameras.values()),
                mcu_connected=self.ctx.mcu.connected,
            )
            if state is RunState.sampling:
                reasons.append(S.SAMPLING_IN_PROGRESS)
            if state is RunState.fault:
                reasons.append(S.IN_FAULT)
            if reasons:
                raise IllegalTransition(state, Event.start, reasons)
            if state is RunState.running:
                return {"run_state": state.value, "mcu_ack": True}
            self._require_backgrounds()
            if state is RunState.paused:
                self.ctx.mcu.run()
                if self.ctx.pipelines:
                    for p in self.ctx.pipelines.values():
                        p.set_blow_enabled(True)
                else:
                    try:
                        self._start_sort_pipelines(blow_enabled=True)
                    except PipelineError as exc:
                        self._stop_pipelines()
                        raise self._pipeline_start_error(exc) from exc
                new = sm.transition(Event.start)
                return {"run_state": new.value, "mcu_ack": True}
            # stopped：运行优先于正在进行的喷阀循环检测（任务收尾会下发 VALVE_TEST_STOP）
            self._cancel_running_job(JobType.valve_test, VALVE_TEST_WAIT_CANCEL_S)
            # 故障抖动后反应线程可能尚未回收旧流水线：先停掉，避免两条流水线共用一台相机
            self._stop_pipelines()
            self.ctx.stats.reset()
            self.ctx.mcu.sync(clear_encoder=True)
            for cam in self.ctx.cameras.values():
                cam.reset_line_counter()
            try:
                self._start_sort_pipelines(blow_enabled=True)
                self.ctx.mcu.run()
            except PipelineError as exc:
                self._stop_pipelines()
                raise self._pipeline_start_error(exc) from exc
            except BaseException:
                self._stop_pipelines()
                raise
            new = sm.transition(Event.start)
            log.info("开始运行", extra={"cameras": list(self.ctx.pipelines)})
            return {"run_state": new.value, "mcu_ack": True}

    def _pipeline_start_error(self, exc: PipelineError) -> ApiError:
        """流水线启动失败 → 带准确原因的 409（背景缺失 / 相机开流失败），其它视为内部错误。"""
        state = self._sm.state.value
        if isinstance(exc, BackgroundUnavailable):
            return ApiError(
                "CONFLICT", str(exc), {"state": state, "reasons": [BACKGROUND_NOT_LEARNED]}
            )
        if isinstance(exc, CameraStreamError):
            return ApiError(
                "CONFLICT",
                str(exc),
                {"state": state, "reasons": [S.CAMERA_DISCONNECTED], "camera_id": exc.camera_id},
            )
        return ApiError("INTERNAL", f"流水线启动失败: {exc}", {"state": state})

    def pause(self) -> dict[str, Any]:
        """``POST /run/pause``：PAUSE → 流水线停吹（继续分类统计）→ paused。"""
        with self._lock:
            sm = self._sm
            if sm.state is RunState.paused:
                return {"run_state": sm.state.value, "mcu_ack": True}
            if not sm.can(Event.pause):
                sm.transition(Event.pause)  # 抛出带原因的 IllegalTransition
            self.ctx.mcu.pause()
            for p in self.ctx.pipelines.values():
                p.set_blow_enabled(False)
            new = sm.transition(Event.pause)
            return {"run_state": new.value, "mcu_ack": True}

    def stop(self) -> dict[str, Any]:
        """``POST /run/stop``：取消采样 → STOP（失败只告警）→ 停流水线 → stopped（fault 态保持 fault）。"""
        sampling = False
        with self._lock:
            sampling = self._sm.state is RunState.sampling
        if sampling:
            # 锁外等待：capture_task 的收尾（end_capture）需要拿锁
            self._cancel_running_job(JobType.capture, CAPTURE_WAIT_CANCEL_S)
        with self._lock:
            mcu_ack = True
            try:
                self.ctx.mcu.stop()
            except McuError as exc:
                mcu_ack = False
                log.warning("下发 STOP 失败", extra={"err": str(exc)})
            self._stop_pipelines()
            new = self._sm.transition(Event.stop)
            return {"run_state": new.value, "mcu_ack": mcu_ack}

    def quiesce(self) -> None:
        """进程退出前的收尾：停流水线、取消采样，不下发 MCU 指令（AppContext.stop 调用）。"""
        with self._lock:
            self._stop_pipelines()

    # ---- 采样生命周期（capture_task 调用）----
    def begin_capture(self, pipeline: Pipeline) -> None:
        """进入 sampling；来源为 paused 时挂起分选流水线（采样独占相机），并把 capture 流水线登记为 ``ctx.pipeline``。"""
        with self._lock:
            self._sm.transition(Event.capture_begin)
            if self._sm.previous is RunState.paused and self.ctx.pipelines:
                self._stop_pipelines()
                self._resume_after_capture = True
            else:
                self._resume_after_capture = False
            self._set_pipelines({pipeline.camera.id: pipeline})

    def end_capture(self, pipeline: Pipeline) -> RunState:
        """离开 sampling（已被 stop / 故障带走时忽略非法迁移）；回到 paused 时恢复分选流水线（不吹）。"""
        with self._lock:
            if self.ctx.pipeline is pipeline:
                self._set_pipelines({})
            try:
                new = self._sm.transition(Event.capture_end)
            except IllegalTransition as exc:
                new = self._sm.state
                log.info("采样结束时状态已改变", extra={"state": new.value, "why": exc.reasons})
            if new is RunState.paused and self._resume_after_capture:
                try:
                    self._start_sort_pipelines(blow_enabled=False)
                except Exception as exc:
                    log.exception("采样后恢复分选流水线失败")
                    self.ctx.faults.set("PIPELINE_ERROR", f"恢复流水线失败: {exc}")
            self._resume_after_capture = False
            return new

    # ---- 故障反应（AppContext 在故障置位后调用）----
    def on_fault_raised(self, code: str) -> None:
        """进入 fault：另起线程取消采样/喷阀检测任务、停流水线、尽力下发 STOP（不能在置故障的线程里 join 自己）。"""
        self.ctx.spawn(f"fault-react-{code}", self._react_to_fault, code)

    def _react_to_fault(self, code: str) -> None:
        """拿到 ``_lock`` 后 **复查** 状态机仍为 fault 再动作：故障抖动（置位后立刻清除）期间若操作员 /
        自动逻辑已重新 start，迟到的反应线程不得再 STOP + 停流水线（否则状态显示 running 而设备已停）。"""
        with self._lock:
            if self._sm.state is not RunState.fault:
                log.info(
                    "故障已清除且状态已改变，跳过故障处理",
                    extra={"fault": code, "state": self._sm.state.value},
                )
                return
            for jt in (JobType.capture, JobType.valve_test):
                job = self.ctx.jobs.running(jt)
                if job is not None:
                    self.ctx.jobs.cancel(job.job_id)
            had_pipelines = bool(self.ctx.pipelines)
            # 先让机器进入安全态（STOP），再回收流水线
            if (had_pipelines or code != S.MCU_DISCONNECTED) and self.ctx.mcu.connected:
                try:
                    self.ctx.mcu.stop()
                except McuError as exc:
                    log.warning("故障后下发 STOP 失败", extra={"fault": code, "err": str(exc)})
            self._stop_pipelines()
        log.warning("故障处理完成：流水线已停止", extra={"fault": code})

    # ==================================================================================
    # 参数
    # ==================================================================================

    def get_params(self) -> dict[str, Any]:
        """``GET /params``。"""
        cfg = self.cfg
        out: dict[str, Any] = {
            sec: getattr(cfg, sec).model_dump(mode="json") for sec in HMI_PARAM_SECTIONS
        }
        out["limits"] = self.ctx.config.limits()
        return out

    def update_params(self, section: str, patch: dict[str, Any]) -> dict[str, Any]:
        """``PUT /params/{section}``：校验 → 持久化 → 下发下位机（MCU 分组）→ 热更新流水线。

        - 下位机分组：成功 ``mcu_ack=True``；超时/断线 ``mcu_ack=False``（已持久化，重连后补发并以
          ``param_sync`` 事件通知）；NACK → 回滚配置并抛 ``McuNack``；
        - blow / capture：``mcu_ack=None``。
        """
        if section not in HMI_PARAM_SECTIONS:
            raise ConfigError("NOT_FOUND", f"未知参数分组: {section}", {"section": section})
        if not isinstance(patch, dict):
            raise ConfigError("VALIDATION", "请求体必须是对象", {"section": section})
        old = getattr(self.cfg, section).model_dump(mode="json")
        with self.ctx.suppress_param_sync():
            applied, warnings = self.ctx.config.update(section, patch)
        warnings = list(warnings)
        mcu_ack: bool | None = None
        if section in MCU_SECTIONS:
            params = P.params_for_section(self.cfg, section)
            try:
                self.ctx.mcu.set_params(params)
                mcu_ack = True
                self.ctx.events.publish(
                    "param_sync", {"section": section, "mcu_ack": True, "error": None}
                )
            except McuNack:
                with self.ctx.suppress_param_sync():
                    self.ctx.config.update(section, old)
                raise
            except (McuTimeout, McuDisconnected) as exc:
                mcu_ack = False
                self.ctx.mark_param_sync_pending(section)
                warnings.append(f"下位机未确认（{exc}），已保存，重连后自动补发")
                self.ctx.events.publish(
                    "param_sync", {"section": section, "mcu_ack": False, "error": str(exc)}
                )
        elif section == "capture":
            for p in list(self.ctx.pipelines.values()):
                try:
                    p.update_capture_settings(applied)  # type: ignore[arg-type]
                except PipelineError as exc:
                    warnings.append(f"相机 {p.camera.id} 流水线未能应用新参数: {exc}")
        elif section == "blow":
            for p in list(self.ctx.pipelines.values()):
                p.update_blow(applied)  # type: ignore[arg-type]
        return {
            "applied": applied.model_dump(mode="json"),
            "mcu_ack": mcu_ack,
            "warnings": warnings,
        }

    # ==================================================================================
    # 喷阀检测
    # ==================================================================================

    @staticmethod
    def _busy_reason(state: RunState) -> str:
        """非 stopped 状态拒绝操作时的 409 原因常量。paused 视为运行会话的一部分（ALREADY_RUNNING）。"""
        if state is RunState.fault:
            return S.IN_FAULT
        if state is RunState.sampling:
            return S.SAMPLING_IN_PROGRESS
        return S.ALREADY_RUNNING

    def _reject_if_busy(self, what: str) -> None:
        """喷阀检测/复位只允许 stopped 或 fault：running / paused / sampling 一律 409。

        paused 也拒绝：VALVE_TEST_CYCLE 会把下位机带到 VALVE_TEST、结束后变 STOPPED，而后端仍是 paused，
        两边状态不一致。调用方须持有 ``_lock``（与 start() 串行）。
        """
        state = self._sm.state
        if state in (RunState.running, RunState.paused, RunState.sampling):
            raise ApiError(
                "CONFLICT",
                f"{state.value} 状态不允许{what}（需先停止）",
                {"state": state.value, "reasons": [self._busy_reason(state)]},
            )

    def _check_hole(self, hole: int, name: str = "hole") -> int:
        total = self.cfg.total_nozzles
        if isinstance(hole, bool) or not isinstance(hole, int) or not (1 <= hole <= total):
            raise ConfigError(
                "PARAM_OUT_OF_RANGE",
                f"{name} = {hole} 超出范围 1~{total}",
                {"field": name, "input": hole, "range": [1, total]},
            )
        return hole

    def valve_test_single(self, hole: int) -> dict[str, Any]:
        """``POST /valve/test/single``（running / paused / sampling → 409）。"""
        with self._lock:
            self._reject_if_busy("单阀检测")
            self._check_hole(hole)
            self.ctx.mcu.valve_test_single(hole)
        with contextlib.suppress(ConfigError), self.ctx.suppress_param_sync():
            self.ctx.config.update("valve_test", {"hole": hole})
        return {"accepted": True}

    def valve_test_cycle(self, frm: int, to: int, interval_ms: int) -> dict[str, Any]:
        """``POST /valve/test/cycle``：后台任务，``valve_test_stop`` / 取消任务结束（running / paused / sampling → 409）。"""
        with self._lock:
            self._reject_if_busy("循环检测")
            self._check_hole(frm, "from")
            self._check_hole(to, "to")
            if frm > to:
                raise ConfigError(
                    "VALIDATION", f"from({frm}) 不能大于 to({to})", {"from": frm, "to": to}
                )
            if (
                isinstance(interval_ms, bool)
                or not isinstance(interval_ms, int)
                or not (1 <= interval_ms <= 65535)
            ):
                raise ConfigError(
                    "PARAM_OUT_OF_RANGE",
                    f"interval_ms = {interval_ms} 超出范围 1~65535",
                    {"field": "interval_ms", "input": interval_ms, "range": [1, 65535]},
                )
            job = self.ctx.jobs.submit(
                JobType.valve_test,
                valve_test_cycle_task(self.ctx, frm, to, interval_ms),
                meta={"from": frm, "to": to, "interval_ms": interval_ms},
            )
        return {"job_id": job.job_id}

    def valve_test_stop(self) -> dict[str, Any]:
        """``POST /valve/test/stop``。"""
        job = self.ctx.jobs.running(JobType.valve_test)
        self._cancel_running_job(JobType.valve_test, VALVE_TEST_WAIT_CANCEL_S)
        try:
            self.ctx.mcu.valve_test_stop()
        except McuError as exc:
            if job is None:
                raise
            log.warning("VALVE_TEST_STOP 失败（任务已取消）", extra={"err": str(exc)})
        return {"stopped": True}

    def valve_reset(self) -> dict[str, Any]:
        """``POST /valve/reset``（running / paused / sampling → 409）。"""
        with self._lock:
            self._reject_if_busy("喷阀复位")
            self.ctx.mcu.valve_reset()
        return {"accepted": True, "mcu_ack": True}

    # ==================================================================================
    # 相机
    # ==================================================================================

    def list_cameras(self) -> list[dict[str, Any]]:
        """``GET /cameras``。"""
        out = []
        for cam_cfg in self.cfg.cameras:
            cam = self.ctx.cameras[cam_cfg.id]
            d = cam.info().to_dict()
            d["name"] = cam_cfg.name
            d["connected"] = cam.connected
            d["streaming"] = cam.streaming
            out.append(d)
        return out

    def _require_online(self, camera_id: int) -> Camera:
        cam = self._camera(camera_id)
        if not cam.connected:
            raise DeviceOffline("camera", f"相机 {camera_id} 离线", camera_id=camera_id)
        return cam

    def camera_params(self, camera_id: int) -> dict[str, Any]:
        """``GET /cameras/{id}/params``（从相机实时读取 + 算力盒侧的起止像素/块高 + limits）。"""
        cam = self._require_online(camera_id)
        try:
            params = cam.get_params()
        except CameraDisconnected as exc:
            raise DeviceOffline("camera", str(exc), camera_id=camera_id) from exc
        cam_cfg = self.cfg.camera(camera_id)
        d = params.to_dict()
        d.update(
            {
                "start_px": cam_cfg.start_px,
                "end_px": cam_cfg.end_px,
                "runtime_chunk_lines": cam_cfg.runtime_chunk_lines,
                "nozzles": cam_cfg.nozzles,
                "limits": camera_limits(),
            }
        )
        return d

    def update_camera_params(self, camera_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        """``PUT /cameras/{id}/params``：校验持久化 → 在线则下发相机 → 起止像素/offset 变化热更新喷头映射。"""
        cam = self._camera(camera_id)
        if not isinstance(patch, dict):
            raise ConfigError("VALIDATION", "请求体必须是对象", {"camera_id": camera_id})
        before = self.cfg.camera(camera_id)
        applied_cfg, warnings = self.ctx.config.update_camera(camera_id, patch)
        warnings = list(warnings)
        device_changed = any(k in patch for k in DEVICE_PARAM_FIELDS)
        if device_changed:
            if cam.connected:
                try:
                    actual = cam.set_params(CameraParams.from_config(applied_cfg))
                except CameraDisconnected as exc:
                    warnings.append(f"相机掉线，参数已保存，连接后生效: {exc}")
                except CameraError as exc:
                    warnings.append(f"相机拒绝参数，已保存但未生效: {exc}")
                else:
                    diffs = {
                        k: v for k, v in actual.to_dict().items() if v != getattr(applied_cfg, k)
                    }
                    if diffs:
                        warnings.append(
                            "相机按步进对齐了部分参数: "
                            + ", ".join(f"{k} → {v}" for k, v in diffs.items())
                        )
                        with contextlib.suppress(ConfigError):
                            applied_cfg, _ = self.ctx.config.update_camera(camera_id, diffs)
            else:
                warnings.append("相机离线，参数已保存，连接后生效")
            if self._pipeline_for(camera_id) is not None and any(
                k in patch for k in ("width", "offset_x", "height", "runtime_chunk_lines")
            ):
                warnings.append("图像尺寸/块高变化需重新开始运行后生效")
        cam.cfg = applied_cfg
        if any(getattr(before, k) != getattr(applied_cfg, k) for k in NOZZLE_FIELDS):
            pipe = self._pipeline_for(camera_id)
            if pipe is not None:
                pipe.update_nozzle_map(
                    NozzleMap.from_camera(applied_cfg, self._camera_index(camera_id)),
                    offset_x=applied_cfg.offset_x,
                )
        out = CameraParams.from_config(applied_cfg).to_dict()
        out.update(
            {
                "start_px": applied_cfg.start_px,
                "end_px": applied_cfg.end_px,
                "runtime_chunk_lines": applied_cfg.runtime_chunk_lines,
            }
        )
        return {"applied": out, "warnings": warnings}

    def awb_once(self, camera_id: int) -> dict[str, Any]:
        """``POST /cameras/{id}/awb-once``：做一次白平衡并把配置锁定为 ``off``。"""
        cam = self._require_online(camera_id)
        try:
            cam.awb_once()
        except CameraDisconnected as exc:
            raise DeviceOffline("camera", str(exc), camera_id=camera_id) from exc
        applied_cfg, _ = self.ctx.config.update_camera(camera_id, {"awb_mode": "off"})
        cam.cfg = applied_cfg
        return {"applied": True}

    def nozzle_map(self, camera_id: int) -> dict[str, Any]:
        """``GET /cameras/{id}/nozzle-map``。"""
        cam_cfg = self.cfg.camera(camera_id)
        return NozzleMap.from_camera(cam_cfg, self._camera_index(camera_id)).to_dict()

    # ---- 图像获取与落盘 ----
    def _collect_timeout(self, camera_id: int, height: int) -> float:
        """运行中拼块的等待上限：约 3 倍"按行频走完 height 行"的时间，夹在 1~5 s。"""
        rate = max(1, int(self.cfg.camera(camera_id).line_rate_hz))
        return min(SNAPSHOT_COLLECT_MAX_S, max(SNAPSHOT_COLLECT_MIN_S, 3.0 * height / rate))

    def _grab_image(self, camera_id: int, height: int) -> tuple[np.ndarray, list[str]]:
        """取一幅 ``height`` 行的图像：流水线在跑时累积其连续块拼到请求高度（不打断采集），否则单次采集。

        返回 ``(image, warnings)``；运行中在时限内拼不够 ``height`` 行时返回已拼到的高度并给出 warning。
        """
        pipe = self._pipeline_for(camera_id)
        if pipe is not None and pipe.running:
            frames = pipe.collect_lines(height, self._collect_timeout(camera_id, height))
            if frames:
                image = (
                    frames[0].image
                    if len(frames) == 1
                    else np.concatenate([f.image for f in frames], axis=0)
                )
                image = np.ascontiguousarray(image[:height])
                warnings: list[str] = []
                if image.shape[0] < height:
                    warnings.append(
                        f"运行中快照取自流水线连续块，时限内仅拼到 {image.shape[0]} 行（请求 {height}）"
                    )
                return image, warnings
            raise ApiError(
                "INTERNAL",
                f"相机 {camera_id} 运行中未能从流水线取得图像",
                {"camera_id": camera_id, "state": self._sm.state.value},
            )
        cam = self._require_online(camera_id)
        try:
            return cam.snapshot(height), []
        except CameraDisconnected as exc:
            raise DeviceOffline("camera", str(exc), camera_id=camera_id) from exc
        except CameraError as exc:
            raise ApiError("INTERNAL", f"相机 {camera_id} 采集失败: {exc}") from exc

    def _background_for(self, camera_id: int, settings: CaptureSettings):
        try:
            return make_background(settings, self.ctx.backgrounds.get(camera_id))
        except ValueError as exc:
            raise ApiError(
                "CONFLICT",
                str(exc),
                {"reasons": ["BACKGROUND_NOT_LEARNED"], "camera_id": camera_id},
            ) from exc

    def _write_jpeg(self, sub_dir: str, name: str, rgb: np.ndarray, max_width: int) -> str:
        """缩放到 ``max_width``、JPEG 编码、原子写入 ``{data_dir}/{sub_dir}/{name}``，并裁剪旧文件；返回 URL。"""
        h, w = rgb.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            rgb = cv2.resize(
                rgb, (max_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA
            )
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_JPEG_QUALITY],
        )
        if not ok:
            raise ApiError("INTERNAL", "JPEG 编码失败")
        target_dir = ensure_dir(self.ctx.data_dir / sub_dir)
        atomic_write_bytes(target_dir / name, buf.tobytes(), fsync=False)
        self._prune_dir(target_dir, int(self.cfg.system.max_snapshots_kept))
        return f"/static/{sub_dir}/{name}"

    @staticmethod
    def _prune_dir(directory: Path, keep: int) -> None:
        try:
            files = sorted(
                (p for p in directory.glob("*.jpg") if p.is_file()), key=lambda p: p.name
            )
        except OSError:
            return
        for p in files[: max(0, len(files) - keep)]:
            with contextlib.suppress(OSError):
                p.unlink()

    def snapshot_image(
        self, camera_id: int, height: int, max_width: int = PREVIEW_MAX_WIDTH, overlay: bool = False
    ) -> dict[str, Any]:
        """``POST /cameras/{id}/snapshot``：运行/采样中把流水线的连续块拼到请求高度，否则单次采集；可叠画分割框。

        响应 ``{"image_url", "width", "height", "warnings", ["count"]}``：``height`` 为实际图像高度，运行中
        时限内拼不满时小于请求值并在 ``warnings`` 说明。
        """
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or not (1 <= height <= MAX_SENSOR_PX)
        ):
            raise ConfigError(
                "PARAM_OUT_OF_RANGE",
                f"height = {height} 超出范围 1~{MAX_SENSOR_PX}",
                {"field": "height", "input": height, "range": [1, MAX_SENSOR_PX]},
            )
        if (
            isinstance(max_width, bool)
            or not isinstance(max_width, int)
            or not (16 <= max_width <= 16384)
        ):
            raise ConfigError(
                "PARAM_OUT_OF_RANGE",
                f"max_width = {max_width} 超出范围 16~16384",
                {"field": "max_width", "input": max_width, "range": [16, 16384]},
            )
        self._camera(camera_id)
        image, warnings = self._grab_image(camera_id, height)
        h, w = image.shape[:2]
        count = None
        if overlay:
            settings = self.cfg.capture
            bg = self._background_for(camera_id, settings)
            particles = segment(image, bg, settings)
            image = draw_overlay(image, particles)
            count = len(particles)
        url = self._write_jpeg("snapshots", f"cam{camera_id}_{_now_stamp()}.jpg", image, max_width)
        out: dict[str, Any] = {
            "image_url": url,
            "width": int(w),
            "height": int(h),
            "warnings": warnings,
        }
        if count is not None:
            out["count"] = count
        return out

    def capture_preview(
        self, camera_id: int, settings_override: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """``POST /capture/preview``：用（临时覆盖的）采集设置分割一帧并叠画，写入 ``previews/``。"""
        cam_cfg = self.cfg.camera(camera_id)
        settings = self.cfg.capture
        if settings_override:
            if not isinstance(settings_override, dict):
                raise ConfigError("VALIDATION", "settings 必须是对象", {})
            try:
                settings = CaptureSettings.model_validate(
                    deep_merge(settings.model_dump(mode="json"), settings_override)
                )
            except ValidationError as exc:
                raise validation_error_to_config_error(exc) from None
        image, warnings = self._grab_image(camera_id, cam_cfg.height)
        bg = self._background_for(camera_id, settings)
        particles = segment(image, bg, settings)
        overlay = draw_overlay(image, particles)
        url = self._write_jpeg(
            "previews", f"cam{camera_id}_{_now_stamp()}.jpg", overlay, PREVIEW_MAX_WIDTH
        )
        return {
            "image_url": url,
            "particles": [{"bbox": list(p.bbox), "area_px": int(p.area)} for p in particles],
            "count": len(particles),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "warnings": warnings,
        }

    # ---- 预览源 ----
    def acquire_preview(self, camera_id: int) -> PreviewHandle:
        """WS 预览取帧句柄（引用计数；流水线在跑时共用其帧，否则按需开流）。"""
        self._camera(camera_id)
        with self._preview_lock:
            src = self._previews.get(camera_id)
            if src is None:
                src = _PreviewSource(self, camera_id)
                self._previews[camera_id] = src
            src.acquire()
        return PreviewHandle(src)

    def preview_holders(self, camera_id: int) -> int:
        with self._preview_lock:
            src = self._previews.get(camera_id)
            return src.refcount if src is not None else 0

    # ==================================================================================
    # 模型
    # ==================================================================================

    def list_models(self) -> list[dict[str, Any]]:
        """``GET /models``。"""
        return self.ctx.store.list(self.ctx.deployed_model_id)

    def current_model(self) -> dict[str, Any] | None:
        """``GET /models/current``。"""
        mid = self.ctx.deployed_model_id
        if mid is None:
            return None
        return self.ctx.store.describe(mid, mid)

    def create_model(self, model_id: int, name: str = "", description: str = "") -> dict[str, Any]:
        """``POST /models/{id}``。"""
        self.ctx.store.create(model_id, name, description)
        return self.ctx.store.describe(model_id, self.ctx.deployed_model_id)

    def patch_model(self, model_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        """``PATCH /models/{id}``（name / description）。"""
        if not isinstance(patch, dict):
            raise StoreError("VALIDATION", "请求体必须是对象", {"id": model_id})
        fields = {k: v for k, v in patch.items() if k in ("name", "description")}
        unknown = sorted(set(patch) - set(fields))
        if unknown:
            raise StoreError("VALIDATION", f"不允许修改的字段: {unknown}", {"unknown": unknown})
        if fields:
            self.ctx.store.update_meta(model_id, **fields)
        else:
            self.ctx.store.meta(model_id)
        return self.ctx.store.describe(model_id, self.ctx.deployed_model_id)

    def _job_touching_model(
        self,
        model_id: int,
        types: Sequence[JobType] = ALL_JOB_TYPES,
        *,
        exclude_job_id: str | None = None,
        backup_target_only: bool = False,
    ) -> str | None:
        """返回正在（排队/运行）操作 ``model_id`` 的任务 id，没有则 None。

        Args:
            types: 只看这些类型。
            exclude_job_id: 忽略该任务（训练任务收尾时提交自动重部署，要排除自己）。
            backup_target_only: 备份任务只按目标模型匹配（备份的源只读，不与部署/训练冲突）。
        """
        for jt in types:
            job = self.ctx.jobs.running(jt)
            if job is None or job.job_id == exclude_job_id:
                continue
            if jt is JobType.backup:
                hit = job.meta.get("target_id") == model_id or (
                    not backup_target_only and job.model_id == model_id
                )
            else:
                hit = job.model_id == model_id
            if hit:
                return job.job_id
        return None

    def _reject_if_model_busy(self, model_id: int, busy: str | None, what: str) -> None:
        if busy is not None:
            raise StoreError(
                "CONFLICT",
                f"模型 {model_id} 有任务在执行（{busy}），不能{what}",
                {"id": model_id, "job_id": busy},
            )

    def delete_model(self, model_id: int) -> None:
        """``DELETE /models/{id}``：当前部署 / 有任务在用 → CONFLICT。"""
        deployed = self.ctx.deployed_model_id
        if deployed == model_id:
            raise StoreError("CONFLICT", f"模型 {model_id} 正在使用，不能删除", {"id": model_id})
        busy = self._job_touching_model(model_id)
        if busy is not None:
            raise StoreError(
                "CONFLICT",
                f"模型 {model_id} 有任务在执行（{busy}）",
                {"id": model_id, "job_id": busy},
            )
        self.ctx.store.delete(model_id, deployed)

    def backup_model(
        self, model_id: int, target_id: int, overwrite: bool = False
    ) -> dict[str, Any]:
        """``POST /models/{id}/backup``。"""
        store = self.ctx.store
        store.meta(model_id)  # NOT_FOUND
        store.model_dir(target_id)  # 范围校验
        if target_id == self.ctx.deployed_model_id:
            raise StoreError(
                "CONFLICT", f"目标模型 {target_id} 正在使用，不能覆盖", {"target_id": target_id}
            )
        if store.exists(target_id) and not overwrite:
            raise StoreError(
                "CONFLICT",
                f"目标模型 {target_id} 已存在，需确认覆盖",
                {"src": model_id, "dst": target_id},
            )
        busy = self._job_touching_model(target_id)
        if busy is not None:
            raise StoreError(
                "CONFLICT", f"目标模型 {target_id} 有任务在执行（{busy}）", {"job_id": busy}
            )
        job = self.ctx.jobs.submit(
            JobType.backup,
            backup_task(self.ctx, model_id, target_id, bool(overwrite)),
            model_id=model_id,
            meta={"target_id": target_id, "overwrite": bool(overwrite)},
        )
        return {"job_id": job.job_id}

    def deploy_model(self, model_id: int, *, exclude_job_id: str | None = None) -> dict[str, Any]:
        """``POST /models/{id}/deploy``：未训练 → CONFLICT；该模型正在训练 / 被备份覆盖 → CONFLICT。

        Args:
            exclude_job_id: 互斥检查时忽略的任务（``train_task`` 收尾提交自动重部署时传自己的 job_id）。
        """
        store = self.ctx.store
        store.meta(model_id)
        if not store.trained(model_id):
            raise StoreError("CONFLICT", f"模型 {model_id} 未训练", {"id": model_id})
        busy = self._job_touching_model(
            model_id,
            (JobType.train, JobType.backup),
            exclude_job_id=exclude_job_id,
            backup_target_only=True,
        )
        self._reject_if_model_busy(model_id, busy, "部署")
        job = self.ctx.jobs.submit(
            JobType.deploy, deploy_task(self.ctx, model_id), model_id=model_id
        )
        return {"job_id": job.job_id}

    def model_metrics(self, model_id: int) -> dict[str, Any]:
        """``GET /models/{id}/metrics``：``train/metrics.json`` 全文。"""
        store = self.ctx.store
        store.meta(model_id)
        path = store.train_dir(model_id) / "metrics.json"
        try:
            doc = read_json(path, default=None)
        except ValueError as exc:
            raise StoreError("INTERNAL", "metrics.json 损坏", {"path": str(path)}) from exc
        if not isinstance(doc, dict):
            raise StoreError("NOT_FOUND", f"模型 {model_id} 尚无训练指标", {"id": model_id})
        return doc

    def train_model(self, model_id: int, classes: Sequence[str] | None = None) -> dict[str, Any]:
        """``POST /models/{id}/train``：预检样本数（``detail.insufficient``）→ 提交训练任务。

        已有训练任务 → ``JobConflict``；该模型正在部署 / 被备份覆盖 → ``StoreError CONFLICT``
        （训练会重写 ``train/`` 产物，不能与读取它们的部署并行）。
        """
        store = self.ctx.store
        store.meta(model_id)
        running = self.ctx.jobs.running(JobType.train)
        if running is not None:
            raise JobConflict(JobType.train, running.job_id)
        busy = self._job_touching_model(
            model_id, (JobType.deploy, JobType.backup), backup_target_only=True
        )
        self._reject_if_model_busy(model_id, busy, "训练")
        class_names = list(self.cfg.system.class_names)
        counts = store.samples(model_id).counts()
        if classes is None:
            chosen = [c for c in class_names if counts.get(c, 0) > 0]
        else:
            chosen = [str(c) for c in classes]
            if not chosen:
                raise TrainError("VALIDATION", "classes 不能为空", {"classes": []})
            if len(set(chosen)) != len(chosen):
                raise TrainError("VALIDATION", f"classes 含重复类别: {chosen}", {"classes": chosen})
            unknown = [c for c in chosen if c not in class_names]
            if unknown:
                raise TrainError(
                    "VALIDATION", f"未知类别: {unknown}", {"unknown": unknown, "classes": chosen}
                )
        need = int(self.cfg.ml.min_samples_per_class)
        insufficient = {c: counts.get(c, 0) for c in chosen if counts.get(c, 0) < need}
        if insufficient:
            raise TrainError(
                "VALIDATION",
                "样本不足: "
                + ", ".join(f"{c} 仅 {n} 张（需 ≥ {need}）" for c, n in insufficient.items()),
                {"insufficient": insufficient, "min_samples_per_class": need},
            )
        if len(chosen) < 2:
            raise TrainError(
                "VALIDATION",
                f"至少需要 2 个有样本的类别，当前 {len(chosen)} 个",
                {"classes": chosen, "insufficient": {}},
            )
        job = self.ctx.jobs.submit(
            JobType.train,
            train_task(self.ctx, model_id, chosen if classes is not None else None),
            model_id=model_id,
            meta={"classes": chosen},
        )
        return {"job_id": job.job_id}

    # ==================================================================================
    # 类别配置
    # ==================================================================================

    def get_classes(self) -> dict[str, Any]:
        """``GET /config/classes``。"""
        return {
            "class_names": list(self.cfg.system.class_names),
            "max_models": int(self.cfg.system.max_models),
        }

    def set_classes(self, class_names: Sequence[str]) -> dict[str, Any]:
        """``PUT /config/classes``：需停止状态且所有模型未训练；整体替换配置并同步样本库与统计。"""
        if not isinstance(class_names, (list, tuple)) or not class_names:
            raise ConfigError("VALIDATION", "class_names 不能为空", {})
        names = [str(c) for c in class_names]
        if len(set(names)) != len(names):
            raise ConfigError("VALIDATION", "class_names 不能重复", {"class_names": names})
        with self._lock:
            if names == list(self.cfg.system.class_names):
                return self.get_classes()
            if self._sm.state is not RunState.stopped:
                raise ApiError(
                    "CONFLICT",
                    "只能在停止状态修改类别",
                    {
                        "state": self._sm.state.value,
                        "reasons": [self._busy_reason(self._sm.state)],
                    },
                )
            trained = [i for i in self.ctx.store.created_ids() if self.ctx.store.trained(i)]
            if trained:
                raise StoreError(
                    "CONFLICT", "需所有模型未训练才能修改类别", {"trained_models": trained}
                )
            data = self.cfg.model_dump(mode="json")
            data["system"]["class_names"] = names
            data["blow"]["classes"] = [c for c in data["blow"]["classes"] if c in names]
            new_cfg = parse_config(data)
            new_cfg.system.data_dir = self.cfg.system.data_dir
            self.ctx.store.set_class_names(names)
            self.ctx.config.replace(new_cfg)
            self.ctx.replace_stats(names)
        self.ctx.events.publish("state", self.ctx.snapshot())
        return self.get_classes()

    # ==================================================================================
    # 样本
    # ==================================================================================

    def start_capture(
        self,
        model_id: int,
        cls: str,
        camera_id: int,
        max_samples: int = 200,
        max_frames: int | None = 50,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        """``POST /models/{id}/classes/{cls}/capture``。"""
        sm = self._sm
        if not sm.can(Event.capture_begin):
            sm.transition(Event.capture_begin)  # 抛出带原因的 IllegalTransition
        self.ctx.store.meta(model_id)
        if cls not in self.cfg.system.class_names:
            raise StoreError(
                "VALIDATION",
                f"未知类别: {cls!r}",
                {"cls": cls, "class_names": list(self.cfg.system.class_names)},
            )
        self._require_online(camera_id)
        for name, value, lo, hi in (
            ("max_samples", max_samples, 1, 100_000),
            ("timeout_s", timeout_s, 1, 86_400),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not (lo <= value <= hi)
            ):
                raise ConfigError(
                    "PARAM_OUT_OF_RANGE",
                    f"{name} = {value} 超出范围 {lo}~{hi}",
                    {"field": name, "input": value, "range": [lo, hi]},
                )
        if max_frames is not None and (
            isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1
        ):
            raise ConfigError(
                "PARAM_OUT_OF_RANGE",
                f"max_frames = {max_frames} 必须 ≥ 1",
                {"field": "max_frames", "input": max_frames},
            )
        free = disk_free_mb(self.ctx.data_dir)
        if free < self.cfg.system.min_free_disk_mb:
            raise StoreError(
                "CONFLICT",
                f"磁盘空间不足：剩余 {free:.0f} MB，低于阈值 {self.cfg.system.min_free_disk_mb} MB",
                {"free_mb": round(free, 1), "min_free_mb": self.cfg.system.min_free_disk_mb},
            )
        job = self.ctx.jobs.submit(
            JobType.capture,
            capture_task(
                self.ctx,
                model_id,
                cls,
                camera_id,
                int(max_samples),
                int(max_frames) if max_frames is not None else None,
                float(timeout_s),
            ),
            model_id=model_id,
            meta={"class": cls, "camera_id": camera_id, "max_samples": int(max_samples)},
        )
        return {"job_id": job.job_id}

    def list_samples(
        self,
        model_id: int,
        cls: str,
        offset: int = 0,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """``GET /models/{id}/classes/{cls}/samples``。"""
        total, items = self.ctx.store.samples(model_id).list(cls, include_deleted, offset, limit)
        return {"total": total, "items": [s.to_dict() for s in items]}

    def delete_sample(self, sample_id: str) -> dict[str, Any]:
        """``DELETE /samples/{sample_id}``：已删除 / 不存在的样本 → ``deleted=False``（``undo_batch`` 为 None）。"""
        mid, _, _ = parse_sample_id(sample_id)
        batch = self.ctx.store.samples(mid).soft_delete([sample_id])
        return {"deleted": batch is not None, "undo_batch": batch}

    def delete_samples(self, ids: Sequence[str]) -> dict[str, Any]:
        """``POST /samples/delete``：按（模型, 类别）分组，各成一个可撤销批次。"""
        if not ids:
            raise StoreError("VALIDATION", "ids 不能为空", {})
        groups: dict[tuple[int, str], list[str]] = {}
        for sid in dict.fromkeys(ids):
            mid, cls, _ = parse_sample_id(sid)
            groups.setdefault((mid, cls), []).append(sid)
        deleted = 0
        batches: list[str] = []
        for (mid, _cls), group in groups.items():
            batch = self.ctx.store.samples(mid).soft_delete(group)
            if batch is not None:
                batches.append(batch)
                deleted += len(group)
        return {"deleted": deleted, "undo_batches": batches}

    def restore_sample(self, sample_id: str) -> dict[str, Any]:
        """``POST /samples/{sample_id}/restore``。"""
        mid, _, _ = parse_sample_id(sample_id)
        return {"restored": self.ctx.store.samples(mid).restore(sample_id)}

    def undo_delete(self, model_id: int, cls: str) -> dict[str, Any]:
        """``POST /models/{id}/classes/{cls}/undo-delete``（无可撤销 → StoreError NOT_FOUND）。"""
        return {"restored": self.ctx.store.samples(model_id).undo_last(cls)}

    def clear_samples(self, model_id: int, cls: str) -> dict[str, Any]:
        """``DELETE /models/{id}/classes/{cls}/samples``。"""
        n, batch = self.ctx.store.samples(model_id).clear(cls)
        return {"deleted": n, "undo_batch": batch}

    # ==================================================================================
    # 任务
    # ==================================================================================

    def get_job(self, job_id: str) -> dict[str, Any]:
        """``GET /jobs/{job_id}``。"""
        job = self.ctx.jobs.get(job_id)
        if job is None:
            raise ApiError("NOT_FOUND", f"任务不存在: {job_id}", {"job_id": job_id})
        return job.to_dict()

    def list_jobs(self, active: bool = False) -> list[dict[str, Any]]:
        """``GET /jobs?active=``。"""
        return [j.to_dict() for j in self.ctx.jobs.list(active_only=active)]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """``POST /jobs/{job_id}/cancel``：不存在/已结束返回 ``cancelled=False``。"""
        return {"cancelled": bool(self.ctx.jobs.cancel(job_id))}

    # ==================================================================================
    # 系统
    # ==================================================================================

    def save(self) -> dict[str, Any]:
        """``POST /system/save``：刷运行时长、显式落盘并导出备份。"""
        self.ctx.flush_uptime()
        path = self.ctx.config.export_backup()
        return {"saved": True, "backup_path": str(path)}

    def shutdown(self, reboot: bool = False) -> dict[str, Any]:
        """``POST /system/shutdown`` / ``/reboot``：先停机，2 s 后在后台执行 ``systemctl poweroff|reboot``。

        只有配置 ``system.allow_power_actions=true``（或环境变量 ``ORESORTER_ALLOW_POWER=1``）时才真正执行，
        否则只记日志（开发机 / 测试安全）。
        """
        try:
            self.stop()
        except Exception as exc:
            log.warning("关机前停止失败，继续", extra={"err": str(exc)})
        action = "reboot" if reboot else "poweroff"
        self.ctx.spawn(f"power-{action}", self._power_action, action)
        return {"accepted": True, "action": action}

    def _power_action(self, action: str) -> None:
        self.ctx.stop_event.wait(POWER_ACTION_DELAY_S)
        cmd = ["systemctl", action]
        allowed = bool(self.ctx.cfg.system.allow_power_actions) or os.environ.get(POWER_ENV) == "1"
        if not allowed:
            log.warning(
                "电源操作未执行（system.allow_power_actions=false；真机请在 jetson.yaml 置 true）",
                extra={"cmd": cmd},
            )
            return
        try:
            self.ctx.flush_uptime()
            subprocess.run(cmd, check=False, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("电源操作失败", extra={"cmd": cmd, "err": str(exc)})
