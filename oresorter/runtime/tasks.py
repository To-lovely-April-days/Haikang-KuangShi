"""后台任务函数（交给 ``JobManager.submit``）：采样、训练、部署、备份、喷阀循环检测。

每个工厂返回 ``fn(handle: JobHandle) -> dict``，参数由闭包绑定。任务内所有阻塞等待都带超时、
可被 ``handle.cancel_event`` 打断；状态机迁移与流水线启停交给 ``ctx.runtime``（RuntimeService）。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from oresorter.data.samples import StoreError
from oresorter.fsutil import atomic_write_text, ensure_dir, read_json
from oresorter.log import get_logger
from oresorter.mcu.link import McuError
from oresorter.ml.train import TrainError
from oresorter.runtime.jobs import JobCancelled, JobConflict, JobHandle
from oresorter.runtime.pipeline import CaptureSink, Pipeline
from oresorter.vision.preprocess import to_tensor_batch

if TYPE_CHECKING:
    from oresorter.runtime.context import AppContext

log = get_logger(__name__)

TaskFn = Callable[[JobHandle], dict[str, Any]]

TRAIN_PROGRESS_POLL_S = 0.5
TRAIN_TERM_GRACE_S = 10.0
TRAIN_NICE = 10
TRASH_PURGE_AGE_S = 7 * 24 * 3600.0
RUNTIME_CONFIG_NAME = "runtime_config.yaml"
TRAIN_SUBPROCESS_LOG = "subprocess.log"
CONSISTENCY_BATCH = 8
CAPTURE_POLL_S = 0.1


# --------------------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------------------


def capture_task(
    ctx: AppContext,
    model_id: int,
    cls: str,
    camera_id: int,
    max_samples: int,
    max_frames: int | None,
    timeout_s: float,
) -> TaskFn:
    """连续采集：在 sampling 状态下跑 capture 流水线，写样本直到任一上限 / 取消 / 超时。

    返回 ``{"count", "frames", "stop_reason"}``；``stop_reason ∈ max_samples | max_frames | timeout |
    cancelled | store_error | pipeline_stopped``。
    """

    def fn(handle: JobHandle) -> dict[str, Any]:
        svc = ctx.runtime
        repo = ctx.store.samples(model_id)
        camera = ctx.cameras[camera_id]

        def on_sample(sample: dict[str, Any], count: int) -> None:
            ctx.events.publish(
                "capture_sample",
                {"model_id": model_id, "class": cls, "sample": sample, "count": count},
            )
            handle.set_progress(count / max_samples, f"{count}/{max_samples}")

        sink = CaptureSink(
            repo,
            cls,
            camera_id,
            max_samples=max_samples,
            max_frames=max_frames,
            batch=handle.job_id,
            on_sample=on_sample,
        )
        pipeline = Pipeline(ctx, camera, mode="capture", sink=sink)
        svc.begin_capture(pipeline)
        stop_reason = "timeout"
        try:
            pipeline.start()
            handle.set_progress(0.0, f"0/{max_samples}")
            deadline = time.monotonic() + float(timeout_s)
            while True:
                if sink.done.wait(CAPTURE_POLL_S):
                    stop_reason = sink.stop_reason or "done"
                    break
                if handle.cancelled:
                    stop_reason = "cancelled"
                    break
                if time.monotonic() >= deadline:
                    stop_reason = "timeout"
                    break
                if not pipeline.running:
                    stop_reason = "pipeline_stopped"
                    break
        finally:
            pipeline.stop()
            svc.end_capture(pipeline)
        handle.log.info(
            "采样结束",
            extra={
                "model_id": model_id,
                "cls": cls,
                "count": sink.count,
                "frames": sink.frames,
                "reason": stop_reason,
            },
        )
        if sink.error:
            handle.log.warning("采样因存储错误提前结束", extra={"err": sink.error})
        return {
            "count": sink.count,
            "frames": sink.frames,
            "stop_reason": stop_reason,
            "error": sink.error,
        }

    return fn


# --------------------------------------------------------------------------------------
# 训练（子进程）
# --------------------------------------------------------------------------------------


def write_runtime_config(ctx: AppContext) -> Path:
    """把当前 **生效** 配置（yaml + HMI 修改）原子写为 ``{data_dir}/runtime_config.yaml`` 供训练子进程读取。

    不直接用启动 yaml：HMI 修改过的 ``capture``（背景色/阈值）与 ``ml`` 分组只存在于 config.json 覆盖层。
    """
    path = ctx.data_dir / RUNTIME_CONFIG_NAME
    data = ctx.cfg.model_dump(mode="json")
    atomic_write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return path


def _package_root() -> Path:
    import oresorter

    return Path(oresorter.__file__).resolve().parent.parent


def _read_json_tolerant(path: Path) -> dict[str, Any] | None:
    try:
        data = read_json(path, default=None)
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _tail_file(path: Path, n: int = 2000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-n:]


def _terminate(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM → 等 ``grace_s`` → SIGKILL。"""
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except OSError:
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        log.warning("训练子进程未响应 SIGTERM，强制结束", extra={"pid": proc.pid})
        try:
            proc.kill()
            proc.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            log.error("训练子进程无法结束", extra={"pid": proc.pid})


def train_task(ctx: AppContext, model_id: int, classes: list[str] | None) -> TaskFn:
    """子进程训练：``python -m oresorter.ml.train --model-dir D --config runtime_config.yaml --nice 10 [--classes]``。

    - 训练前清理 7 天前的回收站样本；
    - 每 0.5 s 轮询 ``train/progress.json`` 转发进度；取消 → SIGTERM，10 s 不退 → SIGKILL；
    - 退出码非 0 → 读 ``train/error.json`` 抛 ``TrainError(code, message, detail)``（样本不足的
      ``detail.insufficient`` 原样透传）；
    - 成功 → ``{"metrics_summary", "artifacts": {"onnx", "engine": False}, ...}``；若该模型就是当前部署
      模型则自动提交 deploy 任务（``redeploy_job_id``）。
    """

    def fn(handle: JobHandle) -> dict[str, Any]:
        store = ctx.store
        model_dir = store.model_dir(model_id)
        if not store.exists(model_id):
            raise StoreError("NOT_FOUND", f"模型 {model_id} 不存在", {"id": model_id})
        try:
            purged = store.samples(model_id).purge(TRASH_PURGE_AGE_S)
            if purged:
                handle.log.info("已清理过期回收站样本", extra={"n": purged})
        except StoreError as exc:
            handle.log.warning("清理回收站失败", extra={"err": exc.message})
        config_path = write_runtime_config(ctx)
        train_dir = ensure_dir(model_dir / "train")
        progress_path = train_dir / "progress.json"
        error_path = train_dir / "error.json"
        progress_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "oresorter.ml.train",
            "--model-dir",
            str(model_dir),
            "--config",
            str(config_path),
            "--nice",
            str(TRAIN_NICE),
        ]
        if classes:
            cmd += ["--classes", ",".join(classes)]
        env = dict(os.environ)
        root = str(_package_root())
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log_path = train_dir / TRAIN_SUBPROCESS_LOG
        handle.set_progress(0.0, "starting")
        handle.log.info("启动训练子进程", extra={"cmd": " ".join(cmd)})
        with log_path.open("wb") as log_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                raise TrainError("INTERNAL", f"无法启动训练子进程: {exc}") from exc
            last_progress: tuple[float, str | None] | None = None
            cancelled = False
            try:
                while True:
                    rc = proc.poll()
                    if rc is not None:
                        break
                    if handle.cancelled and not cancelled:
                        cancelled = True
                        handle.log.info("取消训练，通知子进程", extra={"pid": proc.pid})
                        _terminate(proc, TRAIN_TERM_GRACE_S)
                        continue
                    doc = _read_json_tolerant(progress_path)
                    if doc is not None:
                        cur = (float(doc.get("progress", 0.0) or 0.0), doc.get("stage"))
                        if cur != last_progress:
                            last_progress = cur
                            handle.set_progress(cur[0], str(cur[1]) if cur[1] else None)
                    handle.cancel_event.wait(TRAIN_PROGRESS_POLL_S)
            finally:
                if proc.poll() is None:
                    _terminate(proc, TRAIN_TERM_GRACE_S)
        rc = proc.returncode
        if handle.cancelled:
            raise JobCancelled("训练已取消")
        if rc != 0:
            err = _read_json_tolerant(error_path)
            if err is None:
                raise TrainError(
                    "INTERNAL",
                    f"训练子进程退出码 {rc}（无 error.json）",
                    {"returncode": rc, "log_tail": _tail_file(log_path)},
                )
            detail = dict(err.get("detail") or {})
            detail.setdefault("returncode", rc)
            raise TrainError(
                str(err.get("code") or "INTERNAL"), str(err.get("message") or "训练失败"), detail
            )
        metrics_doc = _read_json_tolerant(train_dir / "metrics.json") or {}
        summary = {
            "val_accuracy": float(metrics_doc.get("accuracy", 0.0) or 0.0),
            "macro_f1": float(metrics_doc.get("macro_f1", 0.0) or 0.0),
        }
        artifacts_doc = metrics_doc.get("artifacts") or {}
        result: dict[str, Any] = {
            "metrics_summary": summary,
            "artifacts": {
                "onnx": bool(artifacts_doc.get("onnx", store.has_onnx(model_id))),
                "engine": False,
            },
            "classes": list(metrics_doc.get("classes") or classes or []),
            "epochs": metrics_doc.get("epochs"),
            "duration_s": metrics_doc.get("duration_s"),
        }
        handle.set_progress(1.0, "done")
        if ctx.deployed_model_id == model_id:
            try:
                # 本训练任务尚未结束（fn 还在跑），要把自己从"同模型训练中"的互斥里排除
                redeploy = ctx.runtime.deploy_model(model_id, exclude_job_id=handle.job_id)
                result["redeploy_job_id"] = redeploy["job_id"]
                handle.log.info("当前部署模型已重训，自动重新部署", extra=redeploy)
            except (JobConflict, StoreError) as exc:
                handle.log.warning("自动重新部署未能提交", extra={"err": str(exc)})
                result["redeploy_error"] = str(exc)
        return result

    return fn


# --------------------------------------------------------------------------------------
# 部署
# --------------------------------------------------------------------------------------


def _synthetic_batch(image_size: int, n: int = CONSISTENCY_BATCH) -> np.ndarray:
    """一致性自检用的合成 RGBA 小图批（确定性随机）。"""
    rng = np.random.default_rng(20261024)
    crops = []
    for _ in range(n):
        h, w = int(rng.integers(24, 64)), int(rng.integers(24, 64))
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((yy - h / 2) ** 2 / (h / 2) ** 2 + (xx - w / 2) ** 2 / (w / 2) ** 2) <= 1.0
        crops.append(np.dstack([rgb, (mask * 255).astype(np.uint8)]))
    return to_tensor_batch(crops, image_size)


def deploy_task(ctx: AppContext, model_id: int) -> TaskFn:
    """部署：（可选）生成 TensorRT 引擎 → 加载分类器 → 预热 → 与 PyTorch 一致性自检 → 原子切换。

    返回 ``{"backend", "engine", "consistency", "model_id"}``。
    """

    def fn(handle: JobHandle) -> dict[str, Any]:
        from oresorter.ml import export
        from oresorter.ml.infer import TorchClassifier, compare_outputs, load_classifier

        store = ctx.store
        cfg = ctx.cfg
        if not store.exists(model_id):
            raise StoreError("NOT_FOUND", f"模型 {model_id} 不存在", {"id": model_id})
        if not store.trained(model_id):
            raise StoreError("CONFLICT", f"模型 {model_id} 未训练", {"id": model_id})
        model_dir = store.model_dir(model_id)
        onnx_path = model_dir / "train" / "model.onnx"
        plan_path = model_dir / "train" / "model.plan"
        engine_ok = False
        warnings: list[str] = []

        handle.set_progress(0.05, "engine")
        trtexec = export.find_trtexec()
        if trtexec is not None and onnx_path.is_file():
            try:
                from oresorter.ml.infer import read_model_meta

                size = int(read_model_meta(model_dir)["image_size"])
                if export.engine_matches(plan_path, onnx_path, image_size=size):
                    engine_ok = True
                else:
                    handle.set_progress(0.1, "building engine")
                    export.build_engine(
                        onnx_path,
                        plan_path,
                        size,
                        timeout_s=int(cfg.ml.engine_build_timeout_s),
                        max_batch=int(cfg.pipeline.infer_batch_max),
                    )
                    engine_ok = True
            except Exception as exc:
                warnings.append(f"引擎生成失败，回退 ONNX/PyTorch: {exc}")
                handle.log.warning("TensorRT 引擎生成失败", extra={"err": str(exc)})
        handle.check_cancel()

        handle.set_progress(0.5, "loading")
        clf = load_classifier(
            model_dir,
            prefer=("trt", "onnx", "torch"),
            device=cfg.ml.device,
            max_batch=int(cfg.pipeline.infer_batch_max),
        )
        consistency: dict[str, Any] = {"checked": False}
        try:
            clf.warmup()
            handle.set_progress(0.7, "consistency check")
            if clf.backend != "torch" and (model_dir / "train" / "best.pt").is_file():
                torch_clf = TorchClassifier(
                    device=cfg.ml.device, max_batch=int(cfg.pipeline.infer_batch_max)
                )
                try:
                    torch_clf.load(model_dir)
                    batch = _synthetic_batch(clf.image_size)
                    cmp = compare_outputs(clf.predict(batch)[0], torch_clf.predict(batch)[0])
                    consistency = {"checked": True, "reference": "torch", **cmp}
                    if cmp["top1_agree_rate"] < 1.0:
                        warnings.append(
                            f"{clf.backend} 与 PyTorch top-1 不一致（{cmp['n_disagree']}/{cmp['n']}），回退 PyTorch"
                        )
                        handle.log.warning("推理后端一致性自检未通过，回退 PyTorch", extra=cmp)
                        clf.close()
                        clf = torch_clf
                        torch_clf = None
                finally:
                    if torch_clf is not None and torch_clf is not clf:
                        torch_clf.close()
            handle.check_cancel()
        except BaseException:
            clf.close()
            raise

        handle.set_progress(0.9, "switching")
        ctx.set_classifier(model_id, clf)
        handle.log.info(
            "模型已部署",
            extra={"model_id": model_id, "backend": clf.backend, "engine": engine_ok},
        )
        return {
            "model_id": model_id,
            "backend": clf.backend,
            "engine": engine_ok,
            "consistency": consistency,
            "image_size": clf.image_size,
            "classes": clf.classes,
            "warnings": warnings,
        }

    return fn


# --------------------------------------------------------------------------------------
# 备份 / 喷阀循环检测
# --------------------------------------------------------------------------------------


def backup_task(ctx: AppContext, src: int, dst: int, overwrite: bool) -> TaskFn:
    """“备份至”：整目录复制（``ModelStore.backup``）。返回 ``{"target_id", "source_id"}``。"""

    def fn(handle: JobHandle) -> dict[str, Any]:
        handle.set_progress(0.1, "copying")
        meta = ctx.store.backup(src, dst, overwrite)
        handle.set_progress(1.0, "done")
        return {"target_id": dst, "source_id": src, "name": meta.name}

    return fn


def valve_test_cycle_task(ctx: AppContext, frm: int, to: int, interval_ms: int) -> TaskFn:
    """喷阀循环检测：下发 VALVE_TEST_CYCLE，等待取消，结束时下发 VALVE_TEST_STOP。"""

    def fn(handle: JobHandle) -> dict[str, Any]:
        mcu = ctx.mcu
        mcu.valve_test_cycle(frm, to, interval_ms)
        handle.set_progress(0.0, f"cycle {frm}-{to} @ {interval_ms}ms")
        try:
            while not handle.cancel_event.wait(0.2):
                pass
        finally:
            try:
                mcu.valve_test_stop()
            except McuError as exc:
                handle.log.warning("结束喷阀循环检测失败", extra={"err": str(exc)})
        return {"stopped": True}

    return fn
