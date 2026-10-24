"""一键训练配方（docs/01 §6.2）与子进程入口。

流程：读样本 → 校验（每类样本数、类别数）→ 分层划分 → 类别均衡采样 →
阶段 A 冻结骨干只训分类头 → 阶段 B 全网微调（cosine 衰减）→ 每 epoch 验证并写进度 →
以验证 macro-F1 选 best → 早停 / 时间上限 / 取消 → 导出 ONNX、写 ``metrics.json``、更新 ``meta.json``。

产物一致性：首次写入新的 ``best.pt`` 之前，先删除上一轮遗留的 ``model.onnx`` / ``model.plan`` / ``model.json``，
因此 ``train/`` 目录里任何时刻都不会同时存在来自两次训练的 checkpoint 与派生产物；ONNX 导出失败只记 warning，
并在返回值与 ``metrics.json`` 的 ``artifacts`` 中标明。

子进程用法::

    python -m oresorter.ml.train --model-dir /data/oresorter/models/1 [--classes A,B,C] [--config config/dev.yaml]

进度写 ``{model_dir}/train/progress.json``；失败时错误写 ``train/error.json``，退出码 1。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from oresorter.config import MlConfig
from oresorter.fsutil import atomic_write_json, ensure_dir, read_json
from oresorter.log import KVFormatter, get_logger
from oresorter.ml import metrics, models
from oresorter.ml.dataset import (
    Item,
    SampleDataset,
    kfold_split,
    list_model_samples,
    make_weighted_sampler,
    stratified_split,
    worker_init_fn,
)

#: train.log 挂在这个父 logger 上，models / dataset 等兄弟模块的日志（如“预训练权重缺失”）也会进文件
_ML_LOGGER_NAME = "oresorter.ml"
# 固定写死名字而不用 __name__：``python -m oresorter.ml.train`` 时 __name__ 是 "__main__"，
# 不是 oresorter.ml 的子 logger，本模块的日志就进不了 train.log
log = get_logger(f"{_ML_LOGGER_NAME}.train")

ProgressCb = Callable[[dict[str, Any]], None]

TRAIN_DIR = "train"
BEST_FILE = "best.pt"
ONNX_FILE = "model.onnx"
PLAN_FILE = "model.plan"
ENGINE_META_FILE = "model.json"
METRICS_FILE = "metrics.json"
PROGRESS_FILE = "progress.json"
LOG_FILE = "train.log"
ERROR_FILE = "error.json"
META_FILE = "meta.json"
#: 由 best.pt 派生、必须与之配套的产物；写新 best.pt 前一并清掉旧的
DERIVED_ARTIFACTS: tuple[str, ...] = (ONNX_FILE, PLAN_FILE, ENGINE_META_FILE)
LOO_THRESHOLD = 50
"""样本数低于此值时交叉验证改为留一法。"""
ALLOWED_IMAGE_SIZES: tuple[int, ...] = (128, 160, 224)
"""docs/01 §9 允许的 ``ml.image_size``；生产入口（CLI）严格限制，进程内调用（测试）可放宽。"""
LOADER_TIMEOUT_S = 60.0
"""``num_workers > 0`` 时 DataLoader 等待一个 batch 的超时，避免 worker 卡死导致无限阻塞。"""


class TrainError(Exception):
    """训练错误，与 ``ConfigError`` 同形：``code`` 取 VALIDATION / NOT_FOUND / CANCELLED / INTERNAL。"""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


@dataclass
class TrainConfig:
    """训练配方参数。前半部分与 ``MlConfig`` 同名；后半部分为固定超参（不暴露给操作工）。"""

    backbone: str = "resnet18"
    image_size: int = 160
    min_samples_per_class: int = 20
    max_train_minutes: float = 10.0
    max_epochs: int = 30
    head_epochs: int = 3
    batch_size: int = 32
    patience: int = 5
    val_ratio: float = 0.2
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    head_lr: float = 1e-3
    lr: float = 1e-4
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    seed: int = 0
    bg_color: tuple[int, int, int] = (0, 0, 255)
    bg_threshold: float = field(default=40.0)

    @classmethod
    def from_ml_config(
        cls, ml: MlConfig, *, strict_image_size: bool = False, **overrides: Any
    ) -> TrainConfig:
        """由配置分组构造，``overrides`` 中值为 None 的项忽略。

        Args:
            strict_image_size: True 时 ``image_size`` 必须是 docs/01 §9 允许的值（生产入口用）。
        """
        names = {f.name for f in fields(cls)}
        data: dict[str, Any] = {k: getattr(ml, k) for k in names if hasattr(ml, k)}
        for k, v in overrides.items():
            if k not in names:
                raise TrainError("VALIDATION", f"未知训练参数: {k}", {"field": k})
            if v is not None:
                data[k] = v
        cfg = cls(**data)
        cfg.validate(strict_image_size=strict_image_size)
        return cfg

    def validate(self, *, strict_image_size: bool = False) -> None:
        """范围校验，越界或类型错误抛 ``TrainError("VALIDATION")`` 并说明原因。

        ``max_train_minutes`` 在这里只要求 > 0（§9 的 ≥ 1 由 ``MlConfig`` 保证），因为进程内调用方
        （交叉验证、测试）需要亚分钟级预算。
        """
        try:
            problems = self._problems(strict_image_size)
        except (TypeError, ValueError) as exc:
            raise TrainError(
                "VALIDATION", f"训练参数类型错误: {exc}", {"errors": [str(exc)]}
            ) from exc
        if problems:
            raise TrainError("VALIDATION", "; ".join(problems), {"errors": problems})

    def _problems(self, strict_image_size: bool) -> list[str]:
        problems: list[str] = []
        if self.backbone not in models.BACKBONES:
            problems.append(f"backbone={self.backbone!r} 不在 {models.BACKBONES}")
        size = int(self.image_size)
        if strict_image_size:
            if size not in ALLOWED_IMAGE_SIZES:
                problems.append(f"image_size={self.image_size} 不在 {list(ALLOWED_IMAGE_SIZES)}")
        elif not (8 <= size <= 1024):
            problems.append(f"image_size={self.image_size} 超出范围 8~1024")
        if int(self.min_samples_per_class) < 1:
            problems.append("min_samples_per_class 必须 ≥ 1")
        if not (float(self.max_train_minutes) > 0):
            problems.append("max_train_minutes 必须 > 0")
        if not (1 <= int(self.max_epochs) <= 1000):
            problems.append(f"max_epochs={self.max_epochs} 超出范围 1~1000")
        if not (0 <= int(self.head_epochs) <= 100):
            problems.append(f"head_epochs={self.head_epochs} 超出范围 0~100")
        if not (1 <= int(self.batch_size) <= 512):
            problems.append(f"batch_size={self.batch_size} 超出范围 1~512")
        if not (1 <= int(self.patience) <= 100):
            problems.append(f"patience={self.patience} 超出范围 1~100")
        if not (0.0 < float(self.val_ratio) < 1.0):
            problems.append(f"val_ratio={self.val_ratio} 必须在 (0, 1) 内")
        if not (0 <= int(self.num_workers) <= 16):
            problems.append(f"num_workers={self.num_workers} 超出范围 0~16")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            problems.append(f"device={self.device!r} 不在 auto/cpu/cuda/mps")
        if not (float(self.head_lr) > 0 and float(self.lr) > 0):
            problems.append("head_lr / lr 必须 > 0")
        if not (0.0 <= float(self.label_smoothing) < 1.0):
            problems.append("label_smoothing 必须在 [0, 1) 内")
        if float(self.weight_decay) < 0:
            problems.append("weight_decay 必须 ≥ 0")
        if int(self.seed) < 0:
            problems.append(f"seed={self.seed} 必须 ≥ 0")
        if len(self.bg_color) != 3 or any(not (0 <= int(c) <= 255) for c in self.bg_color):
            problems.append("bg_color 必须是 3 个 0~255 的整数")
        if float(self.bg_threshold) < 0:
            problems.append(f"bg_threshold={self.bg_threshold} 必须 ≥ 0")
        return problems

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bg_color"] = [int(c) for c in self.bg_color]
        return d


# --------------------------------------------------------------------------------------
# 内部：单次拟合循环
# --------------------------------------------------------------------------------------


@dataclass
class _FitResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_report: dict[str, Any] | None
    best_val_loss: float
    last_report: dict[str, Any] | None
    last_y_true: list[int]
    last_y_pred: list[int]
    stop_reason: str

    @property
    def epochs(self) -> int:
        return len(self.history)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _check_stop(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise TrainError("CANCELLED", "训练已取消")


def _make_loaders(
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[SampleDataset, DataLoader, DataLoader]:
    train_ds = SampleDataset(
        train_items,
        cfg.image_size,
        train=True,
        seed=cfg.seed,
        bg_color=cfg.bg_color,
        bg_threshold=cfg.bg_threshold,
    )
    val_ds = SampleDataset(
        val_items,
        cfg.image_size,
        train=False,
        bg_color=cfg.bg_color,
        bg_threshold=cfg.bg_threshold,
    )
    sampler = make_weighted_sampler(train_ds.labels, seed=cfg.seed)
    pin = device.type == "cuda"
    # BatchNorm 在 train 模式下不接受 batch=1，样本多于一个 batch 时丢弃尾批
    drop_last = len(train_ds) > cfg.batch_size
    workers = int(cfg.num_workers)
    # 多进程读图时必须有超时（worker 卡死不能拖死训练），并在 worker 里关掉 OpenCV 线程池
    mp_kwargs: dict[str, Any] = (
        {"timeout": LOADER_TIMEOUT_S, "worker_init_fn": worker_init_fn} if workers > 0 else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin,
        drop_last=drop_last,
        **mp_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        **mp_kwargs,
    )
    return train_ds, train_loader, val_loader


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    stop_event: threading.Event | None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for xb, yb in loader:
        _check_stop(stop_event)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with ctx:
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item()) * int(yb.shape[0])
        count += int(yb.shape[0])
    return total / count if count else 0.0


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    stop_event: threading.Event | None,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in loader:
        _check_stop(stop_event)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with ctx:
            logits = model(xb)
        loss = criterion(logits.float(), yb)
        total += float(loss.item()) * int(yb.shape[0])
        count += int(yb.shape[0])
        y_true.extend(int(v) for v in yb.cpu().tolist())
        y_pred.extend(int(v) for v in logits.argmax(dim=1).cpu().tolist())
    return (total / count if count else 0.0), y_true, y_pred


def _fit(
    train_items: Sequence[Item],
    val_items: Sequence[Item],
    classes: Sequence[str],
    cfg: TrainConfig,
    *,
    device: torch.device,
    time_budget_s: float,
    stop_event: threading.Event | None,
    select_best: bool,
    on_epoch: Callable[[int, dict[str, Any], bool], None] | None = None,
    on_best: Callable[[nn.Module, int, dict[str, Any]], None] | None = None,
) -> tuple[_FitResult, nn.Module]:
    """按配方训练一次（阶段 A 只训头 → 阶段 B 全网微调）。

    Args:
        select_best: True 时按验证 macro-F1 选 best（相等取更低 val_loss）并启用早停；
            False（交叉验证）时跑满 epoch，只受时间预算约束，结果取最后一轮。
        on_epoch: ``(epoch, history_entry, is_best)``，每轮验证后回调。
        on_best: ``(model, epoch, report)``，best 刷新时回调（用于落盘 checkpoint）。
    """
    _check_stop(stop_event)
    train_ds, train_loader, val_loader = _make_loaders(train_items, val_items, cfg, device)
    model = models.build_model(cfg.backbone, len(classes), pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    head_epochs = min(int(cfg.head_epochs), int(cfg.max_epochs))
    stage = "A" if head_epochs > 0 else "B"
    if stage == "A":
        models.freeze_backbone(model, cfg.backbone, True)
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            models.head_parameters(model, cfg.backbone),
            lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        models.freeze_backbone(model, cfg.backbone, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if stage == "B":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.max_epochs)
        )

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_report: dict[str, Any] | None = None
    last_report: dict[str, Any] | None = None
    last_true: list[int] = []
    last_pred: list[int] = []
    bad_epochs = 0
    stop_reason = "max_epochs"
    t_start = time.monotonic()

    for epoch in range(1, int(cfg.max_epochs) + 1):
        _check_stop(stop_event)
        if stage == "A" and epoch == head_epochs + 1:
            stage = "B"
            models.freeze_backbone(model, cfg.backbone, False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(cfg.max_epochs) - head_epochs)
            )
            log.info("进入阶段 B：解冻全网微调", extra={"epoch": epoch, "lr": cfg.lr})
        t0 = time.monotonic()
        train_ds.set_epoch(epoch)
        lr_now = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, stop_event
        )
        val_loss, y_true, y_pred = _evaluate(model, val_loader, device, use_amp, stop_event)
        rep = metrics.report(y_true, y_pred, classes)
        if scheduler is not None:
            scheduler.step()
        entry = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(rep["accuracy"]),
            "val_macro_f1": float(rep["macro_f1"]),
            "lr": lr_now,
            "seconds": round(time.monotonic() - t0, 3),
        }
        history.append(entry)
        last_report, last_true, last_pred = rep, y_true, y_pred

        is_best = False
        f1 = float(rep["macro_f1"])
        if select_best:
            if f1 > best_f1 or (f1 == best_f1 and val_loss < best_val_loss):
                is_best = True
                best_f1, best_val_loss, best_epoch, best_report = f1, float(val_loss), epoch, rep
                bad_epochs = 0
                if on_best is not None:
                    on_best(model, epoch, rep)
            else:
                bad_epochs += 1
        log.info(
            "epoch 完成",
            extra={
                "epoch": epoch,
                "stage": stage,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(rep["accuracy"], 4),
                "val_f1": round(f1, 4),
                "best": is_best,
            },
        )
        if on_epoch is not None:
            on_epoch(epoch, entry, is_best)

        if select_best and bad_epochs >= int(cfg.patience):
            stop_reason = "early stop"
            log.info("早停", extra={"epoch": epoch, "patience": cfg.patience})
            break
        if time.monotonic() - t_start >= time_budget_s and epoch < int(cfg.max_epochs):
            stop_reason = "time limit"
            log.warning(
                "达到训练时长上限，停止",
                extra={"epoch": epoch, "budget_s": round(time_budget_s, 1)},
            )
            break

    if not select_best:
        # 交叉验证模式：结果取最后一轮
        best_epoch = len(history)
        best_report = last_report
        best_val_loss = float(history[-1]["val_loss"]) if history else float("inf")
    result = _FitResult(
        history=history,
        best_epoch=best_epoch,
        best_report=best_report,
        best_val_loss=best_val_loss,
        last_report=last_report,
        last_y_true=last_true,
        last_y_pred=last_pred,
        stop_reason=stop_reason,
    )
    return result, model


# --------------------------------------------------------------------------------------
# run：模型目录一键训练
# --------------------------------------------------------------------------------------


def _read_meta_tolerant(path: Path) -> dict[str, Any]:
    """读 ``meta.json``；文件缺失、损坏（非 JSON）或不是对象时告警并按空字典处理。"""
    try:
        meta = read_json(path, default={})
    except (ValueError, OSError) as exc:  # json.JSONDecodeError 是 ValueError 的子类
        log.warning("meta.json 无法解析，按空处理", extra={"path": str(path), "err": str(exc)})
        return {}
    if not isinstance(meta, dict):
        log.warning("meta.json 不是对象，按空处理", extra={"path": str(path)})
        return {}
    return meta


def _resolve_classes(
    model_dir: Path,
    classes: Sequence[str] | None,
    class_names: Sequence[str] | None,
) -> list[str]:
    """确定参与训练的类别顺序。非法类别名一律 ``TrainError("VALIDATION")``。"""
    if classes is not None:
        names = [str(c) for c in classes]
        if not names:
            raise TrainError("VALIDATION", "classes 不能为空")
        if len(set(names)) != len(names):
            raise TrainError("VALIDATION", f"classes 含重复类别: {names}", {"classes": names})
        try:
            list_model_samples(model_dir, names)  # 只做类别名合法性检查
        except ValueError as exc:
            raise TrainError("VALIDATION", str(exc), {"classes": names}) from exc
        return names
    if class_names:
        order = [str(c) for c in class_names]
    else:
        meta = _read_meta_tolerant(model_dir / META_FILE)
        order = [str(c) for c in (meta.get("classes") or []) if str(c)]
        samples_dir = model_dir / "samples"
        if samples_dir.is_dir():
            for p in sorted(samples_dir.iterdir()):
                if p.is_dir() and not p.name.startswith(".") and p.name not in order:
                    order.append(p.name)
    try:
        available = list_model_samples(model_dir, order)
    except ValueError as exc:
        raise TrainError("VALIDATION", str(exc), {"classes": order}) from exc
    return [c for c in order if available[c]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@contextlib.contextmanager
def _file_logging(path: Path):
    """给 ``oresorter.ml`` logger 临时挂一个 FileHandler（train.log），退出时移除。

    挂在父 logger 上，models（预训练权重是否加载到）、dataset 等模块的日志也会进 train.log。
    """
    ensure_dir(path.parent)
    target = logging.getLogger(_ML_LOGGER_NAME)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(KVFormatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    prev_level = target.level
    if target.getEffectiveLevel() > logging.INFO:
        target.setLevel(logging.INFO)
    target.addHandler(handler)
    try:
        yield
    finally:
        target.removeHandler(handler)
        handler.close()
        target.setLevel(prev_level)


def _remove_stale_derived(train_dir: Path) -> None:
    """删除上一轮训练遗留的派生产物（onnx / plan / 引擎元信息），它们与即将写入的新 best.pt 不配套。"""
    removed = [name for name in DERIVED_ARTIFACTS if (train_dir / name).exists()]
    for name in removed:
        (train_dir / name).unlink(missing_ok=True)
    if removed:
        log.info("已清理上一轮的派生产物", extra={"removed": removed})


def run(
    model_dir: str | Path,
    classes: Sequence[str] | None,
    cfg: TrainConfig,
    *,
    class_names: Sequence[str] | None = None,
    progress_cb: ProgressCb | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """对一个模型目录执行一键训练。

    Args:
        model_dir: ``{data_dir}/models/{id}``。
        classes: 参与训练的类别；省略 = 目录下有样本的类别（顺序按 ``class_names``，
            再退到 meta.json 的 classes，最后是目录名）。
        cfg: 训练配方。
        class_names: 系统配置的类别全集（决定类别顺序）。
        progress_cb: 进度回调，收到 ``{"progress", "stage", "status", "epoch", "val_macro_f1", "val_accuracy", ...}``。
        stop_event: 置位后尽快停止并抛 ``TrainError("CANCELLED")``（已写的 best.pt 保留）。

    Returns:
        ``{"metrics_summary", "epochs", "best_epoch", "duration_s", "classes", "sample_counts", "stop_reason",
        "artifacts": {"best": True, "onnx": bool[, "onnx_error": str]}}``。

    Raises:
        TrainError: NOT_FOUND（目录不存在）、VALIDATION（样本不足 ``detail.insufficient``、类别数 < 2、
            参数越界）、CANCELLED、INTERNAL（其它任何异常都会被包装，并在 progress.json 记 ``status=failed``）。
    """
    cfg.validate()
    mdir = Path(model_dir)
    if not mdir.is_dir():
        raise TrainError("NOT_FOUND", f"模型目录不存在: {mdir}", {"model_dir": str(mdir)})
    train_dir = ensure_dir(mdir / TRAIN_DIR)
    progress_path = train_dir / PROGRESS_FILE
    # 清掉上一轮的状态文件，避免本轮校验失败时目录里还留着 status=done 的旧 progress.json
    (train_dir / ERROR_FILE).unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    t_start = time.monotonic()
    max_epochs = int(cfg.max_epochs)
    state: dict[str, Any] = {"best_epoch": 0, "best_f1": 0.0, "progress": 0.0, "classes": []}

    def emit(progress: float, stage: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "progress": round(float(progress), 4),
            "stage": stage,
            "status": extra.pop("status", "running"),
            "epochs": max_epochs,
            "elapsed_s": round(time.monotonic() - t_start, 1),
            "best_epoch": state["best_epoch"],
            "classes": list(state["classes"]),
        }
        payload.update(extra)
        state["progress"] = payload["progress"]
        try:
            atomic_write_json(progress_path, payload)
        except Exception:
            log.exception("写 progress.json 失败")
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                log.exception("progress_cb 异常")

    with _file_logging(train_dir / LOG_FILE):
        try:
            return _run_in_dir(
                mdir,
                train_dir,
                classes,
                class_names,
                cfg,
                stop_event=stop_event,
                emit=emit,
                state=state,
                t_start=t_start,
            )
        except TrainError as exc:
            status = "cancelled" if exc.code == "CANCELLED" else "failed"
            log.warning("训练中止", extra={"code": exc.code, "reason": exc.message})
            emit(state["progress"], exc.code.lower(), status=status, error=exc.to_dict())
            raise
        except Exception as exc:
            log.exception("训练异常")
            err = TrainError(
                "INTERNAL",
                f"训练异常: {type(exc).__name__}: {exc}",
                {"type": type(exc).__name__},
            )
            emit(state["progress"], "error", status="failed", error=err.to_dict())
            raise err from exc


def _run_in_dir(
    mdir: Path,
    train_dir: Path,
    classes: Sequence[str] | None,
    class_names: Sequence[str] | None,
    cfg: TrainConfig,
    *,
    stop_event: threading.Event | None,
    emit: Callable[..., None],
    state: dict[str, Any],
    t_start: float,
) -> dict[str, Any]:
    """``run()`` 的主体：校验 → 划分 → 训练 → 产物。所有异常由 ``run()`` 统一转成 TrainError 并记进度。"""
    max_epochs = int(cfg.max_epochs)

    # 1. 类别与样本校验
    class_list = _resolve_classes(mdir, classes, class_names)
    state["classes"] = class_list
    samples = list_model_samples(mdir, class_list)
    sample_counts = {c: len(v) for c, v in samples.items()}
    insufficient = {c: n for c, n in sample_counts.items() if n < int(cfg.min_samples_per_class)}
    if insufficient:
        raise TrainError(
            "VALIDATION",
            "样本不足: "
            + ", ".join(
                f"{c} 仅 {n} 张（需 ≥ {cfg.min_samples_per_class}）"
                for c, n in insufficient.items()
            ),
            {"insufficient": insufficient, "min_samples_per_class": cfg.min_samples_per_class},
        )
    if len(class_list) < 2:
        raise TrainError(
            "VALIDATION",
            f"至少需要 2 个有样本的类别，当前 {len(class_list)} 个",
            {"classes": class_list},
        )
    _seed_everything(cfg.seed)
    device = models.select_device(cfg.device)
    log.info(
        "开始训练",
        extra={
            "model_dir": str(mdir),
            "classes": class_list,
            "sample_counts": sample_counts,
            "device": str(device),
            "backbone": cfg.backbone,
            "image_size": cfg.image_size,
            "pretrained": cfg.pretrained,
            "max_epochs": max_epochs,
            "budget_min": cfg.max_train_minutes,
        },
    )

    # 2. 分层划分
    items: list[Item] = [(p, idx) for idx, c in enumerate(class_list) for p in samples[c]]
    train_items, val_items = stratified_split(items, cfg.val_ratio, cfg.seed)
    val_labels = {y for _, y in val_items}
    no_val = [c for i, c in enumerate(class_list) if i not in val_labels]
    if no_val:
        # 只有 1 张样本的类别全部进训练集：验证集上该类 support=0、F1 记 0（对各 epoch 一致，不影响 best 选择）
        log.warning("以下类别没有验证样本，其验证 F1 按 0 计", extra={"classes": no_val})
    log.info("划分完成", extra={"train": len(train_items), "val": len(val_items)})
    emit(0.0, "prepare", epoch=0)

    best_path = train_dir / BEST_FILE

    def on_best(model: nn.Module, epoch: int, rep: dict[str, Any]) -> None:
        if state["best_epoch"] == 0:
            # 第一次写新 best.pt：先清掉旧 onnx/plan，目录里不留与新模型不配套的派生产物
            _remove_stale_derived(train_dir)
        meta = {
            "backbone": cfg.backbone,
            "classes": class_list,
            "image_size": cfg.image_size,
            "algo": models.algo_name(cfg.backbone),
            "trained_at": _now_iso(),
            "sample_counts": sample_counts,
            "epoch": epoch,
            "val_macro_f1": float(rep["macro_f1"]),
            "val_accuracy": float(rep["accuracy"]),
        }
        models.save_checkpoint(best_path, model, meta)
        state["best_epoch"] = epoch
        state["best_f1"] = float(rep["macro_f1"])

    def on_epoch(epoch: int, entry: dict[str, Any], is_best: bool) -> None:
        emit(
            epoch / max_epochs,
            f"epoch {epoch}/{max_epochs} val_f1={entry['val_macro_f1']:.2f}",
            epoch=epoch,
            val_macro_f1=entry["val_macro_f1"],
            val_accuracy=entry["val_accuracy"],
            train_loss=entry["train_loss"],
            val_loss=entry["val_loss"],
            is_best=is_best,
        )

    # 3~5. 训练
    result, _ = _fit(
        train_items,
        val_items,
        class_list,
        cfg,
        device=device,
        time_budget_s=float(cfg.max_train_minutes) * 60.0,
        stop_event=stop_event,
        select_best=True,
        on_epoch=on_epoch,
        on_best=on_best,
    )
    if result.best_report is None or not best_path.exists():
        raise TrainError("INTERNAL", "未完成任何 epoch，没有可用模型")

    duration_s = round(time.monotonic() - t_start, 3)
    if result.stop_reason == "time limit":
        emit(len(result.history) / max_epochs, "time limit", epoch=len(result.history))

    # 6. 产物：先导出 ONNX（结果记入 metrics.json），再写 metrics.json、合并 meta.json
    best_model, _ = models.load_checkpoint(best_path, device="cpu")
    onnx_ok, onnx_err = _try_export_onnx(best_model, cfg.image_size, train_dir / ONNX_FILE)
    artifacts: dict[str, Any] = {"best": True, "onnx": onnx_ok}
    if onnx_err:
        artifacts["onnx_error"] = onnx_err

    metrics_doc: dict[str, Any] = dict(result.best_report)
    metrics_doc.update(
        {
            "history": result.history,
            "duration_s": duration_s,
            "sample_counts": sample_counts,
            "config": cfg.to_dict(),
            "best_epoch": result.best_epoch,
            "epochs": result.epochs,
            "classes": class_list,
            "device": str(device),
            "stop_reason": result.stop_reason,
            "train_size": len(train_items),
            "val_size": len(val_items),
            "trained_at": _now_iso(),
            "artifacts": artifacts,
        }
    )
    atomic_write_json(train_dir / METRICS_FILE, metrics_doc)

    summary = metrics.summary(result.best_report)
    _merge_meta(mdir / META_FILE, cfg, class_list, sample_counts, summary)

    emit(
        1.0,
        f"done best_epoch={result.best_epoch} val_f1={summary['macro_f1']:.2f}",
        status="done",
        epoch=result.epochs,
        val_macro_f1=summary["macro_f1"],
        val_accuracy=summary["val_accuracy"],
        artifacts=artifacts,
    )
    log.info(
        "训练完成",
        extra={
            "best_epoch": result.best_epoch,
            "epochs": result.epochs,
            "duration_s": duration_s,
            "val_f1": round(summary["macro_f1"], 4),
            "onnx": onnx_ok,
            "stop_reason": result.stop_reason,
        },
    )
    return {
        "metrics_summary": summary,
        "epochs": result.epochs,
        "best_epoch": result.best_epoch,
        "duration_s": duration_s,
        "classes": class_list,
        "sample_counts": sample_counts,
        "stop_reason": result.stop_reason,
        "artifacts": artifacts,
    }


def _merge_meta(
    meta_path: Path,
    cfg: TrainConfig,
    class_list: Sequence[str],
    sample_counts: dict[str, int],
    summary: dict[str, float],
) -> None:
    """把训练结果合并进 ``meta.json``（read → merge → 原子写）。

    ``sample_counts`` 按类别合并而不是整体替换：store 初始化的是全部 ``class_names`` 的计数，
    只训 A、B 两类不能把 C~G 的计数抹掉。文件损坏时告警并重建（只含训练字段）。
    """
    meta = _read_meta_tolerant(meta_path)
    counts = (
        dict(meta.get("sample_counts") or {}) if isinstance(meta.get("sample_counts"), dict) else {}
    )
    counts.update({str(c): int(n) for c, n in sample_counts.items()})
    try:
        version = int(meta.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    meta.update(
        {
            "trained_at": _now_iso(),
            "algo": models.algo_name(cfg.backbone),
            "image_size": int(cfg.image_size),
            "classes": list(class_list),
            "sample_counts": counts,
            "metrics_summary": summary,
            "version": version + 1,
        }
    )
    atomic_write_json(meta_path, meta)


def _try_export_onnx(model: nn.Module, image_size: int, path: Path) -> tuple[bool, str | None]:
    """尝试导出 ONNX；export 模块缺失或导出失败只记 warning，返回 ``(是否成功, 错误描述)``。

    失败时保证 ``path`` 不存在（不留下与当前 best.pt 不配套的文件）。
    """
    try:
        from oresorter.ml.export import to_onnx
    except ImportError as exc:
        log.warning("export 模块不可用，跳过 ONNX 导出", extra={"error": str(exc)})
        path.unlink(missing_ok=True)
        return False, f"export 模块不可用: {exc}"
    try:
        to_onnx(model, image_size, path)
    except Exception as exc:
        log.warning("ONNX 导出失败", extra={"error": f"{type(exc).__name__}: {exc}"})
        path.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"
    if not path.exists():
        log.warning("ONNX 导出未产生文件", extra={"path": str(path)})
        return False, "导出后文件不存在"
    log.info("ONNX 已导出", extra={"path": str(path)})
    return True, None


# --------------------------------------------------------------------------------------
# 交叉验证（供 evaluate / cli eval 调用）
# --------------------------------------------------------------------------------------


def cross_validate(
    items: Sequence[Item],
    classes: Sequence[str],
    cfg: TrainConfig,
    k: int = 5,
    progress_cb: ProgressCb | None = None,
    *,
    loo_threshold: int = LOO_THRESHOLD,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """K 折交叉验证：每折按同样配方训练（跑满 ``cfg.max_epochs``，不按验证折选 best 以免泄漏），
    在验证折上预测，汇总全部折的预测生成一个总 ``metrics.report``。

    样本数 < ``loo_threshold``（默认 50）时改为留一法（k = 样本数）。总时长预算 ``cfg.max_train_minutes``
    平均分到每折。

    Returns:
        ``report(...)`` 的字段 + ``{"k", "mode", "folds": [{"fold", "n_train", "n_val", "accuracy", "epochs"}],
        "duration_s", "sample_counts", "config"}``。
    """
    cfg.validate()
    class_list = [str(c) for c in classes]
    if len(class_list) < 2:
        raise TrainError("VALIDATION", "至少需要 2 个类别", {"classes": class_list})
    n = len(items)
    if n < 2:
        raise TrainError("VALIDATION", f"样本太少无法交叉验证: {n}", {"n": n})
    labels = [int(y) for _, y in items]
    if min(labels) < 0 or max(labels) >= len(class_list):
        raise TrainError("VALIDATION", "标签索引越界", {"n_classes": len(class_list)})
    if k < 2:
        raise TrainError("VALIDATION", f"k 必须 ≥ 2，得到 {k}", {"k": k})
    k_eff = n if n < loo_threshold else min(int(k), n)
    mode = "loo" if k_eff == n else "kfold"
    _seed_everything(cfg.seed)
    device = models.select_device(cfg.device)
    folds = kfold_split(items, k_eff, cfg.seed)
    budget_each = float(cfg.max_train_minutes) * 60.0 / k_eff
    sample_counts = {c: labels.count(i) for i, c in enumerate(class_list)}
    log.info(
        "交叉验证开始",
        extra={"n": n, "k": k_eff, "mode": mode, "device": str(device), "classes": class_list},
    )
    t_start = time.monotonic()
    all_true: list[int] = []
    all_pred: list[int] = []
    fold_results: list[dict[str, Any]] = []
    for i, (tr, va) in enumerate(folds):
        _check_stop(stop_event)
        torch.manual_seed(cfg.seed + i)
        result, _ = _fit(
            tr,
            va,
            class_list,
            cfg,
            device=device,
            time_budget_s=budget_each,
            stop_event=stop_event,
            select_best=False,
        )
        acc = float(result.last_report["accuracy"]) if result.last_report else 0.0
        all_true.extend(result.last_y_true)
        all_pred.extend(result.last_y_pred)
        fold_results.append(
            {
                "fold": i,
                "n_train": len(tr),
                "n_val": len(va),
                "accuracy": acc,
                "epochs": result.epochs,
            }
        )
        log.info("折完成", extra={"fold": i, "k": k_eff, "acc": round(acc, 4)})
        if progress_cb is not None:
            try:
                progress_cb(
                    {
                        "progress": round((i + 1) / k_eff, 4),
                        "stage": f"fold {i + 1}/{k_eff} acc={acc:.2f}",
                        "fold": i,
                        "k": k_eff,
                        "accuracy": acc,
                    }
                )
            except Exception:
                log.exception("progress_cb 异常")
    rep = metrics.report(all_true, all_pred, class_list)
    rep.update(
        {
            "k": k_eff,
            "mode": mode,
            "folds": fold_results,
            "duration_s": round(time.monotonic() - t_start, 3),
            "sample_counts": sample_counts,
            "config": cfg.to_dict(),
        }
    )
    return rep


# --------------------------------------------------------------------------------------
# 子进程入口
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m oresorter.ml.train", description="矿石分类模型一键训练（子进程入口）"
    )
    p.add_argument("--model-dir", required=True, help="模型目录 {data_dir}/models/{id}")
    p.add_argument("--classes", default=None, help="参与训练的类别，逗号分隔；省略 = 有样本的类别")
    p.add_argument("--config", default=None, help="yaml 配置（取 ml 分组、class_names、bg_color）")
    p.add_argument("--epochs", type=int, default=None, help="覆盖 max_epochs")
    p.add_argument("--head-epochs", type=int, default=None)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=f"覆盖 image_size，只允许 {list(ALLOWED_IMAGE_SIZES)}（除非 --allow-any-image-size）",
    )
    p.add_argument(
        "--allow-any-image-size",
        action="store_true",
        help="放开 image_size 的取值限制（仅测试用；训练出的模型与运行时 ml.image_size 可能不匹配）",
    )
    p.add_argument("--backbone", default=None, choices=list(models.BACKBONES))
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--min-samples", type=int, default=None, help="覆盖 min_samples_per_class")
    p.add_argument("--max-minutes", type=float, default=None, help="覆盖 max_train_minutes")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="不加载离线预训练权重（测试用）")
    p.add_argument(
        "--nice",
        type=int,
        default=0,
        help="降低本进程优先级（docs/01 §6.4 建议训练子进程 nice 10）；0 = 不改",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def _apply_nice(increment: int) -> None:
    """按 ``--nice`` 降低进程优先级；非 POSIX 或失败只告警。"""
    if increment <= 0:
        return
    try:
        new = os.nice(increment)
        log.info("已降低训练进程优先级", extra={"nice": new})
    except (AttributeError, OSError) as exc:
        log.warning("设置 nice 失败", extra={"error": str(exc)})


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。退出码 0 成功 / 1 失败。

    失败时错误 JSON 输出到 stderr，并写 ``train/error.json``（模型目录本身不存在时不落盘，
    不凭空创建假模型目录）。
    """
    from oresorter.log import setup_logging

    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    setup_logging(args.log_level)
    _apply_nice(int(args.nice))
    model_dir = Path(args.model_dir)
    train_dir = model_dir / TRAIN_DIR
    class_names: list[str] | None = None
    overrides: dict[str, Any] = {
        "max_epochs": args.epochs,
        "head_epochs": args.head_epochs,
        "image_size": args.image_size,
        "backbone": args.backbone,
        "device": args.device,
        "batch_size": args.batch_size,
        "min_samples_per_class": args.min_samples,
        "max_train_minutes": args.max_minutes,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    if args.no_pretrained:
        overrides["pretrained"] = False
    strict_size = not args.allow_any_image_size

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        log.warning("收到信号，准备停止训练", extra={"signal": signum})
        stop_event.set()

    # 父进程 terminate() 时优雅停止；结束后恢复原 handler（便于在同一进程内复用）
    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):  # 非主线程时不可注册
            previous[sig] = signal.signal(sig, _on_signal)

    try:
        if args.config:
            from oresorter.config import load as load_config

            app_cfg = load_config(args.config)
            class_names = list(app_cfg.system.class_names)
            overrides["bg_color"] = tuple(int(c) for c in app_cfg.capture.bg_color)
            overrides["bg_threshold"] = float(app_cfg.capture.bg_threshold)
            cfg = TrainConfig.from_ml_config(app_cfg.ml, strict_image_size=strict_size, **overrides)
        else:
            cfg = TrainConfig.from_ml_config(MlConfig(), strict_image_size=strict_size, **overrides)
        classes = (
            [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None
        )
        result = run(model_dir, classes, cfg, class_names=class_names, stop_event=stop_event)
    except TrainError as exc:
        _write_error(train_dir, exc.to_dict(), to_disk=model_dir.is_dir())
        return 1
    except Exception as exc:  # 配置加载等其它错误
        log.exception("训练入口异常")
        _write_error(
            train_dir,
            {"code": "INTERNAL", "message": str(exc), "detail": {"type": type(exc).__name__}},
            to_disk=model_dir.is_dir(),
        )
        return 1
    finally:
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError):
                signal.signal(sig, handler)
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


def _write_error(train_dir: Path, err: dict[str, Any], *, to_disk: bool = True) -> None:
    log.error("训练失败", extra={"code": err.get("code"), "reason": err.get("message")})
    if to_disk:
        try:
            atomic_write_json(train_dir / ERROR_FILE, {**err, "ts": _now_iso()})
        except Exception:
            log.exception("写 error.json 失败")
    sys.stderr.write(json.dumps(err, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
