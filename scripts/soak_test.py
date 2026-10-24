#!/usr/bin/env python
"""老化测试：Mock 相机 + Mock 下位机下长时间运行分选流水线，周期记录 RSS / 线程 / 吞吐 / 丢块 / 延迟。

对应 docs/02 §4 的稳定性验收（Mock 下 48 h 连续运行，RSS 增长 < 50 MB、无线程泄漏、丢块 0）与 §5 老化计划。

用法::

    .venv/bin/python scripts/soak_test.py --duration 48h                      # 默认 config/dev.yaml，每 60 s 采样
    .venv/bin/python scripts/soak_test.py --duration 1h --interval 30 --speed 0 --out /tmp/soak.csv
    .venv/bin/python scripts/soak_test.py --duration 10m --model-dir data/models/3

流程：读取配置（强制 ``mcu_kind=mock``、相机 ``kind=mock``、``auto_run_on_boot=false``，``--speed`` 覆盖
Mock 相机速度倍率；column 背景未学习时回退 color 模式）→ ``AppContext.start()`` → 准备模型（``--model-dir``
导入 / 复用已部署模型 / 用 synthetic 样本快速训一个 128 px、2 epoch 的临时模型）→ ``RuntimeService.deploy_model``
→ ``RuntimeService.start()`` → 每 ``--interval`` 秒采样一行写入 JSONL（或 ``.csv``）→ 到时 / SIGINT / SIGTERM /
出现故障或运行状态离开 running 时**立刻**结束（故障与状态机监听直接唤醒采样循环，不等下一个采样点）→
``stop()`` → 删除临时模型并恢复原部署编号 → 打印汇总并写 ``soak_summary.json``。

临时模型槽位在创建之前就登记到清理计划：训练 / 导入 / 部署任何一步失败（含 Ctrl-C 中断训练）都会在收尾时
删除该槽位，不在用户模型库留下残留。启动与模型准备阶段的瞬时故障（例如残留的部署模型权重损坏）只记入
``notes`` / ``startup_faults``，不算“运行期间出现故障”；运行前仍处于 fault 的情况 ``runtime.start()`` 本身会拒绝。

退出码：``0`` 合格；``2`` 超限（RSS 增长 > ``--rss-limit-mb``、线程泄漏、运行期间出现故障、结束时运行状态不是
running 或块数停止增长、或给了 ``--max-dropped`` 且丢块超过）；``1`` 错误（参数非法、训练/部署/启动失败、
运行开始前被中断、异常）。``--help`` 退出 0。

信号：第一次 SIGINT / SIGTERM 优雅结束并输出汇总；第二次恢复默认处理器并抛 ``KeyboardInterrupt`` 强制
退出（退出码 1）。记录文件已存在时拒绝运行，需显式 ``--overwrite``。

每行记录：墙钟时间、运行秒数、RSS（MB）、Python 线程数、OS 线程数、CPU%、运行状态、
``ctx.stats.snapshot()``（p50/p99/max 延迟、块数、丢块、颗粒、吹气、每分钟速率）、各流水线 ``Pipeline.stats()``
（队列长度、丢块、拼接器 dropped/lost/truncated、errors、发送计数、healthy）、相机计数、下位机 counters、故障码。

进程内调用：``main([...]) -> int``（测试用，不另起子进程；根 logger 已有 handler 时不改动日志配置）。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# scripts/ 不是包：把仓库根目录加进 sys.path 以便直接运行
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

from oresorter import __version__
from oresorter.config import AppConfig, ConfigError, ConfigService, load, parse_config
from oresorter.data.samples import StoreError
from oresorter.data.store import ModelStore
from oresorter.fsutil import atomic_write_json, ensure_dir
from oresorter.log import get_logger, setup_logging
from oresorter.ml import train as T
from oresorter.ml.dataset import load_rgba
from oresorter.ml.train import TrainError
from oresorter.runtime.context import AppContext
from oresorter.runtime.jobs import JobStatus
from oresorter.runtime.state import Event, RunState
from oresorter.vision.background import background_path
from oresorter.vision.synthetic import make_sample_set

log = get_logger("oresorter.scripts.soak_test")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LIMIT = 2

DEFAULT_CONFIG = ROOT / "config" / "dev.yaml"
DEFAULT_INTERVAL_S = 60.0
DEFAULT_SPEED = 1.0
DEFAULT_RSS_LIMIT_MB = 50.0
MIN_INTERVAL_S = 0.1
MAX_SPEED = 1000.0
"""与 ``MockCameraConfig.speed`` 的上限一致。"""
SUMMARY_NAME = "soak_summary.json"
TEMP_MODEL_NAME = "soak-temp"
QUICK_IMAGE_SIZE = 128
QUICK_EPOCHS = 2
QUICK_PER_CLASS = 12
QUICK_MAX_CLASSES = 3
QUICK_BATCH_SIZE = 8
DEPLOY_TIMEOUT_S = 900.0
"""部署等待上限（Jetson 上含 TensorRT 引擎生成）。"""
DEPLOY_POLL_S = 0.5
THREAD_SETTLE_S = 2.0
"""停止后等待线程数回到基线的时限。"""
FINAL_SAMPLE_MIN_GAP_S = 0.5
"""结束时距上一次采样超过该间隔才补一次最终采样。"""
GROWTH_MIN_GAP_S = 1.0
"""``running_ok`` 判定中块数增长比较的最小时间间隔（再按最慢相机块周期的 3 倍放大，见 :func:`growth_gap_s`）。"""
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_DURATION_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


class SoakError(RuntimeError):
    """脚本自身的错误（参数、准备阶段失败）→ 退出码 1。"""


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------


def parse_duration(text: str) -> float:
    """``"90"`` / ``"90s"`` / ``"30m"`` / ``"1h"`` / ``"48h"`` / ``"2d"`` → 秒；非法或 ≤ 0 抛 :class:`SoakError`。"""
    m = _DURATION_RE.match(str(text))
    if m is None:
        raise SoakError(f"--duration 格式非法: {text!r}（支持 秒 / 30m / 1h / 48h / 2d）")
    seconds = float(m.group(1)) * _DURATION_UNITS[m.group(2).lower()]
    if not seconds > 0:
        raise SoakError(f"--duration 必须 > 0，得到 {text!r}")
    return seconds


class _Parser(argparse.ArgumentParser):
    """用法错误抛 :class:`SoakError`（退出码 1），不占用表示“超限”的退出码 2。"""

    def error(self, message: str) -> Any:
        raise SoakError(f"参数错误: {message}")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="soak_test",
        description="Mock 相机 + Mock 下位机长时间运行分选流水线，记录 RSS / 线程 / 丢块 / 延迟",
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="yaml 配置（默认 config/dev.yaml）"
    )
    p.add_argument("--duration", default="1h", help="运行时长：秒或 30m / 1h / 48h（默认 1h）")
    p.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S, help="采样间隔秒数（默认 60）"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="记录文件（.jsonl 或 .csv；默认 {data_dir}/logs/soak_YYYYmmdd_HHMMSS.jsonl；给目录则在其中按默认名创建）",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="已训练模型目录（含 train/best.pt）；省略则复用已部署模型，否则快速训一个临时模型",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Mock 相机速度倍率覆盖（默认 1.0 = 9 kHz 节奏；0 = 不限速）",
    )
    p.add_argument(
        "--rss-limit-mb",
        type=float,
        default=DEFAULT_RSS_LIMIT_MB,
        help="RSS 相对基线允许的最大增长 MB（默认 50；超过退出码 2）",
    )
    p.add_argument(
        "--max-dropped",
        type=int,
        default=None,
        help="可选：允许的最大丢块数（流水线 + 相机），超过退出码 2；默认不检查",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="--out 指向的记录文件已存在时允许覆盖（默认拒绝，避免误删上一次老化记录）",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help=f"日志级别 {'/'.join(LOG_LEVELS)}（默认 INFO，不区分大小写）",
    )
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析并校验参数；越界（含 inf / nan）抛 :class:`SoakError` 并说明原因，不静默截断。

    ``--interval`` 必须是有限数、≥ 0.1 s 且 ≤ 时长（否则整个运行只有结束时一次补采样，基线即结束，
    RSS 判定形同虚设）；``--rss-limit-mb`` 必须是有限正数；``--log-level`` 限定为 ``LOG_LEVELS`` 并统一
    为大写；``--out`` 指向已存在的文件且未给 ``--overwrite`` → 拒绝。
    """
    args = build_parser().parse_args(argv)
    args.duration_s = parse_duration(args.duration)
    if not math.isfinite(args.duration_s):
        raise SoakError(f"--duration 必须是有限值，得到 {args.duration!r}")
    if not (math.isfinite(args.interval) and args.interval >= MIN_INTERVAL_S):
        raise SoakError(f"--interval 必须是 ≥ {MIN_INTERVAL_S} s 的有限值，得到 {args.interval}")
    if args.interval > args.duration_s:
        raise SoakError(
            f"--interval ({args.interval:g} s) 不能大于 --duration ({args.duration_s:g} s)，"
            "否则运行期间没有任何周期采样，RSS 基线与结束值相同、判定无意义"
        )
    if not (math.isfinite(args.speed) and 0 <= args.speed <= MAX_SPEED):
        raise SoakError(f"--speed 必须在 0~{MAX_SPEED:g} 内，得到 {args.speed}")
    if not (math.isfinite(args.rss_limit_mb) and args.rss_limit_mb > 0):
        raise SoakError(f"--rss-limit-mb 必须是 > 0 的有限值，得到 {args.rss_limit_mb}")
    if args.max_dropped is not None and args.max_dropped < 0:
        raise SoakError(f"--max-dropped 必须 ≥ 0，得到 {args.max_dropped}")
    level = str(args.log_level).strip().upper()
    if level not in LOG_LEVELS:
        raise SoakError(f"--log-level 必须是 {'/'.join(LOG_LEVELS)} 之一，得到 {args.log_level!r}")
    args.log_level = level
    if not Path(args.config).is_file():
        raise SoakError(f"配置文件不存在: {args.config}")
    if args.model_dir is not None and not Path(args.model_dir).is_dir():
        raise SoakError(f"--model-dir 不是目录: {args.model_dir}")
    if args.out is not None and Path(args.out).is_file() and not args.overwrite:
        raise SoakError(f"记录文件已存在，拒绝覆盖（如确需覆盖请加 --overwrite）: {args.out}")
    return args


# --------------------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------------------


def build_config(path: Path, speed: float) -> tuple[AppConfig, list[str]]:
    """读取 yaml 并改成老化测试需要的形态：Mock 下位机、Mock 相机（速度倍率 ``speed``）、不自动运行；
    column 背景未学习的相机 → 回退 color 模式（背景色取 Mock 相机的合成背景色）。返回 (配置, 改动说明)。"""
    cfg = load(path)
    notes: list[str] = []
    data = cfg.model_dump(mode="json")
    if data["system"]["mcu_kind"] != "mock":
        notes.append(f"system.mcu_kind {data['system']['mcu_kind']} → mock")
    data["system"]["mcu_kind"] = "mock"
    if data["system"]["auto_run_on_boot"]:
        notes.append("system.auto_run_on_boot → false（由脚本控制 start/stop）")
    data["system"]["auto_run_on_boot"] = False
    for cam in data["cameras"]:
        if cam["kind"] != "mock":
            notes.append(f"camera {cam['id']} kind {cam['kind']} → mock")
            cam["kind"] = "mock"
        cam["mock"]["speed"] = float(speed)
    if data["capture"]["bg_mode"] == "column":
        missing = [
            cam["id"]
            for cam in data["cameras"]
            if not background_path(cfg.system.data_dir, cam["id"]).exists()
        ]
        if missing:
            bg = list(data["cameras"][0]["mock"]["bg_color"])
            data["capture"]["bg_mode"] = "color"
            data["capture"]["bg_color"] = bg
            notes.append(
                f"相机 {missing} 列背景未学习：capture.bg_mode column → color，bg_color={bg}"
            )
    new = parse_config(data)
    new.system.data_dir = cfg.system.data_dir
    return new, notes


def resolve_out_path(out: Path | None, data_dir: Path) -> Path:
    """``--out`` 为空 → ``{data_dir}/logs/soak_YYYYmmdd_HHMMSS.jsonl``；为目录 → 目录内按默认名。"""
    name = f"soak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if out is None:
        return Path(data_dir) / "logs" / name
    out = Path(out)
    if out.is_dir():
        return out / name
    return out.resolve()


# --------------------------------------------------------------------------------------
# 模型准备
# --------------------------------------------------------------------------------------


@dataclass
class ModelPlan:
    """本次老化使用的模型：编号、来源、是否结束后删除、运行前的部署编号（结束后恢复）。"""

    model_id: int
    source: str
    """``deployed``（复用已部署）/ ``model-dir``（导入）/ ``quick-train``（快速训练）。"""
    temporary: bool
    prev_deployed: int | None


def find_free_slot(store: ModelStore) -> int | None:
    """第一个未创建且目录不存在的模型编号；没有返回 None。"""
    for mid in range(1, store.max_models + 1):
        if not store.exists(mid) and not store.model_dir(mid).exists():
            return mid
    return None


def train_quick_model(
    store: ModelStore,
    model_id: int,
    class_names: Sequence[str],
    ml_cfg: Any,
    *,
    stop_event: threading.Event | None = None,
    seed: int = 0,
    image_size: int = QUICK_IMAGE_SIZE,
    epochs: int = QUICK_EPOCHS,
    per_class: int = QUICK_PER_CLASS,
) -> dict[str, Any]:
    """在模型库的 ``model_id`` 槽位创建临时模型，用合成样本（前 ≤ 3 个类别 × ``per_class`` 张）
    以 ``image_size`` / ``epochs`` 个 epoch、不加载预训练权重快速训练；返回 ``train.run`` 的结果。"""
    classes = [str(c) for c in class_names][:QUICK_MAX_CLASSES]
    if len(classes) < 2:
        raise SoakError(f"快速训练至少需要 2 个类别，class_names={list(class_names)}")
    store.create(model_id, TEMP_MODEL_NAME, "soak_test 临时模型（结束后自动删除）")
    repo = store.samples(model_id)
    with tempfile.TemporaryDirectory(prefix="soak_samples_") as tmp:
        paths = make_sample_set(Path(tmp), classes, per_class, rng=np.random.default_rng(seed))
        for cls, pngs in paths.items():
            for png in pngs:
                rgba = load_rgba(png)
                h, w = rgba.shape[:2]
                repo.add(cls, rgba, {"camera_id": 1, "bbox": [0, 0, w, h], "batch": "soak"})
    tcfg = T.TrainConfig.from_ml_config(
        ml_cfg,
        image_size=image_size,
        max_epochs=epochs,
        head_epochs=0,
        pretrained=False,
        min_samples_per_class=min(5, per_class),
        batch_size=QUICK_BATCH_SIZE,
        num_workers=0,
        seed=seed,
    )
    log.info(
        "快速训练临时模型",
        extra={
            "model_id": model_id,
            "classes": classes,
            "per_class": per_class,
            "image_size": image_size,
            "epochs": epochs,
            "device": tcfg.device,
        },
    )
    return T.run(
        store.model_dir(model_id),
        classes,
        tcfg,
        class_names=list(class_names),
        stop_event=stop_event,
    )


def import_model_dir(store: ModelStore, model_id: int, src: Path) -> None:
    """把外部模型目录（``train/best.pt`` 或直接 ``best.pt``）的训练产物复制进模型库槽位 ``model_id``。"""
    src = Path(src).resolve()
    if (src / "train" / "best.pt").is_file():
        train_src = src / "train"
    elif (src / "best.pt").is_file():
        train_src = src
    else:
        raise SoakError(f"--model-dir 下找不到 train/best.pt 或 best.pt: {src}")
    store.create(model_id, TEMP_MODEL_NAME, f"soak_test 导入 {src}（结束后自动删除）")
    dst = ensure_dir(store.train_dir(model_id))
    for p in train_src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    log.info("模型目录已导入临时槽位", extra={"src": str(src), "model_id": model_id})


# --------------------------------------------------------------------------------------
# 采样与记录
# --------------------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024.0 * 1024.0)


def take_sample(ctx: AppContext, proc: psutil.Process, elapsed_s: float) -> dict[str, Any]:
    """采集一行指标（全部原生类型，可直接 ``json.dumps``）。"""
    pipelines: dict[str, Any] = {}
    for cid, pipe in dict(ctx.pipelines).items():
        d = pipe.stats()
        d["healthy"] = bool(pipe.healthy())
        pipelines[str(cid)] = d
    cameras = {str(cid): cam.status() for cid, cam in ctx.cameras.items()}
    mcu = ctx.mcu.status.to_dict()
    return {
        "t": _now_iso(),
        "elapsed_s": round(float(elapsed_s), 2),
        "rss_mb": round(rss_mb(proc), 2),
        "threads": threading.active_count(),
        "os_threads": int(proc.num_threads()),
        "cpu_percent": round(float(proc.cpu_percent(None)), 1),
        "run_state": ctx.run_state,
        "stats": ctx.stats.snapshot(),
        "pipelines": pipelines,
        "cameras": cameras,
        "mcu": {
            k: mcu.get(k)
            for k in ("connected", "state", "queue_len", "counters", "ack_timeouts", "reconnects")
        },
        "faults": ctx.faults.codes(),
    }


def _sum_over(items: Sequence[dict[str, Any]], key: str) -> int:
    return sum(int(d.get(key, 0) or 0) for d in items)


def growth_gap_s(cfg: AppConfig, speed: float) -> float:
    """块数增长比较的最小间隔：``max(GROWTH_MIN_GAP_S, 3 × 最慢相机的块周期)``。

    块周期 = ``runtime_chunk_lines / (line_rate_hz × speed)``（``speed`` 为 Mock 速度倍率，0 = 不限速按 0 计）。
    9 kHz / 256 行 → 0.03 s 周期 → 1 s；100 Hz / 1024 行的极慢配置 → 10 s 周期 → 30 s，避免把慢节奏误判为停摆。
    """
    period = 0.0
    if speed > 0:
        for cam in cfg.cameras:
            period = max(period, cam.runtime_chunk_lines / (max(1, cam.line_rate_hz) * speed))
    return max(GROWTH_MIN_GAP_S, 3.0 * period)


class ChunkGrowth:
    """记录每行的 ``(elapsed_s, chunks_total)``，判断最后一行相对 **至少 gap 秒之前** 最近的一行块数是否增长。

    直接与上一行比较会被“补采样紧跟在周期采样之后”（SIGINT / 故障刚好落在采样点后几毫秒）误判为停摆，
    所以参照行必须足够早；没有足够早的参照行时返回 None（无法判断，不计失败）。
    """

    def __init__(self, gap_s: float, interval_s: float):
        self.gap_s = float(gap_s)
        keep = int(self.gap_s / max(float(interval_s), MIN_INTERVAL_S)) + 3
        self.points: deque[tuple[float, int]] = deque(maxlen=keep)

    def add(self, elapsed_s: float, chunks_total: int) -> None:
        self.points.append((float(elapsed_s), int(chunks_total)))

    def reference(self) -> tuple[float, int] | None:
        """最后一行之前、与之相隔 ≥ gap 秒的最近一行；没有返回 None。"""
        if len(self.points) < 2:
            return None
        t_last = self.points[-1][0]
        for t, c in reversed(list(self.points)[:-1]):
            if t <= t_last - self.gap_s:
                return (t, c)
        return None

    def grew(self) -> bool | None:
        """最后一行块数是否比参照行多；无参照行 → None。"""
        ref = self.reference()
        if ref is None:
            return None
        return self.points[-1][1] > ref[1]


def dropped_total(row: dict[str, Any]) -> int:
    """丢块合计：流水线环形队列丢弃（``stats.dropped_chunks``）+ 相机侧缓冲丢块。"""
    cams = list(row.get("cameras", {}).values())
    return int(row["stats"].get("dropped_chunks", 0)) + _sum_over(cams, "dropped_chunks")


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """JSONL 行 → CSV 平铺（多相机的流水线/相机计数求和）。"""
    st = row["stats"]
    pipes = list(row["pipelines"].values())
    cams = list(row["cameras"].values())
    mcu = row["mcu"]
    counters = mcu.get("counters") or {}
    return {
        "t": row["t"],
        "elapsed_s": row["elapsed_s"],
        "rss_mb": row["rss_mb"],
        "threads": row["threads"],
        "os_threads": row["os_threads"],
        "cpu_percent": row["cpu_percent"],
        "run_state": row["run_state"],
        "chunks_total": st["chunks_total"],
        "dropped_chunks": st["dropped_chunks"],
        "particles_total": st["particles_total"],
        "blown_total": st["blown_total"],
        "particles_per_min": st["particles_per_min"],
        "latency_p50_ms": st["chunk_latency_ms_p50"],
        "latency_p99_ms": st["chunk_latency_ms_p99"],
        "latency_max_ms": st["chunk_latency_ms_max"],
        "queue_len": _sum_over(pipes, "queue_len"),
        "queue_dropped": _sum_over(pipes, "queue_dropped"),
        "send_queue_len": _sum_over(pipes, "send_queue_len"),
        "send_dropped": _sum_over(pipes, "send_dropped"),
        "stitch_dropped": _sum_over(pipes, "dropped"),
        "stitch_lost": _sum_over(pipes, "lost"),
        "stitch_truncated": _sum_over(pipes, "truncated"),
        "stitch_resets": _sum_over(pipes, "resets"),
        "errors": _sum_over(pipes, "errors"),
        "predict_errors": _sum_over(pipes, "predict_errors"),
        "eject_sent": _sum_over(pipes, "eject_sent"),
        "eject_failed": _sum_over(pipes, "eject_failed"),
        "eject_skipped": _sum_over(pipes, "eject_skipped"),
        "healthy": all(bool(p.get("healthy", True)) for p in pipes),
        "camera_chunks": _sum_over(cams, "chunks_total"),
        "camera_dropped": _sum_over(cams, "dropped_chunks"),
        "mcu_connected": mcu.get("connected"),
        "mcu_eject_rx": counters.get("eject_rx"),
        "mcu_eject_done": counters.get("eject_done"),
        "mcu_eject_dropped": counters.get("eject_dropped"),
        "mcu_ack_timeouts": mcu.get("ack_timeouts"),
        "mcu_reconnects": mcu.get("reconnects"),
        "faults": ";".join(row["faults"]),
    }


class RowWriter:
    """按行追加写记录文件：``.csv`` 平铺（首行表头），其它 JSONL；每行 flush，崩溃也保留已写部分。

    默认以独占创建（``x``）模式打开：文件已存在抛 :class:`SoakError`，不静默覆盖上一次的老化记录；
    ``overwrite=True`` 才用 ``w`` 模式。
    """

    def __init__(self, path: Path, *, overwrite: bool = False):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.is_csv = self.path.suffix.lower() == ".csv"
        try:
            self._fh = self.path.open("w" if overwrite else "x", encoding="utf-8", newline="")
        except FileExistsError:
            raise SoakError(
                f"记录文件已存在，拒绝覆盖（如确需覆盖请加 --overwrite）: {self.path}"
            ) from None
        self._csv: csv.DictWriter | None = None
        self.rows = 0

    @property
    def format(self) -> str:
        return "csv" if self.is_csv else "jsonl"

    def write(self, row: dict[str, Any]) -> None:
        if self.is_csv:
            flat = flatten_row(row)
            if self._csv is None:
                self._csv = csv.DictWriter(self._fh, fieldnames=list(flat))
                self._csv.writeheader()
            self._csv.writerow(flat)
        else:
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        self.rows += 1

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


# --------------------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------------------


class SoakRunner:
    """一次老化运行。``run()`` 返回退出码；``request_stop()`` 可随时（信号 / 其它线程）要求优雅结束。

    ``plan`` 在临时槽位创建 **之前** 就登记，``run()`` 的 finally 据此清理，训练 / 导入 / 部署失败也不残留。
    故障与状态机监听按阶段生效：``startup``（``runtime.start()`` 之前）的瞬时故障记入 ``startup_faults``；
    ``running`` 期间任何故障或运行状态离开 running 都立刻 ``request_stop``，不必等到下一个采样点；
    ``teardown``（收尾）期间忽略——收尾的 ``runtime.stop()`` 本身会把状态机切到 stopped。
    状态机进入 fault 时同步从故障表取当前故障写入 ``faults_seen``（故障表先于状态机通知我们的
    ``_on_fault``，主线程被唤醒后可能已进入收尾，不能只依赖 ``_on_fault``）。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self.summary: dict[str, Any] | None = None
        self.plan: ModelPlan | None = None
        self.faults_seen: dict[str, str] = {}
        """运行期间置位过的故障 code → detail。"""
        self.startup_faults: dict[str, str] = {}
        """启动 / 模型准备阶段置位过的故障（不计入判定，只写进 notes）。"""
        self._phase = "startup"
        """``startup`` / ``running`` / ``teardown``。"""
        self._ctx: AppContext | None = None

    def request_stop(self, reason: str = "signal") -> None:
        """要求结束：训练 / 部署等待 / 采样循环都会尽快退出并走正常收尾。"""
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stop_event.set()

    # ---- 监听（在故障 / 状态机线程里调用，只记录并置事件，不做阻塞操作）----
    def _record_active_faults(self) -> None:
        """把故障表里当前活动的故障并入 ``faults_seen``（已有的保留首次 detail）。"""
        if self._ctx is not None:
            for f in self._ctx.faults.active():
                self.faults_seen.setdefault(f.code, f.detail)

    def _on_fault(self, code: str, active: bool, detail: str) -> None:
        if not active:
            return
        if self._phase == "running":
            self.faults_seen[code] = detail
            self.request_stop("fault")
        elif self._phase == "startup":
            self.startup_faults[code] = detail

    def _on_state(self, old: RunState, new: RunState, event: Event) -> None:
        if self._phase != "running" or new is RunState.running:
            return
        if new is RunState.fault:
            self._record_active_faults()
            self.request_stop("fault")
        else:
            self.request_stop(f"run_state:{new.value}")

    # ---- 辅助 ----
    def _wait_deploy(self, ctx: AppContext, job_id: str, timeout: float) -> dict[str, Any]:
        """分段等待部署任务结束（可被 ``request_stop`` 打断）。

        除 ``jobs.wait`` 外还轮询任务记录本身：状态先于结束事件落定，即使结束通知环节出错
        （例如日志 ``extra`` 键名与 LogRecord 冲突抛 KeyError 导致 ``done`` 未置位）也不会在这里挂死。
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                job = ctx.jobs.wait(job_id, DEPLOY_POLL_S)
            except TimeoutError:
                job = ctx.jobs.get(job_id)
                if job is None:
                    raise SoakError(f"部署任务 {job_id} 不存在") from None
                if not job.finished:
                    if self.stop_event.is_set():
                        ctx.jobs.cancel(job_id)
                        raise SoakError(
                            f"部署期间被中断（{self.stop_reason}），未开始运行"
                        ) from None
                    if time.monotonic() > deadline:
                        ctx.jobs.cancel(job_id)
                        raise SoakError(f"部署在 {timeout:.0f} s 内未完成") from None
                    continue
            if job.status is not JobStatus.succeeded:
                raise SoakError(f"部署失败（{job.status.value}）: {job.message}")
            return dict(job.result or {})

    def _resolve_model(self, ctx: AppContext) -> ModelPlan:
        """决定本次使用的模型并登记到 ``self.plan``。

        临时槽位（导入 / 快速训练）在 ``store.create`` **之前** 就赋给 ``self.plan``：之后任何一步失败
        （训练被 Ctrl-C 取消、导入复制失败、部署失败……）``run()`` 的 finally 都会删除该槽位。
        """
        store = ctx.store
        prev = ctx.deployed_model_id
        if self.args.model_dir is not None:
            src = Path(self.args.model_dir).resolve()
            if src.parent == store.root.resolve() and src.name.isdigit():
                mid = int(src.name)
                if not (store.exists(mid) and store.trained(mid)):
                    raise SoakError(f"模型库中的模型 {mid} 不存在或未训练: {src}")
                self.plan = ModelPlan(mid, "model-dir", temporary=False, prev_deployed=prev)
                return self.plan
            mid = find_free_slot(store)
            if mid is None:
                raise SoakError(
                    f"模型库 {store.max_models} 个槽位已满，无法导入 --model-dir；"
                    "请直接指定模型库内的目录（{data_dir}/models/{id}）或先删除一个模型"
                )
            self.plan = ModelPlan(mid, "model-dir", temporary=True, prev_deployed=prev)
            import_model_dir(store, mid, src)
            return self.plan
        if prev is not None and ctx.classifier is not None:
            log.info("复用已部署模型", extra={"model_id": prev, "backend": ctx.classifier.backend})
            self.plan = ModelPlan(prev, "deployed", temporary=False, prev_deployed=prev)
            return self.plan
        mid = find_free_slot(store)
        if mid is None:
            raise SoakError(
                f"模型库 {store.max_models} 个槽位已满且无已部署模型；请用 --model-dir 指定模型库内的目录"
            )
        self.plan = ModelPlan(mid, "quick-train", temporary=True, prev_deployed=prev)
        try:
            result = train_quick_model(
                store,
                mid,
                ctx.cfg.system.class_names,
                ctx.cfg.ml,
                stop_event=self.stop_event,
            )
        except TrainError as exc:
            if exc.code == "CANCELLED":
                raise SoakError(
                    f"快速训练被中断（{self.stop_reason or 'cancelled'}），未开始运行"
                ) from exc
            raise SoakError(f"快速训练失败: {exc}") from exc
        log.info(
            "快速训练完成",
            extra={
                "model_id": mid,
                "duration_s": result.get("duration_s"),
                "onnx": result.get("artifacts", {}).get("onnx"),
            },
        )
        return self.plan

    @staticmethod
    def _ensure_blow_classes(ctx: AppContext) -> str | None:
        """``blow.classes`` 为空时改为“模型全部类别、置信度阈值 0”，否则决策→喷头映射→``send_eject``
        这条链路在老化中得不到覆盖（临时模型的置信度也到不了默认阈值）。用户配置了类别则不动。返回说明或 None。"""
        if ctx.cfg.blow.classes or ctx.classifier is None:
            return None
        classes = [c for c in ctx.classifier.classes if c in ctx.cfg.system.class_names]
        if not classes:
            return None
        try:
            ctx.config.update("blow", {"classes": classes, "min_confidence": 0})
        except ConfigError as exc:
            log.warning("设置 blow.classes 失败", extra={"err": str(exc)})
            return None
        return f"blow.classes 为空 → {classes}，min_confidence → 0（仅本次运行，不落盘）"

    @staticmethod
    def _teardown_model(ctx: AppContext, plan: ModelPlan) -> None:
        """恢复运行前的部署编号并删除临时模型（槽位目录存在即删，含训练中断 / 导入失败留下的半成品）。"""
        if plan.source != "deployed":
            try:
                ctx.state_store.update(deployed_model_id=plan.prev_deployed)
            except Exception:
                log.exception("恢复部署编号失败")
        if plan.temporary and ctx.store.model_dir(plan.model_id).exists():
            try:
                ctx.store.delete(plan.model_id, plan.prev_deployed)
                log.info("临时模型已删除", extra={"model_id": plan.model_id})
            except StoreError as exc:
                log.warning("删除临时模型失败", extra={"model_id": plan.model_id, "err": str(exc)})

    @staticmethod
    def _settle_threads(baseline: int) -> None:
        deadline = time.monotonic() + THREAD_SETTLE_S
        while threading.active_count() > baseline and time.monotonic() < deadline:
            time.sleep(0.05)

    # ---- 主流程 ----
    def run(self) -> int:
        args = self.args
        proc = psutil.Process()
        proc.cpu_percent(None)
        threads_baseline = threading.active_count()
        baseline_idents = {t.ident for t in threading.enumerate()}
        rss_start = rss_mb(proc)
        notes: list[str] = []
        faults_seen = self.faults_seen
        first: dict[str, Any] | None = None
        last: dict[str, Any] | None = None
        growth: ChunkGrowth | None = None
        rss_max = 0.0
        threads_running_max = threads_baseline
        started_at: str | None = None
        t_run0: float | None = None
        duration_actual = 0.0
        model_info: dict[str, Any] = {}
        error: str | None = None
        ctx: AppContext | None = None
        writer: RowWriter | None = None
        out_path: Path | None = None
        rc = EXIT_ERROR

        try:
            cfg, notes = build_config(Path(args.config), args.speed)
            out_path = resolve_out_path(args.out, cfg.system.data_dir)
            growth = ChunkGrowth(growth_gap_s(cfg, args.speed), args.interval)
            for n in notes:
                log.warning("配置调整", extra={"note": n})
            ctx = AppContext(ConfigService(cfg, None))
            ctx.config_path = Path(args.config).resolve()
            self._ctx = ctx
            # 两个监听在 running 之前只记录启动阶段故障；running 期间任何故障 / 离开 running 都立刻结束
            ctx.faults.subscribe(self._on_fault)
            ctx.state_machine.subscribe(self._on_state)
            ctx.start()
            if self.stop_event.is_set():
                raise SoakError(f"启动后即被中断（{self.stop_reason}），未开始运行")
            plan = self._resolve_model(ctx)
            if self.stop_event.is_set():
                raise SoakError(f"模型准备后被中断（{self.stop_reason}），未开始运行")
            if plan.source != "deployed":
                result = self._wait_deploy(
                    ctx, ctx.runtime.deploy_model(plan.model_id)["job_id"], DEPLOY_TIMEOUT_S
                )
                log.info("模型已部署", extra={"model_id": plan.model_id, **result})
            clf = ctx.classifier
            model_info = {
                "id": plan.model_id,
                "source": plan.source,
                "temporary": plan.temporary,
                "backend": clf.backend if clf is not None else None,
                "classes": list(clf.classes) if clf is not None else [],
                "image_size": int(clf.image_size) if clf is not None else None,
            }
            note = self._ensure_blow_classes(ctx)
            if note:
                notes.append(note)
                log.warning("配置调整", extra={"note": note})
            if self.startup_faults:
                desc = "；".join(f"{c}: {d}" for c, d in self.startup_faults.items())
                note = f"启动/模型准备阶段出现过故障（已清除，不计入运行判定）: {desc}"
                notes.append(note)
                log.warning("启动阶段故障", extra={"faults": dict(self.startup_faults)})
            if self.stop_event.is_set():
                raise SoakError(f"开始运行前被中断（{self.stop_reason}），未开始运行")

            ctx.runtime.start()
            self._phase = "running"
            started_at = _now_iso()
            t_run0 = time.monotonic()
            writer = RowWriter(out_path, overwrite=bool(args.overwrite))
            log.info(
                "老化开始",
                extra={
                    "duration_s": args.duration_s,
                    "interval_s": args.interval,
                    "speed": args.speed,
                    "out": str(out_path),
                    "model_id": plan.model_id,
                    "rss_start_mb": round(rss_start, 1),
                    "threads_baseline": threads_baseline,
                },
            )

            def _sample(now: float) -> None:
                nonlocal first, last, rss_max, threads_running_max
                row = take_sample(ctx, proc, now - t_run0)
                writer.write(row)
                growth.add(row["elapsed_s"], row["stats"]["chunks_total"])
                if first is None:
                    first = row
                last = row
                rss_max = max(rss_max, row["rss_mb"])
                threads_running_max = max(threads_running_max, row["threads"])
                st = row["stats"]
                log.info(
                    "soak 采样",
                    extra={
                        "n": writer.rows,
                        "elapsed_s": row["elapsed_s"],
                        "rss_mb": row["rss_mb"],
                        "threads": row["threads"],
                        "chunks": st["chunks_total"],
                        "dropped": dropped_total(row),
                        "p99_ms": st["chunk_latency_ms_p99"],
                        "queue_len": _sum_over(list(row["pipelines"].values()), "queue_len"),
                        "state": row["run_state"],
                        "faults": row["faults"],
                    },
                )

            deadline = t_run0 + args.duration_s
            next_t = t_run0 + args.interval
            last_sample_t = t_run0
            # 故障 / 状态监听会直接置 stop_event 唤醒这里；循环内再复查一次兜底
            while not self.stop_event.is_set():
                now = time.monotonic()
                if self.stop_event.wait(max(0.0, min(next_t, deadline) - now)):
                    break
                now = time.monotonic()
                if now + 1e-3 >= next_t:
                    _sample(now)
                    last_sample_t = now
                    next_t += args.interval
                state = ctx.run_state
                if state == "fault":
                    self._record_active_faults()
                if state != "running" or faults_seen:
                    self.request_stop(
                        "fault" if (state == "fault" or faults_seen) else f"run_state:{state}"
                    )
                    break
                if now >= deadline:
                    self.request_stop("deadline")
                    break
            if self.stop_reason not in (None, "deadline"):
                log.error(
                    "运行提前结束",
                    extra={
                        "reason": self.stop_reason,
                        "state": ctx.run_state,
                        "faults": ctx.faults.codes(),
                        "faults_seen": list(faults_seen),
                    },
                )
            now = time.monotonic()
            # 到时结束且刚采过样则不重复；提前结束（信号 / 故障）总是补一行，记下结束时刻的真实状态
            if (
                first is None
                or self.stop_reason != "deadline"
                or now - last_sample_t > FINAL_SAMPLE_MIN_GAP_S
            ):
                _sample(now)
            duration_actual = now - t_run0
            rc = EXIT_OK
        except SoakError as exc:
            error = str(exc)
            log.error("老化测试错误", extra={"err": error})
            rc = EXIT_ERROR
        except Exception as exc:  # 任何未预期异常都收尾并给出退出码 1
            error = f"{type(exc).__name__}: {exc}"
            log.exception("老化测试异常")
            rc = EXIT_ERROR
        finally:
            # 先进入收尾阶段：runtime.stop() 会把状态机切到 stopped，不应再被当作异常结束
            self._phase = "teardown"
            if writer is not None:
                writer.close()
            if ctx is not None:
                if t_run0 is not None:
                    try:
                        ctx.runtime.stop()
                    except Exception:
                        log.exception("停止运行失败")
                try:
                    ctx.stop()
                except Exception:
                    log.exception("AppContext 停止失败")
                if self.plan is not None:
                    self._teardown_model(ctx, self.plan)
            self._settle_threads(threads_baseline)

        threads_end = threading.active_count()
        leaked = [
            t.name for t in threading.enumerate() if t.ident not in baseline_idents and t.is_alive()
        ]
        if t_run0 is None:
            # 运行阶段没有开始：没有汇总可写，只报告错误（被信号中断也归此路径：没有可判定的数据）
            what = "被中断" if self.stop_reason and self.stop_reason != "fault" else "错误"
            print(f"老化测试未开始（{what}）：{error}", file=sys.stderr)
            return rc

        rss_baseline = first["rss_mb"] if first is not None else rss_start
        rss_end = last["rss_mb"] if last is not None else rss_baseline
        rss_growth = rss_end - rss_baseline
        peak_growth = max(rss_max, rss_end) - rss_baseline
        stats = dict(last["stats"]) if last is not None else {}
        dropped = dropped_total(last) if last is not None else 0
        # 结束时仍在 running，且块数相对 ≥ gap 秒前的一行有增长（流水线没有被故障处理静默停掉）
        chunks_grew = growth.grew() if growth is not None else None
        running_ok = (
            last is not None and last["run_state"] == "running" and chunks_grew is not False
        )
        checks: dict[str, Any] = {
            "rss_ok": rss_growth <= args.rss_limit_mb,
            "threads_ok": not leaked,
            "faults_ok": not faults_seen,
            "running_ok": bool(running_ok),
            "running_detail": {
                "run_state_end": last["run_state"] if last is not None else None,
                "chunks_grew": chunks_grew,
                "growth_gap_s": round(growth.gap_s, 3) if growth is not None else None,
            },
            "dropped_ok": None if args.max_dropped is None else dropped <= args.max_dropped,
            "max_dropped": args.max_dropped,
        }
        failed = [k for k, v in checks.items() if v is False]
        if rc == EXIT_OK and failed:
            rc = EXIT_LIMIT
        summary_path = Path(out_path).parent / SUMMARY_NAME
        self.summary = {
            "version": __version__,
            "config": str(Path(args.config).resolve()),
            "out": str(out_path),
            "format": writer.format if writer is not None else None,
            "started_at": started_at,
            "ended_at": _now_iso(),
            "stopped_by": self.stop_reason or ("error" if error else "deadline"),
            "duration_requested_s": round(args.duration_s, 3),
            "duration_actual_s": round(duration_actual, 3),
            "interval_s": args.interval,
            "speed": args.speed,
            "model": model_info,
            "samples": writer.rows if writer is not None else 0,
            "rss": {
                "start_mb": round(rss_start, 2),
                "baseline_mb": round(rss_baseline, 2),
                "end_mb": round(rss_end, 2),
                "max_mb": round(max(rss_max, rss_end), 2),
                "growth_mb": round(rss_growth, 2),
                "peak_growth_mb": round(peak_growth, 2),
                "limit_mb": args.rss_limit_mb,
            },
            "threads": {
                "baseline": threads_baseline,
                "running_max": threads_running_max,
                "end": threads_end,
                "leaked": leaked,
            },
            "chunks_total": int(stats.get("chunks_total", 0)),
            "dropped_total": int(dropped),
            "latency_p99_ms": float(stats.get("chunk_latency_ms_p99", 0.0)),
            "rss_growth_mb": round(rss_growth, 2),
            "threads_leaked": len(leaked),
            "stats": stats,
            "pipelines": dict(last["pipelines"]) if last is not None else {},
            "cameras": dict(last["cameras"]) if last is not None else {},
            "mcu": dict(last["mcu"]) if last is not None else {},
            "faults_seen": dict(faults_seen),
            "startup_faults": dict(self.startup_faults),
            "faults_end": list(last["faults"]) if last is not None else [],
            "checks": checks,
            "failed_checks": failed,
            "passed": rc == EXIT_OK,
            "exit_code": rc,
            "error": error,
            "notes": list(notes),
            "summary_path": str(summary_path),
        }
        try:
            atomic_write_json(summary_path, self.summary)
        except OSError as exc:
            self.summary["summary_path"] = None
            log.error("写 summary 失败", extra={"path": str(summary_path), "err": str(exc)})
        print(format_summary(self.summary))
        return rc


def format_summary(s: dict[str, Any]) -> str:
    """汇总的人读文本（CLI 结果输出）。"""
    rss, th, st, model = s["rss"], s["threads"], s["stats"], s["model"]
    verdict = {EXIT_OK: "合格", EXIT_LIMIT: "超限", EXIT_ERROR: "错误"}.get(s["exit_code"], "?")
    lines = [
        "==== soak_test 汇总 ====",
        f"时长: {s['duration_actual_s']:.1f} s（请求 {s['duration_requested_s']:.0f} s），"
        f"采样 {s['samples']} 次，间隔 {s['interval_s']} s，速度倍率 {s['speed']}，结束原因 {s['stopped_by']}",
        f"模型: {model.get('id')}（{model.get('source')}，{model.get('backend')}，类别 {model.get('classes')}）",
        f"RSS: 启动 {rss['start_mb']:.1f} MB，基线 {rss['baseline_mb']:.1f} MB → 结束 {rss['end_mb']:.1f} MB"
        f"（最大 {rss['max_mb']:.1f}），增长 {rss['growth_mb']:+.1f} MB（峰值 {rss['peak_growth_mb']:+.1f}，"
        f"限 {rss['limit_mb']:g} MB）",
        f"线程: 基线 {th['baseline']} → 运行中最多 {th['running_max']} → 结束 {th['end']}"
        f"（泄漏 {len(th['leaked'])}{': ' + ', '.join(th['leaked']) if th['leaked'] else ''}）",
        f"块: 总计 {s['chunks_total']}，丢块 {s['dropped_total']}，颗粒 {st.get('particles_total', 0)}，"
        f"吹气 {st.get('blown_total', 0)}；延迟 p50 {st.get('chunk_latency_ms_p50', 0)} ms，"
        f"p99 {st.get('chunk_latency_ms_p99', 0)} ms，max {st.get('chunk_latency_ms_max', 0)} ms",
        f"故障: {', '.join(s['faults_seen']) if s['faults_seen'] else '无'}",
    ]
    if s.get("startup_faults"):
        lines.append("启动阶段故障（不计入判定）: " + ", ".join(s["startup_faults"]))
    if s.get("notes"):
        lines.append("配置调整: " + "；".join(s["notes"]))
    if s.get("error"):
        lines.append(f"错误: {s['error']}")
    if s.get("failed_checks"):
        lines.append("未通过: " + ", ".join(s["failed_checks"]))
    lines.append(f"记录: {s['out']}；汇总: {s.get('summary_path') or '(未写入)'}")
    lines.append(f"结论: {verdict}（退出码 {s['exit_code']}）")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def signal_handlers(on_signal: Callable[[str], None]) -> Iterator[None]:
    """在主线程安装 SIGINT / SIGTERM：第一次 → ``on_signal(name)``（优雅结束）；第二次 → 恢复原处理器并抛
    ``KeyboardInterrupt`` 强制退出（收尾卡住时不必 ``kill -9``）。退出时恢复；非主线程不安装。"""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    saved: dict[int, Any] = {}
    hits = 0

    def restore() -> None:
        for sig, prev in saved.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, prev)

    def handler(signum: int, _frame: Any) -> None:
        nonlocal hits
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        hits += 1
        if hits == 1:
            on_signal(name)
            return
        log.warning("再次收到信号，强制退出", extra={"signal": name})
        restore()
        raise KeyboardInterrupt(f"再次收到 {name}，强制退出")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            saved[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # pragma: no cover - 极少数宿主不允许
            continue
    try:
        yield
    finally:
        restore()


def _maybe_setup_logging(level: str) -> None:
    """根 logger 还没有 handler（作为脚本直接运行）时才配置日志；进程内调用（pytest）不动现有配置。"""
    if not logging.getLogger().handlers:
        setup_logging(level)


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并运行；返回退出码（0 合格 / 2 超限 / 1 错误）。"""
    try:
        args = parse_args(argv)
    except SoakError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _maybe_setup_logging(args.log_level)
    runner = SoakRunner(args)
    try:
        with signal_handlers(runner.request_stop):
            return runner.run()
    except KeyboardInterrupt as exc:
        # 第二次信号（或收尾期间再次 Ctrl-C）：收尾已尽力执行，没有可信的汇总，按错误退出
        print(f"强制退出：{exc or 'KeyboardInterrupt'}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
