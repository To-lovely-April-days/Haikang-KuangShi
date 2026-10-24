"""系统与状态接口：/health /state /system/info /system/save（docs/03 §3）。"""

from __future__ import annotations

import os
import platform
import shutil
import socket
from typing import Any

import psutil
from fastapi import APIRouter, Request

from oresorter import __version__
from oresorter.api.schemas import HealthOut, SavedOut
from oresorter.runtime.context import AppContext

router = APIRouter(tags=["system"])


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(ok=True, version=__version__)


@router.get("/state")
def state(request: Request) -> dict[str, Any]:
    """全量状态快照。"""
    return get_ctx(request).snapshot()


@router.get("/system/info")
def system_info(request: Request) -> dict[str, Any]:
    ctx = get_ctx(request)
    du = shutil.disk_usage(str(ctx.data_dir))
    temps: dict[str, float] = {}
    try:
        for name, entries in (psutil.sensors_temperatures() or {}).items():  # type: ignore[attr-defined]
            if entries:
                temps[name] = float(entries[0].current)
    except Exception:
        pass
    gpu = _read_tegra_gpu_load()
    return {
        "version": __version__,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "data_dir": str(ctx.data_dir),
        "disk_total_mb": du.total // (1024 * 1024),
        "disk_free_mb": du.free // (1024 * 1024),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "mem_percent": psutil.virtual_memory().percent,
        "temperatures_c": temps,
        "gpu_percent": gpu,
        "ips": _local_ips(),
        "pid": os.getpid(),
    }


@router.post("/system/save", response_model=SavedOut)
def system_save(request: Request) -> SavedOut:
    ctx = get_ctx(request)
    ctx.flush_uptime()
    path = ctx.config.export_backup()
    return SavedOut(saved=True, backup_path=str(path))


def _local_ips() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    try:
        for name, addrs in psutil.net_if_addrs().items():
            v4 = [
                a.address
                for a in addrs
                if a.family == socket.AF_INET and not a.address.startswith("127.")
            ]
            if v4:
                out[name] = v4
    except Exception:
        pass
    return out


def _read_tegra_gpu_load() -> float | None:
    """Jetson: /sys/devices/gpu.0/load（千分比）。其他平台 None。"""
    for p in ("/sys/devices/gpu.0/load", "/sys/devices/platform/gpu.0/load"):
        try:
            with open(p, encoding="utf-8") as f:
                return int(f.read().strip()) / 10.0
        except OSError:
            continue
    return None
