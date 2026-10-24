"""系统与状态接口：/health /state /system/info /system/save /system/shutdown /system/reboot（docs/03 §3）。

本模块还提供各路由模块共用的 ``get_ctx`` / ``get_runtime``（从 ``request.app.state.ctx`` 取组合根）。
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
from typing import Any

import psutil
from fastapi import APIRouter, Request

from oresorter import __version__
from oresorter.api.schemas import (
    HealthOut,
    PowerOut,
    RequestIdBody,
    SavedOut,
    StateOut,
    SystemInfoOut,
)
from oresorter.runtime.context import AppContext
from oresorter.runtime.service import RuntimeService

router = APIRouter(tags=["system"])


def get_ctx(request: Request) -> AppContext:
    """当前应用的组合根。"""
    return request.app.state.ctx


def get_runtime(request: Request) -> RuntimeService:
    """运行时门面（路由层唯一的业务入口）。"""
    return get_ctx(request).runtime


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """存活探针（systemd watchdog / HMI 轮询）。"""
    return HealthOut(ok=True, version=__version__)


@router.get("/state", response_model=StateOut)
def state(request: Request) -> dict[str, Any]:
    """全量状态快照（HMI 启动与 WS 重连后调用）。"""
    return get_runtime(request).snapshot()


@router.get("/system/info", response_model=SystemInfoOut)
def system_info(request: Request) -> dict[str, Any]:
    """版本、磁盘剩余、温度、GPU/CPU 占用、网口 IP。"""
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
        "preview_max_fps": int(ctx.cfg.system.preview_max_fps),
    }


@router.post("/system/save", response_model=SavedOut)
def system_save(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """显式落盘并导出 ``config.backup.json``（配置本身已随每次修改即时持久化）。"""
    return get_runtime(request).save()


@router.post("/system/shutdown", response_model=PowerOut)
def system_shutdown(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """先停机再关机（``ORESORTER_ALLOW_POWER=1`` 时才真正执行 systemctl）。"""
    return get_runtime(request).shutdown(reboot=False)


@router.post("/system/reboot", response_model=PowerOut)
def system_reboot(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """先停机再重启。"""
    return get_runtime(request).shutdown(reboot=True)


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
