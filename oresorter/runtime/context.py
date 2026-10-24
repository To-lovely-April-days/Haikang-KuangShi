"""组合根 ``AppContext``：按配置装配相机、下位机、数据目录、持久化状态，供 API / CLI / 流水线共用。

P0 阶段只负责装配与设备启停；状态机、流水线、任务系统在 P2 接入（见 docs/02 §3.7）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oresorter import __version__
from oresorter.camera import Camera, build_camera
from oresorter.config import AppConfig, ConfigService
from oresorter.fsutil import atomic_write_json, ensure_dir, read_json
from oresorter.log import get_logger
from oresorter.mcu import McuLink, build_mcu_link

log = get_logger(__name__)


@dataclass
class Fault:
    code: str
    detail: str = ""
    since: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersistedState:
    """``{data_dir}/state.json``：跨重启保留的运行状态。"""

    deployed_model_id: int | None = None
    uptime_total_s: float = 0.0
    bg_version: int = 0
    last_run_state: str = "stopped"


class StateStore:
    """``state.json`` 的读写（原子）。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        data = read_json(path, default=None)
        self.state = PersistedState(
            **{k: v for k, v in (data or {}).items() if k in PersistedState.__dataclass_fields__}
        )

    def save(self) -> None:
        with self._lock:
            atomic_write_json(self.path, asdict(self.state))

    def update(self, **patch: Any) -> PersistedState:
        with self._lock:
            for k, v in patch.items():
                if not hasattr(self.state, k):
                    raise KeyError(k)
                setattr(self.state, k, v)
            atomic_write_json(self.path, asdict(self.state))
            return self.state


class FaultRegistry:
    """当前活动故障集合（线程安全）。"""

    def __init__(self) -> None:
        self._faults: dict[str, Fault] = {}
        self._lock = threading.Lock()
        self._listeners: list[Any] = []

    def set(self, code: str, detail: str = "") -> bool:
        """置故障；返回是否为新故障。"""
        with self._lock:
            if code in self._faults:
                self._faults[code].detail = detail
                return False
            self._faults[code] = Fault(code=code, detail=detail)
            listeners = list(self._listeners)
        log.warning("故障置位", extra={"fault": code, "detail": detail})
        for cb in listeners:
            cb(code, True, detail)
        return True

    def clear(self, code: str) -> bool:
        with self._lock:
            existed = self._faults.pop(code, None) is not None
            listeners = list(self._listeners)
        if existed:
            log.info("故障清除", extra={"fault": code})
            for cb in listeners:
                cb(code, False, "")
        return existed

    def active(self) -> list[Fault]:
        with self._lock:
            return list(self._faults.values())

    def has(self, code: str) -> bool:
        with self._lock:
            return code in self._faults

    def codes(self) -> list[str]:
        with self._lock:
            return list(self._faults)

    def subscribe(self, cb: Any) -> None:
        with self._lock:
            self._listeners.append(cb)


class AppContext:
    """应用上下文。构造只装配不启动；``start()`` 打开设备，``stop()`` 释放。"""

    def __init__(self, config: ConfigService):
        self.config = config
        cfg = config.config
        self.data_dir: Path = ensure_dir(cfg.system.data_dir)
        for sub in ("models", "snapshots", "previews", "background", "logs", "captures"):
            ensure_dir(self.data_dir / sub)
        self.state_store = StateStore(self.data_dir / "state.json")
        self.faults = FaultRegistry()
        self.cameras: dict[int, Camera] = {c.id: build_camera(c) for c in cfg.cameras}
        self.mcu: McuLink = build_mcu_link(cfg)
        self.started_at = time.monotonic()
        self.boot_time = datetime.now(timezone.utc).astimezone()
        self._uptime_anchor = time.monotonic()
        self._uptime_base = self.state_store.state.uptime_total_s
        self._running = False
        # 以下由 P2 的 runtime 模块替换/填充
        self.run_state: str = "stopped"
        self.jobs: Any = None
        self.stats: Any = None
        self.pipeline: Any = None

    # ---- 构造 ----
    @classmethod
    def from_yaml(cls, path: str | Path, *, persist: bool = True) -> AppContext:
        return cls(ConfigService.from_yaml(path, persist=persist))

    @property
    def cfg(self) -> AppConfig:
        return self.config.config

    # ---- 生命周期 ----
    def start(self) -> None:
        """打开相机与下位机。单个设备失败只记故障，不抛出。"""
        if self._running:
            return
        self._running = True
        for cam in self.cameras.values():
            try:
                cam.open()
                self.faults.clear(f"CAMERA_{cam.id}_DISCONNECTED")
            except Exception as exc:
                self.faults.set(f"CAMERA_{cam.id}_DISCONNECTED", str(exc))
                log.error("相机打开失败", extra={"camera": cam.id, "err": str(exc)})
        try:
            self.mcu.connect()
        except Exception as exc:
            self.faults.set("MCU_DISCONNECTED", str(exc))
            log.error("下位机连接失败", extra={"err": str(exc)})
        log.info(
            "AppContext 已启动",
            extra={
                "cameras": list(self.cameras),
                "mcu": self.cfg.system.mcu_kind,
                "data_dir": str(self.data_dir),
            },
        )

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.flush_uptime()
        for cam in self.cameras.values():
            try:
                cam.close()
            except Exception:
                log.exception("相机关闭异常", extra={"camera": cam.id})
        try:
            self.mcu.close()
        except Exception:
            log.exception("下位机关闭异常")

    # ---- 运行时长 ----
    def uptime_total_s(self) -> float:
        return self._uptime_base + (time.monotonic() - self._uptime_anchor)

    def flush_uptime(self) -> None:
        self.state_store.update(uptime_total_s=round(self.uptime_total_s(), 1))

    # ---- 快照 ----
    def snapshot(self) -> dict[str, Any]:
        """GET /state 的全量快照（P2 补齐 jobs / stats）。"""
        mcu_status = self.mcu.status.to_dict()
        return {
            "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "uptime_total_s": int(self.uptime_total_s()),
            "run_state": self.run_state,
            "deployed_model_id": self.state_store.state.deployed_model_id,
            "cameras": [cam.status() for cam in self.cameras.values()],
            "mcu": mcu_status,
            "faults": [f.to_dict() for f in self.faults.active()],
            "jobs": self.jobs.active_list() if self.jobs is not None else [],
            "stats": self.stats.snapshot() if self.stats is not None else {},
            "version": __version__,
        }
