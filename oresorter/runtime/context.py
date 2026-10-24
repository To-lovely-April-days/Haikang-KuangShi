"""组合根 ``AppContext``：按配置装配相机、下位机、数据目录、持久化状态、状态机、任务系统、统计、
模型库、事件总线与运行时门面（``ctx.runtime``），供 API / CLI / 流水线共用。

事件接线（docs/01 §4.3、§7）：

- 任何故障置位 → 状态机 ``fault_raised`` + ``fault`` / ``state`` 事件 + ``RuntimeService.on_fault_raised``
  （另起线程停流水线、尽力下发 STOP）；全部故障清除且处于 fault → ``fault_cleared``（回到 stopped，需人工 start）；
- 下位机事件：connected → 清 ``MCU_DISCONNECTED``、后台补发参数；disconnected / 链路 fault → 置
  ``MCU_DISCONNECTED``；EVENT level 2 → ``MCU_EVENT_{code}``；设备故障位 → ``MCU_FAULT_{code}``；
  counters / status → ``mcu`` 事件（限频 1 s）；
- 状态迁移 → ``state`` 事件 + ``state.json.last_run_state``；
- 配置变更（下位机分组，且不是 ``RuntimeService.update_params`` 自己在同步）→ 后台下发 + ``param_sync`` 事件；
- 每 1 s 的管家线程：running 时广播 ``stats``、每 5 s 广播 ``state``、每 2 s 巡检设备连接、
  启动后 60 s 内重试 ``auto_run_on_boot``（开机自检阶段 TCP 下位机尚未连上会先短暂进入 fault，
  窗口期内 **保留** 重试：故障清除回到 stopped 后下一拍即自动运行；只有被人工操作带入
  running / paused / sampling 才放弃）。
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oresorter import __version__
from oresorter.camera import Camera, build_camera
from oresorter.config import MCU_SECTIONS, AppConfig, ConfigService
from oresorter.data.store import ModelStore
from oresorter.fsutil import atomic_write_json, ensure_dir, read_json
from oresorter.log import get_logger
from oresorter.mcu import McuLink, build_mcu_link
from oresorter.mcu import protocol as P
from oresorter.mcu.link import LinkEvent, McuError, McuNack
from oresorter.runtime.events import EventBus
from oresorter.runtime.jobs import JobManager
from oresorter.runtime.pipeline import CLASSIFIER_ERROR_FAULT
from oresorter.runtime.service import RuntimeService
from oresorter.runtime.state import Event, IllegalTransition, RunState, StateMachine
from oresorter.runtime.state import check_start_preconditions as _check_start
from oresorter.runtime.stats import RunStats
from oresorter.vision.background import ColumnBackground, background_path

if TYPE_CHECKING:
    from oresorter.ml.infer import Classifier
    from oresorter.runtime.pipeline import Pipeline

log = get_logger(__name__)

MCU_DISCONNECTED_FAULT = "MCU_DISCONNECTED"
MODEL_LOAD_FAULT = "MODEL_LOAD_FAILED"
HOUSEKEEPING_TICK_S = 1.0
DEVICE_POLL_EVERY = 2
STATE_BROADCAST_EVERY = 5
UPTIME_FLUSH_EVERY = 60
MCU_EVENT_MIN_INTERVAL_S = 1.0
AUTO_RUN_RETRY_WINDOW_S = 60.0
THREAD_JOIN_S = 2.0


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
            try:
                cb(code, True, detail)
            except Exception:
                log.exception("故障监听回调异常", extra={"fault": code})
        return True

    def clear(self, code: str) -> bool:
        with self._lock:
            existed = self._faults.pop(code, None) is not None
            listeners = list(self._listeners)
        if existed:
            log.info("故障清除", extra={"fault": code})
            for cb in listeners:
                try:
                    cb(code, False, "")
                except Exception:
                    log.exception("故障监听回调异常", extra={"fault": code})
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
    """应用上下文。构造只装配不启动；``start()`` 打开设备并启动管家线程，``stop()`` 释放。"""

    def __init__(self, config: ConfigService):
        self.config = config
        cfg = config.config
        self.config_path: Path | None = None
        self.data_dir: Path = ensure_dir(cfg.system.data_dir)
        for sub in ("models", "snapshots", "previews", "background", "logs", "captures"):
            ensure_dir(self.data_dir / sub)
        self.state_store = StateStore(self.data_dir / "state.json")
        self.faults = FaultRegistry()
        self.events = EventBus()
        self.cameras: dict[int, Camera] = {c.id: build_camera(c) for c in cfg.cameras}
        self.mcu: McuLink = build_mcu_link(cfg)
        self.started_at = time.monotonic()
        self.boot_time = datetime.now(timezone.utc).astimezone()
        self._uptime_anchor = time.monotonic()
        self._uptime_base = self.state_store.state.uptime_total_s
        self._running = False
        self._stopping = False
        self.stop_event = threading.Event()
        self._tls = threading.local()
        self._bg_threads: list[threading.Thread] = []
        self._bg_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_param_sections: set[str] = set()
        self._last_mcu_publish = 0.0
        self._housekeeper: threading.Thread | None = None
        self._auto_run_deadline: float | None = None

        # 运行时组件
        self.state_machine = StateMachine(RunState.stopped)
        self.jobs = JobManager(on_event=self._on_job_event)
        self.stats = RunStats(class_names=cfg.system.class_names)
        self.store = ModelStore(
            self.data_dir / "models",
            cfg.system.max_models,
            list(cfg.system.class_names),
            min_free_mb=cfg.system.min_free_disk_mb,
        )
        self.store.ensure_root()
        self.backgrounds: dict[int, ColumnBackground] = self._load_backgrounds()
        self.classifier: Classifier | None = None
        self.pipeline: Pipeline | None = None
        self.pipelines: dict[int, Pipeline] = {}
        self.runtime = RuntimeService(self)

        # 接线
        self.faults.subscribe(self._on_fault)
        self.state_machine.subscribe(self._on_state_change)
        self.mcu.on_event(self._on_mcu_event)
        self.config.subscribe(self._on_config_change)

    # ---- 构造 ----
    @classmethod
    def from_yaml(cls, path: str | Path, *, persist: bool = True) -> AppContext:
        ctx = cls(ConfigService.from_yaml(path, persist=persist))
        ctx.config_path = Path(path).resolve()
        return ctx

    @property
    def cfg(self) -> AppConfig:
        return self.config.config

    @property
    def run_state(self) -> str:
        """当前运行状态名（docs/03 枚举）。"""
        return self.state_machine.state.value

    @property
    def deployed_model_id(self) -> int | None:
        return self.state_store.state.deployed_model_id

    # ---- 背景 ----
    def _load_backgrounds(self) -> dict[int, ColumnBackground]:
        out: dict[int, ColumnBackground] = {}
        for cam_id in self.cameras:
            path = background_path(self.data_dir, cam_id)
            if not path.exists():
                continue
            try:
                out[cam_id] = ColumnBackground.load(path)
                log.info("列背景已加载", extra={"camera": cam_id, "width": out[cam_id].width})
            except ValueError as exc:
                log.warning("列背景文件无法加载", extra={"camera": cam_id, "err": str(exc)})
        return out

    def reload_background(self, camera_id: int) -> ColumnBackground | None:
        """重新读取某相机的列背景文件（learn-bg 之后调用）。"""
        path = background_path(self.data_dir, camera_id)
        if not path.exists():
            self.backgrounds.pop(camera_id, None)
            return None
        bg = ColumnBackground.load(path)
        self.backgrounds[camera_id] = bg
        return bg

    # ---- 生命周期 ----
    def start(self) -> None:
        """打开相机与下位机、加载部署模型、启动管家线程；单个设备失败只记故障，不抛出。"""
        if self._running:
            return
        self._running = True
        self._stopping = False
        self.stop_event.clear()
        for cam in self.cameras.values():
            try:
                cam.open()
                self.faults.clear(f"CAMERA_{cam.id}_DISCONNECTED")
            except Exception as exc:
                self.faults.set(f"CAMERA_{cam.id}_DISCONNECTED", str(exc))
                log.error("相机打开失败", extra={"camera": cam.id, "err": str(exc)})
        try:
            self.mcu.connect()
            if not self.mcu.connected:
                self.faults.set(MCU_DISCONNECTED_FAULT, "等待下位机连接")
        except Exception as exc:
            self.faults.set(MCU_DISCONNECTED_FAULT, str(exc))
            log.error("下位机连接失败", extra={"err": str(exc)})
        self._load_deployed_model()
        self._housekeeper = threading.Thread(
            target=self._housekeeping_loop, name="ctx-housekeeping", daemon=True
        )
        self._housekeeper.start()
        if self.cfg.system.auto_run_on_boot:
            self._auto_run_deadline = time.monotonic() + AUTO_RUN_RETRY_WINDOW_S
            self._try_auto_run()
        log.info(
            "AppContext 已启动",
            extra={
                "cameras": list(self.cameras),
                "mcu": self.cfg.system.mcu_kind,
                "data_dir": str(self.data_dir),
                "deployed": self.deployed_model_id,
                "state": self.run_state,
            },
        )
        self.events.publish("state", self.snapshot())

    def stop(self) -> None:
        """停流水线与任务、关分类器、关设备、刷运行时长。可重复调用。"""
        if not self._running:
            return
        self._running = False
        self._stopping = True
        self.stop_event.set()
        try:
            self.runtime.quiesce()
        except Exception:
            log.exception("运行时收尾异常")
        self.jobs.shutdown(5.0)
        hk, self._housekeeper = self._housekeeper, None
        if hk is not None:
            hk.join(THREAD_JOIN_S)
        self._join_bg_threads(THREAD_JOIN_S)
        for p in list(self.pipelines.values()):
            with contextlib.suppress(Exception):
                p.stop()
        self.pipelines = {}
        self.pipeline = None
        clf, self.classifier = self.classifier, None
        if clf is not None:
            with contextlib.suppress(Exception):
                clf.close()
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
        log.info("AppContext 已停止")

    # ---- 后台线程 ----
    def spawn(self, name: str, target: Callable[..., Any], *args: Any) -> threading.Thread:
        """启动一个受跟踪的 daemon 线程（``stop()`` 时 join）；异常被捕获记录。"""

        def runner() -> None:
            try:
                target(*args)
            except Exception:
                log.exception("后台线程异常", extra={"thread_name": name})

        t = threading.Thread(target=runner, name=name, daemon=True)
        with self._bg_lock:
            self._bg_threads = [x for x in self._bg_threads if x.is_alive()]
            self._bg_threads.append(t)
        t.start()
        return t

    def _join_bg_threads(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        me = threading.current_thread()
        with self._bg_lock:
            threads = list(self._bg_threads)
        for t in threads:
            if t is me:
                continue
            t.join(max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive() and t is not me]
        if alive:
            log.warning("后台线程未能在时限内退出", extra={"threads": alive})
        with self._bg_lock:
            self._bg_threads = [x for x in self._bg_threads if x.is_alive()]

    def _housekeeping_loop(self) -> None:
        tick = 0
        while not self.stop_event.wait(HOUSEKEEPING_TICK_S):
            tick += 1
            try:
                if self.state_machine.state is RunState.running:
                    self.events.publish("stats", self.stats.snapshot())
                if tick % DEVICE_POLL_EVERY == 0:
                    self._poll_devices()
                if tick % STATE_BROADCAST_EVERY == 0:
                    self.events.publish("state", self.snapshot())
                if tick % UPTIME_FLUSH_EVERY == 0:
                    self.flush_uptime()
                if self._auto_run_deadline is not None:
                    self._try_auto_run()
            except Exception:
                log.exception("管家线程异常")

    def _poll_devices(self) -> None:
        """设备巡检：相机 / 下位机连接状态变化 → 置/清故障（stopped 状态下也能反映掉线）。"""
        if self._stopping:
            return
        for cam in self.cameras.values():
            code = f"CAMERA_{cam.id}_DISCONNECTED"
            if cam.connected:
                self.faults.clear(code)
            elif not self.faults.has(code):
                self.faults.set(code, "相机未连接")
        if self.mcu.connected:
            self.faults.clear(MCU_DISCONNECTED_FAULT)
        elif not self.faults.has(MCU_DISCONNECTED_FAULT):
            self.faults.set(MCU_DISCONNECTED_FAULT, "下位机未连接")

    def _try_auto_run(self) -> None:
        """``auto_run_on_boot``：前置条件满足即 start；失败只记日志，窗口期内重试。

        处于 fault（典型：开机时 TCP 下位机还没连上 → ``MCU_DISCONNECTED``）只等下一拍，不放弃窗口；
        被人工操作带入 running / paused / sampling 才放弃。
        """
        deadline = self._auto_run_deadline
        if deadline is None:
            return
        if time.monotonic() > deadline:
            self._auto_run_deadline = None
            log.warning("开机自动运行超时放弃", extra={"faults": self.faults.codes()})
            return
        state = self.state_machine.state
        if state is RunState.fault:
            return
        if state is not RunState.stopped:
            self._auto_run_deadline = None
            log.info("开机自动运行放弃：状态已被人工操作改变", extra={"state": state.value})
            return
        reasons = _check_start(
            deployed_model_id=self.deployed_model_id,
            cameras_connected=all(c.connected for c in self.cameras.values()),
            mcu_connected=self.mcu.connected,
        )
        if reasons:
            return
        self._auto_run_deadline = None
        try:
            self.runtime.start()
            log.info("开机自动运行已启动")
        except Exception as exc:
            log.warning("开机自动运行失败", extra={"err": str(exc)})

    # ---- 模型 ----
    def _load_deployed_model(self) -> None:
        mid = self.deployed_model_id
        if mid is None:
            return
        if not (self.store.exists(mid) and self.store.trained(mid)):
            log.warning("持久化的部署模型不存在或未训练，清除", extra={"model_id": mid})
            self.state_store.update(deployed_model_id=None)
            return
        try:
            from oresorter.ml.infer import load_classifier

            clf = load_classifier(
                self.store.model_dir(mid),
                prefer=("trt", "onnx", "torch"),
                device=self.cfg.ml.device,
                max_batch=int(self.cfg.pipeline.infer_batch_max),
            )
            clf.warmup()
        except Exception as exc:
            log.error("加载部署模型失败", extra={"model_id": mid, "err": str(exc)})
            self.faults.set(MODEL_LOAD_FAULT, f"模型 {mid} 加载失败: {exc}")
            self.state_store.update(deployed_model_id=None)
            return
        self.classifier = clf
        self.faults.clear(MODEL_LOAD_FAULT)
        log.info("部署模型已加载", extra={"model_id": mid, "backend": clf.backend})

    def set_classifier(self, model_id: int, clf: Classifier) -> None:
        """原子切换分类器（流水线先换引用再关旧的），持久化部署编号并广播 ``state``。

        在 ``runtime.lock`` 下执行：与 ``RuntimeService`` 建流水线的临界区互斥，避免新流水线在
        ``ctx.pipelines`` 发布前拿到的旧分类器被这里关闭。
        """
        with self.runtime.lock:
            old, self.classifier = self.classifier, clf
            for p in list(self.pipelines.values()):
                p.replace_classifier(clf)
            if old is not None and old is not clf:
                try:
                    old.close()
                except Exception:
                    log.exception("关闭旧分类器失败")
            self.state_store.update(deployed_model_id=model_id)
        self.faults.clear(MODEL_LOAD_FAULT)
        self.faults.clear(CLASSIFIER_ERROR_FAULT)
        log.info("分类器已切换", extra={"model_id": model_id, "backend": clf.backend})
        self.events.publish("state", self.snapshot())

    def replace_stats(self, class_names: list[str]) -> None:
        """类别配置变化后重建统计对象。"""
        self.stats = RunStats(class_names=class_names)

    # ---- 参数同步 ----
    @contextlib.contextmanager
    def suppress_param_sync(self) -> Iterator[None]:
        """在本线程的配置更新期间不触发后台下位机同步（调用方自己同步）。"""
        prev = getattr(self._tls, "suppress_param_sync", False)
        self._tls.suppress_param_sync = True
        try:
            yield
        finally:
            self._tls.suppress_param_sync = prev

    def mark_param_sync_pending(self, section: str) -> None:
        """记录一个未被下位机确认的分组，重连后补发并发 ``param_sync``。"""
        with self._pending_lock:
            self._pending_param_sections.add(section)

    def pending_param_sections(self) -> list[str]:
        with self._pending_lock:
            return sorted(self._pending_param_sections)

    def _on_config_change(self, section: str, value: Any) -> None:
        # 链路对象持有的是构造时的配置快照：刷新，使其自己的全量补发（TCP 重连）用到最新值
        self.mcu.cfg = self.cfg
        if section not in MCU_SECTIONS or self._stopping:
            return
        if getattr(self._tls, "suppress_param_sync", False):
            return
        self.spawn(f"param-sync-{section}", self._push_section, section)

    def _push_section(self, section: str) -> None:
        params = P.params_for_section(self.cfg, section)
        try:
            self.mcu.set_params(params)
        except McuNack as exc:
            self.events.publish(
                "param_sync", {"section": section, "mcu_ack": False, "error": str(exc)}
            )
            return
        except McuError as exc:
            self.mark_param_sync_pending(section)
            self.events.publish(
                "param_sync", {"section": section, "mcu_ack": False, "error": str(exc)}
            )
            return
        self.events.publish("param_sync", {"section": section, "mcu_ack": True, "error": None})

    def _resync_mcu_params(self) -> None:
        """下位机（重）连接后：全量补发参数（TCP 链路自己也会推，Mock 不会），并通告挂起分组的结果。"""
        with self._pending_lock:
            pending = sorted(self._pending_param_sections)
            self._pending_param_sections.clear()
        auto = bool(getattr(self.mcu, "auto_push_params", False))
        if auto and not pending:
            return
        ok, err = True, None
        try:
            self.mcu.cfg = self.cfg
            self.mcu.set_params(P.params_from_config(self.cfg))
        except McuError as exc:
            ok, err = False, str(exc)
            log.warning("下位机参数补发失败", extra={"err": err})
            with self._pending_lock:
                self._pending_param_sections.update(pending)
        for section in pending:
            self.events.publish("param_sync", {"section": section, "mcu_ack": ok, "error": err})

    # ---- 事件处理 ----
    def _on_job_event(self, job: dict[str, Any]) -> None:
        self.events.publish("job", job)

    def _on_state_change(self, old: RunState, new: RunState, event: Event) -> None:
        try:
            self.state_store.update(last_run_state=new.value)
        except OSError:
            log.exception("写 state.json 失败")
        self.events.publish("state", self.snapshot())

    def _on_fault(self, code: str, active: bool, detail: str) -> None:
        """故障置位/清除：驱动状态机，并保证每次变化恰好广播 **一条** ``state``。

        发生迁移时 ``_on_state_change`` 已广播快照（故障表此时已更新）；没有迁移（如 fault 态下又来一个
        故障、或清除后仍有其它故障）时在这里补发一条，HMI 才能看到更新后的 faults 列表。
        """
        self.events.publish("fault", {"code": code, "active": active, "detail": detail})
        sm = self.state_machine
        before = sm.state
        if active:
            if before is not RunState.fault:
                with contextlib.suppress(IllegalTransition):
                    sm.transition(Event.fault_raised)
                if not self._stopping:
                    self.runtime.on_fault_raised(code)
        elif before is RunState.fault and not self.faults.active():
            with contextlib.suppress(IllegalTransition):
                sm.transition(Event.fault_cleared)
        if sm.state is before:
            self.events.publish("state", self.snapshot())

    def _publish_mcu(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_mcu_publish < MCU_EVENT_MIN_INTERVAL_S:
            return
        self._last_mcu_publish = now
        self.events.publish("mcu", self.mcu.status.to_dict())

    def _on_mcu_event(self, ev: LinkEvent) -> None:
        if self._stopping:
            return
        kind, data = ev.kind, ev.data
        if kind == "connected":
            self.faults.clear(MCU_DISCONNECTED_FAULT)
            for code in self.faults.codes():
                if code.startswith("MCU_EVENT_"):
                    self.faults.clear(code)
            self.spawn("mcu-resync", self._resync_mcu_params)
            self._publish_mcu(force=True)
        elif kind == "disconnected":
            self.faults.set(MCU_DISCONNECTED_FAULT, str(data.get("reason") or "链路断开"))
            self._publish_mcu(force=True)
        elif kind == "fault":
            code = data.get("code")
            if "active" in data and isinstance(code, int):
                fcode = f"MCU_FAULT_{code:04X}"
                text = (
                    P.EVENT_TEXT.get(P.EventCode(code), fcode)
                    if code in P.EventCode._value2member_map_
                    else fcode
                )
                if data.get("active"):
                    self.faults.set(fcode, text)
                else:
                    self.faults.clear(fcode)
            else:
                self.faults.set(MCU_DISCONNECTED_FAULT, str(code or "链路故障"))
            self._publish_mcu(force=True)
        elif kind == "event":
            level = int(data.get("level", 0) or 0)
            code = int(data.get("code", 0) or 0)
            if level >= 2:
                self.faults.set(f"MCU_EVENT_{code:04X}", str(data.get("text") or f"0x{code:04X}"))
            self._publish_mcu(force=level >= 1)
        elif kind in ("counters", "status"):
            self._publish_mcu(force=kind == "status")
        elif kind == "nack":
            log.debug("下位机 NACK", extra=dict(data))

    # ---- 运行时长 ----
    def uptime_total_s(self) -> float:
        return self._uptime_base + (time.monotonic() - self._uptime_anchor)

    def flush_uptime(self) -> None:
        self.state_store.update(uptime_total_s=round(self.uptime_total_s(), 1))

    # ---- 快照 ----
    def snapshot(self) -> dict[str, Any]:
        """``GET /state`` 的全量快照。"""
        mcu_status = self.mcu.status.to_dict()
        pipeline = self.pipeline
        return {
            "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "uptime_total_s": int(self.uptime_total_s()),
            "run_state": self.run_state,
            "deployed_model_id": self.deployed_model_id,
            "cameras": [cam.status() for cam in self.cameras.values()],
            "mcu": mcu_status,
            "faults": [f.to_dict() for f in self.faults.active()],
            "jobs": self.jobs.active_list(),
            "stats": self.stats.snapshot(),
            "pipeline": pipeline.stats() if pipeline is not None else None,
            "version": __version__,
        }
