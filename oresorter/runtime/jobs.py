"""后台任务系统（docs/02 §3.7、docs/03 §10）。

- :class:`JobManager.submit` 把 ``fn(handle)`` 放到一个 daemon 线程里执行，同类型可互斥；
- :class:`JobHandle` 给任务函数汇报进度、检查取消、写带 ``job_id`` 的日志；
- 任务的创建 / 开始 / 进度（限频）/ 结束都通过 ``on_event(job.to_dict())`` 回调（不持有任务表锁），
  由集成者转发到 WS ``job`` 事件；
- 已结束任务按 **结束顺序** 保留最近 ``max_history`` 条，供 ``GET /jobs/{id}`` 查询。

任务函数约定：

- 正常返回 ``dict``（可 JSON）或 ``None`` → ``succeeded``，``result`` 为返回值，``progress=1``；
  若返回时取消已被请求（``handle.cancelled``），则记为 ``cancelled``（结果仍保留）；
- 抛 :class:`JobCancelled`，或取消已请求且抛任何异常 → ``cancelled``；
- 其他异常 → ``failed``，``message=str(exc)``。

事件语义：

- 事件按状态变更的先后顺序串行送达（内部用 ``_emit_lock`` 串行化：先取事件锁再取任务表锁，
  回调运行时不持有任务表锁），因此 ``on_event`` 内可以调用 ``get()/active_list()/cancel()``，
  但不要在回调里阻塞等待任务（``wait()``），否则会卡住所有任务的事件发送；
- 进度事件限频：同一任务两次进度事件间隔 ≥ ``progress_event_min_interval``，``stage`` 变化与结束不限频；
  被限频压下的最后一次进度会在间隔到期后由一个 daemon 定时器补发（任务结束时取消定时器并在
  结束事件之前补发），HMI 不会长时间看到陈旧进度。
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from oresorter.log import get_logger

log = get_logger(__name__)

JobFn = Callable[["JobHandle"], Any]
EventCallback = Callable[[dict[str, Any]], None]

#: ``wait()`` 的默认超时（秒）：禁止无限阻塞，长任务请显式传更大的值。
DEFAULT_WAIT_TIMEOUT = 30.0


class JobType(str, Enum):
    """任务类型（docs/03 §1 枚举 + backup）。"""

    capture = "capture"
    train = "train"
    deploy = "deploy"
    valve_test = "valve_test"
    backup = "backup"


class JobStatus(str, Enum):
    """任务状态（docs/03 §1 枚举）。"""

    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


FINISHED_STATUSES = frozenset({JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled})
ACTIVE_STATUSES = frozenset({JobStatus.queued, JobStatus.running})


class JobCancelled(Exception):
    """任务函数在检测到取消时抛出（``JobHandle.check_cancel`` 会抛）。"""


class JobConflict(Exception):
    """同类型任务已在排队/运行（HTTP 409）。"""

    code = "CONFLICT"

    def __init__(self, job_type: JobType, running_job_id: str):
        self.job_type = job_type
        self.running_job_id = running_job_id
        self.message = f"已有 {job_type.value} 任务在执行（{running_job_id}）"
        super().__init__(self.message)

    @property
    def detail(self) -> dict[str, Any]:
        return {"type": self.job_type.value, "running_job_id": self.running_job_id}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


@dataclass
class Job:
    """任务记录。``to_dict()`` 字段与 docs/03 ``GET /jobs/{id}`` 一致（另加 created_at、meta）。"""

    job_id: str
    type: JobType
    model_id: int | None = None
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    stage: str | None = None
    message: str | None = None
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def copy(self) -> Job:
        """返回独立副本（result / meta 深拷贝），供外部只读使用。"""
        return Job(
            job_id=self.job_id,
            type=self.type,
            model_id=self.model_id,
            status=self.status,
            progress=self.progress,
            stage=self.stage,
            message=self.message,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            result=copy.deepcopy(self.result),
            meta=copy.deepcopy(self.meta),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "type": self.type.value,
            "model_id": self.model_id,
            "status": self.status.value,
            "progress": round(float(self.progress), 4),
            "stage": self.stage,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": copy.deepcopy(self.result),
            "created_at": self.created_at,
            "meta": copy.deepcopy(self.meta),
        }


class _JobLoggerAdapter(logging.LoggerAdapter):
    """把 ``job_id`` 合并进每条日志的 extra（3.10 的 LoggerAdapter 默认会覆盖调用方 extra）。"""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


class JobHandle:
    """传给任务函数的句柄：进度汇报、取消检查、日志。"""

    def __init__(self, manager: JobManager, job_id: str, cancel_event: threading.Event):
        self._manager = manager
        self._job_id = job_id
        self.cancel_event = cancel_event
        self.log = _JobLoggerAdapter(get_logger("oresorter.jobs"), {"job_id": job_id})

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def job(self) -> Job:
        """当前任务记录的只读副本。"""
        job = self._manager.get(self._job_id)
        if job is None:  # pragma: no cover - 任务被裁剪后仍持有句柄
            raise KeyError(self._job_id)
        return job

    @property
    def cancelled(self) -> bool:
        """是否已请求取消。"""
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        """已请求取消则抛 :class:`JobCancelled`。"""
        if self.cancel_event.is_set():
            raise JobCancelled(self._job_id)

    def set_progress(self, progress: float, stage: str | None = None) -> None:
        """更新进度（0~1，越界截断）与阶段描述；``stage`` 为 None 时保留原值。"""
        self._manager._set_progress(self._job_id, progress, stage)


@dataclass
class _Entry:
    job: Job
    handle: JobHandle
    fn: JobFn
    done: threading.Event
    thread: threading.Thread | None = None
    last_progress_emit: float | None = None
    pending_snapshot: dict[str, Any] | None = None
    flush_timer: threading.Timer | None = None


@dataclass
class _FinishOut:
    """``_finish_locked`` 的产物：锁外需要做的事。"""

    job_id: str
    status: JobStatus
    message: str | None
    pending: dict[str, Any] | None
    snapshot: dict[str, Any]
    done: threading.Event


class JobManager:
    """任务管理器（线程安全，无全局状态）。

    Args:
        on_event: 任务事件回调，参数为 ``job.to_dict()``；回调运行时不持有任务表锁，
            事件按状态变更顺序串行送达，异常被捕获记录。
        max_history: 已结束任务保留条数（按结束顺序，最早结束的先被裁掉）。
        progress_event_min_interval: 进度事件最小间隔（秒）；``stage`` 变化与结束事件不限频，
            被压下的最后一次进度在间隔到期后由定时器补发。

    锁顺序：``_emit_lock`` → ``_lock``；任何持有 ``_lock`` 的代码都不得去取 ``_emit_lock``。
    """

    def __init__(
        self,
        *,
        on_event: EventCallback | None = None,
        max_history: int = 200,
        progress_event_min_interval: float = 0.1,
    ):
        self._on_event = on_event
        self._max_history = max(1, int(max_history))
        self._min_interval = max(0.0, float(progress_event_min_interval))
        self._lock = threading.RLock()  # 保护任务表
        self._emit_lock = threading.RLock()  # 串行化"变更 + 发事件"，保证事件顺序
        self._entries: dict[str, _Entry] = {}
        self._finished_order: deque[str] = deque()  # 已结束任务编号，按结束先后
        self._threads: set[threading.Thread] = set()  # 启动过且可能仍存活的工作线程/定时器
        self._counter = 0
        self._closed = False

    # ---- 提交 ----
    def submit(
        self,
        type: JobType | str,
        fn: JobFn,
        *,
        model_id: int | None = None,
        exclusive: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> Job:
        """创建并启动任务，返回任务记录的副本。

        Args:
            type: 任务类型。
            fn: 任务函数 ``fn(handle) -> dict | None``。
            model_id: 关联模型编号。
            exclusive: True 时同类型已有 queued/running 任务则抛 :class:`JobConflict`。
            meta: 附加信息（随 ``to_dict`` 返回）。
        """
        job_type = JobType(type)
        with self._emit_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("JobManager 已关闭")
                if exclusive:
                    active = self._active_of_locked(job_type)
                    if active is not None:
                        raise JobConflict(job_type, active.job_id)
                job = Job(
                    job_id=self._next_id_locked(),
                    type=job_type,
                    model_id=model_id,
                    meta=dict(meta or {}),
                )
                cancel_event = threading.Event()
                entry = _Entry(
                    job=job,
                    handle=JobHandle(self, job.job_id, cancel_event),
                    fn=fn,
                    done=threading.Event(),
                )
                self._entries[job.job_id] = entry
                snap = job.to_dict()
            log.info("任务创建", extra={"job_id": job.job_id, "type": job_type.value})
            self._emit(snap)
        self._spawn(job.job_id)
        return self.get(job.job_id) or job.copy()

    def _next_id_locked(self) -> str:
        self._counter += 1
        return f"j-{datetime.now().strftime('%Y%m%d')}-{self._counter:04d}"

    def _spawn(self, job_id: str) -> None:
        """启动工作线程（独立方法便于测试替换）。"""
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                return
            t = threading.Thread(
                target=self._run, args=(job_id,), name=f"job-{job_id}", daemon=True
            )
            entry.thread = t
            self._track_locked(t)
        t.start()

    def _track_locked(self, t: threading.Thread) -> None:
        """登记一个即将启动的线程；顺手清掉已结束的，集合规模始终 ≈ 存活线程数。"""
        self._threads = {x for x in self._threads if x.is_alive()}
        self._threads.add(t)

    # ---- 查询 ----
    def get(self, job_id: str) -> Job | None:
        """按编号取任务副本；不存在（或已被历史裁剪）返回 None。"""
        with self._lock:
            entry = self._entries.get(job_id)
            return entry.job.copy() if entry is not None else None

    def list(self, active_only: bool = False) -> list[Job]:
        """按创建顺序列出任务副本。"""
        with self._lock:
            jobs = [e.job for e in self._entries.values()]
            if active_only:
                jobs = [j for j in jobs if j.active]
            return [j.copy() for j in jobs]

    def active_list(self) -> list[dict[str, Any]]:
        """``GET /state`` 的 ``jobs`` 项：queued/running 任务的 ``to_dict()``。"""
        with self._lock:
            return [e.job.to_dict() for e in self._entries.values() if e.job.active]

    def running(self, type: JobType | str) -> Job | None:
        """该类型当前活动（queued/running）任务的副本，无则 None。"""
        job_type = JobType(type)
        with self._lock:
            job = self._active_of_locked(job_type)
            return job.copy() if job is not None else None

    def _active_of_locked(self, job_type: JobType) -> Job | None:
        for e in self._entries.values():
            if e.job.type is job_type and e.job.active:
                return e.job
        return None

    # ---- 控制 ----
    def cancel(self, job_id: str) -> bool:
        """请求取消。queued 在锁内直接标记 cancelled（fn 不会执行）；running 置取消事件由任务函数自行结束。

        已结束或不存在的任务返回 False。
        """
        with self._emit_lock:
            with self._lock:
                entry = self._entries.get(job_id)
                if entry is None or entry.job.finished:
                    return False
                entry.handle.cancel_event.set()
                if entry.job.status is JobStatus.running:
                    log.info("任务取消请求", extra={"job_id": job_id})
                    return True
                # queued：与状态收尾在同一把锁内完成，工作线程随后进入 _run 时只会看到已结束
                out = self._finish_locked(entry, JobStatus.cancelled, None, "已取消")
            self._deliver_finish(out)
        return True

    def wait(self, job_id: str, timeout: float = DEFAULT_WAIT_TIMEOUT) -> Job:
        """阻塞等待任务结束并返回其副本（测试 / 同步调用用）。

        超时抛 ``TimeoutError``，不存在抛 ``KeyError``。默认超时 :data:`DEFAULT_WAIT_TIMEOUT`，
        不提供无限等待。
        """
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise KeyError(job_id)
            done = entry.done
        if not done.wait(max(0.0, float(timeout))):
            raise TimeoutError(f"任务 {job_id} 在 {timeout}s 内未结束")
        job = self.get(job_id)
        if job is None:  # pragma: no cover - done 与 get 之间又有 ≥ max_history 个任务结束
            raise KeyError(job_id)
        return job

    def shutdown(self, timeout: float = 5.0) -> bool:
        """取消全部活动任务并 join 所有工作线程与进度定时器；返回是否全部结束。之后 ``submit`` 会拒绝。"""
        with self._lock:
            self._closed = True
            active_ids = [jid for jid, e in self._entries.items() if e.job.active]
        for jid in active_ids:
            self.cancel(jid)
        me = threading.current_thread()
        with self._lock:
            for e in self._entries.values():
                self._cancel_timer_locked(e)
            threads = [t for t in self._threads if t.is_alive() and t is not me]
        deadline = time.monotonic() + max(0.0, timeout)
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        alive = [t.name for t in threads if t.is_alive()]
        with self._lock:
            self._threads = {t for t in self._threads if t.is_alive()}
        if alive:
            log.warning("任务线程未能在超时内结束", extra={"threads": alive})
        return not alive

    # ---- 内部：执行 ----
    def _run(self, job_id: str) -> None:
        with self._emit_lock:
            with self._lock:
                entry = self._entries.get(job_id)
                if entry is None or entry.job.status is not JobStatus.queued:
                    return  # 排队期间已被取消并收尾
                if entry.handle.cancel_event.is_set():
                    # 取消已请求但尚未收尾（理论上 cancel 已在锁内收尾，这里兜底）：不执行 fn
                    out = self._finish_locked(entry, JobStatus.cancelled, None, "已取消")
                    snap = None
                else:
                    out = None
                    entry.job.status = JobStatus.running
                    entry.job.started_at = _now_iso()
                    snap = entry.job.to_dict()
                handle, fn = entry.handle, entry.fn
            if out is not None:
                self._deliver_finish(out)
                return
            self._emit(snap)
        handle.log.info("任务开始", extra={"type": snap["type"]})
        status, result, message = JobStatus.succeeded, None, None
        try:
            ret = fn(handle)
            result = self._validate_result(ret)
            if handle.cancel_event.is_set():
                status, message = JobStatus.cancelled, "已取消"
        except JobCancelled as exc:
            status, message = JobStatus.cancelled, str(exc) or "已取消"
        except Exception as exc:  # 任务失败必须被记录而不是让线程炸掉
            if handle.cancel_event.is_set():
                status, message = JobStatus.cancelled, str(exc) or "已取消"
            else:
                status, message = JobStatus.failed, str(exc) or exc.__class__.__name__
                handle.log.exception("任务失败")
        except BaseException as exc:  # pragma: no cover - SystemExit 等
            status, message = JobStatus.failed, f"{exc.__class__.__name__}: {exc}"
            handle.log.exception("任务异常终止")
        self._finish(job_id, status, result, message)

    @staticmethod
    def _validate_result(ret: Any) -> dict[str, Any] | None:
        if ret is None:
            return None
        if not isinstance(ret, dict):
            raise TypeError(f"任务返回值必须是 dict 或 None，得到 {type(ret).__name__}")
        json.dumps(ret)  # 不可 JSON 化会抛 TypeError/ValueError
        return ret

    def _finish(
        self,
        job_id: str,
        status: JobStatus,
        result: dict[str, Any] | None,
        message: str | None,
    ) -> None:
        with self._emit_lock:
            with self._lock:
                entry = self._entries.get(job_id)
                if entry is None or entry.job.finished:
                    return
                out = self._finish_locked(entry, status, result, message)
            self._deliver_finish(out)

    def _finish_locked(
        self,
        entry: _Entry,
        status: JobStatus,
        result: dict[str, Any] | None,
        message: str | None,
    ) -> _FinishOut:
        """在 ``_lock`` 内完成状态收尾与历史裁剪；返回锁外要发送的事件。"""
        job = entry.job
        job.status = status
        job.result = result
        job.message = message
        if status is JobStatus.succeeded:
            job.progress = 1.0
        job.finished_at = _now_iso()
        self._cancel_timer_locked(entry)
        pending, entry.pending_snapshot = entry.pending_snapshot, None
        self._finished_order.append(job.job_id)
        self._trim_locked()
        return _FinishOut(
            job_id=job.job_id,
            status=status,
            message=message,
            pending=pending,
            snapshot=job.to_dict(),
            done=entry.done,
        )

    def _deliver_finish(self, out: _FinishOut) -> None:
        """``_finish_locked`` 之后、仍持有 ``_emit_lock`` 时调用：记日志、补发进度、发结束事件、唤醒等待者。"""
        log.info(
            "任务结束",
            extra={"job_id": out.job_id, "status": out.status.value, "reason": out.message},
        )
        if out.pending is not None:
            self._emit(out.pending)
        self._emit(out.snapshot)
        out.done.set()

    def _trim_locked(self) -> None:
        """按结束顺序裁剪历史：刚结束的任务永远在队尾，不会被本次裁掉。"""
        while len(self._finished_order) > self._max_history:
            jid = self._finished_order.popleft()
            self._entries.pop(jid, None)

    # ---- 内部：进度与事件 ----
    def _set_progress(self, job_id: str, progress: float, stage: str | None) -> None:
        now = time.monotonic()
        with self._emit_lock:
            with self._lock:
                entry = self._entries.get(job_id)
                if entry is None or entry.job.finished:
                    return
                job = entry.job
                job.progress = min(1.0, max(0.0, float(progress)))
                stage_changed = stage is not None and stage != job.stage
                if stage is not None:
                    job.stage = stage
                last = entry.last_progress_emit
                due = last is None or now - last >= self._min_interval
                if stage_changed or due:
                    entry.last_progress_emit = now
                    entry.pending_snapshot = None
                    self._cancel_timer_locked(entry)
                    snap: dict[str, Any] | None = job.to_dict()
                else:
                    entry.pending_snapshot = job.to_dict()
                    self._ensure_timer_locked(entry, self._min_interval - (now - (last or now)))
                    snap = None
            if snap is not None:
                self._emit(snap)

    def _ensure_timer_locked(self, entry: _Entry, delay: float) -> None:
        """若没有在途的补发定时器，则安排一个在 ``delay`` 秒后补发被压下的进度。"""
        t = entry.flush_timer
        if t is not None and t.is_alive():
            return
        t = threading.Timer(max(0.001, delay), self._flush_progress, args=(entry.job.job_id,))
        t.name = f"job-flush-{entry.job.job_id}"
        t.daemon = True
        entry.flush_timer = t
        self._track_locked(t)
        t.start()

    @staticmethod
    def _cancel_timer_locked(entry: _Entry) -> None:
        if entry.flush_timer is not None:
            entry.flush_timer.cancel()
            entry.flush_timer = None

    def _flush_progress(self, job_id: str) -> None:
        """定时器回调：补发被限频压下的最后一次进度（任务已结束或已被别的事件覆盖则跳过）。"""
        with self._emit_lock:
            with self._lock:
                entry = self._entries.get(job_id)
                if entry is None or entry.job.finished or entry.pending_snapshot is None:
                    return
                snap, entry.pending_snapshot = entry.pending_snapshot, None
                entry.last_progress_emit = time.monotonic()
            self._emit(snap)

    def _emit(self, payload: dict[str, Any]) -> None:
        cb = self._on_event
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            log.exception("任务事件回调异常", extra={"job_id": payload.get("job_id")})
