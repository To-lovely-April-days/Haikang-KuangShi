"""runtime-core：状态机、任务系统、运行统计。"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import threading
import time
from collections.abc import Callable

import pytest

from oresorter.runtime import state as S
from oresorter.runtime.jobs import (
    Job,
    JobCancelled,
    JobConflict,
    JobHandle,
    JobManager,
    JobStatus,
    JobType,
)
from oresorter.runtime.state import Event, IllegalTransition, RunState, StateMachine
from oresorter.runtime.stats import RunStats

# ======================================================================================
# 状态机
# ======================================================================================

ST, EV = RunState, Event


def make_sm(state: RunState, **kw) -> StateMachine:
    """把一台新状态机驱动到指定状态（sampling 从 stopped 进入）。"""
    sm = StateMachine(**kw)
    steps = {
        ST.stopped: [],
        ST.running: [EV.start],
        ST.paused: [EV.start, EV.pause],
        ST.sampling: [EV.capture_begin],
        ST.fault: [EV.fault_raised],
    }[state]
    for ev in steps:
        sm.transition(ev)
    assert sm.state is state
    return sm


# (state, event) -> 结果状态 / "same"（幂等）/ [reasons]
EXPECTED: dict[tuple[RunState, Event], object] = {
    (ST.stopped, EV.start): ST.running,
    (ST.stopped, EV.pause): [S.NOT_RUNNING],
    (ST.stopped, EV.stop): "same",
    (ST.stopped, EV.capture_begin): ST.sampling,
    (ST.stopped, EV.capture_end): [S.NOT_SAMPLING],
    (ST.stopped, EV.fault_raised): ST.fault,
    (ST.stopped, EV.fault_cleared): "same",
    (ST.running, EV.start): "same",
    (ST.running, EV.pause): ST.paused,
    (ST.running, EV.stop): ST.stopped,
    (ST.running, EV.capture_begin): [S.ALREADY_RUNNING],
    (ST.running, EV.capture_end): [S.NOT_SAMPLING],
    (ST.running, EV.fault_raised): ST.fault,
    (ST.running, EV.fault_cleared): "same",
    (ST.paused, EV.start): ST.running,
    (ST.paused, EV.pause): "same",
    (ST.paused, EV.stop): ST.stopped,
    (ST.paused, EV.capture_begin): ST.sampling,
    (ST.paused, EV.capture_end): [S.NOT_SAMPLING],
    (ST.paused, EV.fault_raised): ST.fault,
    (ST.paused, EV.fault_cleared): "same",
    (ST.sampling, EV.start): [S.SAMPLING_IN_PROGRESS],
    (ST.sampling, EV.pause): [S.SAMPLING_IN_PROGRESS],
    (ST.sampling, EV.stop): ST.stopped,
    (ST.sampling, EV.capture_begin): [S.SAMPLING_IN_PROGRESS],
    (ST.sampling, EV.capture_end): ST.stopped,  # make_sm 从 stopped 进入 sampling
    (ST.sampling, EV.fault_raised): ST.fault,
    (ST.sampling, EV.fault_cleared): "same",
    (ST.fault, EV.start): [S.IN_FAULT],
    (ST.fault, EV.pause): [S.IN_FAULT],
    (ST.fault, EV.stop): "same",  # 故障态允许停止（幂等不清故障，路由层仍下发 STOP）
    (ST.fault, EV.capture_begin): [S.IN_FAULT],
    (ST.fault, EV.capture_end): [S.NOT_SAMPLING],
    (ST.fault, EV.fault_raised): "same",
    (ST.fault, EV.fault_cleared): ST.stopped,
}


def test_transition_table_is_complete():
    assert set(EXPECTED) == {(s, e) for s in RunState for e in Event}


@pytest.mark.parametrize(("state", "event"), sorted(EXPECTED, key=lambda k: (k[0], k[1])))
def test_transition_table(state: RunState, event: Event):
    changes: list[tuple] = []
    sm = make_sm(state, on_change=lambda o, n, e: changes.append((o, n, e)))
    changes.clear()
    hist_before = len(sm.history)
    expected = EXPECTED[(state, event)]
    if isinstance(expected, list):
        assert sm.can(event) is False
        with pytest.raises(IllegalTransition) as ei:
            sm.transition(event)
        err = ei.value
        assert err.reasons == expected
        assert err.code == "CONFLICT"
        d = err.to_dict()
        assert d["code"] == "CONFLICT"
        assert d["detail"] == {"state": state.value, "event": event.value, "reasons": expected}
        assert isinstance(d["message"], str) and d["message"]
        assert sm.state is state
        assert changes == []
        assert len(sm.history) == hist_before
    elif expected == "same":
        assert sm.can(event) is True
        assert sm.transition(event) is state
        assert sm.state is state
        assert changes == [], "幂等迁移不触发 on_change"
        assert len(sm.history) == hist_before, "幂等迁移不记历史"
    else:
        assert sm.can(event) is True
        assert sm.transition(event) is expected
        assert sm.state is expected
        assert changes == [(state, expected, event)]
        rec = sm.history[-1]
        assert (rec.src, rec.dst, rec.event) == (state, expected, event)
        assert isinstance(rec.ts, float)


def test_sampling_returns_to_source_state():
    sm = StateMachine()
    sm.transition(EV.capture_begin)
    assert sm.state is ST.sampling and sm.previous is ST.stopped
    assert sm.transition(EV.capture_end) is ST.stopped

    sm.transition(EV.start)
    sm.transition(EV.pause)
    sm.transition(EV.capture_begin)
    assert sm.state is ST.sampling and sm.previous is ST.paused
    assert sm.transition(EV.capture_end) is ST.paused


def test_sampling_stop_goes_to_stopped_even_from_paused():
    sm = make_sm(ST.paused)
    sm.transition(EV.capture_begin)
    assert sm.transition(EV.stop) is ST.stopped


@pytest.mark.parametrize("state", list(RunState))
def test_fault_from_every_state_then_cleared(state: RunState):
    sm = make_sm(state)
    assert sm.transition(EV.fault_raised) is ST.fault
    assert sm.transition(EV.fault_cleared) is ST.stopped
    # 清除后需人工 start
    assert sm.transition(EV.start) is ST.running


def test_history_capped_and_ordered():
    sm = StateMachine()
    for _ in range(80):
        sm.transition(EV.start)
        sm.transition(EV.pause)
    hist = sm.history
    assert len(hist) == S.HISTORY_LEN == 100
    assert all(h.ts <= hist[i + 1].ts for i, h in enumerate(hist[:-1]))
    assert hist[-1].event is EV.pause and hist[-1].dst is ST.paused
    assert hist[-2].event is EV.start


def test_multiple_listeners_and_unsubscribe():
    got1: list = []
    got2: list = []
    sm = StateMachine(on_change=lambda *a: got1.append(a))
    unsub = sm.subscribe(lambda *a: got2.append(a))
    sm.transition(EV.start)
    assert got1 == [(ST.stopped, ST.running, EV.start)]
    assert got2 == got1
    unsub()
    sm.transition(EV.pause)
    assert len(got1) == 2 and len(got2) == 1


def test_listener_exception_does_not_break_transition():
    def boom(*_):
        raise RuntimeError("listener")

    sm = StateMachine(on_change=boom)
    assert sm.transition(EV.start) is ST.running


def test_transition_accepts_string_event():
    sm = StateMachine()
    assert sm.transition("start") is ST.running
    assert sm.can("pause") is True
    assert sm.to_dict() == {"state": "running", "previous": None}


def test_fault_stop_is_idempotent_not_409():
    changes: list = []
    sm = make_sm(ST.fault, on_change=lambda *a: changes.append(a))
    changes.clear()
    assert sm.can(EV.stop) is True, "HMI 停止按钮在故障态不置灰"
    assert sm.transition(EV.stop) is ST.fault, "停止不清故障"
    assert changes == []
    assert sm.transition(EV.fault_cleared) is ST.stopped


def test_listener_order_matches_transition_order():
    """第一个回调被人为放慢时，另一线程的迁移不得先于它送达（回调顺序 == 迁移顺序）。"""
    sm = StateMachine()
    got: list[RunState] = []
    first_in = threading.Event()

    def cb(_old, new, _ev):
        if new is ST.running:
            first_in.set()
            time.sleep(0.05)
        got.append(new)

    sm.subscribe(cb)
    t = threading.Thread(target=lambda: sm.transition(EV.start), name="sm-start", daemon=True)
    t.start()
    assert first_in.wait(2)
    assert sm.state is ST.running
    assert sm.transition(EV.pause) is ST.paused  # 在第一个回调结束前发起
    t.join(2)
    assert got == [ST.running, ST.paused]
    assert [h.dst for h in sm.history] == got


def test_check_start_preconditions():
    f = S.check_start_preconditions
    assert f(deployed_model_id=1, cameras_connected=True, mcu_connected=True) == []
    assert f(deployed_model_id=None, cameras_connected=True, mcu_connected=True) == [
        S.NO_MODEL_DEPLOYED
    ]
    assert f(deployed_model_id=1, cameras_connected=False, mcu_connected=True) == [
        S.CAMERA_DISCONNECTED
    ]
    assert f(deployed_model_id=1, cameras_connected=True, mcu_connected=False) == [
        S.MCU_DISCONNECTED
    ]
    assert f(deployed_model_id=None, cameras_connected=False, mcu_connected=False) == [
        S.NO_MODEL_DEPLOYED,
        S.CAMERA_DISCONNECTED,
        S.MCU_DISCONNECTED,
    ]


def test_state_machine_thread_safety():
    sm = StateMachine()
    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(200):
                for ev in (EV.start, EV.pause, EV.stop):
                    with contextlib.suppress(IllegalTransition):
                        sm.transition(ev)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, name=f"sm-{i}", daemon=True) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert sm.state in (ST.stopped, ST.running, ST.paused)
    hist = sm.history
    for a, b in itertools.pairwise(hist):
        assert a.dst is b.src, "历史必须首尾相接"


# ======================================================================================
# 任务系统
# ======================================================================================

DOC_JOB_FIELDS = {
    "job_id",
    "type",
    "model_id",
    "status",
    "progress",
    "stage",
    "message",
    "started_at",
    "finished_at",
    "result",
}


@pytest.fixture
def events() -> list[dict]:
    return []


@pytest.fixture
def mgr(events: list[dict]):
    m = JobManager(on_event=events.append, max_history=50)
    yield m
    assert m.shutdown(timeout=3.0)


def _wait_until(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("条件在超时内未满足")
        time.sleep(0.005)


def test_job_succeeds_with_result(mgr: JobManager, events: list[dict]):
    def fn(h: JobHandle):
        h.set_progress(0.5, "half")
        h.log.info("working", extra={"k": 1})
        return {"n": 3}

    job = mgr.submit(JobType.train, fn, model_id=2, meta={"classes": ["A"]})
    assert isinstance(job, Job)
    assert re.fullmatch(r"j-\d{8}-\d{4}", job.job_id)
    done = mgr.wait(job.job_id, timeout=5)
    assert done.status is JobStatus.succeeded
    assert done.result == {"n": 3}
    assert done.progress == 1.0
    assert done.model_id == 2 and done.meta == {"classes": ["A"]}
    assert done.started_at is not None and done.finished_at is not None
    assert done.message is None
    statuses = [e["status"] for e in events if e["job_id"] == job.job_id]
    assert statuses[0] == "queued" and statuses[1] == "running" and statuses[-1] == "succeeded"
    assert events[-1]["result"] == {"n": 3} and events[-1]["progress"] == 1.0
    assert mgr.active_list() == []
    assert mgr.running(JobType.train) is None


def test_to_dict_fields_match_docs(mgr: JobManager):
    job = mgr.submit("deploy", lambda h: None, model_id=1)
    d = mgr.wait(job.job_id, 5).to_dict()
    assert set(d) >= DOC_JOB_FIELDS
    assert set(d) - DOC_JOB_FIELDS <= {"created_at", "meta"}
    assert d["type"] == "deploy" and d["status"] == "succeeded" and d["result"] is None
    json.dumps(d)


def test_progress_rate_limit_last_and_final_delivered(events: list[dict]):
    m = JobManager(on_event=events.append, progress_event_min_interval=10.0)
    try:

        def fn(h: JobHandle):
            for i in range(1, 101):
                h.set_progress(i / 100)
            return None

        job = m.submit(JobType.capture, fn)
        m.wait(job.job_id, 5)
        mine = [e for e in events if e["job_id"] == job.job_id]
        progress_events = [e for e in mine if e["status"] == "running" and e["progress"] > 0]
        assert 1 <= len(progress_events) <= 3, "100 次 set_progress 必须被限频"
        assert progress_events[-1]["progress"] == 1.0, "最后一次 set_progress 必达"
        assert mine[-1]["status"] == "succeeded"
        assert [e["status"] for e in mine][:2] == ["queued", "running"]
    finally:
        assert m.shutdown(2.0)


def test_progress_stage_change_not_rate_limited(events: list[dict]):
    m = JobManager(on_event=events.append, progress_event_min_interval=10.0)
    try:

        def fn(h: JobHandle):
            h.set_progress(0.1, "a")
            h.set_progress(0.2, "b")
            h.set_progress(0.3, "c")
            h.set_progress(0.4)  # 无 stage 变化 → 被限频
            return None

        job = m.submit(JobType.train, fn)
        m.wait(job.job_id, 5)
        stages = [
            e["stage"] for e in events if e["job_id"] == job.job_id and e["status"] == "running"
        ]
        assert stages[1:4] == ["a", "b", "c"]
        assert events[-1]["progress"] == 1.0 and events[-1]["stage"] == "c"
    finally:
        assert m.shutdown(2.0)


def test_exclusive_conflict_and_parallel_types(mgr: JobManager):
    gate = threading.Event()

    def blocker(h: JobHandle):
        gate.wait(5)
        return {"ok": True}

    j1 = mgr.submit(JobType.train, blocker, model_id=1)
    _wait_until(lambda: mgr.get(j1.job_id).status is JobStatus.running)
    with pytest.raises(JobConflict) as ei:
        mgr.submit(JobType.train, blocker, model_id=2)
    err = ei.value
    assert err.code == "CONFLICT"
    assert err.detail == {"type": "train", "running_job_id": j1.job_id}
    assert err.to_dict()["detail"]["running_job_id"] == j1.job_id
    assert mgr.running(JobType.train).job_id == j1.job_id

    j2 = mgr.submit(JobType.capture, blocker)  # 不同类型可并行
    j3 = mgr.submit(JobType.train, blocker, exclusive=False)  # 显式放开互斥
    assert len(mgr.active_list()) == 3
    assert {j["job_id"] for j in mgr.active_list()} == {j1.job_id, j2.job_id, j3.job_id}
    gate.set()
    for j in (j1, j2, j3):
        assert mgr.wait(j.job_id, 5).status is JobStatus.succeeded
    assert mgr.list(active_only=True) == []
    assert [j.job_id for j in mgr.list()] == [j1.job_id, j2.job_id, j3.job_id]


def test_cancel_running_via_check_cancel(mgr: JobManager, events: list[dict]):
    started = threading.Event()

    def fn(h: JobHandle):
        started.set()
        while True:
            h.check_cancel()
            time.sleep(0.005)

    job = mgr.submit(JobType.valve_test, fn)
    assert started.wait(5)
    assert mgr.cancel(job.job_id) is True
    done = mgr.wait(job.job_id, 5)
    assert done.status is JobStatus.cancelled
    assert done.finished_at is not None
    assert events[-1]["status"] == "cancelled"
    assert mgr.cancel(job.job_id) is False, "已结束的任务取消返回 False"


def test_cancel_running_polling_then_return(mgr: JobManager):
    started = threading.Event()

    def fn(h: JobHandle):
        started.set()
        while not h.cancelled:
            time.sleep(0.005)
        return {"partial": 1}

    job = mgr.submit(JobType.capture, fn)
    assert started.wait(5)
    assert mgr.cancel(job.job_id)
    done = mgr.wait(job.job_id, 5)
    assert done.status is JobStatus.cancelled
    assert done.result == {"partial": 1}


def test_cancel_running_with_stop_event_exception(mgr: JobManager):
    def fn(h: JobHandle):
        h.cancel_event.wait(5)
        raise RuntimeError("训练被中断")  # 取消后抛任何异常都算 cancelled

    job = mgr.submit(JobType.train, fn)
    _wait_until(lambda: mgr.get(job.job_id).status is JobStatus.running)
    assert mgr.cancel(job.job_id)
    assert mgr.wait(job.job_id, 5).status is JobStatus.cancelled


def test_cancel_queued(mgr: JobManager, events: list[dict], monkeypatch):
    spawned: list[str] = []
    real_spawn = mgr._spawn
    monkeypatch.setattr(mgr, "_spawn", spawned.append)  # 延迟线程启动，使任务保持 queued
    ran = threading.Event()

    def fn(h: JobHandle):
        ran.set()
        return {"should": "not run"}

    job = mgr.submit(JobType.backup, fn)
    assert mgr.get(job.job_id).status is JobStatus.queued
    assert mgr.cancel(job.job_id) is True
    done = mgr.wait(job.job_id, 1)
    assert done.status is JobStatus.cancelled and done.started_at is None
    assert done.finished_at is not None and done.result is None
    assert events[-1]["status"] == "cancelled"
    n_events = len(events)
    real_spawn(job.job_id)  # 线程此时才启动，必须发现已取消而不执行 fn
    time.sleep(0.05)
    assert not ran.is_set(), "queued 取消后 fn 不得执行"
    assert mgr.get(job.job_id).status is JobStatus.cancelled
    assert len(events) == n_events
    assert mgr.cancel("j-00000000-9999") is False


def test_run_skips_fn_when_cancel_requested_before_start(
    mgr: JobManager, events: list[dict], monkeypatch
):
    """_run 进入 running 前检查取消事件：取消已请求但尚未收尾时不执行 fn。"""
    real_spawn = mgr._spawn
    monkeypatch.setattr(mgr, "_spawn", lambda jid: None)
    ran = threading.Event()

    def fn(h: JobHandle):
        ran.set()
        return None

    job = mgr.submit(JobType.backup, fn)
    mgr._entries[job.job_id].handle.cancel_event.set()  # 模拟 cancel 置位但尚未收尾的窗口
    real_spawn(job.job_id)
    done = mgr.wait(job.job_id, 2)
    assert done.status is JobStatus.cancelled and done.started_at is None
    assert not ran.is_set()
    mine = [e["status"] for e in events if e["job_id"] == job.job_id]
    assert mine == ["queued", "cancelled"]


def test_job_failure_records_message(mgr: JobManager, events: list[dict]):
    def fn(h: JobHandle):
        h.set_progress(0.3, "loading")
        raise ValueError("样本不足")

    job = mgr.submit(JobType.train, fn)
    done = mgr.wait(job.job_id, 5)
    assert done.status is JobStatus.failed
    assert done.message == "样本不足"
    assert done.progress == pytest.approx(0.3) and done.stage == "loading"
    assert events[-1]["status"] == "failed" and events[-1]["message"] == "样本不足"


def test_job_raises_job_cancelled(mgr: JobManager):
    def fn(h: JobHandle):
        raise JobCancelled("自行放弃")

    done = mgr.wait(mgr.submit(JobType.capture, fn).job_id, 5)
    assert done.status is JobStatus.cancelled


def test_non_dict_result_is_failure(mgr: JobManager):
    done = mgr.wait(mgr.submit(JobType.deploy, lambda h: [1, 2]).job_id, 5)
    assert done.status is JobStatus.failed
    assert "dict" in (done.message or "")

    done2 = mgr.wait(mgr.submit(JobType.deploy, lambda h: {"x": object()}).job_id, 5)
    assert done2.status is JobStatus.failed


def test_job_ids_unique_under_concurrent_submit(mgr: JobManager):
    ids: list[str] = []
    lock = threading.Lock()
    go = threading.Event()

    def submitter():
        go.wait(5)
        j = mgr.submit(JobType.capture, lambda h: None, exclusive=False)
        with lock:
            ids.append(j.job_id)

    threads = [threading.Thread(target=submitter, daemon=True, name=f"sub-{i}") for i in range(50)]
    for t in threads:
        t.start()
    go.set()
    for t in threads:
        t.join(10)
    assert len(ids) == 50 and len(set(ids)) == 50
    assert all(re.fullmatch(r"j-\d{8}-\d{4,}", i) for i in ids)
    seq = sorted(int(i.rsplit("-", 1)[1]) for i in ids)
    assert seq == list(range(seq[0], seq[0] + 50)), "进程内单调计数"
    for i in ids:
        assert mgr.wait(i, 5).status is JobStatus.succeeded


def test_shutdown_cancels_and_joins_threads():
    baseline = threading.active_count()
    m = JobManager()

    def fn(h: JobHandle):
        while not h.cancelled:
            time.sleep(0.005)

    jobs = [m.submit(t, fn) for t in (JobType.capture, JobType.train, JobType.valve_test)]
    _wait_until(lambda: all(m.get(j.job_id).status is JobStatus.running for j in jobs))
    assert threading.active_count() == baseline + 3
    t0 = time.monotonic()
    assert m.shutdown(timeout=2.0) is True
    assert time.monotonic() - t0 < 2.0
    assert threading.active_count() == baseline
    assert all(m.get(j.job_id).status is JobStatus.cancelled for j in jobs)
    with pytest.raises(RuntimeError):
        m.submit(JobType.train, fn)


def test_shutdown_joins_thread_still_emitting_finish_event():
    """结束事件回调较慢（模拟 WS 广播）时，shutdown 返回后工作线程必须已退出。"""
    baseline = threading.active_count()

    def slow_on_event(d: dict) -> None:
        if d["status"] in ("succeeded", "cancelled"):
            time.sleep(0.05)

    def one_round() -> None:
        m = JobManager(on_event=slow_on_event)
        job = m.submit(JobType.deploy, lambda h: None)
        # 状态已是 succeeded，但工作线程仍在回调里 sleep
        _wait_until(lambda: m.get(job.job_id).status is JobStatus.succeeded)
        assert m.shutdown(timeout=2.0) is True
        assert threading.active_count() == baseline

    for _ in range(5):
        one_round()


def test_shutdown_joins_progress_flush_timers():
    baseline = threading.active_count()
    m = JobManager(on_event=lambda d: None, progress_event_min_interval=10.0)
    gate = threading.Event()

    def fn(h: JobHandle):
        h.set_progress(0.1, "a")
        h.set_progress(0.2)  # 被限频 → 安排 10 s 后补发的定时器
        gate.wait(5)
        return None

    job = m.submit(JobType.train, fn)
    _wait_until(lambda: m.get(job.job_id).progress >= 0.2)
    assert threading.active_count() == baseline + 2, "工作线程 + 补发定时器"
    gate.set()
    m.wait(job.job_id, 2)
    assert m.shutdown(2.0) is True
    assert threading.active_count() == baseline


def test_progress_suppressed_value_is_flushed_by_timer(events: list[dict]):
    """被限频压下的进度在间隔到期后补发，HMI 不会长期看到陈旧进度；补发不晚于结束事件。"""
    m = JobManager(on_event=events.append, progress_event_min_interval=0.1)
    gate = threading.Event()
    try:

        def fn(h: JobHandle):
            h.set_progress(0.5, "epoch 1")
            h.set_progress(0.9)  # 紧跟其后 → 被限频
            gate.wait(5)  # 长时间无新进度
            return None

        job = m.submit(JobType.train, fn)
        _wait_until(
            lambda: any(
                e["job_id"] == job.job_id and e["status"] == "running" and e["progress"] == 0.9
                for e in events
            ),
            timeout=2.0,
        )
        running = [e["progress"] for e in events if e["status"] == "running"]
        assert running == [0.0, 0.5, 0.9]
        gate.set()
        m.wait(job.job_id, 2)
        mine = [e["status"] for e in events if e["job_id"] == job.job_id]
        assert mine[-1] == "succeeded" and mine.count("succeeded") == 1
        assert "running" not in mine[mine.index("succeeded") :]
    finally:
        assert m.shutdown(2.0)


def test_wait_default_timeout_is_finite():
    import inspect

    default = inspect.signature(JobManager.wait).parameters["timeout"].default
    assert isinstance(default, float) and 0 < default < 600


def test_history_trim_keeps_long_job_that_finishes_last():
    """裁剪按结束顺序：长任务运行期间有 ≥ max_history 个短任务结束，长任务结束后仍可查询。"""
    m = JobManager(max_history=2)
    gate = threading.Event()
    try:

        def long_fn(h: JobHandle) -> dict:
            gate.wait(5)
            return {"ok": 1}

        long_job = m.submit(JobType.train, long_fn)
        _wait_until(lambda: m.get(long_job.job_id).status is JobStatus.running)
        shorts: list[str] = []
        for _ in range(3):
            j = m.submit(JobType.valve_test, lambda h: None, exclusive=False)
            m.wait(j.job_id, 2)
            shorts.append(j.job_id)
        assert m.get(long_job.job_id) is not None, "运行中的任务不参与裁剪"
        assert m.get(shorts[0]) is None and m.get(shorts[1]) is not None
        gate.set()
        done = m.wait(long_job.job_id, 2)
        assert done.status is JobStatus.succeeded and done.result == {"ok": 1}
        assert m.get(long_job.job_id) is not None
        assert {j.job_id for j in m.list()} == {shorts[2], long_job.job_id}
    finally:
        gate.set()
        assert m.shutdown(2.0)


def test_history_trimmed_to_max_history():
    m = JobManager(max_history=3)
    try:
        ids: list[str] = []
        for _ in range(6):
            j = m.submit(JobType.deploy, lambda h: None)
            m.wait(j.job_id, 5)
            ids.append(j.job_id)
            assert len(m.list()) <= 3
        kept = [j.job_id for j in m.list()]
        assert kept == ids[-3:]
        assert m.get(ids[0]) is None
        with pytest.raises(KeyError):
            m.wait(ids[0], 0.1)
    finally:
        assert m.shutdown(2.0)


def test_wait_timeout_raises(mgr: JobManager):
    gate = threading.Event()
    job = mgr.submit(JobType.train, lambda h: gate.wait(5))
    with pytest.raises(TimeoutError):
        mgr.wait(job.job_id, 0.05)
    gate.set()
    mgr.wait(job.job_id, 5)


def test_job_handle_log_carries_job_id(mgr: JobManager, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="oresorter.jobs")

    def fn(h: JobHandle):
        h.log.info("hello", extra={"k": 1})
        return None

    job = mgr.submit(JobType.train, fn)
    mgr.wait(job.job_id, 5)
    recs = [r for r in caplog.records if r.getMessage() == "hello"]
    assert recs and recs[0].job_id == job.job_id and recs[0].k == 1


def test_on_event_exception_is_swallowed():
    def bad(_):
        raise RuntimeError("ws down")

    m = JobManager(on_event=bad)
    try:
        done = m.wait(m.submit(JobType.deploy, lambda h: {"ok": 1}).job_id, 5)
        assert done.status is JobStatus.succeeded
    finally:
        assert m.shutdown(2.0)


# ======================================================================================
# 运行统计
# ======================================================================================


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(1000.0)


def test_stats_per_minute_rates(clock: FakeClock):
    st = RunStats(clock=clock)
    for _ in range(60):
        clock.advance(0.5)
        st.on_chunk(20.0, n_particles=10, blown=3, per_class={"A": 7, "C": 3})
    snap = st.snapshot()
    assert snap["elapsed_s"] == 30.0
    assert snap["particles_per_min"] == 1200
    assert snap["blown_per_min"] == 360
    assert snap["per_class_per_min"] == {"A": 840, "C": 360}
    assert type(snap["particles_per_min"]) is int and type(snap["blown_per_min"]) is int
    assert all(type(v) is int for v in snap["per_class_per_min"].values())
    assert snap["chunks_total"] == 60
    assert snap["particles_total"] == 600 and snap["blown_total"] == 180
    assert snap["per_class_total"] == {"A": 420, "C": 180}
    assert snap["dropped_chunks"] == 0


def test_stats_window_slides(clock: FakeClock):
    st = RunStats(window_s=60.0, clock=clock)
    for _ in range(60):
        clock.advance(0.5)
        st.on_chunk(20.0, 10, 3, per_class={"A": 10})
    clock.advance(70.0)  # 全部样本过期
    snap = st.snapshot()
    assert snap["particles_per_min"] == 0 and snap["blown_per_min"] == 0
    assert snap["per_class_per_min"] == {"A": 0}
    assert snap["particles_total"] == 600, "累计计数不受窗口影响"
    st.on_chunk(5.0, 30, 6, per_class={"B": 30})
    snap = st.snapshot()
    assert snap["particles_per_min"] == 30  # 窗口已满 60 s：30/60*60
    assert snap["blown_per_min"] == 6
    assert snap["per_class_per_min"] == {"A": 0, "B": 30}


def test_stats_short_elapsed_uses_one_second(clock: FakeClock):
    st = RunStats(clock=clock)
    clock.advance(0.25)
    st.on_chunk(1.0, 10, 2)
    snap = st.snapshot()
    assert snap["particles_per_min"] == 600 and snap["blown_per_min"] == 120


def test_stats_latency_percentiles(clock: FakeClock):
    st = RunStats(latency_samples=1000, clock=clock)
    for v in range(100, 0, -1):
        st.on_chunk(float(v), 1, 0)
    snap = st.snapshot()
    assert snap["chunk_latency_ms_p50"] == 50
    assert snap["chunk_latency_ms_p99"] == 99
    assert snap["chunk_latency_ms_max"] == 100
    st2 = RunStats(latency_samples=10, clock=clock)
    for v in range(1, 101):
        st2.on_chunk(float(v), 1, 0)
    assert st2.snapshot()["chunk_latency_ms_p50"] == 95, "只保留最近 10 个样本"


def test_stats_dropped_reset_and_json(clock: FakeClock):
    st = RunStats(clock=clock)
    empty = st.snapshot()
    assert empty["chunk_latency_ms_p50"] == 0.0 and empty["particles_per_min"] == 0
    st.on_dropped()
    st.on_dropped(2)
    st.on_chunk(3.0, 4, 1, per_class={"A": 4}, dropped=1)
    snap = st.snapshot()
    assert snap["dropped_chunks"] == 4 and snap["chunks_total"] == 1
    text = json.dumps(snap)
    assert set(json.loads(text)) == {
        "particles_per_min",
        "blown_per_min",
        "per_class_per_min",
        "chunk_latency_ms_p50",
        "chunk_latency_ms_p99",
        "chunk_latency_ms_max",
        "chunks_total",
        "dropped_chunks",
        "particles_total",
        "blown_total",
        "per_class_total",
        "since",
        "elapsed_s",
    }
    since_before = snap["since"]
    clock.advance(5.0)
    st.reset()
    snap = st.snapshot()
    assert snap["elapsed_s"] == 0.0 and snap["chunks_total"] == 0
    assert snap["dropped_chunks"] == 0 and snap["per_class_total"] == {}
    assert snap["particles_total"] == 0 and snap["chunk_latency_ms_max"] == 0.0
    assert snap["since"] >= since_before


def test_stats_class_names_always_present(clock: FakeClock):
    st = RunStats(clock=clock, class_names=["A", "B", "C"])
    empty = st.snapshot()
    assert empty["per_class_per_min"] == {"A": 0, "B": 0, "C": 0}
    assert empty["per_class_total"] == {"A": 0, "B": 0, "C": 0}
    clock.advance(1.0)
    st.on_chunk(1.0, 5, 1, per_class={"B": 4, "Z": 1})  # Z 未配置
    snap = st.snapshot()
    assert list(snap["per_class_total"]) == ["A", "B", "C", "Z"], "配置顺序优先，额外类别排后"
    assert snap["per_class_total"] == {"A": 0, "B": 4, "C": 0, "Z": 1}
    assert snap["per_class_per_min"] == {"A": 0, "B": 240, "C": 0, "Z": 60}
    st.reset()
    assert st.snapshot()["per_class_total"] == {"A": 0, "B": 0, "C": 0}
    json.dumps(snap)


def test_stats_timer(clock: FakeClock):
    st = RunStats(clock=clock)
    with st.timer() as t:
        clock.advance(0.0315)
    assert t.ms == pytest.approx(31.5)
    st.on_chunk(t.ms, 1, 0)
    assert st.snapshot()["chunk_latency_ms_max"] == 31.5


def test_stats_thread_safety():
    st = RunStats()

    def worker():
        for _ in range(2000):
            st.on_chunk(1.0, 2, 1, per_class={"A": 1, "B": 1})

    threads = [threading.Thread(target=worker, daemon=True, name=f"st-{i}") for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    snap = st.snapshot()
    assert snap["chunks_total"] == 8000
    assert snap["particles_total"] == 16000 and snap["blown_total"] == 8000
    assert snap["per_class_total"] == {"A": 8000, "B": 8000}
