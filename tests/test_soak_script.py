"""``scripts/soak_test.py``：进程内调用 ``main()``（不开子进程），Mock 相机不限速。

覆盖：参数解析与越界拒绝（时长格式、间隔含 inf/nan/大于时长、速度、RSS 上限含 inf、日志级别、已存在的记录
文件、不存在的配置/模型目录；用法错误退出码 1 而不是 argparse 的 2）；快速训练临时模型 → 部署 → 运行 8 s /
每 2 s 采样 → JSONL ≥ 3 行、summary 字段齐全、退出码 0、chunks_total > 0、吹气链路到达 Mock 下位机、临时模型
删除且部署编号恢复、线程数回到基线；``--model-dir`` 导入外部模型 + CSV 记录 + SIGINT 提前优雅结束；复用已部署
模型 / 模型库内目录不删除；``--max-dropped`` 超限退出码 2；快速训练被取消 / 导入失败后临时槽位不残留；启动阶段
的瞬时故障（残留的坏部署模型）不计入判定；运行中故障立刻结束并以退出码 2 报告 faults_ok / running_ok；根 logger
INFO 级别下完整跑通；第二次信号抛 KeyboardInterrupt、main 返回 1；记录文件拒绝覆盖。

模型用脚本自己的 ``train_quick_model``（3 类 × 12 张合成样本，128 px，2 epoch，CPU，约 3 s）。整文件 < 90 s。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from oresorter.config import load
from oresorter.data.store import ModelStore
from oresorter.fsutil import atomic_write_json, read_json

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "soak_test.py"
CAM_WIDTH = 1024
CHUNK = 256
ROW_KEYS = {
    "t",
    "elapsed_s",
    "rss_mb",
    "threads",
    "os_threads",
    "cpu_percent",
    "run_state",
    "stats",
    "pipelines",
    "cameras",
    "mcu",
    "faults",
}
SUMMARY_KEYS = {
    "version",
    "config",
    "out",
    "format",
    "started_at",
    "ended_at",
    "stopped_by",
    "duration_requested_s",
    "duration_actual_s",
    "interval_s",
    "speed",
    "model",
    "samples",
    "rss",
    "threads",
    "chunks_total",
    "dropped_total",
    "latency_p99_ms",
    "rss_growth_mb",
    "threads_leaked",
    "stats",
    "pipelines",
    "cameras",
    "mcu",
    "faults_seen",
    "startup_faults",
    "faults_end",
    "checks",
    "failed_checks",
    "passed",
    "exit_code",
    "error",
    "notes",
    "summary_path",
}
RSS_KEYS = {
    "start_mb",
    "baseline_mb",
    "end_mb",
    "max_mb",
    "growth_mb",
    "peak_growth_mb",
    "limit_mb",
}
THREAD_KEYS = {"baseline", "running_max", "end", "leaked"}
CSV_COLUMNS = {
    "t",
    "elapsed_s",
    "rss_mb",
    "threads",
    "chunks_total",
    "dropped_chunks",
    "latency_p99_ms",
    "queue_len",
    "stitch_dropped",
    "stitch_lost",
    "stitch_truncated",
    "errors",
    "eject_sent",
    "mcu_eject_rx",
    "faults",
}


# --------------------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------------------


def _load_script():
    """按路径加载脚本；dataclass + ``from __future__ import annotations`` 要求模块先登记到 ``sys.modules``。"""
    spec = importlib.util.spec_from_file_location("soak_test_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def soak():
    """脚本模块（按文件路径加载，scripts/ 不是包）。"""
    return _load_script()


@pytest.fixture(autouse=True)
def _quiet_root_logger():
    """不限速时拼接器每次丢块都告警，整文件按 WARNING 以上运行并在结束后恢复级别。"""
    root = logging.getLogger()
    saved = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(saved)


def write_cfg(root: Path) -> Path:
    """临时 yaml：继承 config/default.yaml，Mock 相机 1024 px × 256 行块、color 背景、CPU 快速训练配方。"""
    data = {
        "base": str(REPO / "config" / "default.yaml"),
        "system": {
            "data_dir": str(root / "data"),
            "mcu_kind": "mock",
            "auto_run_on_boot": False,
            "min_free_disk_mb": 10,
        },
        "capture": {"bg_mode": "color", "bg_color": [0, 0, 255], "min_area_px": 100},
        "ml": {"device": "cpu", "pretrained": False, "num_workers": 0},
        "pipeline": {"overlap_lines": 128, "queue_size": 8},
        "cameras": [
            {
                "id": 1,
                "name": "相机1(Mock)",
                "kind": "mock",
                "mock": {
                    "speed": 1.0,
                    "seed": 7,
                    "particles_per_chunk": 4,
                    "particle_px": [24, 80],
                },
                "width": CAM_WIDTH,
                "offset_x": 0,
                "height": 1024,
                "runtime_chunk_lines": CHUNK,
                "start_px": 0,
                "end_px": CAM_WIDTH,
                "line_rate_hz": 9000,
            }
        ],
    }
    path = root / "soak.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def wait_until(pred: Callable[[], bool], timeout: float, step: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        time.sleep(step)
    return True


def assert_threads_back_to(baseline: int) -> None:
    """脚本自己最多等 2 s；这里再给 3 s 余量后断言线程数与开始时相同。"""
    wait_until(lambda: threading.active_count() <= baseline, timeout=3.0)
    names = sorted(t.name for t in threading.enumerate())
    assert threading.active_count() == baseline, f"线程未回到基线 {baseline}: {names}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory, soak) -> Path:
    """用脚本的 ``train_quick_model`` 在独立模型库里训一次（模块级共享），供 --model-dir / 已部署路径复用。"""
    root = tmp_path_factory.mktemp("trained")
    cfg = load(write_cfg(root))
    store = ModelStore(
        root / "models", cfg.system.max_models, list(cfg.system.class_names), min_free_mb=0
    )
    store.ensure_root()
    result = soak.train_quick_model(store, 1, cfg.system.class_names, cfg.ml)
    assert result["artifacts"]["onnx"] is True and store.trained(1)
    return store.model_dir(1)


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------


def test_parse_duration_formats(soak):
    assert soak.parse_duration("90") == 90.0
    assert soak.parse_duration("90s") == 90.0
    assert soak.parse_duration("30m") == 1800.0
    assert soak.parse_duration("1h") == 3600.0
    assert soak.parse_duration("48h") == 48 * 3600.0
    assert soak.parse_duration("2d") == 2 * 86400.0
    assert soak.parse_duration(" 1.5H ") == 5400.0
    for bad in ("0", "-5", "abc", "1x", "", "1h30m"):
        with pytest.raises(soak.SoakError):
            soak.parse_duration(bad)


def test_parse_args_defaults_and_rejections(soak, tmp_path: Path):
    cfg = write_cfg(tmp_path)
    existing = tmp_path / "old.jsonl"
    existing.write_text("{}\n", encoding="utf-8")
    args = soak.parse_args(["--config", str(cfg), "--duration", "48h"])
    assert args.duration_s == 48 * 3600 and args.interval == 60.0 and args.speed == 1.0
    assert args.rss_limit_mb == 50.0 and args.out is None and args.model_dir is None
    assert args.max_dropped is None and args.log_level == "INFO" and args.overwrite is False
    # 间隔恰等于时长允许；日志级别不区分大小写并统一为大写；已存在的记录文件加 --overwrite 才接受
    assert soak.parse_args(["--config", str(cfg), "--duration", "60"]).interval == 60.0
    assert soak.parse_args(["--config", str(cfg), "--log-level", "debug"]).log_level == "DEBUG"
    ok = soak.parse_args(["--config", str(cfg), "--out", str(existing), "--overwrite"])
    assert ok.overwrite is True and ok.out == existing
    bad_cases = [
        (["--duration", "0"], "duration"),
        (["--duration", "1 week"], "duration"),
        (["--interval", "0"], "interval"),
        (["--interval", "-3"], "interval"),
        (["--interval", "inf"], "interval"),
        (["--interval", "1e400"], "interval"),
        (["--interval", "nan"], "interval"),
        (["--duration", "5", "--interval", "10"], "interval"),
        (["--speed", "-0.5"], "speed"),
        (["--speed", "5000"], "speed"),
        (["--speed", "inf"], "speed"),
        (["--speed", "nan"], "speed"),
        (["--rss-limit-mb", "0"], "rss-limit-mb"),
        (["--rss-limit-mb", "inf"], "rss-limit-mb"),
        (["--rss-limit-mb", "nan"], "rss-limit-mb"),
        (["--max-dropped", "-1"], "max-dropped"),
        (["--log-level", "BOGUS"], "log-level"),
        (["--log-level", ""], "log-level"),
        (["--out", str(existing)], "已存在"),
        (["--model-dir", str(tmp_path / "nope")], "model-dir"),
        (["--nonsense"], "参数错误"),
    ]
    for extra, needle in bad_cases:
        with pytest.raises(soak.SoakError) as ei:
            soak.parse_args(["--config", str(cfg), *extra])
        assert needle in str(ei.value), (extra, str(ei.value))
    with pytest.raises(soak.SoakError):
        soak.parse_args(["--config", str(tmp_path / "missing.yaml")])
    assert existing.read_text(encoding="utf-8") == "{}\n", "参数校验不得改动已存在的记录文件"


def test_main_returns_1_on_bad_args_without_running(soak, tmp_path: Path, capsys):
    """用法错误走退出码 1（2 表示“超限”），说明写到 stderr（不是裸 traceback）；``--help`` 仍是 SystemExit(0)。"""
    base = threading.active_count()
    cfg = str(write_cfg(tmp_path))
    assert soak.main(["--config", cfg, "--duration", "nope"]) == 1
    err = capsys.readouterr().err
    assert "duration" in err
    assert soak.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    # --log-level 非法：以“错误: …”说明退出，而不是 setup_logging 的 ValueError traceback
    assert soak.main(["--config", cfg, "--duration", "2m", "--log-level", "BOGUS"]) == 1
    err = capsys.readouterr().err
    assert "错误" in err and "log-level" in err and "Traceback" not in err
    existing = tmp_path / "keep.jsonl"
    existing.write_text("precious\n", encoding="utf-8")
    assert soak.main(["--config", cfg, "--duration", "2m", "--out", str(existing)]) == 1
    assert "已存在" in capsys.readouterr().err
    assert existing.read_text(encoding="utf-8") == "precious\n"
    with pytest.raises(SystemExit) as ei:
        soak.main(["--help"])
    assert ei.value.code == 0
    assert threading.active_count() == base
    assert not (tmp_path / "data").exists(), "参数错误不应碰数据目录"


def test_build_config_forces_mock_and_speed(soak, tmp_path: Path):
    cfg_path = write_cfg(tmp_path)
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data["system"]["mcu_kind"] = "tcp"
    data["system"]["auto_run_on_boot"] = True
    data["capture"] = {"bg_mode": "column"}
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg, notes = soak.build_config(cfg_path, 0.0)
    assert cfg.system.mcu_kind == "mock" and cfg.system.auto_run_on_boot is False
    assert cfg.cameras[0].mock.speed == 0.0
    assert cfg.capture.bg_mode == "color", "列背景未学习 → 回退 color"
    assert list(cfg.capture.bg_color) == list(cfg.cameras[0].mock.bg_color)
    assert cfg.system.data_dir == (tmp_path / "data").resolve()
    assert any("mcu_kind" in n for n in notes) and any("auto_run_on_boot" in n for n in notes)
    assert any("列背景未学习" in n for n in notes)


# --------------------------------------------------------------------------------------
# 完整运行
# --------------------------------------------------------------------------------------


def test_quick_train_run_records_summary_and_cleans_up(soak, tmp_path: Path, capsys):
    cfg_path = write_cfg(tmp_path)
    out = tmp_path / "logs" / "run.jsonl"
    base_threads = threading.active_count()
    t0 = time.monotonic()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "8",
            "--interval",
            "2",
            "--speed",
            "0",
            "--out",
            str(out),
        ]
    )
    wall = time.monotonic() - t0
    assert rc == 0, capsys.readouterr()
    printed = capsys.readouterr().out
    assert "soak_test 汇总" in printed and "结论: 合格（退出码 0）" in printed

    rows = read_jsonl(out)
    assert len(rows) >= 3
    assert all(set(r) >= ROW_KEYS for r in rows)
    elapsed = [r["elapsed_s"] for r in rows]
    # 第一行在 2 s 采样点之后（慢机器上可能晚到，只约束下界与总墙钟）
    assert elapsed == sorted(elapsed) and 1.9 <= elapsed[0] <= wall and 7.5 <= elapsed[-1] <= wall
    last = rows[-1]
    assert last["run_state"] == "running" and last["faults"] == []
    assert last["stats"]["chunks_total"] > 0
    assert last["stats"]["chunks_total"] >= rows[0]["stats"]["chunks_total"] > 0
    assert {
        "chunk_latency_ms_p50",
        "chunk_latency_ms_p99",
        "dropped_chunks",
        "particles_total",
    } <= set(last["stats"])
    pipe = last["pipelines"]["1"]
    assert {
        "queue_len",
        "queue_dropped",
        "dropped",
        "lost",
        "truncated",
        "errors",
        "chunks",
        "healthy",
    } <= set(pipe)
    assert pipe["running"] is True and pipe["healthy"] is True and pipe["errors"] == 0
    assert last["cameras"]["1"]["connected"] is True
    assert last["mcu"]["connected"] is True and {"eject_rx", "eject_done"} <= set(
        last["mcu"]["counters"]
    )
    assert all(isinstance(r["rss_mb"], float) and r["rss_mb"] > 0 for r in rows)
    assert all(r["threads"] > base_threads for r in rows), "运行中应有流水线/相机线程"

    summary_path = out.parent / "soak_summary.json"
    summary = read_json(summary_path)
    assert set(summary) >= SUMMARY_KEYS, SUMMARY_KEYS - set(summary)
    assert set(summary["rss"]) >= RSS_KEYS and set(summary["threads"]) >= THREAD_KEYS
    assert (
        summary["exit_code"] == 0 and summary["passed"] is True and summary["failed_checks"] == []
    )
    assert summary["stopped_by"] == "deadline" and summary["error"] is None
    assert summary["samples"] == len(rows) and summary["format"] == "jsonl"
    assert summary["chunks_total"] == last["stats"]["chunks_total"] > 0
    assert summary["duration_requested_s"] == 8.0 and 7.5 <= summary["duration_actual_s"] <= wall
    assert summary["interval_s"] == 2.0 and summary["speed"] == 0.0
    assert summary["model"]["source"] == "quick-train" and summary["model"]["temporary"] is True
    assert (
        summary["model"]["backend"] in ("onnx", "torch") and summary["model"]["image_size"] == 128
    )
    assert summary["model"]["classes"] == ["A", "B", "C"]
    rss = summary["rss"]
    assert rss["baseline_mb"] == rows[0]["rss_mb"] and rss["end_mb"] == last["rss_mb"]
    assert rss["max_mb"] >= max(rss["end_mb"], rss["baseline_mb"])
    assert rss["growth_mb"] == round(rss["end_mb"] - rss["baseline_mb"], 2)
    assert rss["growth_mb"] <= rss["limit_mb"] == 50.0
    assert summary["checks"]["rss_ok"] is True and summary["checks"]["threads_ok"] is True
    assert summary["checks"]["faults_ok"] is True and summary["checks"]["dropped_ok"] is None
    assert summary["checks"]["running_ok"] is True
    assert summary["checks"]["running_detail"] == {
        "run_state_end": "running",
        "chunks_grew": True,
        "growth_gap_s": 1.0,
    }
    assert summary["threads"]["leaked"] == [] and summary["threads_leaked"] == 0
    assert summary["threads"]["running_max"] > summary["threads"]["baseline"]
    assert summary["faults_seen"] == {} and summary["faults_end"] == []
    assert summary["startup_faults"] == {}
    assert summary["latency_p99_ms"] == last["stats"]["chunk_latency_ms_p99"] > 0
    # 不限速必然丢块（相机远快于处理）：合计 = 流水线丢弃 + 相机侧丢弃，且被如实记录而不影响退出码
    assert (
        summary["dropped_total"]
        == last["stats"]["dropped_chunks"] + last["cameras"]["1"]["dropped_chunks"]
    )
    assert summary["dropped_total"] > 0
    # 吹气链路：blow.classes 为空时脚本改为全类别/阈值 0，喷阀批应到达 Mock 下位机
    assert any("blow.classes" in n for n in summary["notes"])
    assert (
        summary["stats"]["blown_total"] > 0 and pipe["eject_sent"] > 0 and pipe["eject_failed"] == 0
    )
    assert last["mcu"]["counters"]["eject_rx"] > 0
    assert summary["summary_path"] == str(summary_path) and summary["out"] == str(out)

    # 临时模型删除、部署编号恢复为 None、磁盘上没有残留
    data_dir = tmp_path / "data"
    assert not (data_dir / "models" / str(summary["model"]["id"])).exists()
    assert read_json(data_dir / "state.json")["deployed_model_id"] is None
    assert read_json(data_dir / "state.json")["last_run_state"] == "stopped"
    assert not (data_dir / "config.json").exists(), "老化测试不应落盘配置"
    assert_threads_back_to(base_threads)


def test_sigint_stops_early_with_model_dir_and_csv(soak, tmp_path: Path, trained_model_dir: Path):
    """``--model-dir`` 导入外部模型到临时槽位；CSV 记录；运行中 SIGINT → 优雅结束、汇总照常、退出码 0。"""
    cfg_path = write_cfg(tmp_path)
    out = tmp_path / "soak.csv"
    base_threads = threading.active_count()
    fired = threading.Event()

    def fire() -> None:
        # 等第一行数据写出（表头 + 1 行）再发 SIGINT，确保信号落在采样循环内
        ok = wait_until(
            lambda: out.exists() and len(out.read_text(encoding="utf-8").splitlines()) >= 2,
            timeout=60.0,
        )
        if ok:
            fired.set()
            os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Thread(target=fire, name="soak-sigint-timer", daemon=True)
    timer.start()
    t0 = time.monotonic()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "120",
            "--interval",
            "1",
            "--speed",
            "0",
            "--out",
            str(out),
            "--model-dir",
            str(trained_model_dir),
        ]
    )
    timer.join(5.0)
    assert fired.is_set() and not timer.is_alive()
    assert rc == 0
    assert time.monotonic() - t0 < 60, "SIGINT 后应提前结束"

    summary = read_json(out.parent / "soak_summary.json")
    assert (
        summary["stopped_by"] == "SIGINT"
        and summary["exit_code"] == 0
        and summary["passed"] is True
    )
    assert summary["duration_actual_s"] < 30 and summary["duration_requested_s"] == 120.0
    assert summary["format"] == "csv" and summary["samples"] >= 1
    assert summary["model"]["source"] == "model-dir" and summary["model"]["temporary"] is True
    assert summary["model"]["classes"] == ["A", "B", "C"] and summary["chunks_total"] > 0
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == summary["samples"] >= 1
    assert set(rows[0]) >= CSV_COLUMNS
    assert int(rows[-1]["chunks_total"]) == summary["chunks_total"] > 0
    assert rows[-1]["faults"] == "" and rows[-1]["run_state"] == "running"
    assert float(rows[-1]["rss_mb"]) == summary["rss"]["end_mb"]
    # 源目录原样保留；导入的临时槽位已删除；部署编号恢复
    assert (trained_model_dir / "train" / "best.pt").is_file()
    assert not (tmp_path / "data" / "models" / str(summary["model"]["id"])).exists()
    assert read_json(tmp_path / "data" / "state.json")["deployed_model_id"] is None
    # 信号处理器已恢复默认（后续 pytest 不受影响）
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert_threads_back_to(base_threads)


def test_reuses_deployed_model_without_touching_store(
    soak, tmp_path: Path, trained_model_dir: Path
):
    """数据目录已有已部署模型且未给 --model-dir → 直接复用，结束后模型与部署编号原样保留。"""
    cfg_path = write_cfg(tmp_path)
    data_dir = tmp_path / "data"
    shutil.copytree(trained_model_dir, data_dir / "models" / "2")
    atomic_write_json(data_dir / "state.json", {"deployed_model_id": 2})
    out = tmp_path / "reuse.jsonl"
    base_threads = threading.active_count()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "2",
            "--interval",
            "1",
            "--speed",
            "0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    summary = read_json(out.parent / "soak_summary.json")
    assert summary["model"] == {
        "id": 2,
        "source": "deployed",
        "temporary": False,
        "backend": summary["model"]["backend"],
        "classes": ["A", "B", "C"],
        "image_size": 128,
    }
    assert summary["chunks_total"] > 0 and summary["passed"] is True
    assert (data_dir / "models" / "2" / "train" / "best.pt").is_file()
    assert read_json(data_dir / "state.json")["deployed_model_id"] == 2
    assert len(read_jsonl(out)) == summary["samples"] >= 2
    assert_threads_back_to(base_threads)


def test_model_dir_inside_store_kept_and_max_dropped_exit_2(
    soak, tmp_path: Path, trained_model_dir: Path
):
    """--model-dir 指向模型库内目录 → 不复制不删除；不限速必丢块，``--max-dropped 0`` → 退出码 2 且汇总说明原因。"""
    cfg_path = write_cfg(tmp_path)
    data_dir = tmp_path / "data"
    shutil.copytree(trained_model_dir, data_dir / "models" / "3")
    out_dir = tmp_path / "outdir"
    out_dir.mkdir()
    base_threads = threading.active_count()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "2",
            "--interval",
            "1",
            "--speed",
            "0",
            "--out",
            str(out_dir),
            "--model-dir",
            str(data_dir / "models" / "3"),
            "--max-dropped",
            "0",
        ]
    )
    assert rc == 2
    summary = read_json(out_dir / "soak_summary.json")
    assert summary["exit_code"] == 2 and summary["passed"] is False
    assert summary["failed_checks"] == ["dropped_ok"] and summary["checks"]["max_dropped"] == 0
    assert summary["checks"]["dropped_ok"] is False and summary["checks"]["rss_ok"] is True
    assert (
        summary["dropped_total"] > 0
        and summary["error"] is None
        and summary["stopped_by"] == "deadline"
    )
    assert summary["model"]["id"] == 3 and summary["model"]["source"] == "model-dir"
    assert summary["model"]["temporary"] is False
    # 目录型 --out：在其中按默认名创建 jsonl
    logs = sorted(out_dir.glob("soak_*.jsonl"))
    assert (
        len(logs) == 1
        and summary["out"] == str(logs[0])
        and len(read_jsonl(logs[0])) == summary["samples"]
    )
    assert (data_dir / "models" / "3" / "train" / "best.pt").is_file(), "模型库内目录不得删除"
    assert (
        read_json(data_dir / "state.json")["deployed_model_id"] is None
    ), "运行前未部署 → 恢复为 None"
    assert_threads_back_to(base_threads)


# --------------------------------------------------------------------------------------
# 失败路径 / 故障 / 信号
# --------------------------------------------------------------------------------------


def test_failed_quick_train_and_import_leave_no_temp_model(
    soak, tmp_path: Path, trained_model_dir: Path, monkeypatch, capsys
):
    """快速训练被取消（等价于训练中 Ctrl-C）或 --model-dir 导入复制失败 → 退出码 1、“未开始”说明，
    且 store.create 过的 soak-temp 临时槽位（含样本/缩略图）被删除，不占用用户模型库。"""
    cfg_path = write_cfg(tmp_path)
    data_dir = tmp_path / "data"
    base_threads = threading.active_count()
    common = ["--config", str(cfg_path), "--duration", "2", "--interval", "1", "--speed", "0"]

    created: list[Path] = []
    real_run = soak.T.run

    def cancelled_run(model_dir, *args, **kwargs):
        created.append(Path(model_dir))
        assert (Path(model_dir) / "meta.json").is_file(), "T.run 被调用时槽位已创建"
        raise soak.TrainError("CANCELLED", "训练已取消")

    monkeypatch.setattr(soak.T, "run", cancelled_run)
    assert soak.main([*common, "--out", str(tmp_path / "a.jsonl")]) == 1
    err = capsys.readouterr().err
    assert "未开始" in err and "快速训练被中断" in err
    assert created and not created[0].exists(), "取消后临时槽位必须删除"
    assert not any((data_dir / "models").iterdir()), "模型库应无任何残留"
    assert not (tmp_path / "a.jsonl").exists(), "没开始运行不写记录"
    assert read_json(data_dir / "state.json")["deployed_model_id"] is None
    monkeypatch.setattr(soak.T, "run", real_run)

    def failing_copy(src, dst, *a, **k):
        raise OSError("磁盘已满（注入）")

    monkeypatch.setattr(soak.shutil, "copy2", failing_copy)
    rc = soak.main(
        [*common, "--out", str(tmp_path / "b.jsonl"), "--model-dir", str(trained_model_dir)]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "未开始" in err and "磁盘已满" in err
    assert not any((data_dir / "models").iterdir()), "导入失败后临时槽位必须删除"
    assert (trained_model_dir / "train" / "best.pt").is_file(), "源目录不受影响"
    assert_threads_back_to(base_threads)


def test_startup_fault_from_stale_deployed_model_not_counted(
    soak, tmp_path: Path, trained_model_dir: Path
):
    """state.json 残留一个权重损坏的 deployed_model_id → ctx.start() 置 MODEL_LOAD_FAILED（状态进 fault），
    部署新模型清除故障后正常运行：这类启动阶段瞬时故障只进 startup_faults / notes，整次运行判定合格。"""
    cfg_path = write_cfg(tmp_path)
    cfg = load(cfg_path)
    data_dir = tmp_path / "data"
    store = ModelStore(
        data_dir / "models", cfg.system.max_models, list(cfg.system.class_names), min_free_mb=0
    )
    store.ensure_root()
    store.create(2, "坏模型")
    (store.train_dir(2) / "best.pt").write_bytes(b"this is not a checkpoint")
    atomic_write_json(data_dir / "state.json", {"deployed_model_id": 2})
    out = tmp_path / "stale.jsonl"
    base_threads = threading.active_count()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "3",
            "--interval",
            "1",
            "--speed",
            "0",
            "--out",
            str(out),
            "--model-dir",
            str(trained_model_dir),
        ]
    )
    assert rc == 0
    summary = read_json(out.parent / "soak_summary.json")
    assert summary["passed"] is True and summary["failed_checks"] == []
    assert summary["checks"]["faults_ok"] is True and summary["checks"]["running_ok"] is True
    assert "MODEL_LOAD_FAILED" in summary["startup_faults"]
    assert summary["faults_seen"] == {} and summary["faults_end"] == []
    assert any("启动/模型准备阶段" in n and "MODEL_LOAD_FAILED" in n for n in summary["notes"])
    rows = read_jsonl(out)
    assert len(rows) >= 2 and all(r["run_state"] == "running" and r["faults"] == [] for r in rows)
    assert summary["model"]["source"] == "model-dir" and summary["model"]["temporary"] is True
    # 用户的坏模型原样保留（不是脚本创建的），临时槽位删除，部署编号恢复为启动后的 None
    assert (store.train_dir(2) / "best.pt").read_bytes() == b"this is not a checkpoint"
    assert not store.model_dir(summary["model"]["id"]).exists()
    assert read_json(data_dir / "state.json")["deployed_model_id"] is None
    assert_threads_back_to(base_threads)


def test_fault_during_run_stops_immediately_with_exit_2(
    soak, tmp_path: Path, trained_model_dir: Path, monkeypatch
):
    """运行中故障置位又清除（状态机 fault → stopped，流水线已被停掉）：脚本不等下一个采样点、不跑到
    deadline，立刻结束；stopped_by=fault，faults_ok 与 running_ok 都不通过，退出码 2。"""
    cfg_path = write_cfg(tmp_path)
    data_dir = tmp_path / "data"
    shutil.copytree(trained_model_dir, data_dir / "models" / "2")
    atomic_write_json(data_dir / "state.json", {"deployed_model_id": 2})
    out = tmp_path / "fault.jsonl"
    instances: list = []

    class Recording(soak.AppContext):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            instances.append(self)

    monkeypatch.setattr(soak, "AppContext", Recording)
    injected = threading.Event()

    def inject() -> None:
        ok = wait_until(lambda: out.exists() and len(read_jsonl(out)) >= 1, timeout=60.0)
        if ok and instances:
            ctx = instances[-1]
            ctx.faults.set("PIPELINE_ERROR", "注入故障")
            time.sleep(0.4)
            ctx.faults.clear("PIPELINE_ERROR")
            injected.set()

    base_threads = threading.active_count()
    th = threading.Thread(target=inject, name="soak-fault-injector", daemon=True)
    th.start()
    t0 = time.monotonic()
    rc = soak.main(
        [
            "--config",
            str(cfg_path),
            "--duration",
            "60",
            "--interval",
            "1",
            "--speed",
            "0",
            "--out",
            str(out),
        ]
    )
    wall = time.monotonic() - t0
    th.join(5.0)
    assert injected.is_set() and not th.is_alive()
    assert rc == 2 and wall < 30, "故障后应立刻结束，而不是跑满 60 s"
    summary = read_json(out.parent / "soak_summary.json")
    assert summary["exit_code"] == 2 and summary["passed"] is False and summary["error"] is None
    assert summary["stopped_by"] == "fault" and summary["duration_actual_s"] < 15
    assert {"faults_ok", "running_ok"} <= set(summary["failed_checks"])
    assert summary["checks"]["faults_ok"] is False and summary["checks"]["running_ok"] is False
    assert summary["faults_seen"] == {"PIPELINE_ERROR": "注入故障"}
    assert summary["startup_faults"] == {}
    rows = read_jsonl(out)
    assert len(rows) == summary["samples"] >= 2, "提前结束时总会补一行记录结束时刻的状态"
    assert rows[-1]["run_state"] in ("fault", "stopped")
    assert rows[0]["run_state"] == "running"
    assert_threads_back_to(base_threads)


def test_runs_at_info_log_level(soak, tmp_path: Path, trained_model_dir: Path):
    """根 logger INFO（脚本作为 CLI 的默认级别）下完整跑通：脚本自身所有 ``extra`` 键都不与 LogRecord
    冲突，部署等待不依赖任务结束事件（JobManager 的结束日志在 INFO 下抛 KeyError 也不挂死）。"""
    cfg_path = write_cfg(tmp_path)
    data_dir = tmp_path / "data"
    shutil.copytree(trained_model_dir, data_dir / "models" / "2")
    atomic_write_json(data_dir / "state.json", {"deployed_model_id": 2})
    out = tmp_path / "info.jsonl"
    root = logging.getLogger()
    stitch = logging.getLogger("oresorter.vision.stitch")
    saved_root, saved_stitch = root.level, stitch.level
    root.setLevel(logging.INFO)
    stitch.setLevel(logging.ERROR)  # 不限速时每次丢块都告警，单独压掉
    base_threads = threading.active_count()
    try:
        rc = soak.main(
            [
                "--config",
                str(cfg_path),
                "--duration",
                "2",
                "--interval",
                "1",
                "--speed",
                "0",
                "--out",
                str(out),
                "--log-level",
                "info",
            ]
        )
    finally:
        root.setLevel(saved_root)
        stitch.setLevel(saved_stitch)
    assert rc == 0
    summary = read_json(out.parent / "soak_summary.json")
    assert summary["passed"] is True and summary["chunks_total"] > 0
    assert summary["model"]["source"] == "deployed" and summary["samples"] >= 2
    assert_threads_back_to(base_threads)


def test_second_signal_forces_keyboard_interrupt_and_main_returns_1(
    soak, tmp_path: Path, monkeypatch, capsys
):
    """第一次 SIGINT → on_signal（优雅结束）；第二次 → 恢复默认处理器并抛 KeyboardInterrupt；
    ``main`` 捕获 KeyboardInterrupt 打印“强制退出”并返回 1。"""
    calls: list[str] = []
    with soak.signal_handlers(calls.append):
        assert signal.getsignal(signal.SIGINT) is not signal.default_int_handler
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.05)
        assert calls == ["SIGINT"]
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "第二次信号后已恢复"
    assert calls == ["SIGINT"] and signal.getsignal(signal.SIGINT) is signal.default_int_handler

    def interrupted_run(self) -> int:
        raise KeyboardInterrupt("再次收到 SIGINT，强制退出")

    monkeypatch.setattr(soak.SoakRunner, "run", interrupted_run)
    base = threading.active_count()
    rc = soak.main(["--config", str(write_cfg(tmp_path)), "--duration", "2m", "--interval", "1"])
    assert rc == 1
    assert "强制退出" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert threading.active_count() == base


def test_chunk_growth_reference_and_gap(soak, tmp_path: Path):
    """running_ok 的块数增长比较以 ≥ gap 秒前的行为参照：补采样紧跟周期采样之后不误判；真停摆能判出。"""
    g = soak.ChunkGrowth(1.0, 0.5)
    assert g.grew() is None and g.reference() is None
    g.add(0.5, 10)
    assert g.grew() is None, "只有一行"
    g.add(1.0, 20)
    assert g.grew() is None, "相隔 0.5 s < gap，没有参照"
    g.add(1.5, 30)
    assert g.reference() == (0.5, 10) and g.grew() is True
    g.add(1.52, 30)
    assert g.reference() == (0.5, 10) and g.grew() is True, "紧跟其后的补采样仍以 0.5 s 行为参照"
    g.add(2.6, 30)
    assert g.reference() == (1.52, 30) and g.grew() is False, "相对 ≥1 s 前最近一行块数冻结 → 停摆"
    assert g.points.maxlen == int(1.0 / 0.5) + 3
    # gap：max(1 s, 3 × 最慢相机块周期)；测试配置 9 kHz / 256 行
    cfg = load(write_cfg(tmp_path))
    assert soak.growth_gap_s(cfg, 1.0) == 1.0 and soak.growth_gap_s(cfg, 0.0) == 1.0
    assert soak.growth_gap_s(cfg, 0.01) == pytest.approx(3 * 256 / 90.0)


def test_row_writer_refuses_overwrite(soak, tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text("old\n", encoding="utf-8")
    with pytest.raises(soak.SoakError) as ei:
        soak.RowWriter(path)
    assert "已存在" in str(ei.value) and path.read_text(encoding="utf-8") == "old\n"
    w = soak.RowWriter(path, overwrite=True)
    w.close()
    assert path.read_text(encoding="utf-8") == ""
    w2 = soak.RowWriter(tmp_path / "new" / "y.csv")
    assert w2.format == "csv" and w2.rows == 0
    w2.close()
    assert (tmp_path / "new" / "y.csv").is_file()


def test_format_summary_mentions_failures(soak):
    summary = {
        "exit_code": 2,
        "duration_actual_s": 10.0,
        "duration_requested_s": 10,
        "samples": 2,
        "interval_s": 5,
        "speed": 1.0,
        "stopped_by": "deadline",
        "model": {"id": 1, "source": "quick-train", "backend": "onnx", "classes": ["A", "B"]},
        "rss": {
            "start_mb": 100.0,
            "baseline_mb": 200.0,
            "end_mb": 260.0,
            "max_mb": 261.0,
            "growth_mb": 60.0,
            "peak_growth_mb": 61.0,
            "limit_mb": 50.0,
        },
        "threads": {"baseline": 1, "running_max": 6, "end": 2, "leaked": ["ghost"]},
        "chunks_total": 100,
        "dropped_total": 0,
        "stats": {},
        "faults_seen": {"PIPELINE_ERROR": "x"},
        "startup_faults": {"MODEL_LOAD_FAILED": "坏权重"},
        "notes": [],
        "failed_checks": ["rss_ok", "threads_ok", "faults_ok", "running_ok"],
        "out": "/tmp/x.jsonl",
    }
    text = soak.format_summary(summary)
    assert "超限（退出码 2）" in text and "泄漏 1: ghost" in text and "PIPELINE_ERROR" in text
    assert "未通过: rss_ok, threads_ok, faults_ok, running_ok" in text and "+60.0 MB" in text
    assert "启动阶段故障（不计入判定）: MODEL_LOAD_FAILED" in text
