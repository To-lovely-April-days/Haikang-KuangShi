"""训练配方：合成 3 类小样本 → 训练产物、进度、meta 更新、错误路径、数据集工具、子进程入口。

不依赖 scripts/make_synthetic；样本用 camera/mock.draw_particle 现场合成。整文件 CPU 上 < 60 s。
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from oresorter.camera.mock import CLASS_PALETTE, draw_particle
from oresorter.fsutil import read_json
from oresorter.ml import dataset as D
from oresorter.ml import models
from oresorter.ml import train as T
from oresorter.ml.train import TrainConfig, TrainError, cross_validate, main, run

REPO = Path(__file__).resolve().parent.parent
BG = (0, 0, 255)
CLASSES = ["A", "B", "C"]
IMAGE_SIZE = 64


def make_rgba(rng: np.random.Generator, cls: str, size: int = 40) -> np.ndarray:
    """蓝底上画一颗该类颜色的矿石，alpha = 非背景像素。"""
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = np.array(BG, np.uint8)
    draw_particle(canvas, rng, size // 2, size // 2, int(size * 0.8), CLASS_PALETTE[cls])
    alpha = (np.any(canvas != np.array(BG, np.uint8), axis=2)).astype(np.uint8) * 255
    assert alpha.sum() > 0
    return np.dstack([canvas, alpha])


def write_png_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def make_model_dir(tmp_path: Path, counts: dict[str, int], *, seed: int = 1) -> Path:
    """构造 {tmp}/models/1/samples/{cls}/cam1_..._{seq}.png，并放一些 .trash 干扰项。"""
    rng = np.random.default_rng(seed)
    mdir = tmp_path / "models" / "1"
    for cls, n in counts.items():
        for i in range(n):
            write_png_rgba(
                mdir / "samples" / cls / f"cam1_20260101_000000_{i:06d}.png", make_rgba(rng, cls)
            )
        # 软删除目录与临时文件不得被计入
        write_png_rgba(
            mdir / "samples" / ".trash" / cls / "cam1_deleted_000000.png", make_rgba(rng, cls)
        )
    (mdir / "samples" / "A" / ".cam1_x.png.tmp").write_bytes(b"junk")
    return mdir


def small_cfg(**over) -> TrainConfig:
    base = {
        "backbone": "resnet18",
        "image_size": IMAGE_SIZE,
        "max_epochs": 2,
        "head_epochs": 1,
        "batch_size": 8,
        "pretrained": False,
        "device": "cpu",
        "min_samples_per_class": 5,
        "num_workers": 0,
        "patience": 5,
        "val_ratio": 0.2,
        "seed": 3,
    }
    base.update(over)
    cfg = TrainConfig(**base)
    cfg.validate()
    return cfg


def plant_stale_derived(train_dir: Path) -> None:
    """放上一轮训练遗留的派生产物。"""
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "model.onnx").write_bytes(b"stale onnx")
    (train_dir / "model.plan").write_bytes(b"stale plan")
    (train_dir / "model.json").write_text("{}", encoding="utf-8")


# --------------------------------------------------------------------------------------
# 完整训练
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """跑一次完整训练，多个测试复用其产物。"""
    tmp = tmp_path_factory.mktemp("train")
    mdir = make_model_dir(tmp, dict.fromkeys(CLASSES, 12))
    calls: list[dict] = []
    result = run(
        mdir,
        None,
        small_cfg(),
        class_names=["A", "B", "C", "D", "E", "F", "G"],
        progress_cb=calls.append,
    )
    return mdir, result, calls


def test_run_result_structure(trained):
    _, result, _ = trained
    assert set(result) >= {
        "metrics_summary",
        "epochs",
        "best_epoch",
        "duration_s",
        "classes",
        "sample_counts",
        "artifacts",
    }
    assert result["classes"] == CLASSES
    assert result["sample_counts"] == dict.fromkeys(CLASSES, 12)
    assert result["epochs"] == 2 and 1 <= result["best_epoch"] <= 2
    assert set(result["metrics_summary"]) == {"val_accuracy", "macro_f1"}
    assert 0.0 <= result["metrics_summary"]["macro_f1"] <= 1.0
    # ONNX 必须真的导出成功（CLAUDE.md：训练→导出→推理一致性是必测项）
    assert result["artifacts"] == {"best": True, "onnx": True}
    assert result["stop_reason"] == "max_epochs"
    assert result["duration_s"] > 0


def test_artifacts_exist_and_well_formed(trained):
    mdir, result, _ = trained
    train_dir = mdir / "train"
    for name in ("best.pt", "model.onnx", "metrics.json", "progress.json", "train.log"):
        assert (train_dir / name).exists(), name
    assert not list(train_dir.glob(".*.tmp")), "不得残留临时文件"
    assert not (train_dir / "error.json").exists()

    metrics = read_json(train_dir / "metrics.json")
    assert metrics["labels"] == CLASSES
    assert len(metrics["matrix"]) == 3 and all(len(r) == 3 for r in metrics["matrix"])
    assert sum(sum(r) for r in metrics["matrix"]) == 3 * 2  # 每类 round(12*0.2)=2 张进验证集
    assert set(metrics["per_class"]) == set(CLASSES)
    assert set(metrics["per_class"]["A"]) == {"precision", "recall", "f1", "support"}
    assert len(metrics["history"]) == 2
    for h in metrics["history"]:
        assert set(h) >= {
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "lr",
            "seconds",
        }
    assert metrics["history"][0]["stage"] == "A" and metrics["history"][1]["stage"] == "B"
    assert metrics["best_epoch"] == result["best_epoch"]
    assert metrics["sample_counts"] == result["sample_counts"]
    assert metrics["config"]["image_size"] == IMAGE_SIZE and metrics["config"]["bg_color"] == [
        0,
        0,
        255,
    ]
    assert metrics["duration_s"] > 0
    assert metrics["macro_f1"] == pytest.approx(result["metrics_summary"]["macro_f1"])
    assert metrics["artifacts"] == {"best": True, "onnx": True}
    assert metrics["stop_reason"] == "max_epochs"

    progress = read_json(train_dir / "progress.json")
    assert progress["status"] == "done" and progress["progress"] == 1.0
    assert progress["artifacts"]["onnx"] is True

    log_text = (train_dir / "train.log").read_text(encoding="utf-8")
    assert "开始训练" in log_text and "训练完成" in log_text and "epoch" in log_text
    assert "ONNX 已导出" in log_text


def test_onnx_matches_torch_on_same_inputs(trained):
    """同一批小图：PyTorch(best.pt) 与 ONNX(model.onnx) 的 logits 一致、top-1 相同。"""
    ort = pytest.importorskip("onnxruntime")
    mdir, _, _ = trained
    model, meta = models.load_checkpoint(mdir / "train" / "best.pt")
    files = D.list_model_samples(mdir, CLASSES)
    items = [(files[c][i], idx) for idx, c in enumerate(CLASSES) for i in range(2)]
    ds = D.SampleDataset(items, IMAGE_SIZE, train=False)
    x = torch.stack([ds[i][0] for i in range(len(ds))])  # (6,3,S,S)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(
        str(mdir / "train" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    (inp,) = sess.get_inputs()
    assert inp.shape[1:] == [3, IMAGE_SIZE, IMAGE_SIZE]
    out = sess.run(None, {inp.name: x.numpy()})[0]
    assert out.shape == (6, len(meta["classes"]))
    assert np.allclose(out, ref, atol=1e-4), np.abs(out - ref).max()
    assert np.array_equal(out.argmax(1), ref.argmax(1))


def test_progress_callback_monotonic(trained):
    _, _, calls = trained
    assert len(calls) >= 3  # prepare + 2 epoch + done
    values = [c["progress"] for c in calls]
    assert values == sorted(values) and values[0] == 0.0 and values[-1] == 1.0
    assert all(c["status"] in ("running", "done") for c in calls)
    assert calls[0]["stage"] == "prepare" and calls[-1]["status"] == "done"
    epoch_calls = [c for c in calls if c["stage"].startswith("epoch ")]
    assert [c["epoch"] for c in epoch_calls] == [1, 2]
    assert epoch_calls[0]["stage"].startswith("epoch 1/2 val_f1=")
    for c in epoch_calls:
        assert 0.0 <= c["val_macro_f1"] <= 1.0 and 0.0 <= c["val_accuracy"] <= 1.0
        assert c["progress"] == pytest.approx(c["epoch"] / 2)


def test_meta_json_updated(trained):
    mdir, result, _ = trained
    meta = read_json(mdir / "meta.json")
    assert meta["classes"] == CLASSES
    assert meta["version"] == 1
    assert meta["algo"] == "cnn-resnet18" and meta["image_size"] == IMAGE_SIZE
    assert meta["sample_counts"] == dict.fromkeys(CLASSES, 12)
    assert meta["metrics_summary"] == result["metrics_summary"]
    assert isinstance(meta["trained_at"], str) and meta["trained_at"][:2] == "20"


def test_best_checkpoint_loads_and_predicts(trained):
    mdir, _, _ = trained
    model, meta = models.load_checkpoint(mdir / "train" / "best.pt")
    assert meta["classes"] == CLASSES and meta["image_size"] == IMAGE_SIZE
    assert meta["backbone"] == "resnet18" and meta["algo"] == "cnn-resnet18"
    assert 1 <= meta["epoch"] <= 2 and "trained_at" in meta
    with torch.no_grad():
        y = model(torch.zeros(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert y.shape == (4, 3)


def test_file_handler_removed_after_run(trained):
    """train.log 的 FileHandler 训练结束后必须摘掉（否则多次训练会把日志写进别的模型目录）。"""
    ml_logger = logging.getLogger("oresorter.ml")
    assert not any(isinstance(h, logging.FileHandler) for h in ml_logger.handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in T.log.handlers)


def test_meta_merge_preserves_existing_fields_and_other_class_counts(tmp_path):
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    (mdir / "meta.json").write_text(
        json.dumps(
            {
                "id": 1,
                "name": "模型1",
                "description": "说明",
                "version": 4,
                # store 初始化的是全部 class_names 的计数；A 的旧值应被本次计数覆盖，C/D 必须保留
                "sample_counts": {"A": 99, "B": 6, "C": 6, "D": 0},
            }
        ),
        encoding="utf-8",
    )
    run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0))
    meta = read_json(mdir / "meta.json")
    assert meta["id"] == 1 and meta["name"] == "模型1" and meta["description"] == "说明"
    assert meta["version"] == 5 and meta["classes"] == ["A", "B"]
    assert meta["sample_counts"] == {"A": 6, "B": 6, "C": 6, "D": 0}
    assert set(meta["metrics_summary"]) == {"val_accuracy", "macro_f1"}


def test_corrupt_meta_json_is_tolerated(tmp_path):
    """meta.json 不是合法 JSON：不能裸抛 JSONDecodeError，训练照常完成并重建 meta。"""
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    (mdir / "meta.json").write_text("{not json", encoding="utf-8")
    result = run(mdir, None, small_cfg(max_epochs=1, head_epochs=0))
    assert result["classes"] == CLASSES
    meta = read_json(mdir / "meta.json")
    assert meta["classes"] == CLASSES and meta["version"] == 1
    assert read_json(mdir / "train" / "progress.json")["status"] == "done"
    assert "meta.json 无法解析" in (mdir / "train" / "train.log").read_text(encoding="utf-8")


def test_train_log_includes_sibling_module_logs(tmp_path):
    """models.build_model 的“预训练权重”日志（直接决定训练质量）必须进 train.log。"""
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 6})
    ml_logger = logging.getLogger("oresorter.ml")
    level_before = ml_logger.level
    run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0, pretrained=True))
    text = (mdir / "train" / "train.log").read_text(encoding="utf-8")
    assert "oresorter.ml.models" in text and "预训练权重" in text
    assert "oresorter.ml.train" in text and "划分完成" in text
    assert ml_logger.level == level_before
    assert not any(isinstance(h, logging.FileHandler) for h in ml_logger.handlers)


# --------------------------------------------------------------------------------------
# 产物一致性：onnx / plan 必须与 best.pt 配套
# --------------------------------------------------------------------------------------


def test_export_failure_leaves_no_stale_onnx(tmp_path, monkeypatch):
    """导出失败：上一轮的 model.onnx / model.plan / model.json 不得残留，返回值与 metrics.json 标明。"""
    import oresorter.ml.export as E

    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    plant_stale_derived(mdir / "train")

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(E, "to_onnx", boom)
    result = run(mdir, ["A", "B", "C"], small_cfg(max_epochs=1, head_epochs=0))
    train_dir = mdir / "train"
    assert (train_dir / "best.pt").exists()
    for name in ("model.onnx", "model.plan", "model.json"):
        assert not (train_dir / name).exists(), f"{name} 不应残留"
    assert result["artifacts"]["best"] is True and result["artifacts"]["onnx"] is False
    assert "disk full" in result["artifacts"]["onnx_error"]
    metrics = read_json(train_dir / "metrics.json")
    assert (
        metrics["artifacts"]["onnx"] is False and "disk full" in metrics["artifacts"]["onnx_error"]
    )
    progress = read_json(train_dir / "progress.json")
    assert progress["status"] == "done" and progress["artifacts"]["onnx"] is False
    assert "ONNX 导出失败" in (train_dir / "train.log").read_text(encoding="utf-8")


def test_retrain_replaces_stale_plan_and_onnx(tmp_path):
    """重训成功：旧 plan/引擎元信息被清掉（它们基于旧 onnx），onnx 是新导出的。"""
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    plant_stale_derived(mdir / "train")
    result = run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0))
    train_dir = mdir / "train"
    assert result["artifacts"]["onnx"] is True
    assert not (train_dir / "model.plan").exists() and not (train_dir / "model.json").exists()
    assert (train_dir / "model.onnx").read_bytes() != b"stale onnx"
    assert "已清理上一轮的派生产物" in (train_dir / "train.log").read_text(encoding="utf-8")


def test_cancel_before_first_epoch_keeps_old_artifacts(tmp_path):
    """还没写新 best.pt 就取消：旧 best.pt 与旧 onnx 都保留（仍是配套的一组）。"""
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0))
    old_best = (mdir / "train" / "best.pt").read_bytes()
    old_onnx = (mdir / "train" / "model.onnx").read_bytes()
    ev = threading.Event()
    ev.set()
    with pytest.raises(TrainError) as ei:
        run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0), stop_event=ev)
    assert ei.value.code == "CANCELLED"
    assert (mdir / "train" / "best.pt").read_bytes() == old_best
    assert (mdir / "train" / "model.onnx").read_bytes() == old_onnx


# --------------------------------------------------------------------------------------
# 停止条件：早停 / 时间上限
# --------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`:UserWarning")
def test_early_stop_and_best_tie_break_by_val_loss(tmp_path, monkeypatch):
    """用假的训练/验证函数给出确定的指标序列：F1 恒为 1，val_loss 1.0 → 0.5 → 0.5 → 0.5。

    期望：epoch 2 因 loss 更低成为 best（F1 相等取更低 val_loss）；之后连续 patience=2 轮不升 → 早停于 epoch 4。
    """
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    losses = iter([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def fake_train(model, loader, *a, **k):
        return 0.0

    def fake_eval(model, loader, *a, **k):
        y = list(loader.dataset.labels)
        return next(losses), y, list(y)

    monkeypatch.setattr(T, "_train_one_epoch", fake_train)
    monkeypatch.setattr(T, "_evaluate", fake_eval)
    calls: list[dict] = []
    result = run(
        mdir, None, small_cfg(max_epochs=10, head_epochs=0, patience=2), progress_cb=calls.append
    )
    assert result["stop_reason"] == "early stop"
    assert result["epochs"] == 4 and result["best_epoch"] == 2
    assert result["metrics_summary"]["macro_f1"] == 1.0
    metrics = read_json(mdir / "train" / "metrics.json")
    assert [h["val_loss"] for h in metrics["history"]] == [1.0, 0.5, 0.5, 0.5]
    assert metrics["best_epoch"] == 2 and metrics["stop_reason"] == "early stop"
    _, meta = models.load_checkpoint(mdir / "train" / "best.pt")
    assert meta["epoch"] == 2
    values = [c["progress"] for c in calls]
    assert values == sorted(values) and values[-1] == 1.0
    assert [c["is_best"] for c in calls if c["stage"].startswith("epoch ")] == [
        True,
        True,
        False,
        False,
    ]


def test_time_limit_stops_after_first_epoch(tmp_path):
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    calls: list[dict] = []
    result = run(
        mdir,
        None,
        small_cfg(max_epochs=3, head_epochs=0, max_train_minutes=1e-6),
        progress_cb=calls.append,
    )
    assert result["epochs"] == 1 and result["best_epoch"] == 1
    assert result["stop_reason"] == "time limit"
    stages = [c["stage"] for c in calls]
    assert "time limit" in stages
    epoch_idx = next(i for i, s in enumerate(stages) if s.startswith("epoch 1/3 "))
    assert stages.index("time limit") > epoch_idx, "time limit 应在第 1 个 epoch 之后上报"
    assert calls[-1]["status"] == "done"
    metrics = read_json(mdir / "train" / "metrics.json")
    assert metrics["stop_reason"] == "time limit" and len(metrics["history"]) == 1
    assert "达到训练时长上限" in (mdir / "train" / "train.log").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# 错误路径
# --------------------------------------------------------------------------------------


def test_insufficient_samples(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 4, "C": 6})
    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg())
    assert ei.value.code == "VALIDATION"
    assert ei.value.detail["insufficient"] == {"B": 4}
    assert not (mdir / "train" / "best.pt").exists()
    progress = read_json(mdir / "train" / "progress.json")
    assert progress["status"] == "failed" and progress["error"]["code"] == "VALIDATION"


def test_validation_failure_overwrites_stale_done_progress(tmp_path):
    """上一轮 status=done 的 progress.json 不能在本轮校验失败后原样留着。"""
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 6, "C": 6})
    run(mdir, ["A", "B"], small_cfg(max_epochs=1, head_epochs=0))
    assert read_json(mdir / "train" / "progress.json")["status"] == "done"
    with pytest.raises(TrainError):
        run(mdir, ["A", "B", "C"], small_cfg(min_samples_per_class=7))
    assert read_json(mdir / "train" / "progress.json")["status"] == "failed"
    assert (mdir / "train" / "best.pt").exists(), "上一轮的模型不受影响"


def test_explicit_class_without_samples_is_insufficient(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 6})
    with pytest.raises(TrainError) as ei:
        run(mdir, ["A", "B", "C"], small_cfg())
    assert ei.value.detail["insufficient"] == {"C": 0}


def test_illegal_class_names_are_validation_errors(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 6})
    with pytest.raises(TrainError) as ei:
        run(mdir, ["A", "../B"], small_cfg())
    assert ei.value.code == "VALIDATION"
    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg(), class_names=["A", ".trash"])
    assert ei.value.code == "VALIDATION"
    with pytest.raises(TrainError) as ei:
        run(mdir, ["A", "A"], small_cfg())
    assert ei.value.code == "VALIDATION" and "重复" in ei.value.message


def test_too_few_classes(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 6})
    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg())
    assert ei.value.code == "VALIDATION" and "2" in ei.value.message


def test_unexpected_exception_becomes_internal_and_marks_failed(tmp_path, monkeypatch):
    """_fit 之外的步骤抛非 TrainError 异常：也要包成 TrainError(INTERNAL) 并把 progress 置 failed。"""
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))

    def boom(*_a, **_k):
        raise RuntimeError("split exploded")

    monkeypatch.setattr(T, "stratified_split", boom)
    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg())
    assert ei.value.code == "INTERNAL" and "split exploded" in ei.value.message
    assert ei.value.detail == {"type": "RuntimeError"}
    progress = read_json(mdir / "train" / "progress.json")
    assert progress["status"] == "failed" and progress["stage"] == "error"
    assert progress["error"]["code"] == "INTERNAL"


def test_cancel_before_start(tmp_path):
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    ev = threading.Event()
    ev.set()
    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg(), stop_event=ev)
    assert ei.value.code == "CANCELLED"
    assert not (mdir / "train" / "best.pt").exists()
    progress = read_json(mdir / "train" / "progress.json")
    assert progress["status"] == "cancelled"


def test_cancel_mid_training_keeps_best(tmp_path):
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    ev = threading.Event()

    def cb(p: dict) -> None:
        if p.get("epoch") == 1 and p["stage"].startswith("epoch"):
            ev.set()  # 第 1 轮结束后取消

    with pytest.raises(TrainError) as ei:
        run(mdir, None, small_cfg(max_epochs=5, head_epochs=1), progress_cb=cb, stop_event=ev)
    assert ei.value.code == "CANCELLED"
    assert (mdir / "train" / "best.pt").exists(), "已写的 best.pt 应保留"
    assert not (mdir / "train" / "metrics.json").exists()
    assert not (mdir / "train" / "model.onnx").exists()


def test_config_validation_and_from_ml_config():
    from oresorter.config import MlConfig

    cfg = TrainConfig.from_ml_config(MlConfig(), max_epochs=2, device="cpu", bg_color=(1, 2, 3))
    assert cfg.backbone == "resnet18" and cfg.image_size == 160 and cfg.max_epochs == 2
    assert cfg.bg_color == (1, 2, 3) and cfg.pretrained is True
    with pytest.raises(TrainError) as ei:
        TrainConfig.from_ml_config(MlConfig(), max_epochs=0)
    assert ei.value.code == "VALIDATION" and "max_epochs" in ei.value.message
    with pytest.raises(TrainError):
        TrainConfig.from_ml_config(MlConfig(), nonsense=1)
    with pytest.raises(TrainError):
        TrainConfig(backbone="vgg").validate()
    with pytest.raises(TrainError):
        TrainConfig(val_ratio=1.0).validate()
    with pytest.raises(TrainError):
        TrainConfig(bg_color=(0, 0, 999)).validate()
    # 新增校验：seed / bg_threshold / 类型错误都要是 TrainError 而不是 ValueError
    with pytest.raises(TrainError) as ei:
        TrainConfig(seed=-1).validate()
    assert "seed" in ei.value.message
    with pytest.raises(TrainError) as ei:
        TrainConfig(bg_threshold=-5).validate()
    assert "bg_threshold" in ei.value.message
    with pytest.raises(TrainError) as ei:
        TrainConfig(image_size="abc").validate()  # type: ignore[arg-type]
    assert ei.value.code == "VALIDATION" and "类型" in ei.value.message
    # image_size：进程内放宽到 8~1024，生产入口（strict）只允许 §9 的 128/160/224
    TrainConfig(image_size=64).validate()
    with pytest.raises(TrainError) as ei:
        TrainConfig(image_size=64).validate(strict_image_size=True)
    assert "image_size" in ei.value.message
    with pytest.raises(TrainError):
        TrainConfig.from_ml_config(MlConfig(), strict_image_size=True, image_size=64)
    assert (
        TrainConfig.from_ml_config(MlConfig(), strict_image_size=True, image_size=128).image_size
        == 128
    )
    assert TrainError("X", "m", {"a": 1}).to_dict() == {
        "code": "X",
        "message": "m",
        "detail": {"a": 1},
    }


def test_missing_model_dir(tmp_path):
    with pytest.raises(TrainError) as ei:
        run(tmp_path / "nope", None, small_cfg())
    assert ei.value.code == "NOT_FOUND"
    assert not (tmp_path / "nope").exists(), "不能凭空创建模型目录"


def test_make_loaders_sets_timeout_and_worker_init_for_multiprocess(tmp_path):
    """num_workers > 0 时 DataLoader 必须有超时并关闭 worker 内 OpenCV 线程池（不真正起 worker）。"""
    mdir = make_model_dir(tmp_path, {"A": 2, "B": 2})
    files = D.list_model_samples(mdir, ["A", "B"])
    items = [(p, 0) for p in files["A"]] + [(p, 1) for p in files["B"]]
    _, tl, vl = T._make_loaders(items, items, small_cfg(num_workers=2), torch.device("cpu"))
    assert tl.timeout == T.LOADER_TIMEOUT_S and vl.timeout == T.LOADER_TIMEOUT_S
    assert tl.worker_init_fn is D.worker_init_fn and vl.worker_init_fn is D.worker_init_fn
    _, tl0, _ = T._make_loaders(items, items, small_cfg(num_workers=0), torch.device("cpu"))
    assert tl0.num_workers == 0
    D.worker_init_fn(0)  # 可直接调用，不抛


# --------------------------------------------------------------------------------------
# dataset 工具
# --------------------------------------------------------------------------------------


def test_list_model_samples_excludes_trash_and_tmp(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 3, "B": 2})
    out = D.list_model_samples(mdir, ["A", "B", "C"])
    assert [len(out[c]) for c in ("A", "B", "C")] == [3, 2, 0]
    assert all(
        ".trash" not in p.parts and not p.name.startswith(".") for ps in out.values() for p in ps
    )
    assert out["A"] == sorted(out["A"])
    with pytest.raises(ValueError):
        D.list_model_samples(mdir, [".trash"])
    with pytest.raises(ValueError):
        D.list_model_samples(mdir, ["../A"])


def test_list_folder_samples_and_load_rgba_without_alpha(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "folder"
    for cls, ext in (("a", ".bmp"), ("b", ".jpg"), ("c", ".png")):
        for i in range(2):
            rgba = make_rgba(rng, cls.upper())
            p = root / cls / f"{i}{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(p), cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
    (root / ".hidden").mkdir()
    (root / "a" / "notes.txt").write_text("x")
    classes, files = D.list_folder_samples(root)
    assert classes == ["a", "b", "c"]
    assert all(len(files[c]) == 2 for c in classes)
    classes2, files2 = D.list_folder_samples(root, ["c", "a"])
    assert classes2 == ["c", "a"] and set(files2) == {"c", "a"}
    with pytest.raises(ValueError):
        D.list_folder_samples(root, ["zzz"])
    with pytest.raises(ValueError):
        D.list_folder_samples(tmp_path / "missing")

    # 无 alpha → 自动抠 mask：中心是前景、角落是蓝底背景
    rgba = D.load_rgba(files["a"][0], bg_color=BG, bg_threshold=40)
    assert rgba.shape == (40, 40, 4) and rgba.dtype == np.uint8
    assert rgba[20, 20, 3] == 255 and rgba[0, 0, 3] == 0
    assert rgba[0, 0, :3].tolist() == [0, 0, 255], "BGR→RGB 转换"
    with pytest.raises(ValueError):
        D.load_rgba(root / "a" / "notes.txt")


def test_list_folder_samples_rejects_class_names_that_escape_root(tmp_path):
    root = tmp_path / "folder"
    (root / "a").mkdir(parents=True)
    (tmp_path / "outside_cls").mkdir()
    with pytest.raises(ValueError):
        D.list_folder_samples(root, ["../outside_cls"])
    with pytest.raises(ValueError):
        D.list_folder_samples(root, [".hidden"])
    with pytest.raises(ValueError):
        D.list_folder_samples(root, ["a", "a"])
    with pytest.raises(ValueError):
        D.list_folder_samples(root, [""])


def test_load_rgba_preserves_alpha_channel(tmp_path):
    rng = np.random.default_rng(5)
    src = make_rgba(rng, "A")
    p = tmp_path / "x.png"
    write_png_rgba(p, src)
    back = D.load_rgba(p)
    assert np.array_equal(back, src)


def test_augment_shape_dtype_alpha_and_color_only_on_foreground():
    rng = np.random.default_rng(11)
    src = make_rgba(rng, "C", size=48)
    for i in range(20):
        out = D.augment(src, np.random.default_rng(i))
        assert out.shape == src.shape and out.dtype == np.uint8
        assert set(np.unique(out[:, :, 3]).tolist()) <= {0, 255}
        assert out[:, :, 3].sum() > 0
    # 非方形：旋转 90° 时 h/w 互换，通道与 dtype 不变
    rect = np.ascontiguousarray(src[:, :24])
    out = D.augment(rect, np.random.default_rng(2))
    assert out.ndim == 3 and out.shape[2] == 4 and out.dtype == np.uint8
    assert out.shape[:2] in {(48, 24), (24, 48)}
    # 同一 rng 种子 → 可复现
    a = D.augment(src, np.random.default_rng(99))
    b = D.augment(src, np.random.default_rng(99))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        D.augment(src[:, :, :3], rng)


def test_augment_never_mutates_or_aliases_input():
    """任何随机路径下输入数组逐字节不变，输出不与输入共享内存。"""
    src = np.zeros((40, 40, 4), np.uint8)
    src[10:30, 10:30, :3] = (200, 40, 40)
    src[10:30, 10:30, 3] = 255
    src[0, 0, 3] = 100  # 非二值 alpha：若被原地二值化即可察觉
    snapshot = src.copy()
    for i in range(600):
        out = D.augment(src, np.random.default_rng(i))
        assert not np.shares_memory(out, src)
        assert np.array_equal(src, snapshot), f"seed {i} 改写了输入"
        assert out.flags["C_CONTIGUOUS"]


def test_augment_upscale_keeps_particle_inside_canvas():
    """放大不能把颗粒裁出画布（推理小图总带完整 margin，训练分布须一致）；缩小仍然有效。"""
    for shape, center, radius in (((40, 40), (20, 20), 18), ((40, 60), (45, 12), 10)):
        src = np.zeros((*shape, 4), np.uint8)
        cv2.circle(src, center, radius, (200, 40, 40, 255), -1)
        base_area = int((src[:, :, 3] > 0).sum())
        areas = []
        for i in range(300):
            alpha = D.augment(src, np.random.default_rng(i))[:, :, 3] > 0
            assert not (
                alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any()
            )
            areas.append(alpha.sum() / base_area)
        assert min(areas) < 0.75, "缩小分支应仍然生效"
    # 没有前景时不限制放大，也不报错
    empty = np.zeros((32, 32, 4), np.uint8)
    out = D.augment(empty, np.random.default_rng(1))
    assert out.shape == empty.shape and out[:, :, 3].sum() == 0
    assert D._max_upscale(empty) == float("inf")


def test_augment_color_is_multiply_add_only_and_spares_background():
    """前景 RGB 只做统一的 c·x + b（不改色相），背景像素（alpha=0）RGB 不动。"""
    src = np.zeros((40, 40, 4), np.uint8)
    src[:, :, :3] = (200, 40, 40)  # 全图同色，便于分辨"颜色变换"与"几何插值"
    cv2.circle(src, (20, 20), 12, (200, 40, 40, 255), -1)
    for i in range(30):
        out = D.augment(src, np.random.default_rng(i))
        fg = out[:, :, 3] == 255
        bg = ~fg
        assert fg.any() and bg.any()
        # 背景：仍是原色或缩放补零
        bg_vals = {tuple(px) for px in np.unique(out[bg][:, :3], axis=0).tolist()}
        assert bg_vals <= {(200, 40, 40), (0, 0, 0)}, bg_vals
        # 前景：全部像素同一个值，且满足 out = c·src + b（G、B 通道源值相同 → 结果相同）
        fg_vals = np.unique(out[fg][:, :3], axis=0)
        assert len(fg_vals) == 1, fg_vals
        r, g, b = (int(v) for v in fg_vals[0])
        assert abs(g - b) <= 1
        c = (r - g) / 160.0  # (200c+b) - (40c+b) = 160c
        assert 0.88 <= c <= 1.12, c
        offset = g - 40 * c
        assert -27 <= offset <= 27, offset


def test_sample_dataset_getitem_and_epoch_reproducibility(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 2, "B": 2})
    files = D.list_model_samples(mdir, ["A", "B"])
    items = [(p, 0) for p in files["A"]] + [(p, 1) for p in files["B"]]
    ds = D.SampleDataset(items, 32, train=True, seed=1)
    x, y = ds[0]
    assert x.dtype == torch.float32 and x.shape == (3, 32, 32) and y == 0
    x_again, _ = ds[0]
    assert torch.equal(x, x_again), "同 epoch 同 index 增广可复现"
    ds.set_epoch(1)
    x_e1, _ = ds[0]
    assert not torch.equal(x, x_e1), "换 epoch 增广应不同"
    val = D.SampleDataset(items, 32, train=False)
    xv, yv = val[3]
    assert yv == 1 and torch.equal(xv, val[3][0])
    with pytest.raises(ValueError):
        D.SampleDataset(items, 4, train=False)


def test_stratified_split_every_class_in_val():
    items = [(Path(f"{c}_{i}.png"), c) for c in range(3) for i in range(10)] + [(Path("d.png"), 3)]
    tr, va = D.stratified_split(items, 0.2, seed=0)
    assert len(tr) + len(va) == len(items)
    assert set(tr).isdisjoint(va)
    val_labels = {y for _, y in va}
    assert val_labels == {0, 1, 2}, "每个 ≥2 样本的类都要有验证样本"
    assert [y for _, y in va].count(0) == 2
    assert (Path("d.png"), 3) in tr, "只有 1 个样本的类进训练集"
    # 极小类：2 个样本 → 1 训 1 验；val_ratio 越界拒绝
    tr2, va2 = D.stratified_split([(Path("x"), 0), (Path("y"), 0)], 0.9, seed=1)
    assert len(tr2) == 1 and len(va2) == 1
    with pytest.raises(ValueError):
        D.stratified_split(items, 1.0)
    # 确定性
    assert D.stratified_split(items, 0.2, seed=7) == D.stratified_split(items, 0.2, seed=7)


def test_kfold_split_covers_all_without_overlap():
    items = [(Path(f"{c}_{i}.png"), c) for c, n in ((0, 7), (1, 5), (2, 3)) for i in range(n)]
    folds = D.kfold_split(items, 5, seed=0)
    assert len(folds) == 5
    seen: list = []
    for tr, va in folds:
        assert len(tr) + len(va) == len(items)
        assert set(tr).isdisjoint(va)
        assert len(va) in (3,)  # 15 / 5
        seen.extend(va)
    assert sorted(seen) == sorted(items), "每个样本恰好出现在一个验证折"
    # 分层：7 个 0 类分到 5 折，每折 1~2 个
    for _, va in folds:
        assert 1 <= [y for _, y in va].count(0) <= 2
    loo = D.kfold_split(items, len(items), seed=0)
    assert len(loo) == 15 and all(len(va) == 1 for _, va in loo)
    with pytest.raises(ValueError):
        D.kfold_split(items, 1)
    with pytest.raises(ValueError):
        D.kfold_split(items, 16)


def test_weighted_sampler_balances_classes():
    labels = [0] * 90 + [1] * 10
    sampler = D.make_weighted_sampler(labels, seed=0)
    assert len(sampler) == 100
    drawn = [labels[i] for i in list(sampler)]
    assert 30 <= drawn.count(1) <= 70, f"少数类应被均衡抽到，实际 {drawn.count(1)}"
    with pytest.raises(ValueError):
        D.make_weighted_sampler([])


# --------------------------------------------------------------------------------------
# 交叉验证
# --------------------------------------------------------------------------------------


def test_cross_validate_kfold_aggregates_all_folds(tmp_path):
    mdir = make_model_dir(tmp_path, {"A": 4, "B": 4})
    files = D.list_model_samples(mdir, ["A", "B"])
    items = [(p, 0) for p in files["A"]] + [(p, 1) for p in files["B"]]
    calls: list[dict] = []
    rep = cross_validate(
        items,
        ["A", "B"],
        small_cfg(max_epochs=1, head_epochs=0, batch_size=4),
        k=2,
        progress_cb=calls.append,
        loo_threshold=0,
    )
    assert rep["k"] == 2 and rep["mode"] == "kfold"
    assert rep["labels"] == ["A", "B"]
    assert sum(sum(r) for r in rep["matrix"]) == 8, "全部样本都在某一折被预测"
    assert len(rep["folds"]) == 2 and all(
        f["n_val"] == 4 and f["epochs"] == 1 for f in rep["folds"]
    )
    assert rep["sample_counts"] == {"A": 4, "B": 4}
    assert [c["progress"] for c in calls] == [0.5, 1.0]
    assert "accuracy" in rep and "per_class" in rep and rep["duration_s"] > 0


def test_cross_validate_loo_mode_selected_for_small_sets(monkeypatch):
    """样本 < 阈值 → k = 样本数（留一法）。用 monkeypatch 把 _fit 换成假的以免跑很多折。"""
    items = [(Path(f"{i}.png"), i % 2) for i in range(6)]

    def fake_fit(tr, va, classes, cfg, **kw):
        res = T._FitResult(
            history=[{"val_loss": 0.0}],
            best_epoch=1,
            best_report=None,
            best_val_loss=0.0,
            last_report={"accuracy": 1.0},
            last_y_true=[y for _, y in va],
            last_y_pred=[y for _, y in va],
            stop_reason="max_epochs",
        )
        return res, None

    monkeypatch.setattr(T, "_fit", fake_fit)
    rep = cross_validate(items, ["A", "B"], small_cfg(), k=5)
    assert rep["mode"] == "loo" and rep["k"] == 6 and len(rep["folds"]) == 6
    assert rep["accuracy"] == 1.0
    with pytest.raises(TrainError):
        cross_validate(items, ["A"], small_cfg())


# --------------------------------------------------------------------------------------
# 子进程入口
# --------------------------------------------------------------------------------------


def _subprocess_base(mdir: Path, *, epochs: int, extra: list[str] | None = None) -> list[str]:
    return [
        sys.executable,
        "-m",
        "oresorter.ml.train",
        "--model-dir",
        str(mdir),
        "--epochs",
        str(epochs),
        "--head-epochs",
        "0",
        "--image-size",
        str(IMAGE_SIZE),
        "--allow-any-image-size",
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--min-samples",
        "5",
        "--no-pretrained",
        "--config",
        str(REPO / "config" / "dev.yaml"),
        *(extra or []),
    ]


@pytest.mark.slow
def test_main_subprocess_success_and_failure(tmp_path):
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    base = _subprocess_base(mdir, epochs=1)
    proc = subprocess.run(
        [*base, "--classes", "A,B"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["classes"] == ["A", "B"] and out["epochs"] == 1
    assert out["artifacts"] == {"best": True, "onnx": True}
    assert (mdir / "train" / "best.pt").exists() and not (mdir / "train" / "error.json").exists()
    assert (mdir / "train" / "model.onnx").exists()
    assert read_json(mdir / "train" / "progress.json")["status"] == "done"

    # 失败路径：要求的类别没有样本 → 退出码 1 + error.json + progress 置 failed
    proc2 = subprocess.run(
        [*base, "--classes", "A,Z"],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert proc2.returncode == 1
    err = read_json(mdir / "train" / "error.json")
    assert err["code"] == "VALIDATION" and err["detail"]["insufficient"] == {"Z": 0}
    assert read_json(mdir / "train" / "progress.json")["status"] == "failed"
    assert (mdir / "train" / "best.pt").exists(), "上一轮产物保留"


@pytest.mark.slow
def test_main_subprocess_sigterm_cancels_and_keeps_best(tmp_path):
    """jobs 层 terminate() 的路径：SIGTERM → 退出码 1、error.json CANCELLED、progress cancelled、best.pt 保留。"""
    mdir = make_model_dir(tmp_path, dict.fromkeys(CLASSES, 6))
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    cmd = _subprocess_base(mdir, epochs=200, extra=["--patience", "100", "--max-minutes", "10"])
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    progress_path = mdir / "train" / "progress.json"
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"子进程提前结束: rc={proc.returncode}")
            try:
                doc = read_json(progress_path, default={})
            except ValueError:
                doc = {}
            if doc.get("epoch", 0) >= 1 and (mdir / "train" / "best.pt").exists():
                break
            time.sleep(0.05)
        else:
            pytest.fail("等待第 1 个 epoch 超时")
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)
    assert proc.returncode == 1, stderr[-2000:]
    err = read_json(mdir / "train" / "error.json")
    assert err["code"] == "CANCELLED"
    progress = read_json(progress_path)
    assert progress["status"] == "cancelled" and progress["error"]["code"] == "CANCELLED"
    assert (mdir / "train" / "best.pt").exists()
    assert not (mdir / "train" / "metrics.json").exists()
    assert "收到信号" in (mdir / "train" / "train.log").read_text(encoding="utf-8")


def test_main_in_process_bad_args(tmp_path):
    with pytest.raises(SystemExit):
        main(["--epochs", "1"])  # 缺 --model-dir
    # 模型目录不存在：退出码 1，但不得凭空创建目录 / error.json
    rc = main(["--model-dir", str(tmp_path / "missing"), "--device", "cpu", "--epochs", "1"])
    assert rc == 1
    assert not (tmp_path / "missing").exists()
    # 生产入口严格限制 image_size（docs/01 §9），越界在训练前就拒绝并写 error.json
    mdir = make_model_dir(tmp_path, {"A": 6, "B": 6})
    rc = main(["--model-dir", str(mdir), "--device", "cpu", "--epochs", "1", "--image-size", "64"])
    assert rc == 1
    err = read_json(mdir / "train" / "error.json")
    assert err["code"] == "VALIDATION" and "image_size" in err["message"]
    assert not (mdir / "train" / "best.pt").exists()
