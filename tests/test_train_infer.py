"""训练 → 导出 → 推理一致性（CLAUDE.md 必测项；docs/01 §6.3、docs/02 §5 一致性测试）。

端到端使用真实产物：合成样本集 → ``train.run`` → ``best.pt`` / ``model.onnx`` / ``metrics.json`` →
``TorchClassifier`` 与 ``OnnxClassifier`` 对同一批真实样本的输出 top-1 一致、概率误差 < 1e-2。
TensorRT 部分仅能在 Jetson 上验证（``pytest -m jetson``），不在本文件。

整个文件 CPU 上 < 90 s（128 px、30 张样本、2 个 epoch 约 3 s）。标记 slow 但不跳过。
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from oresorter.fsutil import read_json
from oresorter.ml import dataset, infer, models, train
from oresorter.ml.train import TrainConfig
from oresorter.vision import synthetic
from oresorter.vision.preprocess import to_tensor_batch

pytestmark = pytest.mark.slow

CLASSES = ["A", "B", "C"]
PER_CLASS = 10
IMAGE_SIZE = 128
N_COMPARE = 20
MAX_ABS_DIFF = 1e-2  # docs/01 §6.3：概率误差 < 1e-2


def train_cfg(**over) -> TrainConfig:
    """测试用训练配方：128 px、2 epoch（1 个只训头）、CPU、不加载预训练权重。"""
    base = {
        "image_size": IMAGE_SIZE,
        "max_epochs": 2,
        "head_epochs": 1,
        "batch_size": 8,
        "pretrained": False,
        "device": "cpu",
        "min_samples_per_class": 5,
        "num_workers": 0,
        "seed": 0,
    }
    base.update(over)
    cfg = TrainConfig(**base)
    cfg.validate()
    return cfg


def train_synthetic_model(
    model_dir: Path,
    classes: list[str],
    per_class: int,
    *,
    seed: int = 0,
    **cfg_over,
) -> dict:
    """``synthetic.make_sample_set`` 生成 ``model_dir/samples/{cls}/`` 后跑一次 ``train.run``，返回训练结果。"""
    synthetic.make_sample_set(model_dir, classes, per_class, rng=np.random.default_rng(seed))
    return train.run(model_dir, None, train_cfg(**cfg_over), class_names=classes)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _real_batch(model_dir: Path, n: int) -> np.ndarray:
    """从模型目录读 ``n`` 张真实样本（各类轮流取），经训练/推理共用的预处理得到 ``(n,3,S,S)``。"""
    files = dataset.list_model_samples(model_dir, CLASSES)
    paths: list[Path] = []
    i = 0
    while len(paths) < n:
        cls = CLASSES[i % len(CLASSES)]
        k = i // len(CLASSES)
        if k < len(files[cls]):
            paths.append(files[cls][k])
        i += 1
        assert i < 10_000
    crops = [dataset.load_rgba(p) for p in paths]
    assert all(c.shape[2] == 4 for c in crops), "合成样本应为 RGBA"
    x = to_tensor_batch(crops, IMAGE_SIZE)
    assert x.shape == (n, 3, IMAGE_SIZE, IMAGE_SIZE) and x.dtype == np.float32
    return x


def _assert_consistent(
    torch_clf: infer.Classifier, onnx_clf: infer.Classifier, x: np.ndarray
) -> dict:
    pt, ct = torch_clf.predict(x)
    po, co = onnx_clf.predict(x)
    assert ct == co == CLASSES, "两个后端的类别顺序必须相同"
    assert pt.shape == po.shape == (x.shape[0], len(CLASSES))
    assert np.allclose(pt.sum(axis=1), 1.0, atol=1e-4) and np.allclose(
        po.sum(axis=1), 1.0, atol=1e-4
    )
    cmp = infer.compare_outputs(pt, po)
    assert cmp["n"] == x.shape[0] and cmp["n_disagree"] == 0
    assert cmp["top1_agree_rate"] == 1.0, cmp
    assert cmp["max_abs_diff"] < MAX_ABS_DIFF, cmp
    return cmp


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, float]:
    """合成 3 类 × 10 张 → 训练一次；模块内共享、只读（重训测试用副本）。"""
    root = tmp_path_factory.mktemp("e2e")
    model_dir = root / "models" / "1"
    t0 = time.perf_counter()
    result = train_synthetic_model(model_dir, CLASSES, PER_CLASS, seed=0)
    return model_dir, result, time.perf_counter() - t0


@pytest.fixture(scope="module")
def batch20(trained: tuple[Path, dict, float]) -> np.ndarray:
    return _real_batch(trained[0], N_COMPARE)


# --------------------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------------------


def test_sample_set_layout(trained: tuple[Path, dict, float]):
    """make_sample_set 写到 model_dir/samples/{cls}/，训练按 class_names 顺序读到全部样本。"""
    model_dir, result, _ = trained
    for cls in CLASSES:
        pngs = sorted((model_dir / "samples" / cls).glob("*.png"))
        assert len(pngs) == PER_CLASS
        assert all(p.with_suffix(".json").exists() for p in pngs)
    assert result["classes"] == CLASSES
    assert result["sample_counts"] == dict.fromkeys(CLASSES, PER_CLASS)
    assert result["epochs"] == 2 and result["stop_reason"] == "max_epochs"


def test_artifacts_exist(trained: tuple[Path, dict, float]):
    model_dir, result, seconds = trained
    train_dir = model_dir / "train"
    for name in ("best.pt", "model.onnx", "metrics.json"):
        assert (train_dir / name).is_file(), name
    assert result["artifacts"] == {"best": True, "onnx": True}
    assert not list(train_dir.glob(".*.tmp")), "不得残留临时文件"
    metrics = read_json(train_dir / "metrics.json")
    assert metrics["labels"] == CLASSES and metrics["classes"] == CLASSES
    assert metrics["config"]["image_size"] == IMAGE_SIZE
    assert metrics["artifacts"]["onnx"] is True
    meta = read_json(model_dir / "meta.json")
    assert meta["classes"] == CLASSES and meta["image_size"] == IMAGE_SIZE and meta["version"] == 1
    # 训练本身要快：1000 张 ≤ 5 min 的配方，30 张 128 px 2 epoch 在 CPU 上应远小于 60 s
    assert seconds < 60, f"训练耗时 {seconds:.1f} s"


# --------------------------------------------------------------------------------------
# 一致性
# --------------------------------------------------------------------------------------


def test_torch_vs_onnx_on_real_samples(trained: tuple[Path, dict, float], batch20: np.ndarray):
    """同一批真实样本：PyTorch(best.pt) 与 ONNX Runtime(model.onnx) top-1 一致率 100%，概率误差 < 1e-2。"""
    model_dir, _, _ = trained
    with (
        infer.load_classifier(model_dir, prefer=("torch",), device="cpu") as t,
        infer.load_classifier(model_dir, prefer=("onnx",)) as o,
    ):
        assert isinstance(t, infer.TorchClassifier) and t.backend == "torch"
        assert isinstance(o, infer.OnnxClassifier) and o.backend == "onnx"
        assert t.image_size == o.image_size == IMAGE_SIZE
        cmp = _assert_consistent(t, o, batch20)
        # 真实训练产物的 fp32 差异应远小于阈值
        assert cmp["max_abs_diff"] < 1e-4, cmp
        # predict_top 与 predict 的 argmax 一致
        idx_t, conf_t = t.predict_top(batch20)
        idx_o, conf_o = o.predict_top(batch20)
        assert np.array_equal(idx_t, idx_o)
        assert np.abs(conf_t - conf_o).max() < MAX_ABS_DIFF
    assert not t.loaded and not o.loaded


def test_classes_match_metrics_labels(trained: tuple[Path, dict, float]):
    """两个后端的 classes 顺序相同，且与 metrics.json 的 labels、best.pt meta、meta.json 一致。"""
    model_dir, _, _ = trained
    labels = read_json(model_dir / "train" / "metrics.json")["labels"]
    _, ck_meta = models.load_checkpoint(model_dir / "train" / "best.pt")
    with (
        infer.load_classifier(model_dir, prefer=("torch",), device="cpu") as t,
        infer.load_classifier(model_dir, prefer=("onnx",)) as o,
    ):
        assert t.classes == o.classes == labels == CLASSES
        assert ck_meta["classes"] == CLASSES
        assert infer.read_model_meta(model_dir)["classes"] == CLASSES
        assert t.num_classes == o.num_classes == len(CLASSES)


def test_torch_auto_device_matches_onnx(trained: tuple[Path, dict, float], batch20: np.ndarray):
    """运行时默认 device=auto（Mac 上可能是 MPS）：与 ONNX 仍满足一致性指标。"""
    model_dir, _, _ = trained
    with (
        infer.load_classifier(model_dir, prefer=("torch",), device="auto") as t,
        infer.load_classifier(model_dir, prefer=("onnx",)) as o,
    ):
        _assert_consistent(t, o, batch20)


def test_default_prefer_falls_back_to_onnx_on_dev_machine(trained: tuple[Path, dict, float]):
    """默认 prefer=(trt, onnx, torch)：开发机没有 TensorRT，应降级到 onnx（而不是 torch）。"""
    if importlib.util.find_spec("tensorrt") is not None:
        pytest.skip("本机有 TensorRT，默认后端会是 trt（Jetson 用 pytest -m jetson 验证）")
    model_dir, _, _ = trained
    assert infer.DEFAULT_PREFER == ("trt", "onnx", "torch")
    with infer.load_classifier(model_dir) as clf:
        assert clf.backend == "onnx"
        assert clf.classes == CLASSES and clf.image_size == IMAGE_SIZE
        probs, classes = clf.predict(np.zeros((2, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32))
        assert probs.shape == (2, 3) and classes == CLASSES


def test_batched_predict_matches_single(trained: tuple[Path, dict, float], batch20: np.ndarray):
    """max_batch 分批（运行时 infer_batch_max）与一次性推理结果一致；两个后端都如此。"""
    model_dir, _, _ = trained
    for prefer in (("torch",), ("onnx",)):
        with (
            infer.load_classifier(model_dir, prefer=prefer, device="cpu") as whole,
            infer.load_classifier(model_dir, prefer=prefer, device="cpu", max_batch=7) as split,
        ):
            pw, _ = whole.predict(batch20)
            ps, _ = split.predict(batch20)
            assert split.max_batch == 7
            assert np.allclose(pw, ps, atol=1e-5), prefer


# --------------------------------------------------------------------------------------
# 重训：旧 onnx 被替换且仍一致
# --------------------------------------------------------------------------------------


def test_retrain_replaces_onnx_and_stays_consistent(
    trained: tuple[Path, dict, float], batch20: np.ndarray, tmp_path: Path
):
    """再训一轮（max_epochs=1）：model.onnx 内容与 mtime 变化、与新 best.pt 配套，Torch/ONNX 仍一致。"""
    src, _, _ = trained
    model_dir = tmp_path / "models" / "1"
    shutil.copytree(src, model_dir)
    onnx_path = model_dir / "train" / "model.onnx"
    best_path = model_dir / "train" / "best.pt"
    old_onnx_sha, old_best_sha = _sha256(onnx_path), _sha256(best_path)
    old_mtime = onnx_path.stat().st_mtime_ns
    # 旧产物先算一遍参考输出
    with infer.load_classifier(model_dir, prefer=("onnx",)) as o_old:
        p_old, _ = o_old.predict(batch20)

    time.sleep(0.01)  # 保证 mtime 可区分
    result = train.run(
        model_dir, None, train_cfg(max_epochs=1, head_epochs=0, seed=1), class_names=CLASSES
    )
    assert result["artifacts"] == {"best": True, "onnx": True} and result["epochs"] == 1

    assert _sha256(best_path) != old_best_sha, "best.pt 应是新训练的"
    assert _sha256(onnx_path) != old_onnx_sha, "model.onnx 应被新导出替换"
    assert onnx_path.stat().st_mtime_ns > old_mtime
    assert not list((model_dir / "train").glob(".*.tmp"))
    assert read_json(model_dir / "meta.json")["version"] == 2
    assert read_json(model_dir / "train" / "metrics.json")["labels"] == CLASSES

    with (
        infer.load_classifier(model_dir, prefer=("torch",), device="cpu") as t,
        infer.load_classifier(model_dir, prefer=("onnx",)) as o,
    ):
        _assert_consistent(t, o, batch20)
        p_new, _ = o.predict(batch20)
    # 新旧模型权重不同，输出不可能逐位相同（否则说明 onnx 没被真正替换）
    assert not np.allclose(p_new, p_old, atol=1e-6)


def test_file_runtime_budget(trained: tuple[Path, dict, float]):
    """整个文件（含模块 fixture 的训练）须明显低于 90 s；训练部分在这里直接校验。"""
    _, _, seconds = trained
    assert seconds < 45
