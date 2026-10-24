"""ONNX 导出、trtexec 封装、Classifier 各后端一致性与降级、cli eval 辅助函数。

不依赖 train.py / dataset.py，不依赖 GPU / TensorRT；整个文件 < 60 s。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from oresorter.fsutil import atomic_write_json
from oresorter.ml import evaluate, export, infer, models
from oresorter.vision.preprocess import to_tensor_batch

CLASSES = ["A", "B", "C"]
S = 64


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """resnet18(3 类, S=64) 的完整模型目录：train/best.pt、meta.json、train/model.onnx。模块内共享，只读。"""
    root = tmp_path_factory.mktemp("model") / "1"
    torch.manual_seed(0)
    model = models.build_model("resnet18", len(CLASSES), pretrained=False)
    models.save_checkpoint(
        root / "train" / "best.pt",
        model,
        {"backbone": "resnet18", "classes": CLASSES, "image_size": S},
    )
    atomic_write_json(
        root / "meta.json",
        {
            "id": 1,
            "name": "模型1",
            "algo": "cnn-resnet18",
            "image_size": S,
            "classes": CLASSES,
            "version": 1,
        },
    )
    export.to_onnx(model, S, root / "train" / "model.onnx")
    return root


@pytest.fixture
def model_dir(base_model_dir: Path, tmp_path: Path) -> Path:
    """可写副本（删文件做降级测试用）。"""
    dst = tmp_path / "model"
    shutil.copytree(base_model_dir, dst)
    return dst


@pytest.fixture(scope="module")
def batch5() -> np.ndarray:
    """5 张随机 RGBA 小图经真实预处理得到的 (5,3,S,S) 输入。"""
    rng = np.random.default_rng(0)
    crops = []
    for _ in range(5):
        h, w = int(rng.integers(20, 48)), int(rng.integers(20, 48))
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((yy - h / 2) ** 2 / (h / 2) ** 2 + (xx - w / 2) ** 2 / (w / 2) ** 2) <= 1.0
        alpha = (mask * 255).astype(np.uint8)
        crops.append(np.dstack([rgb, alpha]))
    x = to_tensor_batch(crops, S)
    assert x.shape == (5, 3, S, S) and x.dtype == np.float32
    return x


def _write_script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def fake_trtexec(tmp_path: Path) -> Path:
    """假 trtexec：打印版本头、把参数写到 args.txt、往 --saveEngine 写内容、退出 0。

    ``--version`` 探测只打印版本头就退出，不覆盖 args.txt。
    """
    args_file = tmp_path / "args.txt"
    return _write_script(
        tmp_path / "trtexec_ok",
        f"""echo "&&&& RUNNING TensorRT.trtexec [TensorRT v100300] # $0"
case "$1" in --version) exit 0;; esac
printf '%s\\n' "$@" > "{args_file}"
out=""
for a in "$@"; do case "$a" in --saveEngine=*) out="${{a#--saveEngine=}}";; esac; done
if [ -n "$out" ]; then printf 'FAKEPLAN' > "$out"; fi
exit 0
""",
    )


@pytest.fixture
def slow_trtexec(tmp_path: Path) -> Path:
    return _write_script(tmp_path / "trtexec_slow", "exec sleep 5\n")


@pytest.fixture
def failing_trtexec(tmp_path: Path) -> Path:
    return _write_script(
        tmp_path / "trtexec_fail", 'echo "[E] ModelImporter: parse failed" >&2\nexit 3\n'
    )


@pytest.fixture
def no_trtexec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """让 find_trtexec 的所有默认途径都找不到。"""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.delenv("TRTEXEC", raising=False)
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(export, "JETSON_TRTEXEC", tmp_path / "no-such-trtexec")


# --------------------------------------------------------------------------------------
# to_onnx
# --------------------------------------------------------------------------------------


def test_to_onnx_dynamic_batch_and_names(base_model_dir: Path):
    import onnx
    import onnxruntime as ort

    path = base_model_dir / "train" / "model.onnx"
    assert path.is_file() and path.stat().st_size > 1_000_000
    assert not list(path.parent.glob(".model.onnx.*.tmp"))
    proto = onnx.load(str(path))
    (inp,) = [i for i in proto.graph.input if i.name == "input"]
    (out,) = [o for o in proto.graph.output if o.name == "logits"]
    in_dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    out_dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
    assert in_dims[1:] == [3, S, S] and isinstance(in_dims[0], str)
    assert out_dims[1:] == [3] and isinstance(out_dims[0], str)
    assert any(o.domain == "" and o.version == 17 for o in proto.opset_import)
    assert export.onnx_input_size(path) == S

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 7):
        y = sess.run(["logits"], {"input": np.zeros((n, 3, S, S), np.float32)})[0]
        assert y.shape == (n, 3)


def test_to_onnx_does_not_mutate_model_and_rejects_bad_params(tmp_path: Path):
    model = models.build_model("resnet18", 2, pretrained=False)
    model.train()
    export.to_onnx(model, 32, tmp_path / "m.onnx")
    assert model.training, "导出应使用副本，不改调用方模型的模式"
    with pytest.raises(ValueError):
        export.to_onnx(model, 4, tmp_path / "bad.onnx")
    with pytest.raises(ValueError):
        export.to_onnx(model, 32, tmp_path / "bad.onnx", opset=5)
    assert not (tmp_path / "bad.onnx").exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_mobilenet_onnx_and_onnx_classifier(tmp_path: Path):
    torch.manual_seed(1)
    classes = ["A", "B", "C", "D"]
    model = models.build_model("mobilenet_v3_large", len(classes), pretrained=False)
    root = tmp_path / "mb"
    models.save_checkpoint(
        root / "train" / "best.pt",
        model,
        {"backbone": "mobilenet_v3_large", "classes": classes, "image_size": 48},
    )
    export.to_onnx(model, 48, root / "train" / "model.onnx")
    x = np.random.default_rng(3).standard_normal((3, 3, 48, 48)).astype(np.float32)
    onnx_clf = infer.OnnxClassifier()
    onnx_clf.load(root)
    torch_clf = infer.TorchClassifier(device="cpu")
    torch_clf.load(root)
    pa, ca = onnx_clf.predict(x)
    pb, cb = torch_clf.predict(x)
    assert ca == cb == classes and pa.shape == (3, 4)
    cmp = infer.compare_outputs(pa, pb)
    assert cmp["top1_agree_rate"] == 1.0 and cmp["max_abs_diff"] < 1e-3


# --------------------------------------------------------------------------------------
# Classifier 一致性 / 行为
# --------------------------------------------------------------------------------------


def test_torch_vs_onnx_consistency(base_model_dir: Path, batch5: np.ndarray):
    t = infer.TorchClassifier(device="cpu")
    t.load(base_model_dir)
    o = infer.OnnxClassifier()
    o.load(base_model_dir)
    assert t.backend == "torch" and o.backend == "onnx"
    assert t.classes == o.classes == CLASSES
    assert t.image_size == o.image_size == S
    assert t.model_dir == o.model_dir == base_model_dir
    assert o.providers == ["CPUExecutionProvider"]
    assert t.fp16 is False

    pt, ct = t.predict(batch5)
    po, co = o.predict(batch5)
    assert ct == co == CLASSES
    assert pt.shape == po.shape == (5, 3)
    assert pt.dtype == po.dtype == np.float32
    assert np.allclose(pt.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(po.sum(axis=1), 1.0, atol=1e-5)
    assert (pt >= 0).all() and (po >= 0).all()
    cmp = infer.compare_outputs(pt, po)
    assert cmp["n"] == 5 and cmp["top1_agree_rate"] == 1.0 and cmp["max_abs_diff"] < 1e-3

    idx, conf = t.predict_top(batch5)
    assert idx.shape == (5,) and conf.shape == (5,)
    assert (idx == pt.argmax(axis=1)).all()
    assert np.allclose(conf, pt.max(axis=1))
    t.close()
    o.close()


def test_empty_and_dynamic_batches(base_model_dir: Path):
    o = infer.OnnxClassifier()
    o.load(base_model_dir)
    probs, classes = o.predict(np.zeros((0, 3, S, S), np.float32))
    assert probs.shape == (0, 3) and probs.dtype == np.float32 and classes == CLASSES
    idx, conf = o.predict_top(np.zeros((0, 3, S, S), np.float32))
    assert idx.shape == (0,) and conf.shape == (0,)
    for n in (1, 7):
        p, _ = o.predict(np.random.default_rng(n).standard_normal((n, 3, S, S)).astype(np.float32))
        assert p.shape == (n, 3)


def test_max_batch_splitting_matches_single_shot(base_model_dir: Path, batch5: np.ndarray):
    whole = infer.OnnxClassifier()
    whole.load(base_model_dir)
    split = infer.OnnxClassifier(max_batch=2, opt_batch=2)
    split.load(base_model_dir)
    calls: list[int] = []
    orig = split._forward

    def spy(x: np.ndarray) -> np.ndarray:
        calls.append(x.shape[0])
        return orig(x)

    split._forward = spy  # type: ignore[method-assign]
    pw, _ = whole.predict(batch5)
    ps, _ = split.predict(batch5)
    assert calls == [2, 2, 1]
    assert np.allclose(pw, ps, atol=1e-6)


def test_opt_batch_defaults_follow_small_max_batch(
    base_model_dir: Path, model_dir: Path, tmp_path: Path, fake_trtexec: Path
):
    """PipelineConfig.infer_batch_max 允许 1~512：max_batch<32 时默认 opt_batch 自动收缩，不能 ValueError。"""
    assert export.resolve_opt_batch(16, None) == 16
    assert export.resolve_opt_batch(256, None) == 32
    assert export.resolve_opt_batch(16, 4) == 4
    with pytest.raises(ValueError, match="opt_batch"):
        export.resolve_opt_batch(16, 32)
    with pytest.raises(ValueError, match="max_batch"):
        export.resolve_opt_batch(0, None)

    assert infer.OnnxClassifier(max_batch=16).opt_batch == 16
    assert infer.TorchClassifier(max_batch=512).opt_batch == 32
    assert infer.TrtClassifier(max_batch=3).opt_batch == 3
    assert infer.OnnxClassifier(max_batch=1).opt_batch == 1
    clf = infer.load_classifier(base_model_dir, prefer=("onnx",), max_batch=16)
    assert clf.backend == "onnx" and clf.max_batch == 16 and clf.opt_batch == 16
    clf.warmup()
    with pytest.raises(ValueError):
        infer.load_classifier(base_model_dir, prefer=("onnx",), max_batch=16, opt_batch=32)
    with pytest.raises(ValueError):
        infer.load_classifier(base_model_dir, prefer=("onnx",), max_batch=0)

    # build_engine 同步：只给 max_batch=16 时 optShapes 也是 16
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "model.plan"
    export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec), max_batch=16)
    args = (tmp_path / "args.txt").read_text().split("\n")
    assert f"--optShapes=input:16x3x{S}x{S}" in args
    assert f"--maxShapes=input:16x3x{S}x{S}" in args
    meta = json.loads(export.engine_meta_path(plan).read_text(encoding="utf-8"))
    assert meta["opt_batch"] == 16 and meta["max_batch"] == 16
    export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec), max_batch=256)
    assert f"--optShapes=input:32x3x{S}x{S}" in (tmp_path / "args.txt").read_text().split("\n")

    # cli：max_batch 给小值、opt_batch 不给 / None 都可以
    ns = argparse.Namespace(
        model_dir=model_dir, fp16=True, timeout=30, trtexec=str(fake_trtexec), max_batch=8
    )
    assert export.build_engine_cli(ns) == 0
    args = (tmp_path / "args.txt").read_text().split("\n")
    assert f"--optShapes=input:8x3x{S}x{S}" in args and f"--maxShapes=input:8x3x{S}x{S}" in args
    ns_none = argparse.Namespace(
        model_dir=model_dir,
        fp16=True,
        timeout=None,
        trtexec=str(fake_trtexec),
        max_batch=None,
        opt_batch=None,
    )
    assert export.build_engine_cli(ns_none) == 0
    assert f"--maxShapes=input:128x3x{S}x{S}" in (tmp_path / "args.txt").read_text().split("\n")
    ns_bad = argparse.Namespace(
        model_dir=model_dir, fp16=True, timeout=30, trtexec=str(fake_trtexec), max_batch=0
    )
    assert export.build_engine_cli(ns_bad) == 1


def test_warmup_and_validation_errors(base_model_dir: Path):
    clf = infer.TorchClassifier(device="cpu", opt_batch=4, max_batch=8)
    with pytest.raises(infer.ClassifierUnavailable):
        clf.predict(np.zeros((1, 3, S, S), np.float32))
    clf.load(base_model_dir)
    clf.warmup()
    with pytest.raises(ValueError, match="空间尺寸"):
        clf.predict(np.zeros((2, 3, 32, 32), np.float32))
    with pytest.raises(ValueError, match="形状"):
        clf.predict(np.zeros((3, S, S), np.float32))
    with pytest.raises(ValueError):
        clf.predict([[0.0]])  # type: ignore[arg-type]
    p, _ = clf.predict(np.zeros((2, 3, S, S), np.float64))  # float64 自动转 float32
    assert p.dtype == np.float32
    clf.close()
    with pytest.raises(infer.ClassifierUnavailable):
        clf.predict(np.zeros((1, 3, S, S), np.float32))
    with pytest.raises(ValueError):
        infer.TorchClassifier(max_batch=2, opt_batch=4)


def test_torch_fp16_falls_back_on_cpu(base_model_dir: Path):
    clf = infer.TorchClassifier(device="cpu", fp16=True)
    clf.load(base_model_dir)
    assert clf.fp16 is False
    p, _ = clf.predict(np.zeros((1, 3, S, S), np.float32))
    assert np.isfinite(p).all()


def test_torch_classifier_rejects_bad_device(base_model_dir: Path):
    with pytest.raises(ValueError, match="device"):
        infer.TorchClassifier(device="gpu")
    with pytest.raises(ValueError, match="device"):
        infer.load_classifier(base_model_dir, device="gpu")
    for dev in infer.DEVICES:
        assert infer.TorchClassifier(device=dev)._device_pref == dev


def test_predict_and_close_are_serialized(base_model_dir: Path):
    """close 等待在飞的 predict；关闭后 predict 抛 ClassifierUnavailable 而不是 AssertionError；多线程无内部错误。"""
    x = np.zeros((2, 3, S, S), np.float32)

    # 1) close 阻塞到 forward 结束
    clf = infer.OnnxClassifier()
    clf.load(base_model_dir)
    orig = clf._forward
    started = threading.Event()

    def slow_forward(arr: np.ndarray) -> np.ndarray:
        started.set()
        time.sleep(0.3)
        return orig(arr)

    clf._forward = slow_forward  # type: ignore[method-assign]
    outcome: dict[str, object] = {}

    def worker() -> None:
        try:
            outcome["probs"] = clf.predict(x)[0]
        except Exception as exc:
            outcome["error"] = exc

    th = threading.Thread(target=worker)
    th.start()
    assert started.wait(5)
    t0 = time.monotonic()
    clf.close()
    waited = time.monotonic() - t0
    th.join(10)
    assert not th.is_alive()
    assert "error" not in outcome and outcome["probs"].shape == (2, 3)  # type: ignore[union-attr]
    assert waited >= 0.2, "close 应等待在飞的 predict 结束"
    with pytest.raises(infer.ClassifierUnavailable):
        clf.predict(x)
    clf.close()  # 幂等

    # 2) 内部资源被置空时是 ClassifierUnavailable，不是 AssertionError
    t = infer.TorchClassifier(device="cpu")
    t.load(base_model_dir)
    t._model = None
    with pytest.raises(infer.ClassifierUnavailable):
        t.predict(x)
    o = infer.OnnxClassifier()
    o.load(base_model_dir)
    o._session = None
    with pytest.raises(infer.ClassifierUnavailable):
        o.predict(x)

    # 3) 多线程 predict 与 close 并发：结果只能是正常概率或 ClassifierUnavailable
    clf2 = infer.OnnxClassifier()
    clf2.load(base_model_dir)
    errors: list[BaseException] = []
    results: list[object] = []

    def hammer() -> None:
        for _ in range(10):
            try:
                results.append(clf2.predict(x)[0].shape)
            except infer.ClassifierUnavailable:
                results.append("closed")
            except BaseException as exc:
                errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for th in threads:
        th.start()
    time.sleep(0.02)
    clf2.close()
    for th in threads:
        th.join(30)
    assert errors == []
    assert len(results) == 40 and set(results) <= {(2, 3), "closed"}


def test_trt_classifier_unavailable_without_tensorrt(base_model_dir: Path):
    try:
        import tensorrt  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("本机有 tensorrt，由 jetson 标记的测试覆盖")
    with pytest.raises(infer.ClassifierUnavailable):
        infer.TrtClassifier().load(base_model_dir)


class _FakeTrt:
    """tensorrt 模块的最小替身（只有 _resolve_io 用到的部分）。"""

    class TensorIOMode:
        INPUT = "INPUT"
        OUTPUT = "OUTPUT"

    @staticmethod
    def nptype(dt: object) -> type:
        return np.float32


class _FakeEngineV3:
    """TensorRT 10 风格引擎：按张量名访问。"""

    def __init__(self, s: int, c: int, max_batch: int = 64, out_rank: int = 2):
        out_shape = (-1, c) if out_rank == 2 else (-1, c, 1)
        self._t = {"input": ("INPUT", (-1, 3, s, s)), "logits": ("OUTPUT", out_shape)}
        self._max = max_batch
        self.num_io_tensors = 2

    def get_tensor_name(self, i: int) -> str:
        return list(self._t)[i]

    def get_tensor_mode(self, name: str) -> str:
        return self._t[name][0]

    def get_tensor_dtype(self, name: str) -> str:
        return "f32"

    def get_tensor_shape(self, name: str) -> tuple:
        return self._t[name][1]

    def get_tensor_profile_shape(self, name: str, profile: int) -> tuple:
        tail = self._t[name][1][1:]
        return ((1, *tail), (32, *tail), (self._max, *tail))


class _FakeEngineV2:
    """TensorRT 8.x 风格引擎：按 binding 索引访问。"""

    def __init__(self, s: int, c: int, max_batch: int = 64):
        self._b = [("input", True, (-1, 3, s, s)), ("logits", False, (-1, c))]
        self._max = max_batch
        self.num_bindings = 2

    def binding_is_input(self, i: int) -> bool:
        return self._b[i][1]

    def get_binding_name(self, i: int) -> str:
        return self._b[i][0]

    def get_binding_dtype(self, i: int) -> str:
        return "f32"

    def get_binding_shape(self, i: int) -> tuple:
        return self._b[i][2]

    def get_profile_shape(self, profile: int, i: int) -> tuple:
        tail = self._b[i][2][1:]
        return ((1, *tail), (32, *tail), (self._max, *tail))


def _trt_with_fake_engine(engine: object, api_v3: bool, max_batch: int = 128):
    """TrtClassifier + 假引擎；_allocate_buffers 不碰 CUDA，只记录 (S, C, max_batch)。"""
    clf = infer.TrtClassifier(max_batch=max_batch)
    clf._trt = _FakeTrt
    clf._engine = engine
    clf._api_v3 = api_v3
    allocs: list[tuple[int, int, int]] = []
    clf._allocate_buffers = lambda s, c: allocs.append((s, c, clf.max_batch))  # type: ignore[method-assign]
    return clf, allocs


@pytest.mark.parametrize("api_v3", [True, False])
def test_trt_resolve_io_validates_output_classes(api_v3: bool):
    """引擎输出 C 必须等于 meta 类别数，否则加载期拒绝（而不是越界写显存）；输出缓冲按引擎 C 分配。"""
    make = _FakeEngineV3 if api_v3 else _FakeEngineV2
    meta = {"classes": CLASSES, "image_size": S}

    clf, allocs = _trt_with_fake_engine(make(S, 3), api_v3)
    clf._resolve_io(meta)
    assert allocs == [(S, 3, 64)], "max_batch 收缩到引擎 profile 的 64 后再分配"
    assert clf.engine_num_classes == 3 and clf.engine_max_batch == 64
    assert clf.max_batch == 64 and clf.opt_batch == 32
    assert clf._input_name == "input" and clf._output_name == "logits"
    assert clf._in_torch_dtype == torch.float32 and clf._out_torch_dtype == torch.float32

    clf, allocs = _trt_with_fake_engine(make(S, 7), api_v3)
    with pytest.raises(infer.ClassifierUnavailable, match="类别数 7"):
        clf._resolve_io(meta)
    assert allocs == []

    clf, allocs = _trt_with_fake_engine(make(S, 2), api_v3)
    with pytest.raises(infer.ClassifierUnavailable, match="类别数 2"):
        clf._resolve_io(meta)
    assert allocs == []

    clf, allocs = _trt_with_fake_engine(make(32, 3), api_v3)
    with pytest.raises(infer.ClassifierUnavailable, match="image_size"):
        clf._resolve_io(meta)
    assert allocs == []

    # 请求的 max_batch 小于引擎 profile 时按请求值分配
    clf, allocs = _trt_with_fake_engine(make(S, 3, max_batch=256), api_v3, max_batch=16)
    clf._resolve_io(meta)
    assert allocs == [(S, 3, 16)] and clf.opt_batch == 16


def test_trt_resolve_io_rejects_non_2d_output():
    clf, allocs = _trt_with_fake_engine(_FakeEngineV3(S, 3, out_rank=3), True)
    with pytest.raises(infer.ClassifierUnavailable, match=r"\(N,C\)"):
        clf._resolve_io({"classes": CLASSES, "image_size": S})
    assert allocs == []


@pytest.mark.jetson
def test_trt_vs_torch_consistency_on_jetson(base_model_dir: Path, batch5: np.ndarray):
    """Jetson 真机：生成引擎后 TensorRT 与 PyTorch top-1 一致、概率差 < 1e-2。"""
    pytest.importorskip("tensorrt")
    if export.find_trtexec() is None or not torch.cuda.is_available():
        pytest.skip("没有 trtexec / CUDA")
    export.build_engine(
        base_model_dir / "train" / "model.onnx", base_model_dir / "train" / "model.plan", S
    )
    trt_clf = infer.TrtClassifier()
    trt_clf.load(base_model_dir)
    torch_clf = infer.TorchClassifier(device="cuda")
    torch_clf.load(base_model_dir)
    cmp = infer.compare_outputs(trt_clf.predict(batch5)[0], torch_clf.predict(batch5)[0])
    assert cmp["top1_agree_rate"] == 1.0 and cmp["max_abs_diff"] < 1e-2


# --------------------------------------------------------------------------------------
# load_classifier 降级
# --------------------------------------------------------------------------------------


def test_load_classifier_degrades(model_dir: Path, tmp_path: Path):
    clf = infer.load_classifier(model_dir, device="cpu")
    assert clf.backend == "onnx" and clf.classes == CLASSES
    (model_dir / "train" / "model.onnx").unlink()
    clf2 = infer.load_classifier(model_dir, device="cpu")
    assert clf2.backend == "torch"
    clf3 = infer.load_classifier(model_dir, prefer=("torch",), device="cpu")
    assert clf3.backend == "torch"
    (model_dir / "train" / "best.pt").unlink()
    with pytest.raises(infer.ClassifierUnavailable):
        infer.load_classifier(model_dir, device="cpu")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(infer.ClassifierUnavailable):
        infer.load_classifier(empty)
    with pytest.raises(infer.ClassifierUnavailable):
        infer.load_classifier(tmp_path / "missing")
    with pytest.raises(ValueError):
        infer.load_classifier(model_dir, prefer=("tensorflow",))
    with pytest.raises(ValueError):
        infer.load_classifier(model_dir, prefer=())


def test_load_classifier_onnx_requires_meta(model_dir: Path):
    """只有 model.onnx、没有 best.pt 与 meta.json → onnx 后端拿不到类别 → 降级失败。"""
    (model_dir / "train" / "best.pt").unlink()
    (model_dir / "meta.json").unlink()
    with pytest.raises(infer.ClassifierUnavailable):
        infer.load_classifier(model_dir, device="cpu")
    atomic_write_json(model_dir / "meta.json", {"classes": CLASSES, "image_size": S})
    clf = infer.load_classifier(model_dir, device="cpu")
    assert clf.backend == "onnx" and clf.classes == CLASSES


def test_onnx_classifier_rejects_size_mismatch(model_dir: Path):
    (model_dir / "train" / "best.pt").unlink()
    atomic_write_json(model_dir / "meta.json", {"classes": CLASSES, "image_size": 128})
    with pytest.raises(infer.ClassifierUnavailable, match="不一致"):
        infer.OnnxClassifier().load(model_dir)


def test_onnx_class_count_mismatch_detected_at_load(model_dir: Path):
    """ONNX 输出 C ≠ meta 类别数 → load 期 ClassifierUnavailable，load_classifier 能降级到 best.pt。"""
    torch.manual_seed(2)
    four = models.build_model("resnet18", 4, pretrained=False)
    export.to_onnx(four, S, model_dir / "train" / "model.onnx")  # 4 类 ONNX 配 3 类 best.pt
    with pytest.raises(infer.ClassifierUnavailable, match="类别数 4"):
        infer.OnnxClassifier().load(model_dir)
    clf = infer.load_classifier(model_dir, prefer=("onnx", "torch"), device="cpu")
    assert clf.backend == "torch" and clf.classes == CLASSES
    p, _ = clf.predict(np.zeros((1, 3, S, S), np.float32))
    assert p.shape == (1, 3)

    # 审查员的复现：只有 meta.json（2 类）+ 4 类 ONNX → 不能“加载成功然后运行时报错”
    (model_dir / "train" / "best.pt").unlink()
    atomic_write_json(model_dir / "meta.json", {"classes": ["A", "B"], "image_size": S})
    with pytest.raises(infer.ClassifierUnavailable):
        infer.OnnxClassifier().load(model_dir)
    with pytest.raises(infer.ClassifierUnavailable):
        infer.load_classifier(model_dir, prefer=("onnx", "torch"), device="cpu")


# --------------------------------------------------------------------------------------
# read_model_meta
# --------------------------------------------------------------------------------------


def test_read_model_meta_prefers_checkpoint_then_meta_json(model_dir: Path, tmp_path: Path):
    meta = infer.read_model_meta(model_dir)
    assert meta["classes"] == CLASSES and meta["image_size"] == S and meta["backbone"] == "resnet18"

    # best.pt 优先：meta.json 写了不同类别也不生效
    atomic_write_json(model_dir / "meta.json", {"classes": ["X", "Y"], "image_size": 224})
    assert infer.read_model_meta(model_dir)["classes"] == CLASSES

    # 回退 meta.json
    (model_dir / "train" / "best.pt").unlink()
    atomic_write_json(
        model_dir / "meta.json",
        {"classes": ["X", "Y"], "image_size": 224, "algo": "cnn-mobilenet_v3_large"},
    )
    meta = infer.read_model_meta(model_dir)
    assert meta["classes"] == ["X", "Y"] and meta["image_size"] == 224
    assert meta["backbone"] == "mobilenet_v3_large"

    # meta.json 不完整 / 不存在
    atomic_write_json(model_dir / "meta.json", {"classes": [], "image_size": None})
    with pytest.raises(infer.ClassifierUnavailable):
        infer.read_model_meta(model_dir)
    (model_dir / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(infer.ClassifierUnavailable):
        infer.read_model_meta(model_dir)
    with pytest.raises(infer.ClassifierUnavailable):
        infer.read_model_meta(tmp_path / "nothing")


def test_read_model_meta_bad_checkpoint_falls_back(model_dir: Path):
    (model_dir / "train" / "best.pt").write_bytes(b"garbage")
    meta = infer.read_model_meta(model_dir)
    assert meta["classes"] == CLASSES and meta["image_size"] == S


# --------------------------------------------------------------------------------------
# compare_outputs / softmax
# --------------------------------------------------------------------------------------


def test_compare_outputs_and_softmax():
    a = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]], np.float32)
    b = a.copy()
    b[2] = [0.4, 0.3, 0.3]
    r = infer.compare_outputs(a, b)
    assert r["n"] == 4 and r["n_disagree"] == 1 and r["top1_agree_rate"] == pytest.approx(0.75)
    assert r["max_abs_diff"] == pytest.approx(0.1, abs=1e-6)
    assert infer.compare_outputs(a, a) == {
        "top1_agree_rate": 1.0,
        "max_abs_diff": 0.0,
        "n": 4,
        "n_disagree": 0,
    }
    assert infer.compare_outputs(np.zeros((0, 3)), np.zeros((0, 3)))["n"] == 0
    with pytest.raises(ValueError):
        infer.compare_outputs(a, a[:2])
    probs = infer.softmax(np.array([[1000.0, 0.0], [0.0, 0.0]], np.float32))
    assert np.allclose(probs, [[1.0, 0.0], [0.5, 0.5]]) and probs.dtype == np.float32
    assert infer.softmax(np.zeros((0, 3))).shape == (0, 3)


# --------------------------------------------------------------------------------------
# trtexec / build_engine
# --------------------------------------------------------------------------------------


def test_find_trtexec_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_trtexec: None):
    assert export.find_trtexec() is None
    a = _write_script(tmp_path / "a", "exit 0\n")
    b = _write_script(tmp_path / "b", "exit 0\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    c = _write_script(bindir / "trtexec", "exit 0\n")

    monkeypatch.setenv("PATH", str(bindir))
    assert export.find_trtexec() == c
    monkeypatch.setenv("TRTEXEC", str(b))
    assert export.find_trtexec() == b
    assert export.find_trtexec(str(a)) == a.resolve()
    assert export.find_trtexec(str(tmp_path / "missing")) is None, "显式路径无效时不能偷换成别的"
    monkeypatch.setenv("TRTEXEC", str(tmp_path / "missing"))
    assert export.find_trtexec() == c, "$TRTEXEC 无效时继续查默认位置"
    not_exec = tmp_path / "plain"
    not_exec.write_text("x")
    assert export.find_trtexec(str(not_exec)) is None


def test_build_engine_without_trtexec(base_model_dir: Path, tmp_path: Path, no_trtexec: None):
    onnx_path = base_model_dir / "train" / "model.onnx"
    with pytest.raises(export.EngineBuildError, match="Jetson"):
        export.build_engine(onnx_path, tmp_path / "model.plan", S)
    assert not (tmp_path / "model.plan").exists()


def test_build_engine_param_validation(base_model_dir: Path, tmp_path: Path, fake_trtexec: Path):
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "model.plan"
    with pytest.raises(ValueError):
        export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec), max_batch=4, opt_batch=8)
    with pytest.raises(ValueError):
        export.build_engine(onnx_path, plan, 4, trtexec=str(fake_trtexec))
    with pytest.raises(ValueError):
        export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec), timeout_s=0)
    with pytest.raises(export.EngineBuildError, match="不存在"):
        export.build_engine(tmp_path / "missing.onnx", plan, S, trtexec=str(fake_trtexec))
    assert not plan.exists()


def test_build_engine_success_writes_plan_and_meta(
    base_model_dir: Path, tmp_path: Path, fake_trtexec: Path, no_trtexec: None
):
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "out" / "model.plan"
    result = export.build_engine(
        onnx_path, plan, S, trtexec=str(fake_trtexec), max_batch=8, opt_batch=4, timeout_s=30
    )
    assert result == plan and plan.read_bytes() == b"FAKEPLAN"
    assert not list(plan.parent.glob(".model.plan.*.tmp")), "不能留下临时文件"

    args = (tmp_path / "args.txt").read_text().split("\n")
    assert f"--onnx={onnx_path}" in args
    assert "--fp16" in args
    assert f"--minShapes=input:1x3x{S}x{S}" in args
    assert f"--optShapes=input:4x3x{S}x{S}" in args
    assert f"--maxShapes=input:8x3x{S}x{S}" in args
    assert any(a.startswith("--saveEngine=") and a.endswith(".tmp") for a in args)

    meta_path = export.engine_meta_path(plan)
    assert meta_path == plan.with_suffix(".json") and meta_path.is_file()
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["trt_version"] == "10.3.0"
    assert meta["onnx_sha256"] == export.sha256_file(onnx_path)
    assert meta["image_size"] == S and meta["fp16"] is True
    assert meta["min_batch"] == 1 and meta["opt_batch"] == 4 and meta["max_batch"] == 8
    assert meta["plan_bytes"] == len(b"FAKEPLAN") and "built_at" in meta

    # 匹配判断：同 onnx + 同版本 → True；onnx 变了 / 尺寸不同 / 版本查不到 → False
    assert export.engine_matches(plan, onnx_path, trtexec=str(fake_trtexec)) is True
    assert export.engine_matches(plan, onnx_path, trtexec=str(fake_trtexec), image_size=S) is True
    assert (
        export.engine_matches(plan, onnx_path, trtexec=str(fake_trtexec), image_size=128) is False
    )
    other_onnx = tmp_path / "other.onnx"
    other_onnx.write_bytes(onnx_path.read_bytes() + b"\0")
    assert export.engine_matches(plan, other_onnx, trtexec=str(fake_trtexec)) is False
    assert export.engine_matches(plan, onnx_path) is False, "本机查不到 TensorRT 版本时按不匹配"
    assert (
        export.engine_matches(tmp_path / "none.plan", onnx_path, trtexec=str(fake_trtexec)) is False
    )
    meta_path.unlink()
    assert export.engine_matches(plan, onnx_path, trtexec=str(fake_trtexec)) is False

    # 不开 fp16
    export.build_engine(onnx_path, plan, S, fp16=False, trtexec=str(fake_trtexec))
    assert "--fp16" not in (tmp_path / "args.txt").read_text().split("\n")


def test_build_engine_timeout_and_failure(
    base_model_dir: Path, tmp_path: Path, slow_trtexec: Path, failing_trtexec: Path
):
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "model.plan"
    with pytest.raises(export.EngineBuildError, match="超时"):
        export.build_engine(onnx_path, plan, S, trtexec=str(slow_trtexec), timeout_s=1)
    assert not plan.exists() and not list(tmp_path.glob(".model.plan.*.tmp"))
    with pytest.raises(export.EngineBuildError) as ei:
        export.build_engine(onnx_path, plan, S, trtexec=str(failing_trtexec))
    assert "退出码 3" in str(ei.value) and "parse failed" in str(ei.value)
    assert not plan.exists() and not export.engine_meta_path(plan).exists()
    # 退出 0 但没产出文件
    noop = _write_script(tmp_path / "trtexec_noop", "exit 0\n")
    with pytest.raises(export.EngineBuildError, match="未生成"):
        export.build_engine(onnx_path, plan, S, trtexec=str(noop))


def test_build_engine_tolerates_non_utf8_output(base_model_dir: Path, tmp_path: Path):
    """trtexec 输出含非 UTF-8 字节：不能冒出 UnicodeDecodeError，也不能留临时文件。"""
    script = _write_script(
        tmp_path / "trtexec_bin",
        r"""printf '\377\376 [TensorRT v100300]\n'
out=""
for a in "$@"; do case "$a" in --saveEngine=*) out="${a#--saveEngine=}";; esac; done
if [ -n "$out" ]; then printf 'PLAN' > "$out"; fi
exit 0
""",
    )
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "out" / "model.plan"
    assert export.build_engine(onnx_path, plan, S, trtexec=str(script)) == plan
    assert plan.read_bytes() == b"PLAN"
    assert not list(plan.parent.glob(".model.plan.*.tmp"))
    meta = json.loads(export.engine_meta_path(plan).read_text(encoding="utf-8"))
    assert meta["trt_version"] == "10.3.0" and meta["trtexec_version"] == "10.3.0"
    assert export.trtexec_version(script) == "10.3.0"

    # 非零退出且 stderr 非 UTF-8：错误仍是 EngineBuildError，stderr 尾部被替换字符保留
    bad = _write_script(
        tmp_path / "trtexec_bin_fail", "printf '\\377 parse failed\\n' >&2\nexit 2\n"
    )
    with pytest.raises(export.EngineBuildError, match="parse failed"):
        export.build_engine(onnx_path, tmp_path / "x" / "model.plan", S, trtexec=str(bad))
    assert not list((tmp_path / "x").glob(".model.plan.*.tmp"))


def test_build_engine_cleans_tmp_on_unexpected_errors(
    base_model_dir: Path, tmp_path: Path, fake_trtexec: Path, monkeypatch: pytest.MonkeyPatch
):
    """subprocess 之外的意外异常也要删临时文件，并统一成 EngineBuildError；BaseException 原样抛出但同样清理。"""
    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "model.plan"

    def _touch_tmp(cmd: list[str]) -> None:
        for a in cmd:
            if a.startswith("--saveEngine="):
                Path(a[len("--saveEngine=") :]).write_bytes(b"partial")

    def boom(cmd: list[str], timeout_s: int):
        _touch_tmp(cmd)
        raise RuntimeError("boom")

    monkeypatch.setattr(export, "_run_trtexec", boom)
    with pytest.raises(export.EngineBuildError, match="boom"):
        export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec))
    assert not plan.exists() and not list(tmp_path.glob(".model.plan.*.tmp"))
    assert not export.engine_meta_path(plan).exists()

    class _Abort(BaseException):
        pass

    def abort(cmd: list[str], timeout_s: int):
        _touch_tmp(cmd)
        raise _Abort()

    monkeypatch.setattr(export, "_run_trtexec", abort)
    with pytest.raises(_Abort):
        export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec))
    assert not plan.exists() and not list(tmp_path.glob(".model.plan.*.tmp"))


def test_engine_version_recorded_from_same_source_as_matches(
    base_model_dir: Path, tmp_path: Path, fake_trtexec: Path, monkeypatch: pytest.MonkeyPatch
):
    """tensorrt.__version__ 带 4 段、trtexec 头只有 3 段：刚生成的引擎必须立刻 engine_matches。"""
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "10.3.0.26"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    assert export.current_trt_version(str(fake_trtexec)) == "10.3.0.26"

    onnx_path = base_model_dir / "train" / "model.onnx"
    plan = tmp_path / "model.plan"
    export.build_engine(onnx_path, plan, S, trtexec=str(fake_trtexec))
    meta = json.loads(export.engine_meta_path(plan).read_text(encoding="utf-8"))
    assert meta["trt_version"] == "10.3.0.26", "与 current_trt_version 同一来源"
    assert meta["trtexec_version"] == "10.3.0", "trtexec 输出头解析值另存供排查"
    assert export.engine_matches(plan, onnx_path, trtexec=str(fake_trtexec)) is True
    assert export.engine_matches(plan, onnx_path) is True

    # 旧元信息只记了三段，也要能匹配（归一化比较）
    meta["trt_version"] = "10.3.0"
    atomic_write_json(export.engine_meta_path(plan), meta)
    assert export.engine_matches(plan, onnx_path) is True
    fake_trt.__version__ = "10.4.0.1"  # type: ignore[attr-defined]
    assert export.engine_matches(plan, onnx_path) is False

    assert export._normalize_version("10.3.0.26") == "10.3.0"
    assert export._normalize_version("8.6.1") == "8.6.1"
    assert export._normalize_version(" 8.5.2.2 ") == "8.5.2"
    assert export._normalize_version("dev") == "dev"
    assert export._normalize_version("") is None and export._normalize_version(None) is None


def test_parse_trt_version():
    assert export._parse_trt_version("TensorRT.trtexec [TensorRT v100300]") == "10.3.0"
    assert export._parse_trt_version("[TensorRT v8601] # trtexec") == "8.6.1"
    assert export._parse_trt_version("TensorRT version: 8.5.2.2") == "8.5.2.2"
    assert export._parse_trt_version("nothing here") is None
    assert export._parse_trt_version("") is None


def test_build_engine_cli(
    model_dir: Path, fake_trtexec: Path, failing_trtexec: Path, monkeypatch, no_trtexec
):
    ns = argparse.Namespace(model_dir=model_dir, fp16=True, timeout=30, trtexec=str(fake_trtexec))
    assert export.build_engine_cli(ns) == 0
    plan = model_dir / "train" / "model.plan"
    assert plan.read_bytes() == b"FAKEPLAN" and export.engine_meta_path(plan).is_file()

    # 通过 $TRTEXEC 找到（cli 的 argparse 没有 --trtexec 参数）
    plan.unlink()
    monkeypatch.setenv("TRTEXEC", str(fake_trtexec))
    assert (
        export.build_engine_cli(argparse.Namespace(model_dir=model_dir, fp16=True, timeout=30)) == 0
    )
    assert plan.is_file()

    # 生成失败 → 1
    ns_fail = argparse.Namespace(
        model_dir=model_dir, fp16=True, timeout=30, trtexec=str(failing_trtexec)
    )
    assert export.build_engine_cli(ns_fail) == 1

    # 缺 onnx → 2；缺元信息 → 2
    (model_dir / "train" / "model.onnx").unlink()
    assert export.build_engine_cli(ns) == 2
    export.to_onnx(
        models.build_model("resnet18", 3, pretrained=False), S, model_dir / "train" / "model.onnx"
    )
    (model_dir / "train" / "best.pt").unlink()
    (model_dir / "meta.json").unlink()
    assert export.build_engine_cli(ns) == 2


def test_build_engine_cli_detects_size_mismatch(model_dir: Path, fake_trtexec: Path):
    (model_dir / "train" / "best.pt").unlink()
    atomic_write_json(model_dir / "meta.json", {"classes": CLASSES, "image_size": 128})
    ns = argparse.Namespace(model_dir=model_dir, fp16=True, timeout=30, trtexec=str(fake_trtexec))
    assert export.build_engine_cli(ns) == 2


# --------------------------------------------------------------------------------------
# evaluate 辅助函数（不依赖 dataset/train）
# --------------------------------------------------------------------------------------


def test_evaluate_helpers(tmp_path: Path):
    assert evaluate.parse_bg_color("0,0,255") == (0, 0, 255)
    assert evaluate.parse_bg_color(" 10, 20 ,30 ") == (10, 20, 30)
    for bad in ("0,0", "0,0,256", "a,b,c", "1,2,3,4", "-1,0,0"):
        with pytest.raises(ValueError):
            evaluate.parse_bg_color(bad)

    assert evaluate.choose_folds(30, 5) == (30, "loo")
    assert evaluate.choose_folds(49, 5) == (49, "loo")
    assert evaluate.choose_folds(50, 5) == (5, "kfold")
    assert evaluate.choose_folds(120, 10) == (10, "kfold")
    with pytest.raises(ValueError):
        evaluate.choose_folds(1, 5)
    with pytest.raises(ValueError):
        evaluate.choose_folds(100, 1)
    with pytest.raises(ValueError):
        evaluate.choose_folds(100, 101)

    root = tmp_path / "samples"
    for name, files in {
        "a": ["1.bmp", "2.png"],
        "b": ["x.JPG"],
        ".trash": ["z.png"],
        "empty": [],
        "note": ["r.txt"],
    }.items():
        (root / name).mkdir(parents=True)
        for f in files:
            (root / name / f).write_bytes(b"\0")
    (root / "loose.png").write_bytes(b"\0")
    assert evaluate.list_class_dirs(root) == ["a", "b"]


def _eval_args(folder: Path, **overrides) -> argparse.Namespace:
    base = {
        "folder": folder,
        "config": None,
        "image_size": None,
        "backbone": None,
        "folds": None,
        "epochs": None,
        "out": None,
        "bg_color": "0,0,255",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_sample_folder(root: Path, per_class: int, classes=("a", "b", "c")) -> Path:
    for c in classes:
        (root / c).mkdir(parents=True)
        for i in range(per_class):
            (root / c / f"{i}.png").write_bytes(b"\0")
    return root


def _install_fake_train_modules(monkeypatch: pytest.MonkeyPatch, record: dict) -> None:
    """用最小替身代替 oresorter.ml.dataset / train / metrics（本测试文件不依赖真实训练模块）。"""
    dataset = types.ModuleType("oresorter.ml.dataset")

    def list_folder_samples(root: Path, classes: list[str]):
        per = {
            c: sorted(
                str(p)
                for p in (Path(root) / c).iterdir()
                if p.suffix.lower() in evaluate.IMAGE_EXTS
            )
            for c in classes
        }
        return list(classes), per

    dataset.list_folder_samples = list_folder_samples  # type: ignore[attr-defined]

    train = types.ModuleType("oresorter.ml.train")

    class TrainError(Exception):
        pass

    class TrainConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        @classmethod
        def from_ml_config(cls, ml, **overrides):
            base = {
                "backbone": ml.backbone,
                "image_size": ml.image_size,
                "max_epochs": ml.max_epochs,
            }
            base.update(overrides)
            return cls(**base)

        def to_dict(self):
            return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    def cross_validate(items, classes, cfg, k, progress_cb):
        record.update(items=list(items), classes=list(classes), cfg=cfg, k=k)
        progress_cb({"fold": 1, "k": k})
        n = len(classes)
        return {
            "labels": list(classes),
            "matrix": [[1] * n for _ in range(n)],
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "folds": [{"fold": i} for i in range(k)],
        }

    train.TrainConfig = TrainConfig  # type: ignore[attr-defined]
    train.TrainError = TrainError  # type: ignore[attr-defined]
    train.cross_validate = cross_validate  # type: ignore[attr-defined]

    metrics = types.ModuleType("oresorter.ml.metrics")
    metrics.format_report = lambda r: f"acc={r['accuracy']}"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "oresorter.ml.dataset", dataset)
    monkeypatch.setitem(sys.modules, "oresorter.ml.train", train)
    monkeypatch.setitem(sys.modules, "oresorter.ml.metrics", metrics)


def test_evaluate_run_rejects_bad_inputs(tmp_path: Path):
    ns = _eval_args(tmp_path / "missing", folds=5)
    assert evaluate.run(ns) == 2
    folder = tmp_path / "samples"
    (folder / "a").mkdir(parents=True)
    ns.folder = folder
    ns.bg_color = "1,2"
    assert evaluate.run(ns) == 2


def test_evaluate_parse_cli_options_rejects_out_of_range():
    """0 / 负数不能被真值判断静默换成默认值。"""
    folds, ov = evaluate.parse_cli_options(_eval_args(Path(".")))
    assert folds == evaluate.DEFAULT_FOLDS and ov == {}
    folds, ov = evaluate.parse_cli_options(
        _eval_args(Path("."), folds=3, epochs=2, image_size=64, backbone="mobilenet_v3_large")
    )
    assert folds == 3
    assert ov == {"image_size": 64, "max_epochs": 2, "backbone": "mobilenet_v3_large"}
    for bad in (
        {"folds": 0},
        {"folds": 1},
        {"folds": -3},
        {"epochs": 0},
        {"epochs": -1},
        {"image_size": 0},
        {"image_size": 4},
        {"folds": "x"},
    ):
        with pytest.raises(ValueError):
            evaluate.parse_cli_options(_eval_args(Path("."), **bad))


def test_evaluate_run_with_stubbed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """用替身训练模块跑完整 run()：越界参数返回 2，合法参数按真实值传给 cross_validate 并落盘报告。"""
    record: dict = {}
    _install_fake_train_modules(monkeypatch, record)
    folder = _make_sample_folder(tmp_path / "samples", per_class=20)  # 60 张 → kfold

    for bad in ({"folds": 0}, {"epochs": 0}, {"image_size": 0}, {"folds": -3}, {"folds": 100}):
        record.clear()
        assert evaluate.run(_eval_args(folder, **bad)) == 2, bad
        assert record == {}, f"{bad} 不应进入交叉验证"

    out = tmp_path / "report.json"
    rc = evaluate.run(
        _eval_args(folder, folds=4, epochs=2, image_size=64, out=out, bg_color="1,2,3")
    )
    assert rc == 0
    assert record["k"] == 4 and record["classes"] == ["a", "b", "c"]
    assert len(record["items"]) == 60 and {idx for _, idx in record["items"]} == {0, 1, 2}
    cfg = record["cfg"]
    assert cfg.max_epochs == 2 and cfg.image_size == 64 and cfg.bg_color == (1, 2, 3)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["folds"] == 4 and report["method"] == "kfold" and report["n_samples"] == 60
    assert report["sample_counts"] == {"a": 20, "b": 20, "c": 20}
    assert report["config"]["max_epochs"] == 2 and report["config"]["bg_color"] == [1, 2, 3]
    assert report["report"]["accuracy"] == 0.9 and len(report["fold_reports"]) == 4
    assert "duration_s" in report and "created_at" in report

    # 未给数值参数 → 折数默认 5，epochs/image_size 沿用配置
    record.clear()
    assert evaluate.run(_eval_args(folder)) == 0
    assert record["k"] == 5 and "max_epochs" in vars(record["cfg"])
    assert (folder / "eval_report.json").is_file()

    # < 50 张 → 留一法，k = n
    small = _make_sample_folder(tmp_path / "small", per_class=10)
    record.clear()
    assert evaluate.run(_eval_args(small, folds=5)) == 0
    assert record["k"] == 30
    assert json.loads((small / "eval_report.json").read_text(encoding="utf-8"))["method"] == "loo"


def test_evaluate_run_returns_3_without_training_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder = _make_sample_folder(tmp_path / "samples", per_class=20)
    monkeypatch.setitem(sys.modules, "oresorter.ml.train", None)  # import 直接 ImportError
    assert evaluate.run(_eval_args(folder)) == 3


def test_tmp_files_never_left_behind(base_model_dir: Path):
    leftovers = [
        p
        for p in (base_model_dir / "train").iterdir()
        if p.name.startswith(".") and p.suffix == ".tmp"
    ]
    assert leftovers == []
    assert os.path.getsize(base_model_dir / "train" / "best.pt") > 0
