"""运行时分类器：统一的 :class:`Classifier` 接口 + TensorRT / ONNX Runtime / PyTorch 三种实现。

- 输入一律是 :func:`oresorter.vision.preprocess.to_tensor_batch` 的输出 ``(N,3,S,S) float32``；
- 输出 softmax 概率 ``(N,C) float32`` 与训练时的类别顺序；
- :func:`load_classifier` 按 ``prefer`` 顺序尝试，缺文件/缺依赖就告警降级（Jetson: trt → onnx → torch；开发机: onnx → torch）。

线程安全：``predict`` / ``warmup`` / ``close`` 共用一把可重入锁，一个实例同一时刻只跑一次前向
（TensorRT 实现共享输入/输出显存缓冲，本就不能并发）；``close`` 会等在飞的 ``predict`` 结束，
之后再 ``predict`` 抛 :class:`ClassifierUnavailable`。切换模型时先把新实例换到引用上，再 ``close`` 旧实例。
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from oresorter.fsutil import read_json
from oresorter.log import get_logger
from oresorter.ml import models
from oresorter.ml.export import DEFAULT_MAX_BATCH as DEFAULT_MAX_BATCH
from oresorter.ml.export import DEFAULT_OPT_BATCH as DEFAULT_OPT_BATCH
from oresorter.ml.export import resolve_opt_batch

log = get_logger(__name__)

BACKENDS: tuple[str, ...] = ("trt", "onnx", "torch")
DEFAULT_PREFER: tuple[str, ...] = ("trt", "onnx", "torch")
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda", "mps")  # 与 MlConfig.device 的 Literal 一致

CHECKPOINT = Path("train") / "best.pt"
ONNX_FILE = Path("train") / "model.onnx"
PLAN_FILE = Path("train") / "model.plan"
META_FILE = Path("meta.json")


class ClassifierUnavailable(Exception):
    """该推理后端在当前环境/模型目录下不可用（缺依赖、缺文件、元信息不完整、反序列化失败）。"""


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """按行 softmax（数值稳定），返回 float32。"""
    x = np.asarray(logits, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"logits 应为 (N,C)，得到 {x.shape}")
    if x.shape[0] == 0:
        return np.zeros(x.shape, np.float32)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _load_checkpoint_for_meta(path: Path) -> Any:
    """只为读 meta 打开 checkpoint：优先 ``mmap=True``（state_dict 只映射不真正读入内存，
    ResNet18 约 45 MB 的权重不会成为 IO / 内存峰值），旧版 torch 或旧序列化格式不支持时回退普通加载。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=True)


def read_model_meta(model_dir: Path) -> dict[str, Any]:
    """读取模型元信息：优先 ``train/best.pt`` 里的 meta，缺的字段再用 ``meta.json`` 补。

    返回至少含 ``classes``（非空列表）与 ``image_size``（int），有则带 ``backbone`` / ``algo``。
    两处都拿不到 → :class:`ClassifierUnavailable`。
    读 best.pt 走 mmap，只触碰 pickle 头与 meta，不把整份 state_dict 读进内存。
    """
    model_dir = Path(model_dir)
    meta: dict[str, Any] = {}
    ckpt = model_dir / CHECKPOINT
    if ckpt.is_file():
        try:
            ck = _load_checkpoint_for_meta(ckpt)
            m = ck.get("meta") if isinstance(ck, dict) else None
            if isinstance(m, dict):
                meta.update(m)
        except Exception as exc:
            log.warning(
                "读取 best.pt meta 失败，回退 meta.json", extra={"path": str(ckpt), "err": str(exc)}
            )
    if not meta.get("classes") or not meta.get("image_size"):
        try:
            mj = read_json(model_dir / META_FILE, default=None)
        except ValueError as exc:
            log.warning(
                "meta.json 损坏", extra={"path": str(model_dir / META_FILE), "err": str(exc)}
            )
            mj = None
        if isinstance(mj, dict):
            for key in ("classes", "image_size"):
                if not meta.get(key) and mj.get(key):
                    meta[key] = mj[key]
            algo = mj.get("algo")
            if "backbone" not in meta and isinstance(algo, str) and algo.startswith("cnn-"):
                meta["backbone"] = algo[len("cnn-") :]
            if "algo" not in meta and algo:
                meta["algo"] = algo
    classes = meta.get("classes")
    size = meta.get("image_size")
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        raise ClassifierUnavailable(
            f"模型目录 {model_dir} 缺少类别列表（best.pt meta / meta.json 均无 classes）"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size < 8:
        raise ClassifierUnavailable(f"模型目录 {model_dir} 缺少有效 image_size（得到 {size!r}）")
    meta["classes"] = list(classes)
    meta["image_size"] = int(size)
    return meta


def compare_outputs(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """比较两个后端的概率输出：``{"top1_agree_rate", "max_abs_diff", "n", "n_disagree"}``。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"形状不一致或不是 (N,C): {a.shape} vs {b.shape}")
    n = int(a.shape[0])
    if n == 0:
        return {"top1_agree_rate": 1.0, "max_abs_diff": 0.0, "n": 0, "n_disagree": 0}
    agree = np.argmax(a, axis=1) == np.argmax(b, axis=1)
    return {
        "top1_agree_rate": float(agree.mean()),
        "max_abs_diff": float(np.abs(a - b).max()),
        "n": n,
        "n_disagree": int((~agree).sum()),
    }


# --------------------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------------------


class Classifier(ABC):
    """分类器接口。子类实现 :meth:`load`、:meth:`_forward`（logits）与 :meth:`_release`；分批、softmax 与加锁在基类完成。

    线程安全：``predict`` / ``warmup`` / ``close`` 共用一把可重入锁（见模块 docstring）。
    ``opt_batch`` 为 None 时取 ``min(32, max_batch)``，显式给出必须在 1~max_batch 内。
    """

    backend: str = ""

    def __init__(self, *, max_batch: int = DEFAULT_MAX_BATCH, opt_batch: int | None = None):
        self.opt_batch = resolve_opt_batch(max_batch, opt_batch)  # 同时校验 max_batch ≥ 1
        self.max_batch = int(max_batch)
        self._lock = threading.RLock()
        self._classes: list[str] = []
        self._image_size: int = 0
        self._model_dir: Path | None = None
        self._loaded = False

    # ---- 子类实现 ----
    @abstractmethod
    def load(self, model_dir: Path) -> None:
        """加载模型目录；不可用时抛 :class:`ClassifierUnavailable`。"""

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """``x``: ``(n,3,S,S) float32`` 连续数组，``n ≤ max_batch``；返回 logits ``(n,C)``。在锁内调用。"""

    def _release(self) -> None:  # noqa: B027  有意为空的默认钩子，子类按需覆盖
        """释放模型 / 会话 / 显存（在锁内、``_loaded`` 已置 False 之后调用）。子类按需覆盖。"""

    # ---- 公共 ----
    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """``(N,3,S,S) float32`` → ``(probs (N,C) float32, classes)``。空 batch 返回 ``(0,C)``；N 大于 max_batch 自动分批。

        已关闭 / 未加载抛 :class:`ClassifierUnavailable`；形状、dtype 不符抛 ValueError。
        """
        with self._lock:
            self._require_loaded()
            x = self._check_batch(batch)
            n = x.shape[0]
            c = len(self._classes)
            if n == 0:
                return np.zeros((0, c), np.float32), list(self._classes)
            chunks = []
            for start in range(0, n, self.max_batch):
                logits = self._forward(np.ascontiguousarray(x[start : start + self.max_batch]))
                logits = np.asarray(logits, dtype=np.float32)
                if logits.shape != (min(self.max_batch, n - start), c):
                    raise RuntimeError(
                        f"{self.backend} 后端输出形状异常: {logits.shape}，期望 ({min(self.max_batch, n - start)}, {c})"
                    )
                chunks.append(logits)
            probs = softmax(np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0])
            return probs, list(self._classes)

    def predict_top(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """便捷：返回 ``(top1 索引 int64 (N,), 置信度 float32 (N,))``。"""
        probs, _ = self.predict(batch)
        if probs.shape[0] == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.float32)
        idx = np.argmax(probs, axis=1).astype(np.int64)
        conf = probs[np.arange(probs.shape[0]), idx].astype(np.float32)
        return idx, conf

    def warmup(self) -> None:
        """跑一次 batch=1 与 batch=opt_batch，触发懒初始化 / 显存分配 / 内核选择。"""
        with self._lock:
            self._require_loaded()
            t0 = time.monotonic()
            for n in sorted({1, self.opt_batch}):
                self.predict(np.zeros((n, 3, self._image_size, self._image_size), np.float32))
        log.info(
            "分类器预热完成",
            extra={
                "backend": self.backend,
                "opt_batch": self.opt_batch,
                "ms": round((time.monotonic() - t0) * 1000, 1),
            },
        )

    def close(self) -> None:
        """释放资源：等在飞的 ``predict`` 结束后置为未加载并调用 :meth:`_release`。幂等；关闭后再 predict 抛 ClassifierUnavailable。"""
        with self._lock:
            self._loaded = False
            self._release()

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def model_dir(self) -> Path | None:
        return self._model_dir

    @property
    def loaded(self) -> bool:
        return self._loaded

    def __enter__(self) -> Classifier:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} backend={self.backend} classes={self._classes} S={self._image_size}>"

    # ---- 内部 ----
    def _set_meta(self, model_dir: Path, meta: dict[str, Any]) -> None:
        self._model_dir = Path(model_dir)
        self._classes = list(meta["classes"])
        self._image_size = int(meta["image_size"])
        self._loaded = True

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ClassifierUnavailable(f"{type(self).__name__} 尚未加载模型或已关闭")

    def _check_batch(self, batch: np.ndarray) -> np.ndarray:
        if not isinstance(batch, np.ndarray):
            raise ValueError(f"batch 必须是 numpy 数组，得到 {type(batch).__name__}")
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"batch 形状应为 (N,3,S,S)，得到 {batch.shape}")
        s = self._image_size
        if batch.shape[2] != s or batch.shape[3] != s:
            raise ValueError(
                f"batch 空间尺寸应为 {s}x{s}（模型训练尺寸），得到 {batch.shape[2]}x{batch.shape[3]}"
            )
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32)
        return np.ascontiguousarray(batch)


# --------------------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------------------


class TorchClassifier(Classifier):
    """PyTorch 兜底实现：``train/best.pt``；CUDA 上默认 FP16。"""

    backend = "torch"

    def __init__(
        self,
        device: str = "auto",
        fp16: bool | None = None,
        *,
        max_batch: int = DEFAULT_MAX_BATCH,
        opt_batch: int | None = None,
    ):
        super().__init__(max_batch=max_batch, opt_batch=opt_batch)
        if device not in DEVICES:
            raise ValueError(f"device={device!r} 无效，可选 {DEVICES}")
        self._device_pref = device
        self._fp16_pref = fp16
        self._device: torch.device = torch.device("cpu")
        self.fp16 = False
        self._model: torch.nn.Module | None = None

    def load(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        ckpt = model_dir / CHECKPOINT
        if not ckpt.is_file():
            raise ClassifierUnavailable(f"缺少 checkpoint: {ckpt}")
        device = models.select_device(self._device_pref)
        try:
            model, meta = models.load_checkpoint(ckpt, device)
        except Exception as exc:
            raise ClassifierUnavailable(f"加载 checkpoint 失败 {ckpt}: {exc}") from exc
        if not meta.get("classes") or not meta.get("image_size"):
            raise ClassifierUnavailable(f"checkpoint meta 缺少 classes/image_size: {ckpt}")
        fp16 = device.type == "cuda" if self._fp16_pref is None else bool(self._fp16_pref)
        if fp16 and device.type != "cuda":
            log.warning("FP16 仅在 CUDA 上启用，当前设备 %s 使用 FP32", device.type)
            fp16 = False
        if fp16:
            model = model.half()
        self._device = device
        self.fp16 = fp16
        self._model = model.eval()
        self._set_meta(model_dir, meta)
        log.info(
            "PyTorch 分类器已加载",
            extra={
                "model_dir": str(model_dir),
                "device": str(device),
                "fp16": fp16,
                "classes": self._classes,
            },
        )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        model = self._model
        if model is None:
            raise ClassifierUnavailable("TorchClassifier 已关闭")
        t = torch.from_numpy(x).to(self._device)
        if self.fp16:
            t = t.half()
        with torch.inference_mode():
            out = model(t)
        return out.float().cpu().numpy()

    def _release(self) -> None:
        self._model = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------------------


class OnnxClassifier(Classifier):
    """ONNX Runtime 实现：``train/model.onnx``。默认 CPU；有 CUDAExecutionProvider 时优先用它。"""

    backend = "onnx"

    def __init__(
        self,
        providers: list[str] | None = None,
        *,
        max_batch: int = DEFAULT_MAX_BATCH,
        opt_batch: int | None = None,
    ):
        super().__init__(max_batch=max_batch, opt_batch=opt_batch)
        self._providers_pref = providers
        self.providers: list[str] = []
        self._session: Any = None
        self._input_name = ""
        self._output_name = ""

    def load(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        onnx_path = model_dir / ONNX_FILE
        if not onnx_path.is_file():
            raise ClassifierUnavailable(f"缺少 ONNX 文件: {onnx_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ClassifierUnavailable(f"onnxruntime 未安装: {exc}") from exc
        meta = read_model_meta(model_dir)
        providers = self._providers_pref
        if providers is None:
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3  # 只打 error
        try:
            session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
        except Exception as exc:
            raise ClassifierUnavailable(f"创建 onnxruntime 会话失败 {onnx_path}: {exc}") from exc
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ClassifierUnavailable(
                f"ONNX 应为单输入单输出，得到 {len(inputs)} 输入 / {len(outputs)} 输出"
            )
        shape = inputs[0].shape  # 如 ['batch', 3, 160, 160]
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] != meta["image_size"]:
            raise ClassifierUnavailable(
                f"ONNX 输入尺寸 {shape[2]} 与模型元信息 image_size={meta['image_size']} 不一致"
            )
        # 输出 (N,C) 的 C 必须等于 meta 的类别数：加载期就拒绝，让 load_classifier 能降级到自带一致 meta 的 best.pt，
        # 而不是运行时每个 batch 都抛 RuntimeError
        out_shape = outputs[0].shape  # 如 ['batch', 7]
        n_classes = len(meta["classes"])
        if len(out_shape) != 2:
            raise ClassifierUnavailable(f"ONNX 输出应为 (N,C)，得到 {out_shape}")
        if isinstance(out_shape[1], int) and out_shape[1] != n_classes:
            raise ClassifierUnavailable(
                f"ONNX 输出类别数 {out_shape[1]} 与模型元信息 classes 数量 {n_classes} 不一致"
            )
        self._session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.providers = list(session.get_providers())
        self._set_meta(model_dir, meta)
        log.info(
            "ONNX 分类器已加载",
            extra={
                "model_dir": str(model_dir),
                "providers": self.providers,
                "classes": self._classes,
            },
        )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        session = self._session
        if session is None:
            raise ClassifierUnavailable("OnnxClassifier 已关闭")
        (out,) = session.run([self._output_name], {self._input_name: x})
        return np.asarray(out, dtype=np.float32)

    def _release(self) -> None:
        self._session = None


# --------------------------------------------------------------------------------------
# TensorRT
# --------------------------------------------------------------------------------------


class TrtClassifier(Classifier):
    """TensorRT 实现：反序列化 ``train/model.plan``，用 ``torch.cuda`` 张量当输入/输出缓冲（不引入 pycuda）。

    同时覆盖 TensorRT 10（``set_tensor_address`` / ``execute_async_v3``）与 8.x（binding 索引 / ``execute_async_v2``）。
    只能在 Jetson 真机上验证（``pytest -m jetson``）。
    """

    backend = "trt"

    def __init__(self, *, max_batch: int = DEFAULT_MAX_BATCH, opt_batch: int | None = None):
        super().__init__(max_batch=max_batch, opt_batch=opt_batch)
        self._trt: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._input_name = ""
        self._output_name = ""
        self._input_idx = -1
        self._output_idx = -1
        self._in_buf: torch.Tensor | None = None
        self._out_buf: torch.Tensor | None = None
        self._in_torch_dtype: torch.dtype = torch.float32
        self._out_torch_dtype: torch.dtype = torch.float32
        self._api_v3 = True
        self.engine_max_batch = 0
        self.engine_num_classes = 0

    def load(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        try:
            import tensorrt as trt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ClassifierUnavailable(f"tensorrt 不可用（仅 Jetson）: {exc}") from exc
        if not torch.cuda.is_available():
            raise ClassifierUnavailable("TensorRT 需要 CUDA，但 torch.cuda 不可用")
        plan_path = model_dir / PLAN_FILE
        if not plan_path.is_file():
            raise ClassifierUnavailable(f"缺少 TensorRT 引擎: {plan_path}")
        meta = read_model_meta(model_dir)
        self._warn_if_stale(model_dir, plan_path)

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        try:
            engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
        except Exception as exc:
            raise ClassifierUnavailable(f"反序列化引擎失败 {plan_path}: {exc}") from exc
        if engine is None:
            raise ClassifierUnavailable(
                f"反序列化引擎失败（可能与当前 TensorRT 版本不匹配，需重新生成）: {plan_path}"
            )
        context = engine.create_execution_context()
        if context is None:
            raise ClassifierUnavailable("创建 TensorRT 执行上下文失败（显存不足？）")

        self._trt = trt
        self._runtime = runtime
        self._engine = engine
        self._context = context
        self._api_v3 = hasattr(engine, "num_io_tensors")
        self._resolve_io(meta)
        self._stream = torch.cuda.Stream()
        self._set_meta(model_dir, meta)
        log.info(
            "TensorRT 分类器已加载",
            extra={
                "model_dir": str(model_dir),
                "trt": getattr(trt, "__version__", "?"),
                "api": "v3" if self._api_v3 else "v2",
                "engine_max_batch": self.engine_max_batch,
                "max_batch": self.max_batch,
                "classes": self._classes,
            },
        )

    def _warn_if_stale(self, model_dir: Path, plan_path: Path) -> None:
        onnx_path = model_dir / ONNX_FILE
        if not onnx_path.is_file():
            return
        try:
            from oresorter.ml.export import engine_matches

            if not engine_matches(plan_path, onnx_path):
                log.warning(
                    "引擎元信息与当前 ONNX / TensorRT 版本不匹配，建议重新生成",
                    extra={"plan": str(plan_path)},
                )
        except Exception as exc:  # 元信息检查失败不影响加载
            log.debug("引擎匹配检查失败: %s", exc)

    def _resolve_io(self, meta: dict[str, Any]) -> None:
        """找出输入/输出张量名、dtype、profile 最大 batch，核对输入尺寸与输出类别数，再分配显存缓冲。

        输出类别数以引擎自己的输出张量形状为准并要求等于 ``len(meta["classes"])``：
        若只按 meta 分配 ``(max_batch, C)`` 的输出缓冲而引擎实际 C 更大，TensorRT 会越界写显存（静默损坏而非报错）。
        形状核对不依赖 CUDA，便于在开发机上用假引擎测试；真正的显存分配在 :meth:`_allocate_buffers`。
        """
        trt = self._trt
        engine = self._engine
        s = int(meta["image_size"])
        c = len(meta["classes"])
        if self._api_v3:
            names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(ins) != 1 or len(outs) != 1:
                raise ClassifierUnavailable(f"引擎应为单输入单输出，得到 {ins} / {outs}")
            self._input_name, self._output_name = ins[0], outs[0]
            in_dtype = engine.get_tensor_dtype(self._input_name)
            out_dtype = engine.get_tensor_dtype(self._output_name)
            _, _, max_shape = engine.get_tensor_profile_shape(self._input_name, 0)
            in_shape = tuple(engine.get_tensor_shape(self._input_name))
            out_shape = tuple(engine.get_tensor_shape(self._output_name))
        else:
            idx_in = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
            idx_out = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
            if len(idx_in) != 1 or len(idx_out) != 1:
                raise ClassifierUnavailable(f"引擎应为单输入单输出，得到 {idx_in} / {idx_out}")
            self._input_idx, self._output_idx = idx_in[0], idx_out[0]
            self._input_name = engine.get_binding_name(self._input_idx)
            self._output_name = engine.get_binding_name(self._output_idx)
            in_dtype = engine.get_binding_dtype(self._input_idx)
            out_dtype = engine.get_binding_dtype(self._output_idx)
            _, _, max_shape = engine.get_profile_shape(0, self._input_idx)
            in_shape = tuple(engine.get_binding_shape(self._input_idx))
            out_shape = tuple(engine.get_binding_shape(self._output_idx))
        if len(in_shape) == 4 and in_shape[2] > 0 and in_shape[2] != s:
            raise ClassifierUnavailable(f"引擎输入尺寸 {in_shape[2]} 与模型 image_size={s} 不一致")
        if len(out_shape) != 2:
            raise ClassifierUnavailable(f"引擎输出应为 (N,C)，得到 {out_shape}")
        engine_c = int(out_shape[1])
        if engine_c < 1:
            raise ClassifierUnavailable(f"引擎输出类别维不是静态正整数: {out_shape}")
        if engine_c != c:
            raise ClassifierUnavailable(
                f"引擎输出类别数 {engine_c} 与 meta classes 数量 {c} 不一致"
            )
        self.engine_num_classes = engine_c
        self.engine_max_batch = int(max_shape[0])
        if self.engine_max_batch < 1:
            raise ClassifierUnavailable(f"引擎 profile 最大 batch 无效: {max_shape}")
        if self.max_batch > self.engine_max_batch:
            log.warning(
                "引擎最大 batch 小于请求值，自动收缩",
                extra={"engine_max_batch": self.engine_max_batch, "requested": self.max_batch},
            )
            self.max_batch = self.engine_max_batch
            self.opt_batch = min(self.opt_batch, self.max_batch)
        self._in_torch_dtype = _trt_to_torch_dtype(trt, in_dtype)
        self._out_torch_dtype = _trt_to_torch_dtype(trt, out_dtype)
        self._allocate_buffers(s, engine_c)

    def _allocate_buffers(self, image_size: int, num_classes: int) -> None:
        """按 ``max_batch`` 预分配 CUDA 输入 ``(max_batch,3,S,S)`` / 输出 ``(max_batch,C)`` 缓冲，C 取引擎实际值。"""
        dev = torch.device("cuda")
        self._in_buf = torch.empty(
            (self.max_batch, 3, image_size, image_size), dtype=self._in_torch_dtype, device=dev
        )
        self._out_buf = torch.empty(
            (self.max_batch, num_classes), dtype=self._out_torch_dtype, device=dev
        )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if self._context is None or self._in_buf is None or self._out_buf is None:
            raise ClassifierUnavailable("TrtClassifier 已关闭")
        n = x.shape[0]
        ctx = self._context
        with torch.cuda.stream(self._stream):
            self._in_buf[:n].copy_(torch.from_numpy(x).to(self._in_buf.dtype), non_blocking=True)
            if self._api_v3:
                if not ctx.set_input_shape(self._input_name, tuple(self._in_buf[:n].shape)):
                    raise RuntimeError(f"TensorRT 拒绝输入形状 {self._in_buf[:n].shape}")
                ctx.set_tensor_address(self._input_name, self._in_buf.data_ptr())
                ctx.set_tensor_address(self._output_name, self._out_buf.data_ptr())
                ok = ctx.execute_async_v3(self._stream.cuda_stream)
            else:
                if not ctx.set_binding_shape(self._input_idx, tuple(self._in_buf[:n].shape)):
                    raise RuntimeError(f"TensorRT 拒绝输入形状 {self._in_buf[:n].shape}")
                bindings = [0] * self._engine.num_bindings
                bindings[self._input_idx] = self._in_buf.data_ptr()
                bindings[self._output_idx] = self._out_buf.data_ptr()
                ok = ctx.execute_async_v2(bindings=bindings, stream_handle=self._stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async 返回失败")
            out = self._out_buf[:n].float().cpu()  # 隐式同步当前流
        self._stream.synchronize()
        return out.numpy()

    def _release(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        self._in_buf = None
        self._out_buf = None
        self._stream = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _trt_to_torch_dtype(trt: Any, dt: Any) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
        "int8": torch.int8,
        "bool": torch.bool,
    }
    try:
        name = np.dtype(trt.nptype(dt)).name
    except Exception:
        name = "float32"
    return mapping.get(name, torch.float32)


# --------------------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------------------


def load_classifier(
    model_dir: Path,
    prefer: tuple[str, ...] = DEFAULT_PREFER,
    device: str = "auto",
    *,
    max_batch: int = DEFAULT_MAX_BATCH,
    opt_batch: int | None = None,
) -> Classifier:
    """按 ``prefer`` 顺序尝试各后端，第一个加载成功的返回；每次失败记 warning；全部失败抛 :class:`ClassifierUnavailable`。

    Args:
        model_dir: ``{data_dir}/models/{id}``。
        prefer: 后端顺序，元素 ∈ ``("trt", "onnx", "torch")``。
        device: 传给 TorchClassifier 的设备偏好，∈ ``("auto", "cpu", "cuda", "mps")``。
        max_batch: 分批上限（建议传 ``cfg.pipeline.infer_batch_max``，1~512 都合法）。
        opt_batch: 预热 batch；None 取 ``min(32, max_batch)``。

    Raises:
        ValueError: ``prefer`` / ``device`` / ``max_batch`` / ``opt_batch`` 参数无效（不会被当作降级原因吞掉）。
        ClassifierUnavailable: 模型目录不存在或所有后端都加载失败。
    """
    model_dir = Path(model_dir)
    if not prefer:
        raise ValueError("prefer 不能为空")
    bad = [p for p in prefer if p not in BACKENDS]
    if bad:
        raise ValueError(f"未知推理后端 {bad}，可选 {BACKENDS}")
    if device not in DEVICES:
        raise ValueError(f"device={device!r} 无效，可选 {DEVICES}")
    resolve_opt_batch(max_batch, opt_batch)  # 参数错误在这里就抛，不进入降级循环
    if not model_dir.is_dir():
        raise ClassifierUnavailable(f"模型目录不存在: {model_dir}")

    reasons: list[str] = []
    for name in prefer:
        clf: Classifier
        if name == "trt":
            clf = TrtClassifier(max_batch=max_batch, opt_batch=opt_batch)
        elif name == "onnx":
            clf = OnnxClassifier(max_batch=max_batch, opt_batch=opt_batch)
        else:
            clf = TorchClassifier(device=device, max_batch=max_batch, opt_batch=opt_batch)
        try:
            clf.load(model_dir)
        except Exception as exc:  # 缺文件/缺依赖/反序列化失败都降级到下一个
            reason = f"{name}: {type(exc).__name__}: {exc}"
            reasons.append(reason)
            log.warning("推理后端不可用，尝试下一个", extra={"backend": name, "err": str(exc)})
            continue
        log.info("使用推理后端", extra={"backend": clf.backend, "model_dir": str(model_dir)})
        return clf
    raise ClassifierUnavailable(
        f"模型目录 {model_dir} 没有可用的推理后端（尝试顺序 {list(prefer)}）: " + "; ".join(reasons)
    )
