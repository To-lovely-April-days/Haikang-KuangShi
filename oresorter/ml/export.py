"""模型导出：PyTorch → ONNX（动态 batch），ONNX → TensorRT 引擎（调用 ``trtexec``，仅 Jetson）。

产物都先写同目录临时文件再 ``os.replace``，不会留下半成品；引擎旁边写一份 ``model.json``
记录 TensorRT 版本与 ONNX 的 sha256，部署时用 :func:`engine_matches` 判断引擎是否仍然可用。
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from oresorter.fsutil import atomic_write_json, ensure_dir, read_json
from oresorter.log import get_logger

log = get_logger(__name__)

ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAME = "logits"
DEFAULT_OPSET = 17
MIN_OPSET, MAX_OPSET = 11, 20
JETSON_TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
VERSION_PROBE_TIMEOUT_S = 30
_TAIL_CHARS = 2000

# 引擎优化 profile 与运行时分批的默认值（infer.py 复用同一组，保证引擎与分类器口径一致）
DEFAULT_MAX_BATCH = 128
DEFAULT_OPT_BATCH = 32


class ExportError(Exception):
    """ONNX 导出或校验失败。"""


class EngineBuildError(Exception):
    """TensorRT 引擎生成失败（找不到 trtexec、超时、非零退出、产物缺失、意外异常）。"""


def resolve_opt_batch(max_batch: int, opt_batch: int | None) -> int:
    """解析最优 batch：``None`` → ``min(DEFAULT_OPT_BATCH, max_batch)``；显式给出则必须在 1~max_batch 内。

    ``max_batch`` 通常来自 ``PipelineConfig.infer_batch_max``（合法范围 1~512），小于 32 也是合法配置，
    因此默认值不能是固定的 32，否则会拒绝配置表允许的值。

    Raises:
        ValueError: ``max_batch < 1``，或显式 ``opt_batch`` 不在 ``1~max_batch`` 内。
    """
    if max_batch < 1:
        raise ValueError(f"max_batch 必须 ≥ 1，得到 {max_batch}")
    if opt_batch is None:
        return min(DEFAULT_OPT_BATCH, int(max_batch))
    if not (1 <= opt_batch <= max_batch):
        raise ValueError(f"opt_batch={opt_batch} 必须在 1~max_batch({max_batch}) 之间")
    return int(opt_batch)


# --------------------------------------------------------------------------------------
# ONNX
# --------------------------------------------------------------------------------------


def to_onnx(model: nn.Module, image_size: int, path: Path, *, opset: int = DEFAULT_OPSET) -> Path:
    """把分类网络导出为 ONNX：输入 ``input`` (N,3,S,S) float32、batch 维动态，输出 ``logits`` (N,C)。

    导出用模型的深拷贝（搬到 CPU、float32、eval），不改动调用方的模型。
    先写同目录临时文件，``onnx.checker`` 校验通过后再 ``os.replace`` 到 ``path``。

    Args:
        model: 已训练网络。
        image_size: 输入边长 S。
        path: 目标 ``model.onnx``。
        opset: ONNX opset（默认 17）。

    Returns:
        写好的 ``path``。

    Raises:
        ValueError: 参数越界。
        ExportError: 导出或校验失败。
    """
    import onnx

    if image_size < 8:
        raise ValueError(f"image_size 过小: {image_size}（至少 8）")
    if not (MIN_OPSET <= opset <= MAX_OPSET):
        raise ValueError(f"opset={opset} 超出支持范围 {MIN_OPSET}~{MAX_OPSET}")

    target = Path(path)
    ensure_dir(target.parent)
    export_model = copy.deepcopy(model).to("cpu").float().eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    t0 = time.monotonic()
    try:
        _export_legacy(export_model, dummy, tmp, opset)
        proto = onnx.load(str(tmp))
        onnx.checker.check_model(proto)
        _verify_graph_io(proto, image_size)
        os.replace(tmp, target)
    except (ValueError, ExportError):
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # torch/onnx 内部各种异常统一包装
        tmp.unlink(missing_ok=True)
        raise ExportError(f"ONNX 导出失败: {type(exc).__name__}: {exc}") from exc

    size = target.stat().st_size
    log.info(
        "ONNX 导出完成",
        extra={
            "path": str(target),
            "bytes": size,
            "image_size": image_size,
            "opset": opset,
            "seconds": round(time.monotonic() - t0, 2),
        },
    )
    return target


def _export_legacy(model: nn.Module, dummy: torch.Tensor, out: Path, opset: int) -> None:
    """调用 TorchScript 版（legacy）导出器。

    torch ≥ 2.9 的 ``torch.onnx.export`` 默认 ``dynamo=True``：它依赖 ``onnxscript``（本项目依赖里没有，
    Jetson 离线环境也不保证有），且导出慢得多。对 ResNet18 / MobileNetV3 这类纯前馈网络，
    legacy 导出器稳定、0.1 s 级完成、``dynamic_axes`` 直接给出动态 batch，所以显式 ``dynamo=False``。
    旧版 torch 没有 ``dynamo`` 参数时回退为不带该参数的调用（那时本就是 legacy 导出器）。
    """
    kwargs: dict[str, Any] = {
        "input_names": [ONNX_INPUT_NAME],
        "output_names": [ONNX_OUTPUT_NAME],
        "opset_version": opset,
        "dynamic_axes": {ONNX_INPUT_NAME: {0: "batch"}, ONNX_OUTPUT_NAME: {0: "batch"}},
        "do_constant_folding": True,
    }
    with warnings.catch_warnings():
        # legacy 导出器会发两条 DeprecationWarning（“legacy TorchScript-based export”），属已知且有意为之
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # 导出器内部自己切 no_grad / eval，这里不额外包 inference_mode（会干扰 TorchScript 追踪）
        try:
            torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kwargs)
        except TypeError as exc:
            if "dynamo" not in str(exc):
                raise
            torch.onnx.export(model, (dummy,), str(out), **kwargs)


def _verify_graph_io(proto: Any, image_size: int) -> None:
    """确认导出图的输入/输出名与形状符合契约（batch 动态、3×S×S）。"""
    inputs = {i.name: i for i in proto.graph.input}
    outputs = {o.name: o for o in proto.graph.output}
    if ONNX_INPUT_NAME not in inputs or ONNX_OUTPUT_NAME not in outputs:
        raise ExportError(
            f"ONNX 图的输入/输出名不符: inputs={list(inputs)} outputs={list(outputs)}"
        )
    dims = inputs[ONNX_INPUT_NAME].type.tensor_type.shape.dim
    shape = [d.dim_param or d.dim_value for d in dims]
    if len(shape) != 4 or not isinstance(shape[0], str) or shape[1:] != [3, image_size, image_size]:
        raise ExportError(
            f"ONNX 输入形状不符，期望 [batch,3,{image_size},{image_size}]，得到 {shape}"
        )
    odims = outputs[ONNX_OUTPUT_NAME].type.tensor_type.shape.dim
    oshape = [d.dim_param or d.dim_value for d in odims]
    if len(oshape) != 2 or not isinstance(oshape[0], str):
        raise ExportError(f"ONNX 输出形状不符，期望 [batch,C]，得到 {oshape}")


def onnx_input_size(onnx_path: Path) -> int | None:
    """从 ONNX 文件读出输入边长 S（读不到或非静态时返回 None）。"""
    import onnx

    try:
        proto = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as exc:
        log.warning("读取 ONNX 失败", extra={"path": str(onnx_path), "err": str(exc)})
        return None
    for i in proto.graph.input:
        if i.name == ONNX_INPUT_NAME:
            dims = i.type.tensor_type.shape.dim
            if len(dims) == 4 and dims[2].dim_value > 0:
                return int(dims[2].dim_value)
    return None


# --------------------------------------------------------------------------------------
# TensorRT（trtexec）
# --------------------------------------------------------------------------------------


def find_trtexec(explicit: str | None = None) -> Path | None:
    """定位 ``trtexec``：参数 > ``$TRTEXEC`` > ``/usr/src/tensorrt/bin/trtexec`` > ``PATH``。

    显式参数给了但无效时 **不会** 偷换成别的二进制，直接返回 None（并告警）。
    """
    if explicit:
        p = _resolve_executable(explicit)
        if p is None:
            log.warning("指定的 trtexec 不存在或不可执行", extra={"trtexec": explicit})
        return p
    env = os.environ.get("TRTEXEC")
    if env:
        p = _resolve_executable(env)
        if p is not None:
            return p
        log.warning("$TRTEXEC 指向的文件不存在或不可执行，继续查找默认位置", extra={"TRTEXEC": env})
    if JETSON_TRTEXEC.is_file() and os.access(JETSON_TRTEXEC, os.X_OK):
        return JETSON_TRTEXEC
    found = shutil.which("trtexec")
    return Path(found) if found else None


def _resolve_executable(name: str) -> Path | None:
    p = Path(name).expanduser()
    if p.is_file() and os.access(p, os.X_OK):
        return p.resolve()
    found = shutil.which(name)
    return Path(found) if found else None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """文件 sha256（分块读）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _tail(text: str, n: int = _TAIL_CHARS) -> str:
    text = text or ""
    return text if len(text) <= n else text[-n:]


def _parse_trt_version(text: str) -> str | None:
    """从 trtexec 输出里解析 TensorRT 版本：``TensorRT v100300`` → ``10.3.0``，``v8601`` → ``8.6.1``。"""
    m = re.search(r"TensorRT\s+v(\d+)", text or "")
    if not m:
        m = re.search(r"TensorRT\s+version[:\s]+(\d+\.\d+\.\d+(?:\.\d+)?)", text or "")
        return m.group(1) if m else None
    digits = m.group(1)
    if "." in digits:
        return digits
    if len(digits) == 6:  # 10.x：MMmmpp
        return f"{int(digits[:2])}.{int(digits[2:4])}.{int(digits[4:6])}"
    if len(digits) == 4:  # 8.x：Mmpp
        return f"{int(digits[0])}.{int(digits[1])}.{int(digits[2:4])}"
    return digits


def _normalize_version(ver: str | None) -> str | None:
    """版本字符串归一化为 ``major.minor.patch`` 三段（``10.3.0.26`` → ``10.3.0``），便于跨来源比较。

    ``tensorrt.__version__`` 常带第 4 段构建号，而 ``trtexec`` 输出头只有三段；引擎兼容性由前三段决定。
    """
    if not ver:
        return None
    m = re.match(r"\s*(\d+)\.(\d+)\.(\d+)", str(ver))
    if m:
        return ".".join(m.groups())
    return str(ver).strip() or None


def trtexec_version(trtexec: Path, timeout_s: int = VERSION_PROBE_TIMEOUT_S) -> str | None:
    """运行 ``trtexec --version`` 解析版本；失败返回 None（不抛）。输出按 UTF-8 解码，非法字节替换。"""
    try:
        proc = subprocess.run(
            [str(trtexec), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("探测 trtexec 版本失败", extra={"trtexec": str(trtexec), "err": str(exc)})
        return None
    return _parse_trt_version((proc.stdout or "") + "\n" + (proc.stderr or ""))


def current_trt_version(trtexec: str | None = None) -> str | None:
    """当前机器的 TensorRT 版本：优先 ``tensorrt.__version__``，否则 ``trtexec --version``；都没有返回 None。"""
    try:
        import tensorrt  # type: ignore[import-not-found]

        ver = getattr(tensorrt, "__version__", None)
        if ver:
            return str(ver)
    except Exception:
        pass
    exe = find_trtexec(trtexec)
    return trtexec_version(exe) if exe else None


def engine_meta_path(plan_path: Path) -> Path:
    """引擎旁的元信息文件路径（``model.plan`` → ``model.json``）。"""
    return Path(plan_path).with_suffix(".json")


def build_engine(
    onnx_path: Path,
    plan_path: Path,
    image_size: int,
    *,
    fp16: bool = True,
    timeout_s: int = 600,
    trtexec: str | None = None,
    max_batch: int = DEFAULT_MAX_BATCH,
    opt_batch: int | None = None,
) -> Path:
    """用 ``trtexec`` 把 ONNX 编译成 TensorRT 引擎（动态 batch 1~max_batch），原子落盘。

    Args:
        onnx_path: 输入 ``model.onnx``（输入名必须是 ``input``，即 :func:`to_onnx` 的产物）。
        plan_path: 输出 ``model.plan``。
        image_size: 输入边长 S。
        fp16: 是否开 ``--fp16``。
        timeout_s: trtexec 超时秒数（Orin Nano 上 ResNet18 约 30~90 s）。
        trtexec: 显式 trtexec 路径；None 则按 :func:`find_trtexec` 查找。
        max_batch: 优化 profile 的最大 batch（与 ``PipelineConfig.infer_batch_max`` 对齐）。
        opt_batch: 最优 batch；None 取 ``min(32, max_batch)``，显式给出必须在 1~max_batch 内。

    Returns:
        ``plan_path``。旁边同时写 ``plan_path.with_suffix(".json")``（``trt_version`` 与
        :func:`current_trt_version` 同一来源，``trtexec_version`` 为 trtexec 输出头解析值，仅供排查）。

    Raises:
        ValueError: 参数越界。
        EngineBuildError: 找不到 trtexec、超时、非零退出、产物缺失或调用过程中的其他异常；
            任何失败都不会留下临时文件。
    """
    onnx_path = Path(onnx_path)
    plan_path = Path(plan_path)
    if image_size < 8:
        raise ValueError(f"image_size 过小: {image_size}（至少 8）")
    opt_batch = resolve_opt_batch(max_batch, opt_batch)
    if timeout_s <= 0:
        raise ValueError(f"timeout_s 必须 > 0，得到 {timeout_s}")
    if not onnx_path.is_file():
        raise EngineBuildError(f"ONNX 文件不存在: {onnx_path}")

    exe = find_trtexec(trtexec)
    if exe is None:
        raise EngineBuildError(
            "未找到 trtexec：TensorRT 引擎只能在 Jetson 算力盒上生成"
            f"（JetPack 自带 {JETSON_TRTEXEC}，或设置环境变量 TRTEXEC）。"
            "开发机上请直接使用 ONNX / PyTorch 推理。"
        )

    ensure_dir(plan_path.parent)
    tmp = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    shape = f"3x{image_size}x{image_size}"
    cmd = [
        str(exe),
        f"--onnx={onnx_path}",
        f"--saveEngine={tmp}",
        f"--minShapes={ONNX_INPUT_NAME}:1x{shape}",
        f"--optShapes={ONNX_INPUT_NAME}:{opt_batch}x{shape}",
        f"--maxShapes={ONNX_INPUT_NAME}:{max_batch}x{shape}",
    ]
    if fp16:
        cmd.insert(3, "--fp16")

    log.info(
        "开始生成 TensorRT 引擎",
        extra={
            "trtexec": str(exe),
            "onnx": str(onnx_path),
            "plan": str(plan_path),
            "timeout_s": timeout_s,
        },
    )
    t0 = time.monotonic()
    try:
        proc = _run_trtexec(cmd, timeout_s)
        if proc.returncode != 0:
            raise EngineBuildError(
                f"trtexec 退出码 {proc.returncode}。stderr 尾部: {_tail(proc.stderr)}\n"
                f"stdout 尾部: {_tail(proc.stdout)}"
            )
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise EngineBuildError(
                f"trtexec 正常退出但未生成引擎文件。stdout 尾部: {_tail(proc.stdout)}"
            )
        os.replace(tmp, plan_path)
    except BaseException as exc:
        # 统一清理：超时、非零退出、IO 错误、解码等意外异常乃至 KeyboardInterrupt 都不留半成品
        tmp.unlink(missing_ok=True)
        if isinstance(exc, EngineBuildError) or not isinstance(exc, Exception):
            raise
        raise EngineBuildError(f"trtexec 调用过程异常: {type(exc).__name__}: {exc}") from exc
    seconds = round(time.monotonic() - t0, 1)

    # trt_version 必须与 engine_matches 里的 current_trt_version 同一来源，否则刚生成的引擎会被判为不匹配；
    # trtexec 输出头解析出的版本只作排查信息（本机查不到版本时才兜底写入 trt_version）。
    trtexec_reported = _parse_trt_version((proc.stdout or "") + "\n" + (proc.stderr or ""))
    version = current_trt_version(str(exe))
    if version is None:
        version = trtexec_reported
    meta = {
        "trt_version": version,
        "trtexec_version": trtexec_reported,
        "trtexec": str(exe),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "image_size": image_size,
        "fp16": fp16,
        "min_batch": 1,
        "opt_batch": opt_batch,
        "max_batch": max_batch,
        "onnx": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "plan_bytes": plan_path.stat().st_size,
        "build_seconds": seconds,
    }
    atomic_write_json(engine_meta_path(plan_path), meta)
    if not engine_matches(plan_path, onnx_path, trtexec=str(exe), image_size=image_size):
        log.warning(
            "刚生成的引擎未通过 engine_matches 自检（版本字符串来源不一致？），部署时可能被反复重建",
            extra={
                "plan": str(plan_path),
                "trt_version": version,
                "trtexec_version": trtexec_reported,
            },
        )
    log.info(
        "TensorRT 引擎生成完成",
        extra={
            "plan": str(plan_path),
            "bytes": meta["plan_bytes"],
            "trt_version": version,
            "seconds": seconds,
        },
    )
    return plan_path


def _run_trtexec(cmd: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    """运行 trtexec：带超时，输出按 UTF-8 解码并替换非法字节（trtexec 偶有非 UTF-8 输出）。

    超时 / 无法启动 → :class:`EngineBuildError`；调用方负责清理临时文件。
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise EngineBuildError(
            f"trtexec 超时（> {timeout_s} s），已终止。stderr 尾部: {_tail(_as_text(exc.stderr))}"
        ) from exc
    except OSError as exc:
        raise EngineBuildError(f"无法运行 trtexec {cmd[0]}: {exc}") from exc


def _as_text(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def engine_matches(
    plan_path: Path,
    onnx_path: Path,
    *,
    trtexec: str | None = None,
    image_size: int | None = None,
) -> bool:
    """判断已有引擎是否仍可用：文件齐全、由同一份 ONNX 生成、TensorRT 版本与当前一致。

    版本比较前两边都归一化为 ``major.minor.patch``（见 :func:`_normalize_version`）。
    任一信息缺失（没有元信息文件、本机查不到 TensorRT 版本）都按 **不匹配** 处理，由上层重新生成或降级。
    """
    plan_path = Path(plan_path)
    onnx_path = Path(onnx_path)
    if not plan_path.is_file() or not onnx_path.is_file():
        return False
    try:
        meta = read_json(engine_meta_path(plan_path), default=None)
    except ValueError:
        log.warning("引擎元信息损坏", extra={"path": str(engine_meta_path(plan_path))})
        return False
    if not isinstance(meta, dict):
        return False
    if meta.get("onnx_sha256") != sha256_file(onnx_path):
        log.info("引擎与当前 ONNX 不一致", extra={"plan": str(plan_path)})
        return False
    if image_size is not None and meta.get("image_size") != image_size:
        return False
    recorded = meta.get("trt_version")
    current = current_trt_version(trtexec)
    if (
        not recorded
        or not current
        or _normalize_version(str(recorded)) != _normalize_version(current)
    ):
        log.info("引擎 TensorRT 版本与当前不一致", extra={"recorded": recorded, "current": current})
        return False
    return True


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_engine_cli(args: Any) -> int:
    """``cli build-engine``：由 ``{model_dir}/train/model.onnx`` 生成 ``model.plan``。

    ``args`` 需有 ``model_dir``、``fp16``、``timeout``（可选 ``trtexec``、``max_batch``、``opt_batch``；
    缺省或为 None 时分别取 128 / ``min(32, max_batch)``）。
    image_size 从 ``train/best.pt`` 的 meta 或 ``meta.json`` 读取，再与 ONNX 文件交叉核对。
    返回 0 成功，1 生成失败（含参数越界），2 参数/文件问题。
    """
    from oresorter.ml.infer import ClassifierUnavailable, read_model_meta

    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "train" / "model.onnx"
    plan_path = model_dir / "train" / "model.plan"
    if not onnx_path.is_file():
        log.error("缺少 model.onnx，请先训练", extra={"path": str(onnx_path)})
        return 2
    try:
        meta = read_model_meta(model_dir)
    except ClassifierUnavailable as exc:
        log.error("读取模型元信息失败", extra={"model_dir": str(model_dir), "err": str(exc)})
        return 2
    image_size = int(meta["image_size"])
    onnx_size = onnx_input_size(onnx_path)
    if onnx_size is not None and onnx_size != image_size:
        log.error(
            "ONNX 输入尺寸与模型元信息不一致",
            extra={"onnx": onnx_size, "meta": image_size},
        )
        return 2
    timeout_arg = getattr(args, "timeout", None)
    max_batch_arg = getattr(args, "max_batch", None)
    opt_batch_arg = getattr(args, "opt_batch", None)
    try:
        build_engine(
            onnx_path,
            plan_path,
            image_size,
            fp16=bool(getattr(args, "fp16", True)),
            timeout_s=600 if timeout_arg is None else int(timeout_arg),
            trtexec=getattr(args, "trtexec", None),
            max_batch=DEFAULT_MAX_BATCH if max_batch_arg is None else int(max_batch_arg),
            opt_batch=None if opt_batch_arg is None else int(opt_batch_arg),
        )
    except (EngineBuildError, ValueError) as exc:
        log.error("引擎生成失败", extra={"err": str(exc)})
        return 1
    log.info("引擎已生成", extra={"plan": str(plan_path)})
    return 0
