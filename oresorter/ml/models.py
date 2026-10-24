"""骨干网络构建、离线权重加载、checkpoint 读写、设备选择。"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from oresorter.log import get_logger

log = get_logger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHT_FILES: dict[str, str] = {
    "resnet18": "resnet18-f37072fd.pth",
    "mobilenet_v3_large": "mobilenet_v3_large-8738ca79.pth",
}
BACKBONES: tuple[str, ...] = tuple(WEIGHT_FILES)


def algo_name(backbone: str) -> str:
    """meta.json / HMI 显示用的算法名，如 ``cnn-resnet18``。"""
    return f"cnn-{backbone}"


def select_device(pref: str = "auto") -> torch.device:
    """auto: cuda > mps > cpu。"""
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    if pref == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        log.warning("MPS 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(pref)


def build_model(
    backbone: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    weights_dir: Path | None = None,
) -> nn.Module:
    """构建分类网络：ImageNet 预训练骨干（离线文件）+ 替换分类头。

    Args:
        backbone: ``resnet18`` | ``mobilenet_v3_large``。
        num_classes: 类别数。
        pretrained: 是否加载离线权重；文件缺失时告警并随机初始化。
        weights_dir: 权重目录，默认 ``oresorter/ml/weights``。
    """
    import torchvision

    if backbone not in WEIGHT_FILES:
        raise ValueError(f"未知骨干 {backbone!r}，可选 {BACKBONES}")
    if num_classes < 1:
        raise ValueError("num_classes 必须 ≥ 1")
    wdir = weights_dir or WEIGHTS_DIR
    if backbone == "resnet18":
        model: nn.Module = torchvision.models.resnet18(weights=None)
    else:
        model = torchvision.models.mobilenet_v3_large(weights=None)

    if pretrained:
        wpath = wdir / WEIGHT_FILES[backbone]
        if wpath.exists():
            state = torch.load(wpath, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            log.info("已加载离线预训练权重", extra={"backbone": backbone, "path": str(wpath)})
        else:
            log.warning(
                "预训练权重缺失，使用随机初始化", extra={"backbone": backbone, "path": str(wpath)}
            )

    if backbone == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def head_module(model: nn.Module, backbone: str) -> nn.Module:
    """分类头（先只训头、再全网微调时用）。"""
    if backbone == "resnet18":
        return model.fc
    return model.classifier[-1]


def head_parameters(model: nn.Module, backbone: str) -> Iterator[nn.Parameter]:
    return head_module(model, backbone).parameters()


def freeze_backbone(model: nn.Module, backbone: str, frozen: bool = True) -> None:
    """冻结/解冻除分类头外的全部参数。"""
    head_ids = {id(p) for p in head_parameters(model, backbone)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(not frozen)


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict[str, Any]) -> None:
    """原子保存 ``{"state_dict", "meta"}``。``meta`` 至少含 backbone / classes / image_size。"""
    for k in ("backbone", "classes", "image_size"):
        if k not in meta:
            raise ValueError(f"checkpoint meta 缺少 {k}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": sd, "meta": dict(meta)}, tmp)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """读取 checkpoint，按 meta 重建网络（不加载预训练文件）并载入权重，返回 ``(model.eval(), meta)``。"""
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    meta = dict(ck["meta"])
    model = build_model(meta["backbone"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(torch.device(device) if isinstance(device, str) else device)
    model.eval()
    return model, meta


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
