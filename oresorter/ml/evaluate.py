"""``cli eval``：对“每个子目录一类”的样本目录做 k 折交叉验证，打印混淆矩阵并保存 ``eval_report.json``。

直接支持客户现有的 ``a/ b/ c/`` 目录（bmp/png/jpg，无 alpha 的图片按 ``--bg-color`` 自动抠 mask）。
样本总数 < 50 时自动改为留一法。训练/数据集逻辑复用 ``oresorter.ml.dataset`` 与 ``oresorter.ml.train``。
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oresorter.fsutil import atomic_write_json
from oresorter.log import get_logger

log = get_logger(__name__)

LOO_THRESHOLD = 50  # 样本总数低于此值用留一法
DEFAULT_FOLDS = 5
MIN_FOLDS = 2
MIN_IMAGE_SIZE = 8
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "dev.yaml"


def parse_bg_color(text: str) -> tuple[int, int, int]:
    """``"r,g,b"`` → ``(r, g, b)``，每个分量 0~255；格式/范围错误抛 ValueError 并说明原因。"""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"bg_color 应为 'r,g,b' 三个整数，得到 {text!r}")
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"bg_color 分量必须是整数，得到 {text!r}") from exc
    for v in vals:
        if not (0 <= v <= 255):
            raise ValueError(f"bg_color 分量 {v} 超出 0~255")
    return vals  # type: ignore[return-value]


def list_class_dirs(folder: Path) -> list[str]:
    """``folder`` 下的类别子目录名（跳过隐藏目录与空目录），按名字排序。"""
    out = []
    for p in sorted(folder.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir() if f.is_file()):
            out.append(p.name)
    return out


def choose_folds(n_total: int, requested: int) -> tuple[int, str]:
    """决定折数：总样本 < 50 → 留一法（k=n）；否则用请求值（需 2 ≤ k ≤ n）。返回 ``(k, "loo"|"kfold")``。"""
    if n_total < 2:
        raise ValueError(f"样本总数 {n_total} 不足以做交叉验证（至少 2）")
    if n_total < LOO_THRESHOLD:
        return n_total, "loo"
    if requested < 2:
        raise ValueError(f"--folds={requested} 无效，至少 2")
    if requested > n_total:
        raise ValueError(f"--folds={requested} 大于样本总数 {n_total}")
    return requested, "kfold"


def _config_to_dict(cfg: Any) -> Any:
    """TrainConfig → 可 JSON 化的字典（优先其自带 ``to_dict``）。"""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return dict(vars(cfg)) if hasattr(cfg, "__dict__") else str(cfg)


def parse_cli_options(args: Any) -> tuple[int, dict[str, Any]]:
    """解析并校验 ``run`` 的数值参数，返回 ``(folds, overrides)``。

    ``None`` 表示未给（折数用 :data:`DEFAULT_FOLDS`，其余沿用配置文件）；给了就按真实值校验：
    ``folds < 2``、``epochs < 1``、``image_size < 8`` 一律抛 ValueError 说明原因，不静默回退到默认值。
    """
    folds_arg = getattr(args, "folds", None)
    epochs_arg = getattr(args, "epochs", None)
    image_size_arg = getattr(args, "image_size", None)
    backbone_arg = getattr(args, "backbone", None)
    try:
        folds = DEFAULT_FOLDS if folds_arg is None else int(folds_arg)
        epochs = None if epochs_arg is None else int(epochs_arg)
        image_size = None if image_size_arg is None else int(image_size_arg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--folds/--epochs/--image-size 必须是整数: {exc}") from exc
    if folds < MIN_FOLDS:
        raise ValueError(f"--folds={folds} 无效，至少 {MIN_FOLDS}")
    overrides: dict[str, Any] = {}
    if image_size is not None:
        if image_size < MIN_IMAGE_SIZE:
            raise ValueError(f"--image-size={image_size} 无效，至少 {MIN_IMAGE_SIZE}")
        overrides["image_size"] = image_size
    if epochs is not None:
        if epochs < 1:
            raise ValueError(f"--epochs={epochs} 无效，至少 1")
        overrides["max_epochs"] = epochs
    if backbone_arg is not None and str(backbone_arg):
        overrides["backbone"] = str(backbone_arg)
    return folds, overrides


def run(args: Any) -> int:
    """``cli eval`` 入口。

    ``args``: ``folder``、``config``、``image_size``、``backbone``、``folds``、``epochs``、``out``、``bg_color``
    （数值项为 None 表示未给）。返回 0 成功；1 交叉验证失败；2 参数/目录问题；3 训练模块不可用。
    """
    folder = Path(args.folder)
    if not folder.is_dir():
        log.error("样本目录不存在", extra={"folder": str(folder)})
        return 2
    try:
        bg_color = parse_bg_color(getattr(args, "bg_color", None) or "0,0,255")
        folds, overrides = parse_cli_options(args)
    except ValueError as exc:
        log.error("参数错误", extra={"err": str(exc)})
        return 2
    overrides["bg_color"] = tuple(bg_color)

    try:
        from oresorter.ml.dataset import list_folder_samples
        from oresorter.ml.metrics import format_report
        from oresorter.ml.train import TrainConfig, TrainError, cross_validate
    except (ImportError, AttributeError) as exc:
        log.error("训练/数据集模块不可用，无法评估", extra={"err": str(exc)})
        return 3

    from oresorter.config import ConfigError, load

    config_path = Path(getattr(args, "config", None) or DEFAULT_CONFIG)
    try:
        cfg = load(config_path)
    except ConfigError as exc:
        log.error("配置加载失败", extra={"config": str(config_path), "err": exc.message})
        return 2

    classes = list_class_dirs(folder)
    if len(classes) < 2:
        log.error(
            "至少需要两个含图片的类别子目录", extra={"folder": str(folder), "classes": classes}
        )
        return 2

    classes, per_class = list_folder_samples(folder, classes)
    sample_counts = {c: len(per_class.get(c, [])) for c in classes}
    empty = [c for c, n in sample_counts.items() if n == 0]
    if empty:
        log.error("有类别没有可用图片", extra={"classes": empty})
        return 2
    items: list[tuple[Path, int]] = [
        (Path(p), idx) for idx, c in enumerate(classes) for p in per_class[c]
    ]
    n_total = len(items)
    try:
        k, method = choose_folds(n_total, folds)
    except ValueError as exc:
        log.error("折数无效", extra={"err": str(exc)})
        return 2

    try:
        train_cfg = TrainConfig.from_ml_config(cfg.ml, **overrides)
    except (TrainError, ValueError, TypeError) as exc:
        log.error("训练参数无效", extra={"overrides": overrides, "err": str(exc)})
        return 2

    log.info(
        "开始交叉验证",
        extra={"folder": str(folder), "classes": classes, "n": n_total, "k": k, "method": method},
    )

    def progress_cb(info: Any) -> None:
        log.info("评估进度", extra={"progress": info})

    t0 = time.monotonic()
    try:
        report = cross_validate(items, classes, train_cfg, k, progress_cb)
    except TrainError as exc:
        log.error("交叉验证失败", extra={"err": str(exc)})
        return 1
    duration = round(time.monotonic() - t0, 1)

    print(format_report(report))  # CLI 结果输出，允许 print
    out_path = Path(args.out) if getattr(args, "out", None) else folder / "eval_report.json"
    payload = {
        "folder": str(folder),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": classes,
        "sample_counts": sample_counts,
        "n_samples": n_total,
        "folds": k,
        "method": method,
        "bg_color": list(bg_color),
        "config": _config_to_dict(train_cfg),
        "duration_s": duration,
        "report": report,
        "fold_reports": report.get("folds") if isinstance(report, dict) else None,
    }
    atomic_write_json(out_path, payload)
    log.info("评估完成", extra={"report": str(out_path), "seconds": duration})
    return 0
