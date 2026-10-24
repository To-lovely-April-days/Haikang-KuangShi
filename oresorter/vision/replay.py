"""``cli replay``：回放大图目录走一遍 分割 →（可选）分类，输出标注图与 ``replay_report.json``。

用途：无硬件时用合成大图（``scripts/make_synthetic.py``）或现场保存的整帧，验证分割参数、跨块拼接与
训练好的模型（docs/02 §1 P1 验收：``cli replay`` 跑通 分割→分类）。

两种回放方式：

- **整图**（默认）：每张大图直接 ``segment.segment``；每张图独立处理、不跨图拼接，顶到图片上下边界的颗粒
  按完整颗粒计数（报告条目带 ``touches_top`` / ``touches_bottom``，``totals.edge`` 给出边界颗粒数，便于过滤）；
- **按块**（``--chunk-lines N``，128~4096）：把每张大图按 N 行切成 ``Frame`` 序列，经
  ``ChunkStitcher(cfg.pipeline.overlap_lines)`` push / select，与运行时流水线同一条路径；每张图末尾再
  推入一帧纯背景“冲洗帧”，让被延后的触底颗粒得到处理（有限长度的皮带图没有“下一块”；因此
  ``stitch.frames`` 比实际块数多 1）。块内坐标统一换算回大图坐标，因此两种方式的结果可以直接比较。
  ``overlap_lines=0`` 时跨块颗粒会被切成两半且不标记截断，回放时给出 warning。

背景：``--learn-bg N > 0`` 用前 N 张图学习 ``ColumnBackground``（N 大于图片数按越界拒绝）；否则用
``ColorBackground(cfg.capture.bg_color)``。报告 ``config.bg_info`` 记录 ``frames_used`` / ``mean_rgb``。

产物（``--out``，默认与图片目录同级的 ``replay_out/``；不允许是图片目录本身或其子目录，否则标注图会被下一次
回放当作输入；文件名以 ``_overlay.jpg`` 结尾的图片会被跳过并告警）：

- ``{stem}_overlay.jpg``：``segment.draw_overlay`` 画出 bbox 与编号，有模型时附 ``类别 置信度``；
  目录里同名不同扩展名的图片（``a.png`` / ``a.jpg``）改用 ``{stem}_{ext}_overlay.jpg`` 以免互相覆盖；
- ``replay_report.json``：每张图的颗粒列表（bbox / area / cls / conf / trailing_line / touches_*）、每类数量、
  平均面积、耗时分位数、配置摘要；图片目录含 ``labels.json``（``synthetic.make_frames`` 产物）时，另给出
  IoU ≥ 0.5 匹配的召回 / 误检数，有模型时再给匹配颗粒上的分类准确率；``labels.json`` 中缺键或值不是列表的
  图片不参与统计并告警，``totals.labels.images_without_labels`` 计数。

参数错误（目录不存在、``chunk_lines`` 越界、模型目录无产物、配置加载失败、输出目录落在图片目录内）记日志
error 并返回 1，不抛堆栈。单张图回放中的任何异常（分割 / 分类 / 标注写入，包括 onnxruntime / torch /
cv2 的运行时错误）只记录为该图的 error 条目（带堆栈日志），其余图片继续，报告照常写出，最终返回 1。
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from oresorter.camera.base import Frame
from oresorter.config import AppConfig, CaptureSettings, ConfigError
from oresorter.config import load as load_config
from oresorter.fsutil import atomic_write_bytes, atomic_write_json, ensure_dir, read_json
from oresorter.log import get_logger
from oresorter.vision import preprocess
from oresorter.vision import segment as seg
from oresorter.vision.background import Background, ColorBackground, ColumnBackground
from oresorter.vision.segment import Particle
from oresorter.vision.stitch import ChunkStitcher

log = get_logger(__name__)

IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
LABELS_FILE = "labels.json"
REPORT_FILE = "replay_report.json"
REPORT_VERSION = 2
OVERLAY_SUFFIX = "_overlay.jpg"
"""标注图文件名后缀；输入目录里以此结尾的文件视为上次回放的产物而跳过。"""
CHUNK_LINES_MIN, CHUNK_LINES_MAX = 128, 4096
"""``--chunk-lines`` 允许范围（相机 ``runtime_chunk_lines`` 下限 128；上限为传感器最大行数）。"""
IOU_MATCH = 0.5
"""分割结果与标注 bbox 的匹配阈值。"""
MIN_FLUSH_ROWS = 16
"""按块回放时末尾冲洗帧的最少行数（须大于 ``dilate_px``，使触底颗粒在冲洗帧中不再触底）。"""
OVERLAY_JPEG_QUALITY = 85
CLASSIFIER_PREFER: tuple[str, ...] = ("onnx", "torch")
REPLAY_CAMERA_ID = 1
MAX_NAMES_IN_LOG = 5
"""告警日志里最多列出的文件名个数。"""


class ReplayError(Exception):
    """参数 / 环境错误：由 :func:`run` 转成日志 error + 返回码 1。"""


@dataclass(frozen=True)
class ReplayOptions:
    """``run`` 解析后的参数。"""

    images: Path
    config: Path
    model_dir: Path | None
    chunk_lines: int | None
    out: Path
    learn_bg: int


# --------------------------------------------------------------------------------------
# 参数与输入
# --------------------------------------------------------------------------------------


def _is_within(path: Path, folder: Path) -> bool:
    """``path`` 是否就是 ``folder`` 或位于其内（按解析后的绝对路径比较，二者都可以不存在）。"""
    p, f = path.resolve(), folder.resolve()
    return p == f or f in p.parents


def default_out_dir(images: Path) -> Path:
    """默认输出目录：图片目录同级的 ``replay_out/``；若图片目录本身就叫 ``replay_out`` 则加上目录名前缀。"""
    parent = images.resolve().parent
    out = parent / "replay_out"
    if _is_within(out, images):
        out = parent / f"{images.resolve().name}_replay_out"
    return out


def parse_options(args: argparse.Namespace) -> ReplayOptions:
    """校验并整理 CLI 参数（``images`` 必填；其余缺省时取默认值）。

    Raises:
        ReplayError: 图片目录不存在、``chunk_lines`` 不在 128~4096、``learn_bg`` 为负、模型目录不存在、
            输出目录是图片目录本身或其子目录。
    """
    images_arg = getattr(args, "images", None)
    if not images_arg:
        raise ReplayError("必须指定 --images 大图目录")
    images = Path(images_arg)
    if not images.is_dir():
        raise ReplayError(f"大图目录不存在: {images}")

    config_arg = getattr(args, "config", None)
    if not config_arg:
        raise ReplayError("必须指定 --config 配置文件")
    config = Path(config_arg)

    chunk_arg = getattr(args, "chunk_lines", None)
    chunk_lines: int | None = None
    if chunk_arg is not None:
        try:
            chunk_lines = int(chunk_arg)
        except (TypeError, ValueError) as exc:
            raise ReplayError(f"chunk_lines 必须是整数，得到 {chunk_arg!r}") from exc
        if not (CHUNK_LINES_MIN <= chunk_lines <= CHUNK_LINES_MAX):
            raise ReplayError(
                f"chunk_lines={chunk_lines} 超出范围 {CHUNK_LINES_MIN}~{CHUNK_LINES_MAX}"
            )

    learn_arg = getattr(args, "learn_bg", 0)
    try:
        learn_bg = int(learn_arg or 0)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"learn_bg 必须是整数，得到 {learn_arg!r}") from exc
    if learn_bg < 0:
        raise ReplayError(f"learn_bg 不能为负数，得到 {learn_bg}")

    model_arg = getattr(args, "model_dir", None)
    model_dir: Path | None = None
    if model_arg:
        model_dir = Path(model_arg)
        if not model_dir.is_dir():
            raise ReplayError(f"模型目录不存在: {model_dir}")

    out_arg = getattr(args, "out", None)
    out = Path(out_arg) if out_arg else default_out_dir(images)
    if _is_within(out, images):
        raise ReplayError(
            f"输出目录不能是图片目录本身或其子目录（标注图会被下一次回放当作输入）: out={out} images={images}"
        )
    return ReplayOptions(
        images=images,
        config=config,
        model_dir=model_dir,
        chunk_lines=chunk_lines,
        out=out,
        learn_bg=learn_bg,
    )


def list_images(folder: Path) -> list[Path]:
    """目录下的图片（png/jpg/jpeg/bmp，跳过隐藏文件），按文件名排序。

    以 ``_overlay.jpg`` 结尾的文件视为上次回放写进来的标注图，跳过并告警。
    """
    files: list[Path] = []
    overlays: list[str] = []
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.name.lower().endswith(OVERLAY_SUFFIX):
            overlays.append(p.name)
            continue
        files.append(p)
    if overlays:
        log.warning(
            "图片目录中有上次回放的标注图，已跳过",
            extra={"skipped": len(overlays), "names": overlays[:MAX_NAMES_IN_LOG]},
        )
    return files


def overlay_names(files: Sequence[Path]) -> dict[Path, str]:
    """每张图的标注图文件名：``{stem}_overlay.jpg``。

    同名不同扩展名的图片（``a.png`` / ``a.jpg``）改用 ``{stem}_{ext}_overlay.jpg``，仍重复时再加序号，
    保证每张图的标注图文件名唯一，并告警。
    """
    stems = Counter(p.stem for p in files)
    dup_stems = sorted(s for s, n in stems.items() if n > 1)
    names: dict[Path, str] = {}
    for p in files:
        if p.stem in dup_stems:
            names[p] = f"{p.stem}_{p.suffix.lstrip('.').lower()}{OVERLAY_SUFFIX}"
        else:
            names[p] = f"{p.stem}{OVERLAY_SUFFIX}"
    taken = Counter(names.values())
    seen: Counter[str] = Counter()
    for p, name in list(names.items()):
        if taken[name] > 1:
            seen[name] += 1
            names[p] = f"{name[: -len(OVERLAY_SUFFIX)]}_{seen[name]}{OVERLAY_SUFFIX}"
    if dup_stems:
        log.warning(
            "图片目录中有同名不同扩展名的图片，标注图文件名加入扩展名以免覆盖",
            extra={"stems": dup_stems[:MAX_NAMES_IN_LOG], "n": len(dup_stems)},
        )
    return names


def read_rgb(path: Path) -> np.ndarray | None:
    """读一张图为 ``(h, w, 3) uint8`` RGB；无法解码返回 None。"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_labels(folder: Path) -> dict[str, list[dict[str, Any]]] | None:
    """读取 ``labels.json``（``{"0000.png": [{"cls": "A", "bbox": [x, y, w, h]}, ...]}``）。

    文件不存在返回 None；损坏或顶层不是对象时告警并返回 None。某张图的值不是列表时告警并**不收录该图**
    （回放时按“无标注”处理，不参与召回统计）；单条标注缺少合法 bbox 时跳过该条并告警。
    """
    path = Path(folder) / LABELS_FILE
    if not path.is_file():
        return None
    try:
        raw = read_json(path)
    except (ValueError, OSError) as exc:
        log.warning("labels.json 无法解析，忽略标注", extra={"path": str(path), "err": str(exc)})
        return None
    if not isinstance(raw, dict):
        log.warning("labels.json 顶层不是对象，忽略标注", extra={"path": str(path)})
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    bad_values: list[str] = []
    for name, items in raw.items():
        if not isinstance(items, list):
            bad_values.append(str(name))
            continue
        entries: list[dict[str, Any]] = []
        for item in items:
            bbox = item.get("bbox") if isinstance(item, dict) else None
            if (
                not isinstance(bbox, (list, tuple))
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
            ):
                skipped += 1
                continue
            cls = item.get("cls")
            entries.append(
                {"cls": str(cls) if cls is not None else None, "bbox": [int(v) for v in bbox]}
            )
        out[str(name)] = entries
    if bad_values:
        log.warning(
            "labels.json 中有图片的标注不是列表，这些图按无标注处理",
            extra={"n": len(bad_values), "names": bad_values[:MAX_NAMES_IN_LOG]},
        )
    if skipped:
        log.warning("labels.json 中有标注缺少合法 bbox，已跳过", extra={"skipped": skipped})
    return out


# --------------------------------------------------------------------------------------
# 分割：整图 / 按块
# --------------------------------------------------------------------------------------


def split_frames(
    image: np.ndarray, chunk_lines: int, *, camera_id: int = REPLAY_CAMERA_ID
) -> Iterator[Frame]:
    """把一张大图按 ``chunk_lines`` 行切成 ``Frame`` 序列（``line_start`` 即大图行号）。"""
    if chunk_lines < 1:
        raise ValueError(f"chunk_lines 必须 ≥ 1，得到 {chunk_lines}")
    h = int(image.shape[0])
    for seq, y0 in enumerate(range(0, h, chunk_lines)):
        yield Frame(
            seq=seq,
            line_start=y0,
            image=image[y0 : y0 + chunk_lines],
            ts=time.monotonic(),
            camera_id=camera_id,
        )


def _to_image_coords(p: Particle, y_offset: int, img_h: int) -> Particle:
    """处理图坐标 → 大图坐标（y 加 ``y_offset``），把冲洗帧带来的越界行裁回图内，
    并按大图边界重算 ``touches_top`` / ``touches_bottom``。"""
    x, y, w, h = p.bbox
    y_full = y + y_offset
    h_full = max(1, min(h, img_h - y_full))
    trailing = y_full + h_full - 1
    return replace(
        p,
        bbox=(x, y_full, w, h_full),
        centroid_y=p.centroid_y + y_offset,
        trailing_y=trailing,
        trailing_line=trailing,
        touches_top=(y_full <= 0),
        touches_bottom=(y_full + h_full >= img_h),
    )


def segment_whole(image: np.ndarray, bg: Background, settings: CaptureSettings) -> list[Particle]:
    """整图分割；大图即整条皮带，``trailing_line`` 直接取图内尾沿行。"""
    particles = seg.segment(image, bg, settings)
    for p in particles:
        p.trailing_line = int(p.trailing_y)
    return particles


def segment_chunked(
    image: np.ndarray,
    bg: Background,
    settings: CaptureSettings,
    *,
    chunk_lines: int,
    overlap: int,
) -> tuple[list[Particle], dict[str, Any]]:
    """按块分割：切块 → ``ChunkStitcher.process`` → 坐标换算回大图 → 末尾冲洗帧。

    Returns:
        ``(大图坐标系下的颗粒列表（按尾沿行、x 升序）, stitcher.stats())``；``stats["frames"]`` 含冲洗帧。
    """
    img_h, img_w = int(image.shape[0]), int(image.shape[1])
    stitcher = ChunkStitcher(overlap)

    def segment_fn(proc: np.ndarray) -> list[Particle]:
        return seg.segment(proc, bg, settings)

    out: list[Particle] = []
    last_seq = -1
    for frame in split_frames(image, chunk_lines):
        selected = stitcher.process(frame, segment_fn)
        out.extend(_to_image_coords(p, stitcher.y_offset, img_h) for p in selected)
        last_seq = frame.seq

    # 冲洗帧：纯背景行，使最后一块里触底而被延后的颗粒得到处理（有限皮带图没有“下一块”）
    flush_rows = max(MIN_FLUSH_ROWS, 2 * int(settings.dilate_px) + 2)
    flush = np.ascontiguousarray(
        np.broadcast_to(bg.row_rgb(0, img_w)[None, :, :], (flush_rows, img_w, 3))
    )
    flush_frame = Frame(
        seq=last_seq + 1,
        line_start=img_h,
        image=flush,
        ts=time.monotonic(),
        camera_id=REPLAY_CAMERA_ID,
    )
    selected = stitcher.process(flush_frame, segment_fn)
    out.extend(_to_image_coords(p, stitcher.y_offset, img_h) for p in selected)

    out.sort(key=lambda p: (p.trailing_y, p.bbox[0]))
    return out, stitcher.stats()


# --------------------------------------------------------------------------------------
# 分类与标注匹配
# --------------------------------------------------------------------------------------


def classify_particles(particles: Sequence[Particle], classifier: Any) -> None:
    """用分类器给每颗颗粒打 ``extra["cls_name"]`` / ``extra["conf"]``，并置 ``extra["cls"]`` 为 ``"A 0.97"`` 供 ``draw_overlay`` 显示。"""
    if not particles:
        return
    x = preprocess.to_tensor_batch([p.crop_rgba for p in particles], int(classifier.image_size))
    probs, classes = classifier.predict(x)
    idx = np.argmax(probs, axis=1)
    for i, p in enumerate(particles):
        k = int(idx[i])
        conf = float(probs[i, k])
        p.extra["cls_name"] = str(classes[k])
        p.extra["conf"] = conf
        p.extra["cls"] = f"{classes[k]} {conf:.2f}"


def iou_xywh(a: Sequence[int], b: Sequence[int]) -> float:
    """两个 ``(x, y, w, h)`` 框的 IoU。"""
    ax, ay, aw, ah = (int(v) for v in a)
    bx, by, bw, bh = (int(v) for v in b)
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_labels(
    detections: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    *,
    iou_min: float = IOU_MATCH,
) -> dict[str, Any]:
    """分割结果与标注做一对一贪心匹配（IoU 降序，≥ ``iou_min``），给出召回 / 误检 / 分类准确率。

    Returns:
        ``{"n_labels", "n_detections", "matched", "recall", "false_positives", "cls_evaluated",
        "cls_correct", "cls_accuracy"}``；没有标注时 ``recall`` 为 None，没有带类别的匹配颗粒时
        ``cls_accuracy`` 为 None。
    """
    pairs: list[tuple[float, int, int]] = []
    for li, lab in enumerate(labels):
        for di, det in enumerate(detections):
            v = iou_xywh(lab["bbox"], det["bbox"])
            if v >= iou_min:
                pairs.append((v, li, di))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_l: set[int] = set()
    used_d: set[int] = set()
    matched = cls_eval = cls_correct = 0
    for _, li, di in pairs:
        if li in used_l or di in used_d:
            continue
        used_l.add(li)
        used_d.add(di)
        matched += 1
        det_cls = detections[di].get("cls")
        if det_cls is not None:
            cls_eval += 1
            if det_cls == labels[li].get("cls"):
                cls_correct += 1
    n_l, n_d = len(labels), len(detections)
    return {
        "n_labels": n_l,
        "n_detections": n_d,
        "matched": matched,
        "recall": round(matched / n_l, 4) if n_l else None,
        "false_positives": n_d - matched,
        "cls_evaluated": cls_eval,
        "cls_correct": cls_correct,
        "cls_accuracy": round(cls_correct / cls_eval, 4) if cls_eval else None,
    }


def _sum_label_stats(
    items: Iterable[dict[str, Any]], *, images_without_labels: int
) -> dict[str, Any]:
    """把各图的 ``match_labels`` 结果累加成总计，并附带有 / 无标注的图片数。"""
    total = {
        "n_labels": 0,
        "n_detections": 0,
        "matched": 0,
        "false_positives": 0,
        "cls_evaluated": 0,
        "cls_correct": 0,
    }
    n_images = 0
    for it in items:
        n_images += 1
        for k in total:
            total[k] += int(it.get(k, 0))
    n_l, n_e = total["n_labels"], total["cls_evaluated"]
    return {
        **total,
        "recall": round(total["matched"] / n_l, 4) if n_l else None,
        "cls_accuracy": round(total["cls_correct"] / n_e, 4) if n_e else None,
        "images_with_labels": n_images,
        "images_without_labels": int(images_without_labels),
    }


# --------------------------------------------------------------------------------------
# 单张回放与报告
# --------------------------------------------------------------------------------------


def particle_record(p: Particle) -> dict[str, Any]:
    """颗粒 → 报告条目：``to_dict()`` + ``cls`` / ``conf`` / ``truncated`` / ``touches_top`` / ``touches_bottom``。"""
    d = p.to_dict()
    d["cls"] = p.extra.get("cls_name")
    conf = p.extra.get("conf")
    d["conf"] = round(float(conf), 4) if conf is not None else None
    d["truncated"] = bool(p.truncated)
    d["touches_top"] = bool(p.touches_top)
    d["touches_bottom"] = bool(p.touches_bottom)
    return d


def _per_class(particles: Sequence[Particle]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in particles:
        name = p.extra.get("cls_name")
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def replay_image(
    image: np.ndarray,
    *,
    name: str,
    bg: Background,
    settings: CaptureSettings,
    overlap: int,
    chunk_lines: int | None,
    classifier: Any | None,
    labels: Sequence[dict[str, Any]] | None,
    overlay_path: Path | None,
) -> dict[str, Any]:
    """回放一张大图：分割 →（有模型时）分类 → 写标注图 →（有标注时）匹配，返回该图的报告条目。

    任何异常（分割层 ValueError、分类器 / onnxruntime / torch 的 RuntimeError、``cv2.error``、标注图
    写入 OSError）原样抛出，由 :func:`run` 记录为该图的 error 条目。
    """
    t0 = time.perf_counter()
    stitch_stats: dict[str, Any] | None = None
    if chunk_lines is None:
        particles = segment_whole(image, bg, settings)
    else:
        particles, stitch_stats = segment_chunked(
            image, bg, settings, chunk_lines=chunk_lines, overlap=overlap
        )
    t1 = time.perf_counter()
    if classifier is not None:
        classify_particles(particles, classifier)
    t2 = time.perf_counter()

    if overlay_path is not None:
        overlay = seg.draw_overlay(image, particles)
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), OVERLAY_JPEG_QUALITY],
        )
        if not ok:
            raise OSError(f"标注图 JPEG 编码失败: {overlay_path}")
        atomic_write_bytes(overlay_path, buf.tobytes(), fsync=False)

    records = [particle_record(p) for p in particles]
    stats = seg.particle_stats(particles)
    rec: dict[str, Any] = {
        "name": name,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "count": len(particles),
        "per_class": _per_class(particles),
        "area_mean": stats["area_mean"],
        "area_min": stats["area_min"],
        "area_max": stats["area_max"],
        "truncated": sum(1 for p in particles if p.truncated),
        "edge": sum(1 for p in particles if p.touches_top or p.touches_bottom),
        "segment_ms": round((t1 - t0) * 1000, 2),
        "infer_ms": round((t2 - t1) * 1000, 2),
        "ms": round((t2 - t0) * 1000, 2),
        "overlay": overlay_path.name if overlay_path is not None else None,
        "stitch": stitch_stats,
        "labels": match_labels(records, labels) if labels is not None else None,
        "particles": records,
    }
    return rec


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, np.float64), q)), 2)


def build_report(
    records: Sequence[dict[str, Any]],
    *,
    opts: ReplayOptions,
    cfg: AppConfig,
    bg_kind: str,
    bg_info: dict[str, Any],
    classifier: Any | None,
    errors: int,
    labels_present: bool,
    images_without_labels: int,
) -> dict[str, Any]:
    """汇总各图条目为 ``replay_report.json`` 的内容。

    ``labels_present`` 为 False（目录没有可用的 ``labels.json``）时 ``totals.labels`` 为 None。
    """
    ok = [r for r in records if "error" not in r]
    durations = [float(r["ms"]) for r in ok]
    per_class: dict[str, int] = {}
    area_sum = 0.0
    n_particles = 0
    for r in ok:
        for k, v in r["per_class"].items():
            per_class[k] = per_class.get(k, 0) + int(v)
        n_particles += int(r["count"])
        area_sum += float(r["area_mean"]) * int(r["count"])
    labels_total: dict[str, Any] | None = None
    if labels_present:
        labels_total = _sum_label_stats(
            (r["labels"] for r in ok if r.get("labels") is not None),
            images_without_labels=images_without_labels,
        )
    return {
        "version": REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "images_dir": str(opts.images),
        "out_dir": str(opts.out),
        "config": {
            "path": str(opts.config),
            "capture": cfg.capture.model_dump(mode="json"),
            "overlap_lines": int(cfg.pipeline.overlap_lines),
            "chunk_lines": opts.chunk_lines,
            "mode": "chunked" if opts.chunk_lines is not None else "whole",
            "bg": bg_kind,
            "bg_info": dict(bg_info),
            "learn_bg": opts.learn_bg,
            "model_dir": str(opts.model_dir) if opts.model_dir is not None else None,
            "backend": classifier.backend if classifier is not None else None,
            "classes": classifier.classes if classifier is not None else None,
            "image_size": int(classifier.image_size) if classifier is not None else None,
        },
        "totals": {
            "images": len(records),
            "images_ok": len(ok),
            "images_failed": errors,
            "particles": n_particles,
            "per_class": dict(sorted(per_class.items())),
            "area_mean": round(area_sum / n_particles, 1) if n_particles else 0.0,
            "truncated": sum(int(r["truncated"]) for r in ok),
            "edge": sum(int(r["edge"]) for r in ok),
            "labels": labels_total,
        },
        "timing_ms": {
            "p50": _percentile(durations, 50),
            "p99": _percentile(durations, 99),
            "mean": round(float(np.mean(durations)), 2) if durations else 0.0,
            "max": round(max(durations), 2) if durations else 0.0,
            "total": round(sum(durations), 2),
        },
        "images": list(records),
    }


def format_summary(report: dict[str, Any]) -> str:
    """控制台简表（CLI 结果输出）。"""
    cols = f"{'image':<24}{'size':>11}{'n':>5}  {'per_class':<24}{'area':>9}{'ms':>9}{'recall':>8}{'fp':>5}{'cls_acc':>9}"
    lines = [cols, "-" * len(cols)]
    for r in report["images"]:
        if "error" in r:
            lines.append(f"{r['name']:<24}  错误: {r['error']}")
            continue
        pc = " ".join(f"{k}:{v}" for k, v in r["per_class"].items()) or "-"
        lab = r.get("labels") or {}
        recall = lab.get("recall")
        acc = lab.get("cls_accuracy")
        lines.append(
            f"{r['name']:<24}{r['width']}x{r['height']:<5}{r['count']:>5}  {pc:<24}"
            f"{r['area_mean']:>9.1f}{r['ms']:>9.1f}"
            f"{(f'{recall:.2f}' if recall is not None else '-'):>8}"
            f"{(str(lab['false_positives']) if lab else '-'):>5}"
            f"{(f'{acc:.2f}' if acc is not None else '-'):>9}"
        )
    t = report["totals"]
    tm = report["timing_ms"]
    pc = " ".join(f"{k}:{v}" for k, v in t["per_class"].items()) or "-"
    lines.append("-" * len(cols))
    lines.append(
        f"总计 {t['images_ok']}/{t['images']} 张, {t['particles']} 颗 [{pc}], 平均面积 {t['area_mean']:.1f} px, "
        f"边界颗粒 {t['edge']}, 耗时 p50={tm['p50']:.1f} ms p99={tm['p99']:.1f} ms"
    )
    if t["labels"] is not None:
        lab = t["labels"]
        recall = lab["recall"]
        acc = lab["cls_accuracy"]
        line = (
            f"标注 {lab['n_labels']} 颗: 召回 {recall if recall is None else f'{recall:.3f}'}"
            f"（匹配 {lab['matched']}）, 误检 {lab['false_positives']}, "
            f"分类准确率 {acc if acc is None else f'{acc:.3f}'}（{lab['cls_evaluated']} 颗）"
        )
        if lab.get("images_without_labels"):
            line += f", {lab['images_without_labels']} 张无标注未计入"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------


def _load_cfg(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        raise ReplayError(f"加载配置失败 {path}: {exc}") from exc


def _load_classifier(model_dir: Path, cfg: AppConfig) -> Any:
    from oresorter.ml.infer import ClassifierUnavailable, load_classifier

    try:
        return load_classifier(
            model_dir,
            prefer=CLASSIFIER_PREFER,
            device=cfg.ml.device,
            max_batch=int(cfg.pipeline.infer_batch_max),
        )
    except (ClassifierUnavailable, ValueError) as exc:
        raise ReplayError(f"模型目录无可用产物 {model_dir}: {exc}") from exc


def _iter_rgb(paths: Sequence[Path]) -> Iterator[np.ndarray]:
    for p in paths:
        img = read_rgb(p)
        if img is None:
            raise ReplayError(f"学习背景的图片无法读取: {p}")
        yield img


def _build_background(
    settings: CaptureSettings, files: Sequence[Path], learn_bg: int
) -> tuple[Background, str, dict[str, Any]]:
    """``learn_bg > 0`` 用前 N 张图学 ``ColumnBackground``，否则 ``ColorBackground``。

    Returns:
        ``(背景, "column"|"color", {"frames_used", "width", "mean_rgb"})``；``mean_rgb`` 便于排查用含颗粒的
        回放图学背景时是否学偏。
    """
    if learn_bg > 0:
        if learn_bg > len(files):
            raise ReplayError(f"learn_bg={learn_bg} 大于图片数 {len(files)}")
        bg = ColumnBackground()
        try:
            bg.learn(_iter_rgb(files[:learn_bg]))
        except ValueError as exc:
            raise ReplayError(f"学习背景失败: {exc}") from exc
        mean = bg.rgb.reshape(-1, 3).mean(axis=0)
        info = {
            "frames_used": int(bg.frames_used),
            "width": bg.width,
            "mean_rgb": [round(float(v), 1) for v in mean],
        }
        return bg, "column", info
    if settings.bg_mode == "column":
        log.info(
            "配置 bg_mode=column 但未指定 --learn-bg，回放改用固定颜色背景",
            extra={"bg_color": list(settings.bg_color)},
        )
    color = ColorBackground(settings.bg_color)
    info = {"frames_used": 0, "width": None, "mean_rgb": [int(v) for v in color.bg_color]}
    return color, "color", info


def run(args: argparse.Namespace) -> int:
    """``cli replay`` 实现。

    ``args``：``images``（大图目录）、``config``（yaml）、``model_dir``（可选）、``chunk_lines``（可选，
    128~4096）、``out``（可选，默认 ``images`` 同级 ``replay_out/``）、``learn_bg``（前 N 张学背景，0 = 颜色背景）。

    Returns:
        0 成功；1 参数 / 环境错误，或有图片回放失败（单张失败只影响该图：原因与堆栈写日志、报告里记一条
        error 条目，其余图片继续，报告仍会写出）。
    """
    t_start = time.perf_counter()
    classifier: Any | None = None
    try:
        opts = parse_options(args)
        cfg = _load_cfg(opts.config)
        files = list_images(opts.images)
        if not files:
            raise ReplayError(f"大图目录中没有图片（{'/'.join(IMAGE_EXTS)}）: {opts.images}")
        if opts.model_dir is not None:
            classifier = _load_classifier(opts.model_dir, cfg)
        bg, bg_kind, bg_info = _build_background(cfg.capture, files, opts.learn_bg)
        try:
            ensure_dir(opts.out)
        except OSError as exc:
            raise ReplayError(f"无法创建输出目录 {opts.out}: {exc}") from exc
    except ReplayError as exc:
        log.error("replay 参数错误", extra={"err": str(exc)})
        if classifier is not None:
            classifier.close()
        return 1

    overlap = int(cfg.pipeline.overlap_lines)
    if opts.chunk_lines is not None and overlap == 0:
        log.warning(
            "pipeline.overlap_lines=0：按块回放时跨块颗粒会被切成两半分别计数，且不会标记 truncated",
            extra={"chunk_lines": opts.chunk_lines},
        )
    labels_all = load_labels(opts.images)
    if labels_all is not None:
        unused = sorted(set(labels_all) - {p.name for p in files})
        if unused:
            log.warning(
                "labels.json 中有标注对应的图片不在目录里",
                extra={"n": len(unused), "names": unused[:MAX_NAMES_IN_LOG]},
            )
    overlays = overlay_names(files)
    log.info(
        "开始回放",
        extra={
            "images": len(files),
            "dir": str(opts.images),
            "out": str(opts.out),
            "mode": "chunked" if opts.chunk_lines is not None else "whole",
            "chunk_lines": opts.chunk_lines,
            "overlap": overlap,
            "bg": bg_kind,
            "backend": classifier.backend if classifier is not None else None,
            "labels": labels_all is not None,
        },
    )

    records: list[dict[str, Any]] = []
    errors = 0
    images_without_labels = 0
    try:
        for path in files:
            labels: list[dict[str, Any]] | None = None
            if labels_all is not None:
                labels = labels_all.get(path.name)
                if labels is None:
                    images_without_labels += 1
                    log.warning(
                        "labels.json 中没有该图的标注，不计入召回统计", extra={"image": path.name}
                    )
            try:
                image = read_rgb(path)
                if image is None:
                    errors += 1
                    log.error("无法读取图片", extra={"image": str(path)})
                    records.append({"name": path.name, "error": "无法读取图片"})
                    continue
                rec = replay_image(
                    image,
                    name=path.name,
                    bg=bg,
                    settings=cfg.capture,
                    overlap=overlap,
                    chunk_lines=opts.chunk_lines,
                    classifier=classifier,
                    labels=labels,
                    overlay_path=opts.out / overlays[path],
                )
            except (
                Exception
            ) as exc:  # 单张失败（含 onnxruntime/torch/cv2 运行时错误）不拖垮整次回放
                errors += 1
                log.error(
                    "回放失败",
                    extra={"image": path.name, "err": str(exc), "type": type(exc).__name__},
                    exc_info=True,
                )
                records.append(
                    {"name": path.name, "error": str(exc), "error_type": type(exc).__name__}
                )
                continue
            records.append(rec)
            log.info(
                "回放一张完成",
                extra={
                    "image": path.name,
                    "particles": rec["count"],
                    "ms": rec["ms"],
                    "recall": (rec["labels"] or {}).get("recall"),
                },
            )
    finally:
        if classifier is not None:
            classifier.close()

    report = build_report(
        records,
        opts=opts,
        cfg=cfg,
        bg_kind=bg_kind,
        bg_info=bg_info,
        classifier=classifier,
        errors=errors,
        labels_present=labels_all is not None,
        images_without_labels=images_without_labels,
    )
    report["elapsed_s"] = round(time.perf_counter() - t_start, 3)
    try:
        atomic_write_json(opts.out / REPORT_FILE, report)
    except OSError as exc:
        log.error("写 replay_report.json 失败", extra={"path": str(opts.out), "err": str(exc)})
        return 1
    print(format_summary(report))  # CLI 结果输出
    print(f"报告: {opts.out / REPORT_FILE}")
    log.info(
        "回放完成",
        extra={
            "images": len(records),
            "failed": errors,
            "particles": report["totals"]["particles"],
            "p50_ms": report["timing_ms"]["p50"],
            "p99_ms": report["timing_ms"]["p99"],
            "seconds": report["elapsed_s"],
        },
    )
    return 1 if errors else 0
