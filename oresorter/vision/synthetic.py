"""合成数据：蓝底矿石小图样本集、多颗粒大图帧、长皮带图（``scripts/make_synthetic.py`` 的实现）。

画风与 ``camera.mock`` 一致（复用 ``synth_block`` / ``draw_particle`` / ``CLASS_PALETTE``），
保证用合成数据训练出的模型能在 Mock 相机流上跑通。

产物布局（``out_dir`` 下）::

    samples/{cls}/cam0_{YYYYmmdd_HHMMSS}_{seq:06d}.png   RGBA，alpha = mask
    samples/{cls}/cam0_{YYYYmmdd_HHMMSS}_{seq:06d}.json  元数据（camera_id/bbox/area_px/created_at/synthetic）
    frames/{i:04d}.png                                    RGB 大图
    frames/labels.json                                    {"0000.png": [{"cls": "A", "bbox": [x, y, w, h]}, ...]}
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from oresorter.camera.mock import CLASS_PALETTE, draw_particle, synth_block
from oresorter.fsutil import atomic_write_bytes, atomic_write_json, ensure_dir
from oresorter.log import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_BG_COLOR: tuple[int, int, int] = (0, 0, 255)
MIN_DIAMETER_PX = 8
SYNTHETIC_CAMERA_ID = 0


def palette_for(classes: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """给每个类别一个 RGB：``CLASS_PALETTE`` 里有的直接用，没有的按顺序循环借用调色板颜色。

    借用会导致两个类别同色（仅影响合成数据的可分性），在日志里给出 warning。
    """
    if not classes:
        raise ValueError("classes 不能为空")
    if len(set(classes)) != len(classes):
        raise ValueError(f"classes 有重复: {list(classes)}")
    fallback = list(CLASS_PALETTE.values())
    out: dict[str, tuple[int, int, int]] = {}
    k = 0
    for c in classes:
        if c in CLASS_PALETTE:
            out[c] = CLASS_PALETTE[c]
        else:
            out[c] = fallback[k % len(fallback)]
            log.warning("类别不在内置调色板中，借用颜色", extra={"cls": c, "rgb": list(out[c])})
            k += 1
    return out


def encode_png(image: np.ndarray) -> bytes:
    """RGB / RGBA uint8 → PNG 字节（内部转 BGR / BGRA 交给 OpenCV）。"""
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ValueError(f"期望 (h, w, 3|4) uint8 图像，得到 {image.shape} {image.dtype}")
    code = cv2.COLOR_RGBA2BGRA if image.shape[2] == 4 else cv2.COLOR_RGB2BGR
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(image), code))
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return buf.tobytes()


def _check_size_range(size_range: tuple[int, int]) -> tuple[int, int]:
    lo, hi = int(size_range[0]), int(size_range[1])
    if lo < MIN_DIAMETER_PX or hi < lo:
        raise ValueError(f"size_range 必须满足 {MIN_DIAMETER_PX} ≤ lo ≤ hi，得到 {(lo, hi)}")
    return lo, hi


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_sample_set(
    out_dir: Path,
    classes: list[str],
    per_class: int,
    *,
    rng: np.random.Generator,
    size_range: tuple[int, int] = (40, 160),
    margin: int = 8,
    bg_color: tuple[int, int, int] = DEFAULT_BG_COLOR,
) -> dict[str, list[Path]]:
    """生成 ``out_dir/samples/{cls}/*.png`` RGBA 小图样本（alpha = mask）与同名 ``.json`` 元数据。

    每张样本：在干净的 ``bg_color`` 小画布中央用 ``draw_particle`` 画一颗该类颜色的矿石，
    与背景色不同的像素即 mask；按 bbox 向外扩 ``margin`` 裁出 RGBA。
    元数据 ``bbox`` 是颗粒在小图内的 ``[x, y, w, h]``。

    Returns:
        ``{cls: [png 路径, ...]}``。
    """
    if per_class < 0:
        raise ValueError(f"per_class 必须 ≥ 0，得到 {per_class}")
    if margin < 0:
        raise ValueError(f"margin 必须 ≥ 0，得到 {margin}")
    lo, hi = _check_size_range(size_range)
    palette = palette_for(classes)
    bg = np.array(bg_color, np.uint8)
    samples_dir = ensure_dir(Path(out_dir) / "samples")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = _now_iso()
    out: dict[str, list[Path]] = {}
    seq = 0
    for cls in classes:
        cls_dir = ensure_dir(samples_dir / cls)
        paths: list[Path] = []
        for _ in range(per_class):
            diameter = int(rng.integers(lo, hi + 1))
            side = diameter + 2 * margin + 8
            canvas = np.empty((side, side, 3), np.uint8)
            canvas[:] = bg
            x, y, w, h = draw_particle(canvas, rng, side // 2, side // 2, diameter, palette[cls])
            if w == 0 or h == 0:
                continue
            mask = np.any(canvas != bg, axis=2)
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(side, x + w + margin), min(side, y + h + margin)
            rgba = np.dstack([canvas[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.uint8) * 255])
            name = f"cam{SYNTHETIC_CAMERA_ID}_{stamp}_{seq:06d}"
            png = cls_dir / f"{name}.png"
            atomic_write_bytes(png, encode_png(rgba))
            atomic_write_json(
                cls_dir / f"{name}.json",
                {
                    "camera_id": SYNTHETIC_CAMERA_ID,
                    "cls": cls,
                    "bbox": [x - x0, y - y0, w, h],
                    "area_px": int(mask.sum()),
                    "created_at": created_at,
                    "synthetic": True,
                    "seq": seq,
                    "diameter_px": diameter,
                },
            )
            paths.append(png)
            seq += 1
        out[cls] = paths
    log.info(
        "合成样本集完成",
        extra={"dir": str(samples_dir), "classes": list(classes), "total": seq},
    )
    return out


def make_frames(
    out_dir: Path,
    n_frames: int,
    *,
    width: int = 4096,
    height: int = 2048,
    rng: np.random.Generator,
    n_particles: int = 30,
    classes: list[str] | None = None,
    bg_color: tuple[int, int, int] = DEFAULT_BG_COLOR,
    diameter_range: tuple[int, int] = (40, 160),
) -> Path:
    """生成 ``out_dir/frames/{i:04d}.png``（RGB 蓝底多颗粒大图）与 ``frames/labels.json``。

    Returns:
        ``frames`` 目录路径。
    """
    if n_frames < 0:
        raise ValueError(f"n_frames 必须 ≥ 0，得到 {n_frames}")
    if width < 64 or height < 64:
        raise ValueError(f"width/height 必须 ≥ 64，得到 {(width, height)}")
    if n_particles < 0:
        raise ValueError(f"n_particles 必须 ≥ 0，得到 {n_particles}")
    _check_size_range(diameter_range)
    palette = palette_for(classes) if classes else dict(CLASS_PALETTE)
    frames_dir = ensure_dir(Path(out_dir) / "frames")
    labels: dict[str, list[dict[str, object]]] = {}
    for i in range(n_frames):
        img, labs = synth_block(
            rng,
            width,
            height,
            bg_color=bg_color,
            n_particles=n_particles,
            diameter_range=diameter_range,
            palette=palette,
        )
        name = f"{i:04d}.png"
        atomic_write_bytes(frames_dir / name, encode_png(img))
        labels[name] = [{"cls": c, "bbox": list(b)} for c, b in labs]
    atomic_write_json(frames_dir / "labels.json", labels)
    log.info(
        "合成大图完成",
        extra={"dir": str(frames_dir), "frames": n_frames, "size": [width, height]},
    )
    return frames_dir


def make_belt(
    rng: np.random.Generator,
    width: int,
    total_lines: int,
    n_particles: int,
    *,
    classes: list[str] | None = None,
    bg_color: tuple[int, int, int] = DEFAULT_BG_COLOR,
    diameter_range: tuple[int, int] = (40, 160),
) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int, int]]]]:
    """一条长皮带图 ``(total_lines, width, 3)``（跨块拼接测试用），返回 ``(image, [(cls, bbox), ...])``。"""
    if width < 64 or total_lines < 64:
        raise ValueError(f"width/total_lines 必须 ≥ 64，得到 {(width, total_lines)}")
    if n_particles < 0:
        raise ValueError(f"n_particles 必须 ≥ 0，得到 {n_particles}")
    _check_size_range(diameter_range)
    palette = palette_for(classes) if classes else dict(CLASS_PALETTE)
    return synth_block(
        rng,
        width,
        total_lines,
        bg_color=bg_color,
        n_particles=n_particles,
        diameter_range=diameter_range,
        palette=palette,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_synthetic", description="生成合成蓝底矿石样本集与多颗粒大图"
    )
    p.add_argument("--out", type=Path, required=True, help="输出目录")
    p.add_argument("--classes", default="A,B,C", help="类别名，逗号分隔（默认 A,B,C）")
    p.add_argument("--per-class", type=int, default=50, help="每类样本数（0 = 不生成样本）")
    p.add_argument("--frames", type=int, default=5, help="大图帧数（0 = 不生成大图）")
    p.add_argument("--width", type=int, default=4096, help="大图宽度")
    p.add_argument("--frame-height", type=int, default=2048, help="大图高度")
    p.add_argument("--particles", type=int, default=30, help="每帧颗粒数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``scripts/make_synthetic.py`` 入口。返回退出码。"""
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    if not classes:
        log.error("--classes 不能为空")
        return 2
    rng = np.random.default_rng(int(args.seed))
    try:
        out_dir = Path(args.out)
        if args.per_class > 0:
            sample_paths = make_sample_set(out_dir, classes, int(args.per_class), rng=rng)
            # 类别名不能直接当 LogRecord 的字段名（name/msg/args… 是保留字），放在固定键下
            log.info(
                "样本",
                extra={"per_class": {cls: len(paths) for cls, paths in sample_paths.items()}},
            )
        if args.frames > 0:
            frames_dir = make_frames(
                out_dir,
                int(args.frames),
                width=int(args.width),
                height=int(args.frame_height),
                rng=rng,
                n_particles=int(args.particles),
                classes=classes,
            )
            log.info("大图", extra={"dir": str(frames_dir)})
    except (ValueError, OSError) as exc:
        log.error("生成失败", extra={"err": str(exc)})
        return 1
    print(str(Path(args.out).resolve()))  # CLI 结果输出：产物根目录
    return 0


if __name__ == "__main__":
    sys.exit(main())
