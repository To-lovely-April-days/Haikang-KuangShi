"""背景模型：学习 / 距离 / 保存加载 / 宽度校验 / 自适应 / 固定颜色 / 选择逻辑 / 性能 / learn-bg CLI。"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from oresorter.config import CaptureSettings
from oresorter.vision.background import (
    DISTANCE_CHUNK_ROWS,
    ColorBackground,
    ColumnBackground,
    background_path,
    expand_lab_rows,
    lab_distance,
    learn_bg_cli,
    make_background,
)
from tests.conftest import make_cfg

BLUE = (0, 0, 255)
RED = (200, 40, 40)


def _belt(h: int, w: int, rng: np.random.Generator | None = None, *, noise: int = 3) -> np.ndarray:
    """带强烈列向渐变（模拟像元响应差异）与小噪声的空皮带帧：左边纯蓝，右边 B 降 100、G 升 60。"""
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BLUE
    g = np.linspace(0, 1, w)
    img[:, :, 2] = (255 - 100 * g).round().astype(np.uint8)[None, :]
    img[:, :, 1] = (60 * g).round().astype(np.uint8)[None, :]
    if rng is not None and noise > 0:
        n = rng.integers(-noise, noise + 1, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return img


def _learned(w: int = 256, seed: int = 0) -> ColumnBackground:
    rng = np.random.default_rng(seed)
    bg = ColumnBackground()
    bg.learn([_belt(64, w, rng), _belt(32, w, rng), _belt(100, w, rng)])
    return bg


def _uniform(h: int, w: int, color=BLUE) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    return img


# ---------------------------------------------------------------------------- 学习与距离


def test_learn_sets_properties():
    bg = ColumnBackground()
    assert not bg.ready and bg.width is None and bg.learned_at is None and bg.frames_used == 0
    bg = _learned(256)
    assert bg.ready and bg.width == 256 and bg.frames_used == 3
    assert isinstance(bg.learned_at, str)
    datetime.fromisoformat(bg.learned_at)  # 可解析的 ISO 时间
    assert bg.rgb.shape == (256, 3) and bg.rgb.dtype == np.float32
    assert bg.lab.shape == (256, 3)
    # 学到的列背景应跟真实渐变一致（噪声中值滤掉）
    truth = _belt(1, 256)[0].astype(np.float32)
    assert np.abs(bg.rgb - truth).max() < 2.0


def test_distance_small_on_background_large_on_foreground():
    bg = _learned(256)
    clean = _belt(48, 256)
    d = bg.distance(clean)
    assert d.shape == (48, 256) and d.dtype == np.float32
    assert d.max() < 8.0, "背景像素的距离应接近 0"
    noisy = _belt(48, 256, np.random.default_rng(5))
    assert np.percentile(bg.distance(noisy), 99) < 15.0
    fg = clean.copy()
    fg[10:30, 100:140] = RED
    d2 = bg.distance(fg)
    assert d2[10:30, 100:140].min() > 100.0
    assert d2[:10].max() < 8.0


def test_column_model_absorbs_column_gradient_but_color_model_does_not():
    """列背景存在的意义：同一帧，逐列模型距离≈0，固定颜色模型在右侧列远超阈值。"""
    bg = _learned(256)
    frame = _belt(16, 256)
    assert bg.distance(frame).max() < 8.0
    color = ColorBackground(BLUE).distance(frame)
    assert color[:, -1].min() > 40.0


def test_learn_errors():
    bg = ColumnBackground()
    with pytest.raises(ValueError):
        bg.learn([])
    with pytest.raises(ValueError, match="宽度"):
        bg.learn([_uniform(8, 64), _uniform(8, 32)])
    with pytest.raises(ValueError):
        bg.learn([np.zeros((8, 64, 3), np.float32)])
    with pytest.raises(ValueError):
        bg.learn([np.zeros((8, 64), np.uint8)])
    assert not bg.ready


def test_distance_requires_ready_and_matching_width():
    bg = ColumnBackground()
    with pytest.raises(ValueError, match="请先学习背景"):
        bg.distance(_uniform(8, 64))
    bg = _learned(128)
    with pytest.raises(ValueError, match="宽度"):
        bg.distance(_uniform(8, 100))
    with pytest.raises(ValueError):
        bg.distance(_uniform(8, 128).astype(np.float32))


def test_learn_accepts_tall_frames_with_row_subsampling():
    bg = ColumnBackground()
    bg.learn([_uniform(1200, 32)])
    assert bg.ready and bg.width == 32
    assert np.allclose(bg.rgb, np.array(BLUE, np.float32))


# ---------------------------------------------------------------------------- 持久化


def test_save_load_roundtrip(tmp_path: Path):
    bg = _learned(200)
    p = tmp_path / "bg" / "cam1.npz"
    bg.save(p)
    assert p.exists()
    assert not list(p.parent.glob("*.tmp")), "不能留下临时文件"
    with np.load(p, allow_pickle=False) as z:
        assert int(z["version"]) == 1
        assert int(z["width"]) == 200
        assert int(z["frames_used"]) == 3
        assert str(z["learned_at"]) == bg.learned_at
    loaded = ColumnBackground.load(p)
    assert loaded.ready and loaded.width == 200
    assert loaded.frames_used == 3 and loaded.learned_at == bg.learned_at
    assert np.allclose(loaded.rgb, bg.rgb)
    img = _belt(20, 200, np.random.default_rng(1))
    assert np.array_equal(loaded.distance(img), bg.distance(img))


def test_save_requires_ready():
    with pytest.raises(ValueError):
        ColumnBackground().save("/nonexistent/should/not/be/written.npz")


def test_load_missing_or_corrupt(tmp_path: Path):
    with pytest.raises(ValueError, match="不存在"):
        ColumnBackground.load(tmp_path / "nope.npz")
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"this is not an npz file")
    with pytest.raises(ValueError):
        ColumnBackground.load(bad)


def test_loaded_background_rejects_other_width(tmp_path: Path):
    bg = _learned(64)
    bg.save(tmp_path / "b.npz")
    loaded = ColumnBackground.load(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="宽度"):
        loaded.distance(_uniform(4, 65))


# ---------------------------------------------------------------------------- 自适应


def test_adapt_updates_only_columns_without_foreground():
    bg = _learned(128)
    before = bg.rgb
    img = _belt(32, 128)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + 40, 0, 255).astype(np.uint8)  # 整体偏绿
    fg = np.zeros((32, 128), bool)
    fg[5:10, 40:61] = True  # 列 40~60 有前景
    n = bg.adapt(img, fg, rate=0.5)
    after = bg.rgb
    assert n == 128 - 21
    assert np.array_equal(after[40:61], before[40:61]), "有前景的列不得更新"
    clean = np.ones(128, bool)
    clean[40:61] = False
    expected = 0.5 * before[clean] + 0.5 * img[:, clean].mean(axis=0)
    assert np.allclose(after[clean], expected, atol=1e-3)
    # Lab 也同步更新：对新背景的距离变小
    assert bg.distance(img)[:, :40].max() < 25.0


def test_adapt_accepts_uint8_mask_and_returns_zero_when_all_columns_busy():
    bg = _learned(32)
    img = _belt(8, 32)
    full = np.full((8, 32), 255, np.uint8)
    before = bg.rgb
    assert bg.adapt(img, full) == 0
    assert np.array_equal(bg.rgb, before)


def test_adapt_rejects_bad_args():
    bg = _learned(32)
    img = _belt(8, 32)
    mask = np.zeros((8, 32), bool)
    with pytest.raises(ValueError):
        bg.adapt(img, mask, rate=0.0)
    with pytest.raises(ValueError):
        bg.adapt(img, mask, rate=1.5)
    with pytest.raises(ValueError):
        bg.adapt(img, np.zeros((8, 31), bool))
    with pytest.raises(ValueError):
        bg.adapt(_uniform(8, 33), np.zeros((8, 33), bool))
    with pytest.raises(ValueError):
        ColumnBackground().adapt(img, mask)
    # 三维掩码曾被 shape[:2] 校验放过，跨通道混合后静默写出错误背景并返回 96 列
    before = bg.rgb
    with pytest.raises(ValueError, match="二维"):
        bg.adapt(img, np.zeros((8, 32, 3), np.uint8))
    with pytest.raises(ValueError, match="二维"):
        bg.adapt(img, np.zeros((8, 32, 1), bool))
    assert np.array_equal(bg.rgb, before)


# ---------------------------------------------------------------------------- 固定颜色背景 / row_rgb


def test_color_background():
    cb = ColorBackground(BLUE)
    assert cb.ready and cb.width is None and cb.bg_color == BLUE
    img = _uniform(10, 20)
    img[2:5, 3:9] = RED
    d = cb.distance(img)
    assert d.shape == (10, 20) and d.dtype == np.float32
    assert d[0, 0] == 0.0 and d[6:].max() == 0.0
    assert d[2:5, 3:9].min() > 100.0
    row = cb.row_rgb(0, 5)
    assert row.shape == (5, 3) and row.dtype == np.uint8 and (row == np.array(BLUE)).all()
    assert cb.row_rgb(3, 3).shape == (0, 3)
    with pytest.raises(ValueError):
        ColorBackground((0, 0))
    with pytest.raises(ValueError):
        ColorBackground((0, 0, 256))


def test_column_row_rgb_clamps_out_of_range_columns():
    bg = ColumnBackground()
    bg.learn([_belt(4, 50)])
    row = bg.row_rgb(-5, 3)
    assert row.shape == (8, 3) and row.dtype == np.uint8
    assert (row[:5] == row[5]).all(), "左侧越界列取第 0 列"
    tail = bg.row_rgb(48, 53)
    assert (tail[2:] == tail[1]).all(), "右侧越界列取最后一列"
    assert bg.row_rgb(10, 10).shape == (0, 3)


def test_column_matches_color_on_uniform_belt():
    bg = ColumnBackground()
    bg.learn([_uniform(16, 64), _uniform(8, 64)])
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(20, 64, 3), dtype=np.uint8)
    assert np.allclose(bg.distance(img), ColorBackground(BLUE).distance(img), atol=1e-4)


# ---------------------------------------------------------------------------- make_background


def test_make_background_selects_by_mode():
    color = CaptureSettings(bg_mode="color", bg_color=[10, 20, 30])
    cb = make_background(color, None)
    assert isinstance(cb, ColorBackground) and cb.bg_color == (10, 20, 30)
    column = CaptureSettings(bg_mode="column")
    with pytest.raises(ValueError, match="请先学习背景"):
        make_background(column, None)
    with pytest.raises(ValueError, match="请先学习背景"):
        make_background(column, ColumnBackground())
    bg = _learned(32)
    assert make_background(column, bg) is bg


# ---------------------------------------------------------------------------- 性能


def test_distance_speed_4096x512():
    """4096×512 的距离计算目标 ≤ 60 ms（开发机 ≈ 4 ms）；这里给宽松上限防止抖动误报。"""
    bg = ColumnBackground()
    bg.learn([_uniform(16, 4096)])
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(512, 4096, 3), dtype=np.uint8)
    bg.distance(img)  # 预热
    best = min(_timed(bg.distance, img) for _ in range(5))
    assert best < 0.12, f"distance 耗时 {best * 1000:.1f} ms 过长"


def _timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def test_distance_processes_in_chunks_with_bounded_memory():
    """4096×768 一块：旧实现把背景 broadcast 成整幅 (h, w, 3) float32 再做差/平方，峰值分配 ≈150 MB；
    分块后 ≈45 MB，且与整幅直接计算逐像素一致（含不足一块、恰好一块、跨块边界的高度）。"""
    bg = ColumnBackground()
    bg.learn([_uniform(16, 4096)])
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(768, 4096, 3), dtype=np.uint8)
    bg.distance(img)  # 预热：建立分块缓存
    tracemalloc.start()
    try:
        bg.distance(img)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 80e6, f"distance 峰值分配 {peak / 1e6:.0f} MB"
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.sqrt(((lab - bg.lab[None, :, :]) ** 2).sum(axis=2))
    color = ColorBackground(BLUE)
    for h in (1, 255, DISTANCE_CHUNK_ROWS, DISTANCE_CHUNK_ROWS + 1, 768):
        assert np.allclose(bg.distance(img[:h]), ref[:h], atol=1e-3)
        assert np.allclose(color.distance(img[:h]), ref[:h], atol=1e-3)


def test_lab_distance_accepts_single_row_and_rejects_width_mismatch():
    row = np.ascontiguousarray(np.broadcast_to(ColorBackground(BLUE).lab, (1, 20, 3)))
    img = _uniform(300, 20)
    img[5:9, 3:7] = RED
    d = lab_distance(img, row)
    assert d.shape == (300, 20) and d.dtype == np.float32
    assert d[0].max() == 0.0 and d[5:9, 3:7].min() > 100.0
    # 任意分块行数（含最后一块不满）结果相同
    assert np.array_equal(lab_distance(img, expand_lab_rows(row, 7)), d)
    with pytest.raises(ValueError, match="宽度"):
        lab_distance(_uniform(4, 21), row)


# ---------------------------------------------------------------------------- learn-bg CLI


def _write_yaml(tmp_path: Path, **cam_overrides) -> Path:
    cfg = make_cfg(tmp_path)
    data = cfg.model_dump(mode="json")
    data["cameras"][0].update(cam_overrides)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_learn_bg_cli_from_images(tmp_path: Path):
    img_dir = tmp_path / "belt"
    img_dir.mkdir()
    for i in range(2):
        rgb = _belt(40, 300, np.random.default_rng(i))
        cv2.imwrite(str(img_dir / f"{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (img_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    cfg_path = _write_yaml(tmp_path)
    args = argparse.Namespace(config=cfg_path, camera=1, frames=None, images=img_dir)
    assert learn_bg_cli(args) == 0
    out = background_path(tmp_path / "data", 1)
    assert out.exists()
    bg = ColumnBackground.load(out)
    assert bg.width == 300 and bg.frames_used == 2
    assert bg.distance(_belt(8, 300)).max() < 8.0


def test_learn_bg_cli_from_mock_camera(tmp_path: Path):
    cfg_path = _write_yaml(tmp_path, height=256)
    args = argparse.Namespace(config=cfg_path, camera=1, frames=2, images=None)
    assert learn_bg_cli(args) == 0
    bg = ColumnBackground.load(background_path(tmp_path / "data", 1))
    assert bg.width == 4096 and bg.frames_used == 2
    # Mock 合成图是蓝底：列中值应接近蓝色
    assert np.abs(bg.rgb.mean(axis=0) - np.array(BLUE, np.float32)).max() < 10.0


def test_learn_bg_cli_errors(tmp_path: Path):
    cfg_path = _write_yaml(tmp_path)
    assert learn_bg_cli(argparse.Namespace(config=cfg_path, camera=9, frames=1, images=None)) == 1
    missing = tmp_path / "no_such_dir"
    assert (
        learn_bg_cli(argparse.Namespace(config=cfg_path, camera=1, frames=1, images=missing)) == 1
    )
    assert learn_bg_cli(argparse.Namespace(config=cfg_path, camera=1, frames=0, images=None)) == 1
    assert (
        learn_bg_cli(
            argparse.Namespace(config=tmp_path / "nope.yaml", camera=1, frames=1, images=None)
        )
        == 1
    )
    # --camera 0 曾被 `or 1` 静默改成 1 并写出 cam1.npz；相机编号 ≥ 1，越界必须拒绝
    assert learn_bg_cli(argparse.Namespace(config=cfg_path, camera=0, frames=1, images=None)) == 1
    assert not background_path(tmp_path / "data", 1).exists()
    assert not background_path(tmp_path / "data", 0).exists()
