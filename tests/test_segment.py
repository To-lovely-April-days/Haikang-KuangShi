"""分割：数量/位置/面积精度、margin 与越界填充、alpha 只含本颗粒、触边、面积过滤、color/column 一致、
拆粘连、排序、辅助函数、性能，以及合成数据（make_sample_set / make_frames / make_belt / CLI）产物。"""

from __future__ import annotations

import importlib.util
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from oresorter.camera.mock import CLASS_PALETTE, draw_particle, synth_block
from oresorter.config import CaptureSettings
from oresorter.vision import segment as sg
from oresorter.vision import synthetic as syn
from oresorter.vision.background import ColorBackground, ColumnBackground

BLUE = (0, 0, 255)
RED = CLASS_PALETTE["A"]
GREEN = CLASS_PALETTE["B"]
REPO = Path(__file__).resolve().parent.parent


def _settings(**over) -> CaptureSettings:
    base: dict = {"bg_mode": "color", "bg_color": list(BLUE)}
    base.update(over)
    return CaptureSettings(**base)


def _canvas(h: int, w: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BLUE
    return img


def _truth(img: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    """由"与背景色不同的像素"得到的真值连通域 [(bbox, area)]（要求 bg_noise=0）。"""
    mask = np.any(img != np.array(BLUE, np.uint8), axis=2).astype(np.uint8) * 255
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        ((int(st[i][0]), int(st[i][1]), int(st[i][2]), int(st[i][3])), int(st[i][4]))
        for i in range(1, n)
    ]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / (aw * ah + bw * bh - inter)


def _grid_image(
    rng: np.random.Generator, rows: int, cols: int, *, cell: int = 120, diameter=(60, 100)
) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int, int]]]]:
    """确定性的不重叠布局：每个格子中央一颗矿石。"""
    img = _canvas(rows * cell, cols * cell)
    names = list(CLASS_PALETTE)
    labels = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            d = int(rng.integers(diameter[0], diameter[1] + 1))
            cls = names[k % len(names)]
            bbox = draw_particle(
                img, rng, c * cell + cell // 2, r * cell + cell // 2, d, CLASS_PALETTE[cls]
            )
            labels.append((cls, bbox))
            k += 1
    return img, labels


def _assert_matches_truth(img: np.ndarray, particles: list[sg.Particle]) -> None:
    truth = _truth(img)
    assert len(particles) == len(truth)
    for bbox, area in truth:
        best = max(particles, key=lambda p: _iou(bbox, p.bbox))
        assert _iou(bbox, best.bbox) > 0.8, (bbox, best.bbox)
        assert abs(best.area - area) / area < 0.15, (best.area, area)


# ---------------------------------------------------------------------------- 精度


def test_grid_layout_exact_count_and_boxes():
    img, labels = _grid_image(np.random.default_rng(1), 3, 5)
    particles = sg.segment(img, None, _settings())
    assert len(particles) == len(labels) == 15
    for _, bbox in labels:
        best = max(particles, key=lambda p: _iou(bbox, p.bbox))
        assert _iou(bbox, best.bbox) > 0.8
    _assert_matches_truth(img, particles)


def test_synth_block_matches_truth_components():
    rng = np.random.default_rng(3)
    img, labels = synth_block(rng, 1024, 768, n_particles=12, diameter_range=(60, 160), bg_noise=0)
    assert len(labels) == 12
    particles = sg.segment(img, None, _settings())
    _assert_matches_truth(img, particles)  # 粘连颗粒在真值里也是一个连通域，数量应完全一致
    for p in particles:
        assert p.area == int((p.crop_rgba[:, :, 3] == 255).sum())


def test_symmetric_morphology_recovers_near_exact_masks():
    """erode=dilate 为开运算：只会抹掉零星凸角像素，bbox 不变、面积最多少 1%。"""
    img, _ = _grid_image(np.random.default_rng(4), 2, 3)
    particles = sg.segment(img, None, _settings(erode_px=1, dilate_px=1))
    for bbox, area in _truth(img):
        best = max(particles, key=lambda p: _iou(bbox, p.bbox))
        assert best.bbox == bbox
        assert 0 <= area - best.area < 0.01 * area


def test_noisy_empty_belt_has_no_false_positives():
    img, _ = synth_block(np.random.default_rng(0), 1024, 256, n_particles=0, bg_noise=6)
    assert sg.segment(img, None, _settings()) == []
    assert sg.foreground_mask(img, None, _settings()).max() == 0


# ---------------------------------------------------------------------------- Particle 字段


def test_particle_fields_and_crop_geometry():
    img = _canvas(200, 300)
    cv2.circle(img, (150, 100), 30, RED, -1)
    (p,) = sg.segment(img, None, _settings(margin_px=8))
    x, y, w, h = p.bbox
    assert p.trailing_y == y + h - 1
    assert x <= p.centroid_x <= x + w and y <= p.centroid_y <= y + h
    assert abs(p.centroid_x - 150) < 1.0 and abs(p.centroid_y - 100) < 1.0
    assert p.crop_rgba.shape == (h + 16, w + 16, 4) and p.crop_rgba.dtype == np.uint8
    assert p.area == int((p.crop_rgba[:, :, 3] == 255).sum())
    assert set(np.unique(p.crop_rgba[:, :, 3]).tolist()) <= {0, 255}
    assert np.array_equal(p.crop_rgba[8 : 8 + h, 8 : 8 + w, :3], img[y : y + h, x : x + w])
    assert not p.touches_top and not p.touches_bottom and not p.truncated
    assert p.trailing_line == -1 and isinstance(p.extra["label"], int)
    assert p.x_range == (x, x + w - 1) and p.width == w and p.height == h
    d = p.to_dict()
    assert d["bbox"] == [x, y, w, h] and d["area_px"] == p.area and d["trailing_line"] == -1


def test_margin_out_of_bounds_filled_with_background():
    img = _canvas(100, 100)
    cv2.circle(img, (12, 10), 8, RED, -1)
    m = 8
    (p,) = sg.segment(img, None, _settings(margin_px=m, min_area_px=0))
    x, y, w, h = p.bbox
    ox, oy = m - x, m - y  # 越界的列数/行数
    assert ox > 0 and oy > 0
    crop = p.crop_rgba
    assert crop.shape == (h + 2 * m, w + 2 * m, 4)
    assert (crop[:, :ox, :3] == np.array(BLUE, np.uint8)).all()
    assert (crop[:oy, :, :3] == np.array(BLUE, np.uint8)).all()
    assert crop[:, :ox, 3].max() == 0 and crop[:oy, :, 3].max() == 0
    # 图内部分原样复制
    assert np.array_equal(crop[oy:, ox:, :3], img[: y + h + m, : x + w + m])
    # margin=0 → crop 恰好是 bbox
    (p0,) = sg.segment(img, None, _settings(margin_px=0, min_area_px=0))
    assert p0.crop_rgba.shape == (p0.bbox[3], p0.bbox[2], 4)


def test_margin_out_of_bounds_uses_column_background_color():
    """column 模式下越界填充用的是该列学到的背景色，而不是固定蓝色。"""
    w = 120
    belt = _canvas(16, w)
    belt[:, :, 1] = np.linspace(0, 120, w).astype(np.uint8)[None, :]  # 列向渐变
    bg = ColumnBackground()
    bg.learn([belt])
    img = belt[:1].repeat(100, axis=0).copy()
    cv2.circle(img, (60, 6), 5, RED, -1)  # 顶部越界
    (p,) = sg.segment(img, bg, _settings(bg_mode="column", margin_px=8, min_area_px=0))
    x, y, w_, _h = p.bbox
    oy = 8 - y
    assert oy > 0
    expected = bg.row_rgb(x - 8, x + w_ + 8)
    assert np.array_equal(p.crop_rgba[0, :, :3], expected)
    assert p.crop_rgba[:oy, :, 3].max() == 0


def test_alpha_contains_only_own_particle():
    img = _canvas(100, 140)
    cv2.circle(img, (40, 50), 15, RED, -1)
    cv2.circle(img, (76, 50), 15, GREEN, -1)  # 间隔 6 px，不接触，但 margin=8 会把对方框进来
    # 用对称形态学（开运算），mask ⊆ 颗粒像素，便于断言 alpha=255 的像素都是前景色
    left, right = sg.segment(img, None, _settings(margin_px=8, erode_px=1, dilate_px=1))
    assert left.bbox[0] < right.bbox[0]
    for p in (left, right):
        assert p.area == int((p.crop_rgba[:, :, 3] == 255).sum())
        rgb = p.crop_rgba[:, :, :3].astype(int)
        alpha = p.crop_rgba[:, :, 3]
        not_blue = np.abs(rgb - np.array(BLUE)).sum(axis=2) > 60
        assert (not_blue & (alpha == 0)).sum() > 20, "框内应含对方颗粒的像素且 alpha 为 0"
        assert not (alpha == 255)[~not_blue].any(), "alpha=255 的像素必须是前景颜色"
    # 左颗粒 crop 右侧含绿色像素，右颗粒 crop 左侧含红色像素
    lc = left.crop_rgba[:, :, :3].reshape(-1, 3)
    rc = right.crop_rgba[:, :, :3].reshape(-1, 3)
    assert (np.abs(lc.astype(int) - np.array(GREEN)).sum(axis=1) < 30).any()
    assert (np.abs(rc.astype(int) - np.array(RED)).sum(axis=1) < 30).any()


def test_touches_top_and_bottom_flags():
    img = _canvas(200, 300)
    cv2.circle(img, (50, 0), 20, RED, -1)  # 跨上边
    cv2.circle(img, (150, 100), 20, RED, -1)  # 中间
    cv2.circle(img, (250, 199), 20, RED, -1)  # 跨下边
    ps = sorted(sg.segment(img, None, _settings()), key=lambda p: p.bbox[0])
    assert [p.touches_top for p in ps] == [True, False, False]
    assert [p.touches_bottom for p in ps] == [False, False, True]
    assert ps[0].bbox[1] == 0 and ps[2].bbox[1] + ps[2].bbox[3] == 200


def test_area_filter():
    img = _canvas(200, 200)
    cv2.circle(img, (50, 50), 5, RED, -1)  # ≈ 80 px
    cv2.circle(img, (140, 140), 30, RED, -1)  # ≈ 2800 px
    assert len(sg.segment(img, None, _settings(min_area_px=0))) == 2
    big = sg.segment(img, None, _settings(min_area_px=400))
    assert len(big) == 1 and big[0].area > 2000
    small = sg.segment(img, None, _settings(min_area_px=0, max_area_px=1000))
    assert len(small) == 1 and small[0].area < 200
    assert sg.segment(img, None, _settings(min_area_px=5000)) == []


# ---------------------------------------------------------------------------- 背景模式


def test_column_mode_matches_color_mode():
    img, _ = synth_block(
        np.random.default_rng(7), 512, 384, n_particles=8, diameter_range=(50, 120)
    )
    blank = _canvas(32, 512)
    bg = ColumnBackground()
    bg.learn([blank, blank[:16]])
    a = sg.segment(img, None, _settings())
    b = sg.segment(img, bg, _settings(bg_mode="column"))
    assert len(a) == len(b) > 0
    for pa, pb in zip(a, b, strict=True):
        assert pa.bbox == pb.bbox and pa.area == pb.area
        assert np.array_equal(pa.crop_rgba, pb.crop_rgba)
    # 显式传 ColorBackground 也等价
    c = sg.segment(img, ColorBackground(BLUE), _settings())
    assert [p.bbox for p in c] == [p.bbox for p in a]


def test_column_mode_requires_learned_background():
    img = _canvas(32, 64)
    with pytest.raises(ValueError, match="背景"):
        sg.segment(img, None, _settings(bg_mode="column"))
    with pytest.raises(ValueError, match="背景"):
        sg.segment(img, ColumnBackground(), _settings(bg_mode="column"))
    bg = ColumnBackground()
    bg.learn([_canvas(4, 32)])
    with pytest.raises(ValueError, match="宽度"):
        sg.segment(img, bg, _settings(bg_mode="column"))


def test_invalid_images_rejected():
    with pytest.raises(ValueError):
        sg.segment(np.zeros((32, 64), np.uint8), None, _settings())
    with pytest.raises(ValueError):
        sg.segment(np.zeros((32, 64, 3), np.float32), None, _settings())
    with pytest.raises(ValueError):
        sg.foreground_mask(np.zeros((0, 64, 3), np.uint8), None, _settings())


def test_foreground_mask_values():
    img, _ = _grid_image(np.random.default_rng(9), 2, 2)
    mask = sg.foreground_mask(img, None, _settings())
    assert mask.dtype == np.uint8 and mask.shape == img.shape[:2]
    assert set(np.unique(mask).tolist()) == {0, 255}
    truth_area = sum(a for _, a in _truth(img))
    assert abs(int((mask > 0).sum()) - truth_area) / truth_area < 0.15


# ---------------------------------------------------------------------------- 拆粘连


def _two_tangent_circles(r1: int = 40, r2: int = 40) -> np.ndarray:
    img = _canvas(200, 300)
    cv2.circle(img, (100, 100), r1, RED, -1)
    cv2.circle(img, (100 + r1 + r2 - 2, 100), r2, RED, -1)  # 重叠 2 px，保证 8 连通粘连
    return img


def test_split_touching_separates_tangent_equal_circles():
    img = _two_tangent_circles()
    off = sg.segment(img, None, _settings())
    on = sg.segment(img, None, _settings(split_touching=True))
    assert len(off) == 1
    assert len(on) >= len(off)
    assert len(on) == 2
    areas = sorted(p.area for p in on)
    assert areas[0] / areas[1] > 0.9, f"等大相切圆应被对称切开，得到 {areas}"
    assert sum(areas) == off[0].area, "拆分不丢像素"
    for p in on:
        assert p.extra["split_from"] == off[0].extra["label"]
        assert p.area == int((p.crop_rgba[:, :, 3] == 255).sum())
    assert on[0].bbox[0] < on[1].bbox[0]
    assert on[0].bbox[0] + on[0].bbox[2] <= on[1].bbox[0] + 3


def test_split_touching_unequal_circles_and_tangent_ellipses():
    img = _two_tangent_circles(50, 20)
    off = sg.segment(img, None, _settings())
    on = sg.segment(img, None, _settings(split_touching=True))
    assert len(on) >= len(off) and len(on) == 2
    big = max(on, key=lambda p: p.area)
    assert abs(big.area - np.pi * 50**2) / (np.pi * 50**2) < 0.1
    ell = _canvas(300, 400)
    cv2.ellipse(ell, (120, 150), (60, 40), 0, 0, 360, RED, -1)
    cv2.ellipse(ell, (238, 150), (60, 40), 0, 0, 360, GREEN, -1)
    assert len(sg.segment(ell, None, _settings())) == 1
    assert len(sg.segment(ell, None, _settings(split_touching=True))) == 2


def test_split_touching_does_not_oversplit_isolated_particles():
    img, labels = _grid_image(np.random.default_rng(11), 3, 4)
    off = sg.segment(img, None, _settings())
    on = sg.segment(img, None, _settings(split_touching=True))
    assert len(on) == len(off) == len(labels)
    assert [p.bbox for p in on] == [p.bbox for p in off]
    ell = _canvas(300, 300)
    cv2.ellipse(ell, (150, 150), (80, 30), 30, 0, 360, RED, -1)
    assert len(sg.segment(ell, None, _settings(split_touching=True))) == 1
    single = np.zeros((60, 60), np.uint8)
    cv2.circle(single, (30, 30), 20, 255, -1)
    assert sg.split_touching_mask(single) is None
    assert sg.split_touching_mask(np.zeros((2, 2), np.uint8)) is None


def test_split_touching_mask_partition_properties():
    mask = np.zeros((200, 300), np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    cv2.circle(mask, (178, 100), 40, 255, -1)
    parts = sg.split_touching_mask(mask)
    assert parts is not None and len(parts) == 2
    union = np.zeros_like(mask, bool)
    for p in parts:
        assert p.dtype == bool and p.shape == mask.shape
        assert not (union & p).any(), "碎片互不重叠"
        union |= p
    assert np.array_equal(union, mask > 0), "碎片并集等于原掩码"


def test_split_touching_solid_blocks_do_not_crash_and_stay_fast():
    """掩码里没有 0 像素（矩形、贯穿块的长条、整帧前景）时 distanceTransform 曾返回 ≈65535 →
    巨型核 → cv2.error（1 s+ 后从 segment 冒出）。现在 bbox 外补一圈 0：这些连通域只有一个峰，
    判为不可拆，且耗时与颗粒大小无关。"""
    s_on = _settings(split_touching=True, erode_px=0, dilate_px=0)  # 不做形态学，便于断言精确 bbox
    rect = _canvas(200, 400)
    rect[40:160, 100:260] = RED
    (p,) = sg.segment(rect, None, s_on)
    assert p.bbox == (100, 40, 160, 120) and p.area == 160 * 120
    strip = _canvas(200, 400)
    strip[:, 100:260] = RED  # 贯穿上下边的长条
    (p,) = sg.segment(strip, None, s_on)
    assert p.touches_top and p.touches_bottom and p.area == 160 * 200
    assert sg.split_touching_mask(np.full((200, 300), 255, np.uint8)) is None

    full = np.empty((768, 4096, 3), np.uint8)
    full[:] = RED
    sg.segment(full, None, s_on)  # 预热
    t0 = time.perf_counter()
    assert sg.segment(full, None, s_on) == []  # 3.1M px > 默认 max_area_px，拆不开则丢弃
    (p,) = sg.segment(full, None, _settings(split_touching=True, max_area_px=16_000_000))
    elapsed = time.perf_counter() - t0
    assert p.area == 768 * 4096 and p.touches_top and p.touches_bottom
    assert elapsed < 0.5, f"整帧前景走拆粘连路径耗时 {elapsed * 1000:.0f} ms"


def test_split_touching_large_particles_use_downsampled_peak_search():
    """峰值半径大时局部极大在 max-pool 下采样图上搜索（否则椭圆核膨胀 ∝ 半径²，半径 150 要 400 ms）：
    结果仍对称、不过拆、耗时有界。"""
    img = _canvas(600, 900)
    cv2.circle(img, (250, 300), 120, RED, -1)
    cv2.circle(img, (250 + 238, 300), 120, RED, -1)
    off = sg.segment(img, None, _settings())
    t0 = time.perf_counter()
    on = sg.segment(img, None, _settings(split_touching=True))
    elapsed = time.perf_counter() - t0
    assert len(off) == 1 and len(on) == 2
    areas = sorted(p.area for p in on)
    assert areas[0] / areas[1] > 0.95 and sum(areas) == off[0].area
    assert elapsed < 0.3, f"大颗粒拆分耗时 {elapsed * 1000:.0f} ms"
    ell = _canvas(600, 900)
    cv2.ellipse(ell, (450, 300), (250, 100), 20, 0, 360, RED, -1)
    assert len(sg.segment(ell, None, _settings(split_touching=True))) == 1


def test_split_touching_does_not_oversplit_irregular_single_particles():
    """ "椭圆 + 3 个凸起"的不规则单颗：鞍点深度判据下不得被拆开（旧实现 50 例中拆错 8 例）。"""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        img = _canvas(300, 300)
        axes = (int(rng.integers(40, 91)), int(rng.integers(30, 61)))
        cv2.ellipse(img, (150, 150), axes, float(rng.uniform(0, 180)), 0, 360, RED, -1)
        for _ in range(3):
            c = (int(rng.integers(110, 191)), int(rng.integers(110, 191)))
            cv2.circle(img, c, int(rng.integers(20, 46)), RED, -1)
        off = sg.segment(img, None, _settings())
        if len(off) != 1:
            continue
        on = sg.segment(img, None, _settings(split_touching=True))
        assert len(on) == 1, f"不规则单颗被拆成 {len(on)} 颗"
        assert on[0].bbox == off[0].bbox and on[0].area == off[0].area
        checked += 1
    assert checked >= 30


def test_split_touching_chain_of_three_and_mixed_sizes():
    """三颗一排（等大 / 大中小 / 两大夹一小）：各颗的鞍点水位不同，按水位从高到低长种子后仍全部拆开、不丢像素。"""
    for radii in ((40, 40, 40), (50, 35, 25), (60, 25, 60)):
        img = _canvas(300, 600)
        x = 80
        for i, r in enumerate(radii):
            if i > 0:
                x += radii[i - 1] + r - 2
            cv2.circle(img, (x, 150), r, RED, -1)
        off = sg.segment(img, None, _settings())
        on = sg.segment(img, None, _settings(split_touching=True))
        assert len(off) == 1 and len(on) == 3, radii
        assert sum(p.area for p in on) == off[0].area
        for p, r in zip(sorted(on, key=lambda q: q.bbox[0]), radii, strict=True):
            assert abs(p.area - np.pi * r**2) / (np.pi * r**2) < 0.2, (radii, p.area)


def test_split_touching_mask_max_area_guard_and_input_check():
    mask = np.zeros((200, 400), np.uint8)
    for cx in (60, 138, 216):
        cv2.circle(mask, (cx, 100), 40, 255, -1)
    parts = sg.split_touching_mask(mask)
    assert parts is not None and len(parts) == 3
    union = np.zeros_like(mask, bool)
    for p in parts:
        assert not (union & p).any()
        union |= p
    assert np.array_equal(union, mask > 0)
    # 内切圆面积已超过单颗上限 → 实心大块，不拆
    assert sg.split_touching_mask(mask, max_area=int(np.pi * 39**2)) is None
    assert len(sg.split_touching_mask(mask, max_area=int(np.pi * 41**2))) == 3
    with pytest.raises(ValueError):
        sg.split_touching_mask(np.zeros((4, 4, 3), np.uint8))


# ---------------------------------------------------------------------------- 排序 / 辅助


def test_sorted_by_trailing_y_then_x():
    img = _canvas(300, 300)
    cv2.circle(img, (250, 60), 20, RED, -1)  # 尾沿 80
    cv2.circle(img, (60, 150), 20, RED, -1)  # 尾沿 170
    cv2.circle(img, (200, 150), 20, RED, -1)  # 尾沿 170，x 更大
    cv2.circle(img, (150, 40), 10, RED, -1)  # 尾沿 50
    ps = sg.segment(img, None, _settings(min_area_px=0))
    ty = [p.trailing_y for p in ps]
    assert ty == sorted(ty)
    assert [round(p.centroid_x) for p in ps] == [150, 250, 60, 200]


def test_draw_overlay_and_stats():
    img, _ = _grid_image(np.random.default_rng(2), 1, 3)
    ps = sg.segment(img, None, _settings())
    before = img.copy()
    out = sg.draw_overlay(img, ps, color=(255, 255, 255), thickness=2)
    assert np.array_equal(img, before), "不得修改输入"
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out == 255).all(axis=2).any(), "应画出白色框线"
    x, y, w, _h = ps[0].bbox
    assert (out[y, x : x + w] == 255).all(axis=1).any()
    ps[0].extra["cls"] = "A"
    out2 = sg.draw_overlay(img, ps)
    assert out2.shape == img.shape
    assert np.array_equal(sg.draw_overlay(img, []), img)
    with pytest.raises(ValueError):
        sg.draw_overlay(img, ps, thickness=0)

    st = sg.particle_stats(ps)
    areas = [p.area for p in ps]
    assert st["count"] == 3 and st["area_min"] == min(areas) and st["area_max"] == max(areas)
    assert abs(st["area_mean"] - sum(areas) / 3) < 0.1
    assert sg.particle_stats([]) == {"count": 0, "area_min": 0, "area_max": 0, "area_mean": 0.0}


def test_segment_speed_4096x768():
    """整条流程（不含 split_touching）目标 ≤ 40 ms（开发机 ≈ 7 ms）；宽松上限防抖动。"""
    img, _labels = synth_block(np.random.default_rng(1), 4096, 768, n_particles=12)
    settings = _settings()
    sg.segment(img, None, settings)  # 预热
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        ps = sg.segment(img, None, settings)
        best = min(best, time.perf_counter() - t0)
    assert len(ps) > 0
    assert best < 0.15, f"segment 耗时 {best * 1000:.1f} ms 过长"


# ---------------------------------------------------------------------------- 合成数据


def test_make_sample_set_layout(tmp_path: Path):
    rng = np.random.default_rng(0)
    out = syn.make_sample_set(tmp_path, ["A", "B"], 3, rng=rng, size_range=(40, 80), margin=8)
    assert set(out) == {"A", "B"} and all(len(v) == 3 for v in out.values())
    name_re = re.compile(r"^cam0_\d{8}_\d{6}_\d{6}\.png$")
    seen = set()
    for cls, paths in out.items():
        for png in paths:
            assert png.parent == tmp_path / "samples" / cls
            assert name_re.match(png.name), png.name
            assert png.name not in seen
            seen.add(png.name)
            bgra = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            assert bgra is not None and bgra.ndim == 3 and bgra.shape[2] == 4
            alpha = bgra[:, :, 3]
            assert set(np.unique(alpha).tolist()) == {0, 255}
            meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
            assert meta["camera_id"] == 0 and meta["synthetic"] is True and meta["cls"] == cls
            assert meta["area_px"] == int((alpha == 255).sum())
            x, y, w, h = meta["bbox"]
            assert (x, y) == (8, 8) and bgra.shape[:2] == (h + 16, w + 16)
            assert alpha[y : y + h, x : x + w].any()
            assert alpha[:8].max() == 0 and alpha[:, :8].max() == 0
            assert "created_at" in meta
            # 颗粒像素颜色接近该类调色板颜色（BGR → RGB）
            rgb = bgra[:, :, :3][alpha == 255][:, ::-1].astype(float).mean(axis=0)
            assert np.abs(rgb - np.array(CLASS_PALETTE[cls])).max() < 40
    assert not list((tmp_path / "samples").rglob("*.tmp"))


def test_make_sample_set_unknown_class_and_bad_args(tmp_path: Path):
    rng = np.random.default_rng(1)
    out = syn.make_sample_set(tmp_path, ["X"], 2, rng=rng, size_range=(30, 40))
    assert len(out["X"]) == 2
    assert syn.make_sample_set(tmp_path / "e", ["A"], 0, rng=rng) == {"A": []}
    with pytest.raises(ValueError):
        syn.make_sample_set(tmp_path, ["A"], -1, rng=rng)
    with pytest.raises(ValueError):
        syn.make_sample_set(tmp_path, ["A"], 1, rng=rng, size_range=(50, 20))
    with pytest.raises(ValueError):
        syn.make_sample_set(tmp_path, ["A", "A"], 1, rng=rng)
    with pytest.raises(ValueError):
        syn.make_sample_set(tmp_path, [], 1, rng=rng)


def test_make_frames_layout_and_segmentable(tmp_path: Path):
    rng = np.random.default_rng(2)
    frames_dir = syn.make_frames(
        tmp_path, 2, width=512, height=256, rng=rng, n_particles=4, classes=["A", "B"]
    )
    assert frames_dir == tmp_path / "frames"
    labels = json.loads((frames_dir / "labels.json").read_text(encoding="utf-8"))
    assert set(labels) == {"0000.png", "0001.png"}
    for name, items in labels.items():
        bgr = cv2.imread(str(frames_dir / name), cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.shape == (256, 512, 3)
        assert len(items) == 4
        for it in items:
            assert it["cls"] in {"A", "B"}
            x, y, w, h = it["bbox"]
            assert x >= 0 and y >= 0 and x + w <= 512 and y + h <= 256 and w > 0 and h > 0
    rgb = cv2.cvtColor(cv2.imread(str(frames_dir / "0000.png")), cv2.COLOR_BGR2RGB)
    ps = sg.segment(rgb, None, _settings())
    assert 1 <= len(ps) <= 4
    with pytest.raises(ValueError):
        syn.make_frames(tmp_path, 1, width=16, height=256, rng=rng)
    with pytest.raises(ValueError):
        syn.make_frames(tmp_path, -1, rng=rng)


def test_make_belt():
    rng = np.random.default_rng(3)
    img, labels = syn.make_belt(rng, 256, 1024, 10, classes=["A", "C"])
    assert img.shape == (1024, 256, 3) and img.dtype == np.uint8
    assert len(labels) == 10
    for cls, (x, y, w, h) in labels:
        assert cls in {"A", "C"}
        assert x >= 0 and y >= 0 and x + w <= 256 and y + h <= 1024
    with pytest.raises(ValueError):
        syn.make_belt(rng, 256, 32, 1)


def test_cli_main_and_script_wrapper(tmp_path: Path, capsys):
    rc = syn.main(
        [
            "--out",
            str(tmp_path / "synth"),
            "--classes",
            "A,B",
            "--per-class",
            "2",
            "--frames",
            "1",
            "--width",
            "128",
            "--frame-height",
            "64",
            "--particles",
            "2",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == str((tmp_path / "synth").resolve())
    assert len(list((tmp_path / "synth" / "samples" / "A").glob("*.png"))) == 2
    assert len(list((tmp_path / "synth" / "samples" / "B").glob("*.json"))) == 2
    assert (tmp_path / "synth" / "frames" / "0000.png").exists()
    assert (tmp_path / "synth" / "frames" / "labels.json").exists()
    assert syn.main(["--out", str(tmp_path / "x"), "--classes", " , "]) == 2
    assert syn.main(["--out", str(tmp_path / "y"), "--per-class", "0", "--width", "8"]) == 1

    script = REPO / "scripts" / "make_synthetic.py"
    spec = importlib.util.spec_from_file_location("make_synthetic_script", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main is syn.main


def test_cli_main_accepts_logging_reserved_class_names(tmp_path: Path, capsys):
    """类别名曾被直接当作 LogRecord 字段名：name / msg / args 等保留字会让 logging 抛 KeyError。"""
    argv = ["--out", str(tmp_path / "s"), "--classes", "name,msg,args", "--per-class", "1"]
    assert syn.main([*argv, "--frames", "0"]) == 0
    capsys.readouterr()
    for cls in ("name", "msg", "args"):
        assert len(list((tmp_path / "s" / "samples" / cls).glob("*.png"))) == 1
