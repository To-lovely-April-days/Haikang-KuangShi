"""预处理：确定性、尺寸、mask 应用、补方、批处理、自动抠 mask。"""

from __future__ import annotations

import numpy as np
import pytest

from oresorter.vision import preprocess as pp


def _rgba(h, w, color=(200, 40, 40), alpha_pattern="full"):
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = color
    if alpha_pattern == "full":
        img[:, :, 3] = 255
    elif alpha_pattern == "half":
        img[:, : w // 2, 3] = 255
    return img


def test_shape_dtype_and_determinism():
    crop = _rgba(30, 50)
    t1 = pp.to_tensor(crop, 64)
    t2 = pp.to_tensor(crop.copy(), 64)
    assert t1.shape == (3, 64, 64) and t1.dtype == np.float32
    assert np.array_equal(t1, t2)
    assert t1.flags["C_CONTIGUOUS"]


def test_mask_sets_background_black_before_normalize():
    crop = _rgba(20, 20, alpha_pattern="half")
    t = pp.to_tensor(crop, 20)
    img = pp.denormalize(t)
    assert img[10, 2].tolist() == [200, 40, 40]
    assert img[10, 17].tolist() == [0, 0, 0], "alpha=0 的像素应为背景黑"


def test_pad_square_centers_content():
    crop = _rgba(10, 40, color=(10, 200, 10))
    sq = pp.pad_square(pp.apply_mask(crop))
    assert sq.shape == (40, 40, 3)
    assert sq[0, 0].tolist() == [0, 0, 0] and sq[20, 20].tolist() == [10, 200, 10]


@pytest.mark.parametrize("size", [128, 160, 224])
def test_sizes(size):
    assert pp.to_tensor(_rgba(300, 120), size).shape == (3, size, size)
    assert pp.to_tensor(_rgba(5, 7), size).shape == (3, size, size)


def test_rgb_input_treated_as_full_mask():
    rgb = np.full((16, 16, 3), 100, np.uint8)
    t = pp.to_tensor(rgb, 16)
    assert np.allclose(pp.denormalize(t), 100, atol=1)


def test_batch_and_empty():
    b = pp.to_tensor_batch([_rgba(10, 10), _rgba(20, 30)], 32)
    assert b.shape == (2, 3, 32, 32)
    assert pp.to_tensor_batch([], 32).shape == (0, 3, 32, 32)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pp.to_tensor(np.zeros((0, 5, 4), np.uint8), 32)
    with pytest.raises(ValueError):
        pp.to_tensor(np.zeros((5, 5), np.uint8), 32)
    with pytest.raises(ValueError):
        pp.to_tensor(_rgba(5, 5), 4)


def test_mask_from_background_blue():
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:] = (0, 0, 255)
    rgb[10:30, 10:30] = (200, 40, 40)
    alpha = pp.mask_from_background(rgb, (0, 0, 255), 40)
    assert alpha[20, 20] == 255 and alpha[2, 2] == 0
    rgba = pp.rgb_to_rgba(rgb)
    assert rgba.shape == (40, 40, 4)
