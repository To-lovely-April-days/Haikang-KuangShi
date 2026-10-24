"""训练与运行时推理 **共用** 的预处理（docs/01 §5.6）。这是小图进入网络前的唯一入口。

步骤：RGBA 小图 → 背景像素置黑 → 按长边补成正方形 → 缩放到 ``image_size`` → RGB [0,1] → ImageNet 归一化 → ``float32 CHW``。
训练增广（翻转/旋转/缩放、±10% 亮度对比度）必须在调用 ``to_tensor`` **之前** 对 RGBA 做；禁止色相/饱和度抖动。
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BG_FILL = 0  # 背景与补边填充值（归一化前的 0~255 灰度）
MASK_THRESHOLD = 128  # alpha ≥ 此值视为前景


def apply_mask(crop_rgba: np.ndarray, fill: int = BG_FILL) -> np.ndarray:
    """RGBA → RGB，alpha < 阈值的像素置为 ``fill``。输入若已是 RGB 则原样返回拷贝。"""
    if crop_rgba.ndim != 3 or crop_rgba.shape[2] not in (3, 4):
        raise ValueError(f"期望 (h, w, 3|4) 图像，得到 {crop_rgba.shape}")
    if crop_rgba.shape[0] == 0 or crop_rgba.shape[1] == 0:
        raise ValueError("空图像")
    if crop_rgba.shape[2] == 3:
        return np.ascontiguousarray(crop_rgba)
    rgb = crop_rgba[:, :, :3].copy()
    bg = crop_rgba[:, :, 3] < MASK_THRESHOLD
    rgb[bg] = fill
    return rgb


def pad_square(rgb: np.ndarray, fill: int = BG_FILL) -> np.ndarray:
    """按长边居中补成正方形。"""
    h, w = rgb.shape[:2]
    side = max(h, w)
    if h == w:
        return rgb
    out = np.full((side, side, rgb.shape[2]), fill, rgb.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = rgb
    return out


def resize_square(rgb: np.ndarray, image_size: int) -> np.ndarray:
    """正方形图缩放到 ``image_size``：缩小用 INTER_AREA，放大用 INTER_LINEAR（确定性）。"""
    side = rgb.shape[0]
    if side == image_size:
        return rgb
    interp = cv2.INTER_AREA if side > image_size else cv2.INTER_LINEAR
    return cv2.resize(rgb, (image_size, image_size), interpolation=interp)


def normalize(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 HWC RGB → float32 CHW，ImageNet 均值方差归一化。"""
    x = rgb_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def to_tensor(crop_rgba: np.ndarray, image_size: int) -> np.ndarray:
    """RGBA/RGB 小图 → ``(3, image_size, image_size) float32``。训练与推理唯一入口。"""
    if image_size < 8:
        raise ValueError(f"image_size 过小: {image_size}")
    rgb = apply_mask(crop_rgba)
    rgb = pad_square(rgb)
    rgb = resize_square(rgb, image_size)
    return normalize(rgb)


def to_tensor_batch(crops: Sequence[np.ndarray], image_size: int) -> np.ndarray:
    """多张小图 → ``(N, 3, S, S) float32``；空列表返回 shape ``(0, 3, S, S)``。"""
    if len(crops) == 0:
        return np.zeros((0, 3, image_size, image_size), np.float32)
    return np.stack([to_tensor(c, image_size) for c in crops], axis=0)


def denormalize(chw: np.ndarray) -> np.ndarray:
    """``to_tensor`` 的逆（调试/可视化用）：CHW float → HWC uint8。"""
    x = chw.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def mask_from_background(
    rgb: np.ndarray,
    bg_color: Sequence[int] = (0, 0, 255),
    threshold: float = 40.0,
    *,
    close_px: int = 2,
) -> np.ndarray:
    """无 alpha 的小图（客户现有 a/ b/ c/ 目录）按固定背景色自动抠 mask（Lab 颜色距离）。返回 uint8 alpha。"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = cv2.cvtColor(np.array([[list(bg_color)]], np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[
        0, 0
    ]
    dist = np.linalg.norm(lab - bg, axis=2)
    alpha = (dist > threshold).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)
    return alpha


def rgb_to_rgba(
    rgb: np.ndarray, bg_color: Sequence[int] = (0, 0, 255), threshold: float = 40.0
) -> np.ndarray:
    """RGB + 自动 mask → RGBA。"""
    alpha = mask_from_background(rgb, bg_color, threshold)
    return np.dstack([rgb, alpha])
