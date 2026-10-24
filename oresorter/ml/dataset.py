"""读样本目录 → ``torch.utils.data.Dataset``（含增广）、分层划分、K 折、类别均衡采样。

增广只做几何变换 + 小幅亮度/对比度（±10%），**禁止色相/饱和度抖动**（docs/01 §5.6），
且全部在 ``preprocess.to_tensor`` 之前对 RGBA 进行；张量只经 ``to_tensor`` 得到，保证训练/推理一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from oresorter.log import get_logger
from oresorter.vision import preprocess

log = get_logger(__name__)

IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
"""通用目录模式支持的图片后缀（客户现有 a/ b/ c/ 目录多为 bmp）。"""
SAMPLE_EXT = ".png"
"""模型目录 ``samples/{cls}/`` 中的样本固定为 RGBA PNG。"""
TRASH_DIR = ".trash"

Item = tuple[Path, int]
"""``(图片路径, 类别索引)``。"""


def _is_visible_file(p: Path) -> bool:
    return p.is_file() and not p.name.startswith(".")


def _check_class_name(name: str) -> str:
    """类别名只能是一个普通目录名：非空、不以 ``.`` 开头、不含路径分隔符（防止 ``../x`` 逃出根目录）。"""
    if not name or name.startswith(".") or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"非法类别名: {name!r}")
    return name


def worker_init_fn(worker_id: int) -> None:
    """``DataLoader`` worker 初始化：关闭 OpenCV 内部线程池。

    fork 出的 worker 继承父进程的 OpenCV 线程池状态，是已知的死锁来源；每个 worker 只处理一张小图，
    单线程足够。
    """
    cv2.setNumThreads(0)


def list_model_samples(model_dir: str | Path, classes: Sequence[str]) -> dict[str, list[Path]]:
    """列出模型目录下各类别的有效样本（``samples/{cls}/*.png``，排除 ``.trash`` 与临时文件）。

    Args:
        model_dir: 模型目录（含 ``samples/``）。
        classes: 要读取的类别名；目录不存在的类别返回空列表。

    Returns:
        ``{类别: [按文件名排序的路径]}``，键顺序与 ``classes`` 一致。

    Raises:
        ValueError: 类别名非法（空、以 ``.`` 开头或含路径分隔符）。
    """
    root = Path(model_dir) / "samples"
    out: dict[str, list[Path]] = {}
    for cls in classes:
        name = _check_class_name(str(cls))
        d = root / name
        if not d.is_dir():
            out[name] = []
            continue
        out[name] = sorted(
            p for p in d.iterdir() if _is_visible_file(p) and p.suffix.lower() == SAMPLE_EXT
        )
    return out


def list_folder_samples(
    root: str | Path, classes: Sequence[str] | None = None
) -> tuple[list[str], dict[str, list[Path]]]:
    """通用"每个子目录一类"目录（如客户现有 ``a/ b/ c/``，图片 bmp/png/jpg/jpeg）。

    Args:
        root: 根目录。
        classes: 指定类别（子目录名）；省略 = 排序后的全部非隐藏子目录。

    Returns:
        ``(classes, {类别: [路径]})``。

    Raises:
        ValueError: 根目录不存在、没有任何类别子目录、指定的类别目录不存在或类别名非法
            （含路径分隔符等，防止逃出 ``root``）。
    """
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"样本目录不存在: {base}")
    if classes is None:
        names = sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
        if not names:
            raise ValueError(f"目录下没有类别子目录: {base}")
    else:
        names = [_check_class_name(str(c)) for c in classes]
        if len(set(names)) != len(names):
            raise ValueError(f"classes 含重复类别: {names}")
        missing = [c for c in names if not (base / c).is_dir()]
        if missing:
            raise ValueError(f"类别目录不存在: {missing}")
    out: dict[str, list[Path]] = {}
    for name in names:
        d = base / name
        out[name] = sorted(
            p for p in d.iterdir() if _is_visible_file(p) and p.suffix.lower() in IMAGE_EXTS
        )
    return names, out


def load_rgba(
    path: str | Path,
    bg_color: Sequence[int] = (0, 0, 255),
    bg_threshold: float = 40.0,
) -> np.ndarray:
    """读取一张样本为 ``(h, w, 4) uint8`` RGBA。

    - 4 通道文件：BGRA → RGBA，alpha 原样保留；
    - 3 通道 / 灰度文件：转 RGB 后用 ``preprocess.rgb_to_rgba`` 按背景色自动抠 mask。

    Raises:
        ValueError: 文件不存在、无法解码或通道数不受支持。
    """
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {p}")
    if img.dtype != np.uint8:
        # 16 位 PNG 等：按比例压到 8 位，不静默截断高位
        maxv = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
        img = np.clip(img.astype(np.float32) * (255.0 / maxv) + 0.5, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return np.ascontiguousarray(preprocess.rgb_to_rgba(rgb, bg_color, bg_threshold))
    if img.ndim != 3:
        raise ValueError(f"不支持的图片维度 {img.shape}: {p}")
    ch = img.shape[2]
    if ch == 4:
        return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    if ch == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(preprocess.rgb_to_rgba(rgb, bg_color, bg_threshold))
    raise ValueError(f"不支持的通道数 {ch}: {p}")


# --------------------------------------------------------------------------------------
# 增广
# --------------------------------------------------------------------------------------

SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (-0.10, 0.10)
CONTRAST_RANGE = (0.90, 1.10)
UPSCALE_MARGIN_PX = 1.0
"""放大后前景外接框距画布边缘至少保留的像素数（见 :func:`_max_upscale`）。"""


def _max_upscale(rgba: np.ndarray) -> float:
    """以画布中心为缩放中心时，使 alpha 前景的外接框放大后仍完整落在画布内的最大倍率（≥ 1.0）。

    推理时的小图总是带完整 margin、颗粒不触边；若训练增广把颗粒边缘裁掉，训练分布就与推理不一致。
    所以放大倍率取 ``min(SCALE_RANGE[1], 本函数返回值)``。没有前景时不限制。
    """
    h, w = rgba.shape[:2]
    ys, xs = np.nonzero(rgba[:, :, 3] >= preprocess.MASK_THRESHOLD)
    if len(ys) == 0:
        return float("inf")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    # 到画布中心的最远前景像素外沿距离（+0.5 取像素外边界）
    dy = max(cy - float(ys.min()), float(ys.max()) - cy) + 0.5
    dx = max(cx - float(xs.min()), float(xs.max()) - cx) + 0.5
    limit_y = (h / 2.0 - UPSCALE_MARGIN_PX) / dy if dy > 0 else float("inf")
    limit_x = (w / 2.0 - UPSCALE_MARGIN_PX) / dx if dx > 0 else float("inf")
    return max(1.0, min(limit_y, limit_x))


def _rescale_keep_canvas(rgba: np.ndarray, scale: float) -> np.ndarray:
    """按 ``scale`` 缩放内容（RGB 与 alpha 一起），再居中放回原尺寸画布：放大则中心裁切，缩小则补零。

    保持输出形状不变，使缩放真正改变"颗粒相对画幅的大小"（否则 ``to_tensor`` 的补方+缩放会把整体缩放抵消）。
    尺寸不变时直接返回输入本身（调用方须保证传入的是自己的副本）。
    """
    h, w = rgba.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) == (h, w):
        return rgba
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(rgba, (nw, nh), interpolation=interp)
    if scaled.ndim == 2:  # 单通道时 cv2 会降维（此处不会发生，防御）
        scaled = scaled[:, :, None]
    out = np.zeros((h, w, rgba.shape[2]), rgba.dtype)
    # 源区域与目标区域的居中对齐
    sy, sx = max(0, (nh - h) // 2), max(0, (nw - w) // 2)
    dy, dx = max(0, (h - nh) // 2), max(0, (w - nw) // 2)
    ch, cw = min(h, nh), min(w, nw)
    out[dy : dy + ch, dx : dx + cw] = scaled[sy : sy + ch, sx : sx + cw]
    return out


def augment(rgba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """训练增广（docs/01 §6.2）：水平/垂直翻转、随机 90° 旋转、随机缩放 0.8~1.2、亮度/对比度 ±10%。

    - 几何变换对 RGB 与 alpha 同步进行；缩放保持画幅不变（缩小补零、放大中心裁切），但放大倍率
      另受 :func:`_max_upscale` 限制，保证颗粒不被裁出画布；缩放后 alpha 重新二值化到 {0, 255}；
    - 颜色变换只对 alpha > 0 的像素做 ``rgb * contrast + brightness`` 并 clip，**不改色相/饱和度**；
    - 输入数组不会被修改，返回的总是新分配的数组。

    Returns:
        ``(h, w, 4)`` 或（旋转 90°/270° 时）``(w, h, 4)`` 的 uint8 RGBA；方形输入形状不变。
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(f"augment 期望 (h, w, 4) uint8，得到 {rgba.shape} {rgba.dtype}")
    out = rgba
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out[::-1, :]
    k = int(rng.integers(0, 4))
    if k:
        # 旋转 90°/270° 会交换 h/w（非方形小图形状变为 (w, h, 4)），to_tensor 会补方，无需裁回
        out = np.rot90(out, k)
    scale = float(rng.uniform(*SCALE_RANGE))
    # 翻转/旋转只产生视图；这里强制拷贝，后面的原地写入才不会污染调用者的数组
    out = np.array(out, dtype=np.uint8, copy=True, order="C")
    if scale > 1.0:
        scale = min(scale, _max_upscale(out))
    out = _rescale_keep_canvas(out, scale)
    alpha = out[:, :, 3]
    alpha = np.where(alpha >= preprocess.MASK_THRESHOLD, 255, 0).astype(np.uint8)
    out[:, :, 3] = alpha

    contrast = float(rng.uniform(*CONTRAST_RANGE))
    brightness = float(rng.uniform(*BRIGHTNESS_RANGE)) * 255.0
    fg = alpha > 0
    if fg.any():
        rgb = out[:, :, :3].astype(np.float32)
        rgb[fg] = rgb[fg] * contrast + brightness
        out[:, :, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class SampleDataset(Dataset):
    """样本数据集：``__getitem__`` 返回 ``(torch.float32 (3, S, S), int label)``。

    - ``train=True`` 时做增广，随机源为 ``default_rng([seed, epoch, index])``，
      调用 ``set_epoch(e)`` 后每个 epoch 的增广不同且可复现（多 worker 下同样成立）；
    - 张量只通过 ``preprocess.to_tensor`` 得到。
    """

    def __init__(
        self,
        items: Sequence[Item],
        image_size: int,
        *,
        train: bool,
        seed: int = 0,
        bg_color: Sequence[int] = (0, 0, 255),
        bg_threshold: float = 40.0,
    ):
        if image_size < 8:
            raise ValueError(f"image_size 过小: {image_size}")
        self.items: list[Item] = [(Path(p), int(y)) for p, y in items]
        self.image_size = int(image_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.bg_color = tuple(int(c) for c in bg_color)
        self.bg_threshold = float(bg_threshold)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch（影响增广随机数）。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> list[int]:
        return [y for _, y in self.items]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[index]
        rgba = load_rgba(path, self.bg_color, self.bg_threshold)
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, int(index)])
            rgba = augment(rgba, rng)
        x = preprocess.to_tensor(rgba, self.image_size)
        return torch.from_numpy(x), int(label)


# --------------------------------------------------------------------------------------
# 划分与采样
# --------------------------------------------------------------------------------------


def _group_by_label(items: Sequence[Item]) -> dict[int, list[Item]]:
    groups: dict[int, list[Item]] = {}
    for it in items:
        groups.setdefault(int(it[1]), []).append((Path(it[0]), int(it[1])))
    return groups


def stratified_split(
    items: Sequence[Item], val_ratio: float, seed: int = 0
) -> tuple[list[Item], list[Item]]:
    """分层划分训练/验证集。

    每类按 ``val_ratio`` 取整（四舍五入）进验证集，但 **至少 1 个、至多 n-1 个**；
    类只有 1 个样本时该样本进训练集。

    Raises:
        ValueError: ``val_ratio`` 不在 (0, 1) 内或 ``items`` 为空。
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio 必须在 (0, 1) 内，得到 {val_ratio}")
    if len(items) == 0:
        raise ValueError("items 为空")
    rng = np.random.default_rng(seed)
    train_items: list[Item] = []
    val_items: list[Item] = []
    groups = _group_by_label(items)
    for label in sorted(groups):
        group = groups[label]
        n = len(group)
        order = rng.permutation(n)
        if n < 2:
            train_items.extend(group)
            continue
        n_val = round(n * val_ratio)
        n_val = min(max(n_val, 1), n - 1)
        val_items.extend(group[i] for i in order[:n_val])
        train_items.extend(group[i] for i in order[n_val:])
    return train_items, val_items


def kfold_split(
    items: Sequence[Item], k: int, seed: int = 0
) -> list[tuple[list[Item], list[Item]]]:
    """分层 K 折：每类内部打乱后轮转分配到 k 个折。返回 ``[(train, val), ...]``，共 k 项。

    每个样本恰好出现在一个验证折中；``k == len(items)`` 即留一法。

    Raises:
        ValueError: ``k < 2`` 或 ``k > len(items)``。
    """
    n_total = len(items)
    if k < 2:
        raise ValueError(f"k 必须 ≥ 2，得到 {k}")
    if k > n_total:
        raise ValueError(f"k({k}) 不能大于样本数 {n_total}")
    rng = np.random.default_rng(seed)
    fold_of: list[int] = [0] * n_total
    indexed = [(Path(p), int(y)) for p, y in items]
    by_label: dict[int, list[int]] = {}
    for i, (_, y) in enumerate(indexed):
        by_label.setdefault(y, []).append(i)
    offset = 0
    for label in sorted(by_label):
        idxs = by_label[label]
        order = rng.permutation(len(idxs))
        for j, oi in enumerate(order):
            fold_of[idxs[oi]] = (j + offset) % k
        offset += len(idxs)  # 让小类别不总落在前几个折
    folds: list[tuple[list[Item], list[Item]]] = []
    for f in range(k):
        val = [indexed[i] for i in range(n_total) if fold_of[i] == f]
        train = [indexed[i] for i in range(n_total) if fold_of[i] != f]
        folds.append((train, val))
    return folds


def make_weighted_sampler(
    labels: Sequence[int], *, seed: int | None = None
) -> WeightedRandomSampler:
    """类别均衡采样器：每个样本权重 = 1 / 该类样本数，有放回抽 ``len(labels)`` 次。"""
    if len(labels) == 0:
        raise ValueError("labels 为空")
    arr = np.asarray(labels, dtype=np.int64)
    if arr.min() < 0:
        raise ValueError("标签必须 ≥ 0")
    counts = np.bincount(arr)
    weights = 1.0 / counts[arr].astype(np.float64)
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(arr),
        replacement=True,
        generator=gen,
    )
