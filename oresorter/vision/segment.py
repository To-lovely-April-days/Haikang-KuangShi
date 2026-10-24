"""分割：背景差分 → 阈值 → 形态学 → 连通域 → 带 mask 裁切，输出 ``Particle`` 列表（docs/01 §5.3）。

**本文件的 ``Particle`` 是 vision / runtime / data / api 共用契约**，字段只增不删。
坐标系：bbox / 质心 / trailing_y 均在传入 ``segment()`` 的图像坐标系内（可能是 stitcher 拼好的处理图），
``trailing_line`` 为全局行号，由 ``stitch.ChunkStitcher.select`` 填写。

流程（``segment``）：

1. ``dist = bg.distance(image)``（Lab 颜色距离，见 ``vision.background``）；
2. ``mask = dist > bg_threshold``，再按 ``erode_px`` 腐蚀（去噪）、``dilate_px`` 膨胀（填洞/连接），椭圆核；
3. ``cv2.connectedComponentsWithStats``（8 连通），按 ``[min_area_px, max_area_px]`` 过滤
   （``max_area_px`` 实际在第 4 步之后对每个碎片应用，使粘成一团而超限的合格矿石仍能被拆开，
   见 ``_find_components``）；
4. ``split_touching`` 为真时对每个连通域做 距离变换 → 局部极大作种子 → 鞍点水位分离，拆开粘连颗粒
   （``split_touching_mask``）；
5. 每个连通域生成一个 ``Particle``：bbox 不含 margin；``crop_rgba`` 以 bbox 向外扩 ``margin_px`` 裁切，
   越界部分 RGB 用背景填充、alpha=0；alpha 只对 **本连通域** 的像素为 255；
6. 返回按 ``trailing_y``（尾沿行）升序，同行按 x 升序。

性能：所有逐颗粒操作只在 bbox ROI 上进行，不对整图重复运算。
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from oresorter.log import get_logger
from oresorter.vision.background import ColorBackground, ColumnBackground

if TYPE_CHECKING:
    from oresorter.config import CaptureSettings

log = get_logger(__name__)

Background = ColumnBackground | ColorBackground

SPLIT_MIN_PEAK_PX = 3.0
"""拆粘连：距离变换峰值小于此值（颗粒半径太小）不尝试拆分。"""

SPLIT_SADDLE_RATIO = 0.7
"""拆粘连：一个局部极大只有在它与更高峰之间的分离水位（鞍点 + 1）≤ 该比例 × 自身峰值时才算一颗
独立颗粒，否则视为同一颗矿石轮廓上的凸起。合成实验（默认 erode 1 / dilate 2）：相切/轻微重叠的
两颗圆颗粒比例中位数 0.47、九成以内 ≤ 0.65；"椭圆 + 3 个凸起"的不规则单颗最低 0.75。
待现场图确认的调参项。"""

SPLIT_LOCAL_MAX_RADIUS_PX = 16
"""拆粘连：局部极大搜索窗口在原分辨率上允许的最大半径；更大时先按整数倍 max-pool 下采样再搜索，
使代价与颗粒大小无关（椭圆核膨胀的代价 ∝ 半径²，半径 150 时单次已要 400 ms）。"""


@dataclass(slots=True)
class Particle:
    """一颗矿石。"""

    bbox: tuple[int, int, int, int]
    """``(x, y, w, h)``，不含 margin 外扩，图像坐标。"""
    centroid_x: float
    centroid_y: float
    area: int
    """mask 像素数。"""
    trailing_y: int
    """块内尾沿行 = ``y + h - 1``（最后被扫到的一行）。"""
    crop_rgba: np.ndarray
    """``(h + 2·margin, w + 2·margin, 4) uint8``，alpha 即 mask（255 前景）。"""
    touches_top: bool = False
    touches_bottom: bool = False
    truncated: bool = False
    """对象长度超过 overlap 被截断（stitch 标记）。"""
    trailing_line: int = -1
    """全局尾沿行号（stitch.select 填写；-1 = 未知）。"""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def x_range(self) -> tuple[int, int]:
        """``(x0, x1)`` 闭区间，用于喷头 span。"""
        return self.bbox[0], self.bbox[0] + self.bbox[2] - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "centroid": [round(self.centroid_x, 1), round(self.centroid_y, 1)],
            "area_px": int(self.area),
            "trailing_line": int(self.trailing_line),
        }


@dataclass(slots=True)
class _Component:
    """连通域中间结果（ROI 内的 mask + 统计量）。"""

    label: int
    bbox: tuple[int, int, int, int]
    area: int
    cx: float
    cy: float
    mask: np.ndarray
    """bbox 大小的 bool 掩码。"""
    split_from: int | None = None


# --------------------------------------------------------------------------------------
# 背景与前景掩码
# --------------------------------------------------------------------------------------


def resolve_background(bg: Background | None, settings: CaptureSettings) -> Background:
    """确定分割用的背景对象。

    - ``bg`` 为 ``None``：``color`` 模式用 ``ColorBackground(settings.bg_color)``；``column`` 模式抛 ``ValueError``；
    - ``bg`` 是未学习的 ``ColumnBackground``：抛 ``ValueError``。
    """
    if bg is None:
        if settings.bg_mode == "color":
            return ColorBackground(settings.bg_color)
        raise ValueError("bg_mode 为 column 但未提供列背景，请先学习背景")
    if not bg.ready:
        raise ValueError("列背景尚未学习，请先学习背景")
    return bg


def _ellipse_kernel(radius_px: int) -> np.ndarray:
    size = 2 * int(radius_px) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _check_rgb(image: np.ndarray) -> tuple[int, int]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 (h, w, 3) RGB 图像，得到 shape={getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        raise ValueError(f"期望 uint8 图像，得到 {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("空图像")
    return int(image.shape[0]), int(image.shape[1])


def mask_from_distance(dist: np.ndarray, settings: CaptureSettings) -> np.ndarray:
    """距离图 → 前景掩码 ``uint8 (0/255)``：阈值 → 腐蚀（``erode_px``）→ 膨胀（``dilate_px``），椭圆核。"""
    mask = cv2.compare(dist, float(settings.bg_threshold), cv2.CMP_GT)
    if settings.erode_px > 0:
        mask = cv2.erode(mask, _ellipse_kernel(settings.erode_px))
    if settings.dilate_px > 0:
        mask = cv2.dilate(mask, _ellipse_kernel(settings.dilate_px))
    return mask


def foreground_mask(
    image: np.ndarray, bg: Background | None, settings: CaptureSettings
) -> np.ndarray:
    """前景掩码 ``uint8 (h, w)``，0/255。供预览 / 调参 / ``ColumnBackground.adapt`` 使用。"""
    _check_rgb(image)
    bg = resolve_background(bg, settings)
    return mask_from_distance(bg.distance(image), settings)


# --------------------------------------------------------------------------------------
# 拆粘连
# --------------------------------------------------------------------------------------


def _maxpool(a: np.ndarray, s: int) -> np.ndarray:
    """按 ``s×s`` 块取最大值下采样（右/下边缘补 0 到整数倍）。"""
    h, w = a.shape
    hp, wp = -(-h // s) * s, -(-w // s) * s
    if (hp, wp) != (h, w):
        a = cv2.copyMakeBorder(a, 0, hp - h, 0, wp - w, cv2.BORDER_CONSTANT, value=0)
    return a.reshape(hp // s, s, wp // s, s).max(axis=(1, 3))


def _seed_mask(dist: np.ndarray, fg: np.ndarray, r: int) -> np.ndarray:
    """局部极大种子掩码 ``uint8 (h, w)``。

    窗口半径 ``r`` 内的最大值且 ≥ ``SPLIT_MIN_PEAK_PX`` 的像素，再按半径 ``r // 2`` 膨胀合并相邻极大
    （椭圆中心的平台 / 离散误差不会产生多个种子），最后与前景 ``fg`` 相交。

    ``r > SPLIT_LOCAL_MAX_RADIUS_PX`` 时先按整数倍 ``s`` max-pool 下采样再搜索（块内任一像素达到
    邻域最大值即整块为候选），然后放大回原分辨率。种子只决定"有几个峰、各在哪一带"，峰的精确位置
    之后在原分辨率的 ``dist`` 上重新取 argmax，所以下采样不影响鞍点与边界的计算。
    """
    h, w = dist.shape
    s = max(1, -(-r // SPLIT_LOCAL_MAX_RADIUS_PX))
    small = dist if s == 1 else _maxpool(dist, s)
    local_max = cv2.dilate(small, _ellipse_kernel(max(1, -(-r // s))))
    peaks = cv2.compare(small, local_max, cv2.CMP_GE)
    peaks = cv2.bitwise_and(peaks, cv2.compare(small, SPLIT_MIN_PEAK_PX, cv2.CMP_GE))
    peaks = cv2.dilate(peaks, _ellipse_kernel(max(1, (r // 2) // s)))
    if s > 1:
        peaks = np.ascontiguousarray(np.repeat(np.repeat(peaks, s, axis=0), s, axis=1)[:h, :w])
    return cv2.bitwise_and(peaks, fg)


class _LevelLabels:
    """``dist >= level`` 二值图上各峰所在的连通域编号，按水位缓存（鞍点搜索会反复查询同一水位）。"""

    def __init__(self, dist: np.ndarray, peak_pos: Sequence[tuple[int, int]]) -> None:
        self._dist = dist
        self._peaks = list(peak_pos)
        self._cache: dict[int, list[int]] = {}

    def ids(self, level: int) -> list[int]:
        got = self._cache.get(level)
        if got is None:
            _, cc = cv2.connectedComponents(
                cv2.compare(self._dist, float(level), cv2.CMP_GE),
                connectivity=8,
                ltype=cv2.CV_32S,
            )
            got = [int(cc[y, x]) for y, x in self._peaks]
            self._cache[level] = got
        return got

    def isolated(self, level: int, i: int, others: Sequence[int]) -> bool:
        """峰 ``i`` 在该水位上是否与 ``others`` 中的每个峰都不在同一连通域。"""
        ids = self.ids(level)
        return ids[i] > 0 and all(ids[j] != ids[i] for j in others)

    def lowest_isolating(self, i: int, others: Sequence[int], hi: int) -> int | None:
        """``[1, hi]`` 内使峰 ``i`` 与 ``others`` 分开的最低整数水位（鞍点水位 + 1）；``hi`` 也分不开则 ``None``。

        分离性随水位单调（水位越高连通域越小），故可二分。
        """
        lo = 1
        if hi < lo or not self.isolated(hi, i, others):
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            if self.isolated(mid, i, others):
                hi = mid
            else:
                lo = mid + 1
        return lo


def split_touching_mask(
    roi_mask: np.ndarray, *, max_area: int | None = None
) -> list[np.ndarray] | None:
    """用 距离变换 + 局部极大种子 + 鞍点水位 把一个连通域拆成多个粘连颗粒。

    Args:
        roi_mask: 单个连通域的 bool / uint8 二维掩码（bbox 大小，bbox 外视为背景）。
        max_area: 单颗面积上限（``CaptureSettings.max_area_px``）。给出时，内切圆面积已超过它的实心大块
            （皮带边条、石板、整帧前景）不是合格矿石的粘连，直接判为不可拆。

    Returns:
        拆出的若干 bool 掩码（与输入同形状、互不重叠、并集 = 输入）；不可拆（只有一个独立的峰、太小、
        实心大块）返回 ``None``。

    Raises:
        ValueError: 掩码不是二维。

    算法（等价于在距离图上做标记分水岭，但对凸颗粒的边界位置更稳定）：

    1. 掩码补一圈 0 后做 ``cv2.distanceTransform``（L2）：每个前景像素到边缘的距离，峰值 ``dmax`` ≈
       最大内切圆半径。补圈是因为 bbox 外本就是背景；否则矩形颗粒 / 贯穿处理图的长条 / 整帧前景这类
       掩码里没有 0 像素，``distanceTransform`` 会返回 ≈65535，后续窗口半径随之爆炸；
    2. 局部极大作种子（``_seed_mask``，窗口半径 ``r = round(dmax / 2)``，大窗口在下采样图上搜索）；
       每个种子取其内距离最大的像素作为"峰"；
    3. **鞍点深度判据**：按峰值降序，每个峰二分查找它与所有更高峰分开的最低水位 ``L_i``（= 鞍点 + 1）；
       只有 ``L_i ≤ SPLIT_SADDLE_RATIO × 峰值`` 的峰才算一颗独立颗粒，否则它只是同一颗矿石轮廓上的
       凸起（不规则单颗不过拆）；最高峰总是保留；
    4. 按 ``L_i`` 从高到低，把 ``dist >= L_i`` 中含峰 ``i`` 的连通域（去掉已被更高水位的种子占用的像素）
       作为长大后的种子，即每颗颗粒去掉接触颈部的主体；分离性保证各种子互不重叠且都含自己的峰；
    5. 其余像素按 **最近种子** 归属（``cv2.distanceTransformWithLabels``），并集不丢像素。

    不用 ``cv2.watershed``：它以相邻像素颜色差作为淹没优先级，在平滑的距离图上边界位置取决于
    处理顺序，两颗等大相切圆会被切成明显不等的两半。
    """
    m = np.asarray(roi_mask)
    if m.ndim != 2:
        raise ValueError(f"roi_mask 必须是二维掩码，得到 shape={m.shape}")
    m = np.ascontiguousarray((m > 0).astype(np.uint8) * 255)
    h, w = m.shape
    if h < 3 or w < 3:
        return None
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = np.ascontiguousarray(cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1])
    dmax = float(dist.max())
    if dmax < SPLIT_MIN_PEAK_PX or dmax > max(h, w):  # 后者补圈后不可能出现，纯防御
        return None
    if max_area is not None and math.pi * dmax * dmax > max_area:
        return None
    r = max(1, round(dmax / 2))
    n_seeds, seeds, sstats, _ = cv2.connectedComponentsWithStats(
        _seed_mask(dist, m, r), connectivity=8, ltype=cv2.CV_32S
    )
    if n_seeds <= 2:  # 含背景 0
        return None

    # 每个种子取其内距离最大的像素作为"峰"（只在种子 bbox 内找）
    peak_pos: list[tuple[int, int]] = []
    peak_val: list[float] = []
    for k in range(1, int(n_seeds)):
        sx, sy, sw, sh = (int(v) for v in sstats[k][:4])
        sub = dist[sy : sy + sh, sx : sx + sw]
        masked = np.where(seeds[sy : sy + sh, sx : sx + sw] == k, sub, -1.0)
        dy, dx = divmod(int(np.argmax(masked)), sw)
        peak_pos.append((sy + dy, sx + dx))
        peak_val.append(float(dist[sy + dy, sx + dx]))

    # 鞍点深度判据：峰值降序，每个峰与所有更高峰的最低分离水位 L_i 必须 ≤ ratio × 峰值
    levels = _LevelLabels(dist, peak_pos)
    order = sorted(range(len(peak_val)), key=lambda i: -peak_val[i])
    kept: list[tuple[int, int]] = []  # (L_i, 峰下标)
    for rank, i in enumerate(order):
        if rank == 0:
            continue
        level = levels.lowest_isolating(i, order[:rank], math.floor(peak_val[i]))
        if level is not None and level <= SPLIT_SADDLE_RATIO * peak_val[i]:
            kept.append((level, i))
    if not kept:
        return None
    kept.append((max(lv for lv, _ in kept), order[0]))  # 最高峰：与所有保留峰分开的最低水位

    # 按水位从高到低长出种子，互不重叠
    taken = np.zeros((h, w), bool)
    seed_masks: list[np.ndarray] = []
    for level, i in sorted(kept, key=lambda t: -t[0]):
        _, cc = cv2.connectedComponents(
            cv2.compare(dist, float(level), cv2.CMP_GE), connectivity=8, ltype=cv2.CV_32S
        )
        y, x = peak_pos[i]
        seed = (cc == cc[y, x]) & ~taken
        taken |= seed
        seed_masks.append(seed)

    # 其余像素按最近种子归属
    _, vlabels = cv2.distanceTransformWithLabels(
        cv2.bitwise_not(taken.astype(np.uint8) * 255),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_CCOMP,
    )
    inside = m > 0
    parts: list[np.ndarray] = []
    for seed in seed_masks:
        part = inside & np.isin(vlabels, np.unique(vlabels[seed]))
        if part.any():
            parts.append(part)
    if len(parts) < 2:
        return None
    return parts


def _component_from_mask(
    mask: np.ndarray, x_off: int, y_off: int, label: int, split_from: int | None
) -> _Component:
    """由 ROI 内的 bool 掩码构造 ``_Component``（bbox 收紧到掩码范围）。"""
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return _Component(
        label=label,
        bbox=(x_off + x0, y_off + y0, x1 - x0, y1 - y0),
        area=len(xs),
        cx=float(xs.mean()) + x_off,
        cy=float(ys.mean()) + y_off,
        mask=np.ascontiguousarray(mask[y0:y1, x0:x1]),
        split_from=split_from,
    )


# --------------------------------------------------------------------------------------
# 分割主流程
# --------------------------------------------------------------------------------------


def _find_components(mask: np.ndarray, settings: CaptureSettings) -> list[_Component]:
    """连通域 + 面积过滤 + 可选拆粘连。

    面积过滤顺序：``min_area_px`` 在拆分之前（太小的不值得拆），``max_area_px`` 在拆分 **之后**
    （几颗合格矿石粘成一团可能超过单颗上限，拆开后各碎片再按 ``[min, max]`` 过滤）；
    实心大块由 ``split_touching_mask`` 用内切圆面积与 ``max_area_px`` 比较直接判为不可拆，
    不会为整帧前景这类连通域付出拆分代价。拆分过程中的 OpenCV 异常按"不可拆"处理并告警，
    不让流水线的一个块因此失败。
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    min_area = int(settings.min_area_px)
    max_area = int(settings.max_area_px)
    out: list[_Component] = []
    next_label = int(n)
    for lbl in range(1, int(n)):
        x, y, w, h, area = (int(v) for v in stats[lbl])
        if area < min_area:
            continue
        roi_mask = labels[y : y + h, x : x + w] == lbl
        if settings.split_touching:
            try:
                parts = split_touching_mask(roi_mask, max_area=max_area)
            except cv2.error as exc:
                log.warning(
                    "拆粘连失败，按单个连通域处理",
                    extra={"label": lbl, "bbox": [x, y, w, h], "area": area, "err": str(exc)},
                )
                parts = None
            if parts is not None:
                for p in parts:
                    comp = _component_from_mask(p, x, y, next_label, split_from=lbl)
                    next_label += 1
                    if min_area <= comp.area <= max_area:
                        out.append(comp)
                continue
        if area > max_area:
            continue
        out.append(
            _Component(
                label=lbl,
                bbox=(x, y, w, h),
                area=area,
                cx=float(centroids[lbl][0]),
                cy=float(centroids[lbl][1]),
                mask=roi_mask,
            )
        )
    return out


def _make_crop(image: np.ndarray, comp: _Component, bg: Background, margin: int) -> np.ndarray:
    """按 bbox 外扩 ``margin`` 裁切成 RGBA；越界部分 RGB 用背景填充、alpha=0；alpha 只含本连通域。"""
    img_h, img_w = image.shape[:2]
    x, y, w, h = comp.bbox
    x0, y0, x1, y1 = x - margin, y - margin, x + w + margin, y + h + margin
    crop = np.empty((h + 2 * margin, w + 2 * margin, 4), np.uint8)
    crop[:, :, :3] = bg.row_rgb(x0, x1)[None, :, :]
    crop[:, :, 3] = 0
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, img_w), min(y1, img_h)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, :3] = image[sy0:sy1, sx0:sx1]
    crop[margin : margin + h, margin : margin + w, 3] = comp.mask.astype(np.uint8) * 255
    return crop


def segment(
    image: np.ndarray,
    bg: Background | None,
    settings: CaptureSettings,
) -> list[Particle]:
    """把一幅 RGB 图分割成矿石列表（流程见模块 docstring）。

    Args:
        image: ``(h, w, 3) uint8`` RGB。
        bg: 背景模型；``settings.bg_mode == "color"`` 时可为 None（用 ``settings.bg_color``），
            ``column`` 模式必须给出已学习的 ``ColumnBackground``。
        settings: 采集设置（阈值、形态学、面积过滤、margin、分水岭）。

    Returns:
        按 ``trailing_y`` 升序（同值按 x 升序）的 ``Particle`` 列表；``extra["label"]`` 为连通域编号，
        拆粘连得到的碎片另有 ``extra["split_from"]``（原连通域编号）。

    Raises:
        ValueError: 图像格式不对、背景缺失/未学习、图像宽度与列背景不一致。
    """
    t0 = time.perf_counter()
    img_h, img_w = _check_rgb(image)
    bg = resolve_background(bg, settings)
    mask = mask_from_distance(bg.distance(image), settings)
    comps = _find_components(mask, settings)
    margin = int(settings.margin_px)

    particles: list[Particle] = []
    for comp in comps:
        _, y, _, h = comp.bbox
        extra: dict[str, Any] = {"label": int(comp.label)}
        if comp.split_from is not None:
            extra["split_from"] = int(comp.split_from)
        particles.append(
            Particle(
                bbox=comp.bbox,
                centroid_x=comp.cx,
                centroid_y=comp.cy,
                area=comp.area,
                trailing_y=y + h - 1,
                crop_rgba=_make_crop(image, comp, bg, margin),
                touches_top=(y == 0),
                touches_bottom=(y + h == img_h),
                extra=extra,
            )
        )
    particles.sort(key=lambda p: (p.trailing_y, p.bbox[0]))
    log.debug(
        "分割完成",
        extra={
            "size": [img_w, img_h],
            "particles": len(particles),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        },
    )
    return particles


# --------------------------------------------------------------------------------------
# 辅助：叠加绘制、统计
# --------------------------------------------------------------------------------------


def draw_overlay(
    image: np.ndarray,
    particles: Sequence[Particle],
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """在 RGB 副本上画每颗矿石的 bbox 与编号（从 1 起；``extra["cls"]`` 存在时附带类别），返回副本。

    供 ``/capture/preview`` 与 snapshot overlay 使用；不修改输入。
    """
    _check_rgb(image)
    if thickness < 1:
        raise ValueError(f"thickness 必须 ≥ 1，得到 {thickness}")
    out = np.ascontiguousarray(image.copy())
    img_h, img_w = out.shape[:2]
    col = (int(color[0]), int(color[1]), int(color[2]))
    font_scale = max(0.4, 0.3 * thickness)
    for i, p in enumerate(particles, start=1):
        x, y, w, h = p.bbox
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), col, thickness)
        text = str(i)
        cls = p.extra.get("cls") if p.extra else None
        if cls is not None:
            text = f"{i}:{cls}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y - 4 if y - 4 - th >= 0 else min(img_h - 2, y + th + 4)
        tx = min(max(0, x), max(0, img_w - tw - 1))
        cv2.putText(
            out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness, cv2.LINE_AA
        )
    return out


def particle_stats(particles: Sequence[Particle]) -> dict[str, Any]:
    """``{"count", "area_min", "area_max", "area_mean"}``；空列表时面积项为 0。"""
    if not particles:
        return {"count": 0, "area_min": 0, "area_max": 0, "area_mean": 0.0}
    areas = np.array([p.area for p in particles], np.int64)
    return {
        "count": len(particles),
        "area_min": int(areas.min()),
        "area_max": int(areas.max()),
        "area_mean": round(float(areas.mean()), 1),
    }
