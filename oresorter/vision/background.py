"""背景模型（docs/01 §5.1 / §5.3）：逐列背景 ``ColumnBackground`` 与固定颜色背景 ``ColorBackground``。

线阵相机每一列的响应不同（镜头暗角、像元增益差异），皮带又有纵向纹路，所以 ``column`` 模式
为每一列单独保存一个背景颜色；``color`` 模式用一个固定 RGB（合成图 / 客户现有小图）。

两种背景都提供同样的接口：

- ``distance(image) -> float32 (h, w)``：每个像素到所在列背景的 **Lab 欧氏距离**；
- ``row_rgb(x0, x1) -> uint8 (x1 - x0, 3)``：列区间的背景 RGB（裁切越界填充用）；
- ``width``（``None`` 表示任意宽）、``ready``。

Lab 采用 OpenCV 8 位约定（L ∈ [0, 255]，a/b 偏移 128），与 ``vision.preprocess.mask_from_background``
一致，因此 ``CaptureSettings.bg_threshold``（默认 40）在两处含义相同。
"""

from __future__ import annotations

import argparse
import io
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from oresorter.fsutil import atomic_write_bytes, ensure_dir
from oresorter.log import get_logger

if TYPE_CHECKING:
    from oresorter.config import CaptureSettings

log = get_logger(__name__)

BACKGROUND_FORMAT_VERSION = 1
"""npz 文件格式版本；字段变化时递增，``load`` 拒绝更高版本。"""

MAX_MEDIAN_ROWS = 512
"""``learn`` 时单帧参与列中值统计的最多行数（更高的帧等间隔抽行，省内存省时间）。"""

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_CHANNEL_SUM = np.ones((1, 3), np.float32)

DISTANCE_CHUNK_ROWS = 256
"""``lab_distance`` 每次处理的行数：临时数组只有块大小（4096 宽时每块约 13 MB），而不是整幅
``(h, w, 3) float32``（4096×768 一块的瞬时分配约 150 MB，Jetson 共享内存吃不消）。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _check_image(image: np.ndarray) -> tuple[int, int]:
    """校验 ``(h, w, 3) uint8``，返回 ``(h, w)``。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"期望 (h, w, 3) RGB 图像，得到 shape={getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        raise ValueError(f"期望 uint8 图像，得到 {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("空图像")
    return int(image.shape[0]), int(image.shape[1])


def rgb_row_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """``(n, 3) uint8`` RGB → ``(n, 3) float32`` Lab（OpenCV 8 位约定）。"""
    row = np.ascontiguousarray(rgb_u8, dtype=np.uint8).reshape(1, -1, 3)
    return cv2.cvtColor(row, cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def expand_lab_rows(bg_lab_row: np.ndarray, rows: int) -> np.ndarray:
    """把一行列背景 Lab ``(w, 3)`` / ``(1, w, 3)`` 复制成 ``(rows, w, 3) float32`` 连续数组（供 ``lab_distance`` 分块复用）。"""
    row = np.asarray(bg_lab_row, np.float32).reshape(1, -1, 3)
    return np.ascontiguousarray(np.broadcast_to(row, (int(rows), row.shape[1], 3)))


def lab_distance(image: np.ndarray, bg_lab_block: np.ndarray) -> np.ndarray:
    """每像素与所在列背景的 Lab 欧氏距离，返回 ``float32 (h, w)``。

    Args:
        image: ``(h, w, 3) uint8`` RGB。
        bg_lab_block: 列背景 Lab，``(k, w, 3) float32``，各行相同；``k`` 即分块行数（背景对象缓存一份
            ``DISTANCE_CHUNK_ROWS`` 行的块反复使用）。``k == 1`` 时在本函数内展开成
            ``min(h, DISTANCE_CHUNK_ROWS)`` 行。

    全部用 OpenCV 向量化算子，按 ``k`` 行一块处理：转 Lab → 减背景 → 原地平方 → 三通道求和
    （``cv2.transform``）→ 原地开方写入输出。4096×512 在开发机 CPU 上约 3~4 ms；
    峰值临时内存只有一块的大小。
    """
    h, w = _check_image(image)
    if bg_lab_block.ndim != 3 or bg_lab_block.shape[1:] != (w, 3):
        raise ValueError(f"背景 Lab 块形状 {bg_lab_block.shape} 与图像宽度 {w} 不匹配")
    if bg_lab_block.shape[0] == 1 and h > 1:
        bg_lab_block = expand_lab_rows(bg_lab_block, min(h, DISTANCE_CHUNK_ROWS))
    if bg_lab_block.dtype != np.float32 or not bg_lab_block.flags.c_contiguous:
        bg_lab_block = np.ascontiguousarray(bg_lab_block, np.float32)
    rows = int(bg_lab_block.shape[0])
    out = np.empty((h, w), np.float32)
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        lab = cv2.cvtColor(np.ascontiguousarray(image[y0:y1]), cv2.COLOR_RGB2LAB)
        diff = cv2.subtract(lab, bg_lab_block[: y1 - y0], dtype=cv2.CV_32F)
        cv2.multiply(diff, diff, dst=diff)
        dist = cv2.transform(diff, _CHANNEL_SUM)
        # out[y0:y1] 是连续的行切片，OpenCV 会原地写入
        cv2.sqrt(dist, dst=out[y0:y1])
    return out


class ColumnBackground:
    """逐列背景模型。

    统计方法：每一帧空皮带图 **按列取中值**（抗偶发颗粒/噪点；行数超过 ``MAX_MEDIAN_ROWS`` 时
    等间隔抽行），再对各帧的列中值 **取均值**，得到 ``(w, 3)`` 的 RGB 列背景；同时保存一份由
    其四舍五入后的 uint8 值转换来的 Lab 列背景，供 ``distance`` 用（与图像像素走同一条
    uint8 → Lab 的转换路径，使恰好等于背景色的像素距离为 0）。
    """

    def __init__(self) -> None:
        self._rgb: np.ndarray | None = None  # (w, 3) float32
        self._rgb_u8: np.ndarray | None = None  # (w, 3) uint8
        self._lab: np.ndarray | None = None  # (1, w, 3) float32
        self._lab_block: np.ndarray | None = (
            None  # (DISTANCE_CHUNK_ROWS, w, 3) float32，distance 复用
        )
        self._learned_at: str | None = None
        self._frames_used: int = 0

    # ---- 属性 ----
    @property
    def ready(self) -> bool:
        """是否已学习（可以调用 ``distance``）。"""
        return self._rgb is not None

    @property
    def width(self) -> int | None:
        """列数；未学习时 ``None``。"""
        return None if self._rgb is None else int(self._rgb.shape[0])

    @property
    def learned_at(self) -> str | None:
        """学习完成时间（ISO 8601，本地时区）。"""
        return self._learned_at

    @property
    def frames_used(self) -> int:
        """学习时用到的帧数。"""
        return self._frames_used

    @property
    def rgb(self) -> np.ndarray:
        """``(w, 3) float32`` 列背景 RGB 的只读拷贝。"""
        self._require_ready()
        assert self._rgb is not None
        return self._rgb.copy()

    @property
    def lab(self) -> np.ndarray:
        """``(w, 3) float32`` 列背景 Lab 的只读拷贝。"""
        self._require_ready()
        assert self._lab is not None
        return self._lab[0].copy()

    # ---- 学习 ----
    def learn(self, frames: Iterable[np.ndarray]) -> None:
        """用若干空皮带帧学习列背景。

        Args:
            frames: 每帧 ``(h, w, 3) uint8`` RGB，高度可不同，宽度必须一致。

        Raises:
            ValueError: 没有帧、帧格式不对或宽度不一致。
        """
        acc: np.ndarray | None = None
        n = 0
        width = -1
        for frame in frames:
            h, w = _check_image(frame)
            if width < 0:
                width = w
            elif w != width:
                raise ValueError(f"学习帧宽度不一致: {w} != {width}")
            if h > MAX_MEDIAN_ROWS:
                idx = np.linspace(0, h - 1, MAX_MEDIAN_ROWS).round().astype(np.int64)
                frame = frame[idx]
            med = np.median(frame, axis=0)  # (w, 3) float64
            acc = med if acc is None else acc + med
            n += 1
        if acc is None or n == 0:
            raise ValueError("学习背景至少需要 1 帧空皮带图")
        rgb = (acc / n).astype(np.float32)
        self._set_rgb(rgb)
        self._learned_at = _now_iso()
        self._frames_used = n
        log.info(
            "列背景学习完成",
            extra={"width": width, "frames": n, "mean_rgb": rgb.mean(axis=0).round(1).tolist()},
        )

    def _set_rgb(self, rgb: np.ndarray) -> None:
        rgb = np.asarray(rgb, np.float32).reshape(-1, 3)
        self._rgb = rgb
        self._rgb_u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        self._lab = rgb_row_to_lab(self._rgb_u8)[None]
        self._lab_block = None  # 背景变了，分块缓存作废（下次 distance 时重建）

    def _require_ready(self) -> None:
        if not self.ready:
            raise ValueError("列背景尚未学习，请先学习背景")

    # ---- 使用 ----
    def distance(self, image: np.ndarray) -> np.ndarray:
        """每像素与所在列背景的 Lab 欧氏距离，``float32 (h, w)``。

        Raises:
            ValueError: 未学习，或图像宽度与背景列数不一致。
        """
        self._require_ready()
        assert self._lab is not None
        _, w = _check_image(image)
        if w != self._lab.shape[1]:
            raise ValueError(f"图像宽度 {w} 与背景列数 {self._lab.shape[1]} 不一致，请重新学习背景")
        if self._lab_block is None:
            self._lab_block = expand_lab_rows(self._lab, DISTANCE_CHUNK_ROWS)
        return lab_distance(image, self._lab_block)

    def row_rgb(self, x0: int, x1: int) -> np.ndarray:
        """列区间 ``[x0, x1)`` 的背景 RGB ``(x1 - x0, 3) uint8``；越界列取最近的有效列。"""
        self._require_ready()
        assert self._rgb_u8 is not None
        if x1 <= x0:
            return np.zeros((0, 3), np.uint8)
        idx = np.clip(np.arange(x0, x1), 0, self._rgb_u8.shape[0] - 1)
        return self._rgb_u8[idx]

    def adapt(self, image: np.ndarray, fg_mask: np.ndarray, rate: float = 0.01) -> int:
        """慢速自适应：只在本帧 **完全没有前景** 的列上，用该列均值对背景做 EMA 更新。

        Args:
            image: ``(h, w, 3) uint8``。
            fg_mask: ``(h, w)`` 前景掩码（非零 = 前景），通常是 ``segment.foreground_mask`` 的输出。
            rate: EMA 系数 ``(0, 1]``；新背景 = ``(1 - rate)·旧 + rate·列均值``。

        Returns:
            被更新的列数。
        """
        self._require_ready()
        assert self._rgb is not None
        h, w = _check_image(image)
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"rate 必须在 (0, 1]，得到 {rate}")
        if w != self._rgb.shape[0]:
            raise ValueError(f"图像宽度 {w} 与背景列数 {self._rgb.shape[0]} 不一致")
        fg_mask = np.asarray(fg_mask)
        if fg_mask.ndim != 2 or fg_mask.shape != (h, w):
            raise ValueError(
                f"fg_mask 必须是 (h, w) 二维掩码，得到 {fg_mask.shape}，图像为 {(h, w)}"
            )
        clean_cols = ~fg_mask.astype(bool).any(axis=0)
        n = int(clean_cols.sum())
        if n == 0:
            return 0
        col_mean = image[:, clean_cols].mean(axis=0, dtype=np.float64).astype(np.float32)
        rgb = self._rgb.copy()
        rgb[clean_cols] = (1.0 - rate) * rgb[clean_cols] + rate * col_mean
        self._set_rgb(rgb)
        return n

    # ---- 持久化 ----
    def save(self, path: str | Path) -> None:
        """原子保存为 npz（``version/width/learned_at/frames_used/rgb/lab``）。"""
        self._require_ready()
        assert self._rgb is not None and self._lab is not None
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            version=np.int32(BACKGROUND_FORMAT_VERSION),
            width=np.int32(self._rgb.shape[0]),
            learned_at=np.str_(self._learned_at or ""),
            frames_used=np.int32(self._frames_used),
            rgb=self._rgb,
            lab=self._lab[0],
        )
        atomic_write_bytes(path, buf.getvalue())
        log.info("列背景已保存", extra={"path": str(path), "width": self.width})

    @classmethod
    def load(cls, path: str | Path) -> ColumnBackground:
        """从 ``save`` 产生的 npz 读取。文件不存在/格式不对/版本过高时抛 ``ValueError``。"""
        p = Path(path)
        if not p.exists():
            raise ValueError(f"背景文件不存在: {p}")
        try:
            with np.load(p, allow_pickle=False) as z:
                version = int(z["version"])
                if version > BACKGROUND_FORMAT_VERSION:
                    raise ValueError(
                        f"背景文件版本 {version} 高于支持的 {BACKGROUND_FORMAT_VERSION}"
                    )
                rgb = np.asarray(z["rgb"], np.float32)
                width = int(z["width"])
                learned_at = str(z["learned_at"]) or None
                frames_used = int(z["frames_used"])
        except (KeyError, OSError, ValueError, EOFError) as exc:
            if isinstance(exc, ValueError) and "版本" in str(exc):
                raise
            raise ValueError(f"背景文件损坏或格式不对: {p} ({exc})") from exc
        if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] != width or width <= 0:
            raise ValueError(f"背景文件内容不一致: rgb={rgb.shape}, width={width}")
        obj = cls()
        obj._set_rgb(rgb)
        obj._learned_at = learned_at
        obj._frames_used = frames_used
        return obj


class ColorBackground:
    """固定颜色背景（``bg_mode == "color"``）：所有列共用 ``bg_color`` 的 Lab 值。"""

    def __init__(self, bg_color: Sequence[int]) -> None:
        if len(bg_color) != 3 or any(not (0 <= int(c) <= 255) for c in bg_color):
            raise ValueError(f"bg_color 必须是 3 个 0~255 的整数，得到 {list(bg_color)}")
        self.bg_color: tuple[int, int, int] = (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))
        self._lab = rgb_row_to_lab(np.array([self.bg_color], np.uint8))[0]  # (3,) float32
        self._lab_block: np.ndarray | None = None  # (DISTANCE_CHUNK_ROWS, w, 3)，按最近一次宽度缓存

    @property
    def ready(self) -> bool:
        return True

    @property
    def width(self) -> None:
        """``None``：适用于任意宽度。"""
        return None

    @property
    def lab(self) -> np.ndarray:
        """背景色的 Lab ``(3,) float32``。"""
        return self._lab.copy()

    def distance(self, image: np.ndarray) -> np.ndarray:
        """每像素与 ``bg_color`` 的 Lab 欧氏距离，``float32 (h, w)``。"""
        _, w = _check_image(image)
        if self._lab_block is None or self._lab_block.shape[1] != w:
            row = np.broadcast_to(self._lab[None, :], (w, 3))
            self._lab_block = expand_lab_rows(row, DISTANCE_CHUNK_ROWS)
        return lab_distance(image, self._lab_block)

    def row_rgb(self, x0: int, x1: int) -> np.ndarray:
        """``(x1 - x0, 3) uint8``，全部为 ``bg_color``。"""
        n = max(0, x1 - x0)
        out = np.empty((n, 3), np.uint8)
        out[:] = np.array(self.bg_color, np.uint8)
        return out


Background = ColumnBackground | ColorBackground
"""分割接受的背景类型。"""


def make_background(
    settings: CaptureSettings, column_bg: ColumnBackground | None
) -> ColumnBackground | ColorBackground:
    """按 ``settings.bg_mode`` 选背景对象。

    - ``color``：返回 ``ColorBackground(settings.bg_color)``；
    - ``column``：返回 ``column_bg``；为 ``None`` 或未学习时抛 ``ValueError``（"请先学习背景"）。
    """
    if settings.bg_mode == "color":
        return ColorBackground(settings.bg_color)
    if settings.bg_mode == "column":
        if column_bg is None or not column_bg.ready:
            raise ValueError("bg_mode 为 column 但列背景尚未学习，请先学习背景")
        return column_bg
    raise ValueError(f"未知 bg_mode: {settings.bg_mode!r}")


def background_path(data_dir: str | Path, camera_id: int) -> Path:
    """``{data_dir}/background/cam{id}.npz``。"""
    return Path(data_dir) / "background" / f"cam{int(camera_id)}.npz"


def load_images_rgb(folder: str | Path) -> list[np.ndarray]:
    """读取目录下全部图片为 RGB uint8（按文件名排序）。没有可读图片时抛 ``ValueError``。"""
    d = Path(folder)
    if not d.is_dir():
        raise ValueError(f"图片目录不存在: {d}")
    out: list[np.ndarray] = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("无法读取图片，跳过", extra={"path": str(p)})
            continue
        out.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not out:
        raise ValueError(f"目录中没有可读图片: {d}")
    return out


def learn_bg_cli(args: argparse.Namespace) -> int:
    """``cli learn-bg`` 实现：从相机快照或图片目录学习列背景并保存到 ``{data_dir}/background/cam{id}.npz``。

    参数（见 ``oresorter.cli``）：``args.config``、``args.camera``、``args.frames``（覆盖
    ``capture.bg_learn_frames``）、``args.images``（给出时从目录学习，不碰相机）。
    返回进程退出码：0 成功，1 失败（原因写日志）。
    """
    from oresorter.runtime.context import AppContext

    try:
        ctx = AppContext.from_yaml(args.config, persist=False)
    except Exception as exc:
        log.error("加载配置失败", extra={"config": str(args.config), "err": str(exc)})
        return 1
    cfg = ctx.cfg
    camera_arg = getattr(args, "camera", None)
    camera_id = 1 if camera_arg is None else int(camera_arg)
    if camera_id < 1:
        log.error("相机编号必须 ≥ 1", extra={"camera": camera_id})
        return 1
    frames_arg = getattr(args, "frames", None)
    n_frames = cfg.capture.bg_learn_frames if frames_arg is None else int(frames_arg)
    if n_frames < 1:
        log.error("帧数必须 ≥ 1", extra={"frames": n_frames})
        return 1
    images_dir = getattr(args, "images", None)

    bg = ColumnBackground()
    t0 = time.perf_counter()
    try:
        if images_dir:
            frames = load_images_rgb(images_dir)[:n_frames]
            log.info("从目录学习背景", extra={"dir": str(images_dir), "frames": len(frames)})
            bg.learn(frames)
        else:
            cam = ctx.cameras.get(camera_id)
            if cam is None:
                log.error("相机不存在", extra={"camera": camera_id, "known": list(ctx.cameras)})
                return 1
            height = cam.cfg.height
            log.info(
                "从相机学习背景", extra={"camera": camera_id, "frames": n_frames, "height": height}
            )
            cam.open()
            try:
                bg.learn(cam.snapshot(height) for _ in range(n_frames))
            finally:
                cam.close()
    except Exception as exc:
        log.error("学习背景失败", extra={"camera": camera_id, "err": str(exc)})
        return 1

    out = background_path(cfg.system.data_dir, camera_id)
    try:
        ensure_dir(out.parent)
        bg.save(out)
    except OSError as exc:
        log.error("保存背景失败", extra={"path": str(out), "err": str(exc)})
        return 1
    log.info(
        "learn-bg 完成",
        extra={
            "camera": camera_id,
            "path": str(out),
            "width": bg.width,
            "frames": bg.frames_used,
            "seconds": round(time.perf_counter() - t0, 2),
        },
    )
    return 0
