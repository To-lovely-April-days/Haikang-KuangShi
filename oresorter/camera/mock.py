"""Mock 相机：目录回放或合成蓝底矿石图，按 ``chunk_lines`` 切块输出，模拟行频节奏，可注入丢块/断线。

合成图生成函数（``synth_block`` / ``draw_particle`` / ``CLASS_PALETTE``）也供 ``scripts/make_synthetic.py``
与测试复用，保证训练数据与 Mock 流画风一致。
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from oresorter.camera.base import Camera, CameraDisconnected, CameraError, CameraParams, Frame
from oresorter.config import CameraConfig
from oresorter.log import get_logger

log = get_logger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 合成模式各类别颗粒的基准颜色（RGB）。颜色是主要判据，类间尽量分开。
CLASS_PALETTE: dict[str, tuple[int, int, int]] = {
    "A": (200, 40, 40),
    "B": (40, 160, 60),
    "C": (230, 200, 40),
    "D": (150, 80, 200),
    "E": (240, 140, 30),
    "F": (60, 200, 200),
    "G": (120, 120, 120),
}


def draw_particle(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cx: int,
    cy: int,
    diameter: int,
    color: tuple[int, int, int],
    *,
    noise: float = 12.0,
) -> tuple[int, int, int, int]:
    """在 RGB 画布上画一颗带纹理的椭圆矿石，返回 bbox ``(x, y, w, h)``（裁到画布内）。

    只在以颗粒为中心、边长 2·diameter 的 ROI 内运算，大图上也很快。
    """
    h, w = canvas.shape[:2]
    ax = max(4, int(diameter * rng.uniform(0.35, 0.5)))
    ay = max(4, int(diameter * rng.uniform(0.35, 0.5)))
    angle = float(rng.uniform(0, 180))
    r = max(ax, ay) + 2
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(mask, (int(cx - x0), int(cy - y0)), (ax, ay), angle, 0, 360, 255, -1)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    base = np.array(color, np.float32)
    gain = rng.uniform(0.8, 1.15)  # 整体明暗 + 像素级噪声，保持色相
    pix = base[None, :] * gain + rng.normal(0, noise, size=(len(ys), 3)).astype(np.float32)
    canvas[ys + y0, xs + x0] = np.clip(pix, 0, 255).astype(np.uint8)
    bx0, bx1 = int(xs.min()) + x0, int(xs.max()) + x0
    by0, by1 = int(ys.min()) + y0, int(ys.max()) + y0
    return (bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1)


def synth_block(
    rng: np.random.Generator,
    width: int,
    height: int,
    *,
    bg_color: tuple[int, int, int] = (0, 0, 255),
    n_particles: int = 10,
    diameter_range: tuple[int, int] = (40, 160),
    palette: dict[str, tuple[int, int, int]] | None = None,
    bg_noise: float = 4.0,
    margin: int = 8,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, int, int, int]]]]:
    """合成一块皮带图（RGB uint8），随机撒 ``n_particles`` 颗不同类别的矿石。

    Returns:
        (image, [(class_name, bbox), ...])
    """
    pal = palette or CLASS_PALETTE
    names = list(pal)
    img = np.empty((height, width, 3), np.uint8)
    img[:] = np.array(bg_color, np.uint8)
    if bg_noise > 0:
        # 用一块小噪声贴片纵向平铺（随机行偏移），避免对整幅大图做高斯采样
        tile_h = min(height, 128)
        tile = rng.integers(
            -int(bg_noise), int(bg_noise) + 1, size=(tile_h, width, 3), dtype=np.int16
        )
        reps = height // tile_h + 2
        off = int(rng.integers(0, tile_h))
        noise = np.tile(tile, (reps, 1, 1))[off : off + height]
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    labels: list[tuple[str, tuple[int, int, int, int]]] = []
    for _ in range(n_particles):
        d = int(rng.integers(diameter_range[0], diameter_range[1] + 1))
        cx = int(rng.integers(margin + d // 2, max(margin + d // 2 + 1, width - margin - d // 2)))
        cy = int(rng.integers(margin + d // 2, max(margin + d // 2 + 1, height - margin - d // 2)))
        cls = names[int(rng.integers(0, len(names)))]
        bbox = draw_particle(img, rng, cx, cy, d, pal[cls])
        if bbox[2] > 0:
            labels.append((cls, bbox))
    return img, labels


def load_images(source_dir: str | Path, width: int) -> list[np.ndarray]:
    """读取目录下全部图片为 RGB，按宽度缩放到 ``width``（保持纵横比）。"""
    d = Path(source_dir)
    if not d.is_dir():
        raise CameraError(f"Mock 相机回放目录不存在: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise CameraError(f"Mock 相机回放目录没有图片: {d}")
    out = []
    for p in files:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("无法读取图片，跳过", extra={"path": str(p)})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != width:
            scale = width / rgb.shape[1]
            rgb = cv2.resize(
                rgb, (width, max(1, round(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA
            )
        out.append(np.ascontiguousarray(rgb))
    if not out:
        raise CameraError(f"Mock 相机回放目录没有可读图片: {d}")
    return out


class MockCamera(Camera):
    """Mock 线阵相机。

    - ``cfg.mock.source_dir`` 非空：循环回放目录中的大图（纵向无缝拼接成无限皮带）；
    - 否则：用 ``synth_block`` 合成蓝底矿石图；
    - ``cfg.mock.speed``：1.0 按 ``line_rate_hz`` 节奏出块，0 不限速；
    - 故障注入：``inject_drop(n)`` 丢掉接下来 n 块（行号照常推进），``set_offline(True)`` 模拟掉线。
    """

    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self.m = cfg.mock
        self._params = CameraParams.from_config(cfg)
        self._connected = False
        self._offline = False
        self._streaming = False
        self._chunk_lines = cfg.runtime_chunk_lines
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=self.m.queue_size)
        self._lock = threading.RLock()
        self._belt_lock = threading.Lock()
        self._seq = 0
        self._line = 0
        self._rng = np.random.default_rng(self.m.seed)
        self._snap_rng = np.random.default_rng(self.m.seed + 10_007)
        self._belt = np.zeros((0, self._params.width, 3), np.uint8)
        self._images: list[np.ndarray] = []
        self._img_idx = 0
        self._drop_pending = 0
        self.palette = (
            {k: tuple(v) for k, v in self.m.class_colors.items()}
            if self.m.class_colors
            else dict(CLASS_PALETTE)
        )
        self.bg_color: tuple[int, int, int] = tuple(int(c) for c in self.m.bg_color)  # type: ignore[assignment]

    # ---- 连接 ----
    def open(self) -> None:
        with self._lock:
            if self.m.source_dir:
                self._images = load_images(self.m.source_dir, self._params.width)
                log.info(
                    "Mock 相机回放目录已加载",
                    extra={"camera": self.id, "images": len(self._images)},
                )
            else:
                log.info("Mock 相机使用合成图", extra={"camera": self.id, "seed": self.m.seed})
            self._connected = True
            self._offline = False

    def close(self) -> None:
        self.stop_stream()
        with self._lock:
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected and not self._offline

    @property
    def streaming(self) -> bool:
        return self._streaming

    # ---- 参数 ----
    def get_params(self) -> CameraParams:
        self._require_connected()
        with self._lock:
            return CameraParams(**self._params.to_dict())

    def set_params(self, p: CameraParams) -> CameraParams:
        self._require_connected()
        with self._lock:
            if p.width != self._params.width:
                with self._belt_lock:
                    self._belt = np.zeros((0, p.width, 3), np.uint8)
                if self.m.source_dir:
                    self._images = load_images(self.m.source_dir, p.width)
            self._params = CameraParams(**p.to_dict())
            return CameraParams(**self._params.to_dict())

    def awb_once(self) -> None:
        self._require_connected()
        with self._lock:
            self._params.awb_mode = "off"

    # ---- 采集 ----
    def snapshot(self, height: int) -> np.ndarray:
        self._require_connected()
        with self._lock:
            width = self._params.width
            if self._images:
                rows = []
                total = 0
                idx = self._img_idx
                while total < height:
                    img = self._images[idx % len(self._images)]
                    rows.append(img)
                    total += img.shape[0]
                    idx += 1
                return np.ascontiguousarray(np.concatenate(rows, axis=0)[:height])
            n = max(1, round(self.m.particles_per_chunk * height / max(1, self._chunk_lines)))
            img, _ = synth_block(
                self._snap_rng,
                width,
                height,
                bg_color=self.bg_color,
                n_particles=n,
                diameter_range=(int(self.m.particle_px[0]), int(self.m.particle_px[1])),
                palette=self.palette,
            )
            return img

    def start_stream(self, chunk_lines: int) -> None:
        self._require_connected()
        with self._lock:
            if self._streaming:
                return
            self._chunk_lines = int(chunk_lines)
            self._seq = 0
            self._stop_evt.clear()
            self._drain_queue()
            self._streaming = True
            self._thread = threading.Thread(
                target=self._produce, name=f"mockcam{self.id}", daemon=True
            )
            self._thread.start()

    def stop_stream(self) -> None:
        with self._lock:
            if not self._streaming:
                return
            self._streaming = False
            self._stop_evt.set()
            t = self._thread
            self._thread = None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)

    def get_frame(self, timeout: float) -> Frame | None:
        if not self._connected or self._offline:
            raise CameraDisconnected(f"Mock 相机 {self.id} 已掉线")
        try:
            f = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        self.stats.on_frame(f.ts)
        return f

    def reset_line_counter(self) -> None:
        with self._lock:
            self._line = 0
            self._drain_queue()

    # ---- 故障注入（测试用）----
    def inject_drop(self, n: int) -> None:
        with self._lock:
            self._drop_pending += int(n)

    def set_offline(self, offline: bool) -> None:
        with self._lock:
            was = self._offline
            self._offline = bool(offline)
            if was and not offline:
                self.stats.reconnects += 1

    # ---- 内部 ----
    def _require_connected(self) -> None:
        if not self._connected or self._offline:
            raise CameraDisconnected(f"Mock 相机 {self.id} 未连接")

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _make_block(self, min_rows: int) -> np.ndarray:
        if self._images:
            img = self._images[self._img_idx % len(self._images)]
            self._img_idx += 1
            return img
        height = max(2048, 4 * min_rows)
        n = max(0, round(self.m.particles_per_chunk * height / max(1, self._chunk_lines)))
        img, _ = synth_block(
            self._rng,
            self._params.width,
            height,
            bg_color=self.bg_color,
            n_particles=n,
            diameter_range=(int(self.m.particle_px[0]), int(self.m.particle_px[1])),
            palette=self.palette,
        )
        return img

    def _next_rows(self, n: int) -> np.ndarray:
        """从“无限皮带”取 n 行。块生成在锁外进行，避免阻塞 stop_stream / 参数查询。"""
        while True:
            with self._belt_lock:
                if self._belt.shape[0] >= n:
                    rows = self._belt[:n]
                    self._belt = self._belt[n:]
                    return np.ascontiguousarray(rows)
            blk = self._make_block(n)
            with self._belt_lock:
                if blk.shape[1] != self._belt.shape[1]:
                    self._belt = np.zeros((0, blk.shape[1], 3), np.uint8)
                self._belt = np.concatenate([self._belt, blk], axis=0)

    def _produce(self) -> None:
        chunk = self._chunk_lines
        interval = (
            (chunk / max(1, self._params.line_rate_hz)) / self.m.speed if self.m.speed > 0 else 0.0
        )
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            if self._offline:
                self._stop_evt.wait(0.02)
                next_t = time.monotonic()
                continue
            rows = self._next_rows(chunk)
            with self._lock:
                frame = Frame(
                    seq=self._seq,
                    line_start=self._line,
                    image=rows,
                    ts=time.monotonic(),
                    camera_id=self.id,
                )
                self._seq += 1
                self._line += chunk
                drop = self._drop_pending > 0
                if drop:
                    self._drop_pending -= 1
            if drop:
                self.stats.dropped_chunks += 1
            else:
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    self.stats.dropped_chunks += 1
            if interval > 0:
                next_t += interval
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop_evt.wait(delay)
                else:
                    next_t = time.monotonic()
