"""样本库：写入、列表、软删除/撤销/清空、缩略图、过期清理、类别改名迁移（docs/01 §8、docs/02 §3.4、docs/03 §9）。

目录布局（``model_dir`` = ``{data_dir}/models/{id}``）::

    samples/{cls}/{stem}.png          RGBA PNG，alpha 即 mask
    samples/{cls}/{stem}.json         元数据（camera_id、bbox、area_px、created_at、seq、batch、extra）
    samples/.trash/{cls}/{stem}.png   软删除后的 png/json
    samples/.trash/journal.json       删除批次栈 {"next": n, "batches": [...]}
    thumbs/{cls}/{stem}.jpg           缩略图（长边 thumb_size）
    thumbs/.trash/{cls}/{stem}.jpg    软删除后的缩略图

文件名 ``cam{camera_id}_{YYYYmmdd_HHMMSS}_{seq:06d}``：``seq`` 是仓库级单调计数器，
启动时扫描全部 png（含 .trash）取最大值初始化；``seq`` 与时间戳在同一临界区内取得，
保证同秒/并发写入文件名唯一、且 seq 顺序 == 写入顺序。

约定：
- 样本 json 不存 sample_id / 类别名等可由路径推导的字段：整目录复制（备份至）或类别改名后
  无需逐个重写 json。
- 列表排序按 seq 升序（最早写入在前）。seq 单调递增、与写入顺序一致，不受系统时钟回跳
  （Jetson 无 RTC / NTP 校时）影响；created_at 仅作展示。排序键直接取自文件名，分页无需读 json。
  同一 stem 同时出现在正常目录与 .trash（移动中断/人工操作）时以正常目录为准，只列一条。
- 软删除时把 png/json 移到 ``samples/.trash/{cls}/``、jpg 移到 ``thumbs/.trash/{cls}/``，
  并用 ``os.utime`` 把 mtime 刷成删除时刻，``purge`` 据此判断过期。
- journal 批次记录完整 sample_id，但匹配/移除一律按 ``(cls, stem)``，模型目录被整体复制到
  别的编号后仍能对上；``rewrite_journal_model_id`` 供“备份至”把前缀改为目标编号。
- 一次 ``soft_delete`` 只允许同一类别（HMI 的删除都在某一类别视图内进行），作为一个可撤销批次；
  ids 去重。``undo_last`` 弹出的批次若一张都恢复不了（文件已被外部清理）视为僵尸批次丢弃，
  继续找更早的批次；全部为空才抛 NOT_FOUND。
- ``clear(cls)`` 该类无样本时返回 ``(0, None)``，不产生批次。
- 类别名校验：``add`` 必须在 ``class_names`` 内；查看/删除/恢复/清空还接受磁盘上已存在的
  历史类别目录（类别改名/配置变更后遗留的数据仍可查看与清理）。类别改名用 ``rename_class``
  整体迁移（四个目录 + journal）。``counts()`` 只统计 ``class_names``，遗留目录见 ``orphan_classes()``。
- 所有公共方法由一把 ``RLock`` 保护。文件系统失败统一抛 ``StoreError(INTERNAL)``。
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oresorter.fsutil import (
    atomic_write_bytes,
    atomic_write_json,
    disk_free_mb,
    ensure_dir,
    read_json,
)
from oresorter.log import get_logger

log = get_logger(__name__)

TRASH_DIR = ".trash"
JOURNAL_NAME = "journal.json"
MAX_LIST_LIMIT = 1000
THUMB_JPEG_QUALITY = 85

# cam{camera_id}_{YYYYmmdd_HHMMSS}_{seq}
_STEM_RE = re.compile(r"^cam(\d+)_(\d{8}_\d{6})_(\d+)$")
# 类别名 / stem 不能含路径分隔符、不能以 . 开头
_SEGMENT_RE = re.compile(r"^[^/\\\x00]+$")


class StoreError(Exception):
    """数据层错误，与 ``config.ConfigError`` 同形。``code`` ∈ VALIDATION / NOT_FOUND / CONFLICT / INTERNAL。"""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


# --------------------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------------------


@dataclass
class Sample:
    """一张样本（文件三件套：png / json / jpg）。"""

    model_id: int
    cls: str
    stem: str
    path: Path
    """RGBA PNG 路径。"""
    json_path: Path
    thumb_path: Path
    created_at: str
    """ISO 8601 带时区。"""
    camera_id: int
    bbox: list[int]
    """原图上的 ``[x, y, w, h]``。"""
    area_px: int
    deleted: bool = False
    batch: str | None = None
    """采集批次号（如 capture 任务 id），非删除批次。"""
    seq: int = 0

    @property
    def sample_id(self) -> str:
        """``{model_id}/{cls}/{stem}``。"""
        return f"{self.model_id}/{self.cls}/{self.stem}"

    def to_dict(self) -> dict[str, Any]:
        """docs/03 §9 样本项。URL 按实际位置生成：已软删除的样本带 ``.trash/`` 段，HMI 仍可显示。"""
        trash = f"{TRASH_DIR}/" if self.deleted else ""
        base = f"/static/models/{self.model_id}"
        return {
            "sample_id": self.sample_id,
            "thumb_url": f"{base}/thumbs/{trash}{self.cls}/{self.stem}.jpg",
            "image_url": f"{base}/samples/{trash}{self.cls}/{self.stem}.png",
            "created_at": self.created_at,
            "camera_id": self.camera_id,
            "bbox": list(self.bbox),
            "area_px": int(self.area_px),
            "deleted": self.deleted,
        }


def _check_segment(value: Any, what: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or not _SEGMENT_RE.match(value)
    ):
        raise StoreError("VALIDATION", f"非法的{what}: {value!r}", {what: value})


def validate_class_name(name: Any) -> None:
    """类别名校验：非空字符串、不含路径分隔符、不以 ``.`` 开头；非法 → ``StoreError(VALIDATION)``。"""
    _check_segment(name, "类别名")


def parse_sample_id(sample_id: str) -> tuple[int, str, str]:
    """解析 ``"{model_id}/{cls}/{stem}"``；非法 → ``StoreError(VALIDATION)``。"""
    if not isinstance(sample_id, str):
        raise StoreError("VALIDATION", "sample_id 必须是字符串", {"sample_id": sample_id})
    parts = sample_id.split("/")
    if len(parts) != 3:
        raise StoreError(
            "VALIDATION",
            "sample_id 格式应为 {model_id}/{cls}/{stem}",
            {"sample_id": sample_id},
        )
    mid_s, cls, stem = parts
    if not mid_s.isdigit() or int(mid_s) <= 0:
        raise StoreError(
            "VALIDATION", f"sample_id 中模型编号非法: {mid_s!r}", {"sample_id": sample_id}
        )
    _check_segment(cls, "类别名")
    _check_segment(stem, "文件名")
    if stem.endswith((".png", ".json", ".jpg")):
        raise StoreError("VALIDATION", "sample_id 中文件名不应带扩展名", {"sample_id": sample_id})
    return int(mid_s), cls, stem


def _id_key(sample_id: Any) -> tuple[str, str] | None:
    """journal 中的 id → ``(cls, stem)``（忽略模型前缀）；非法返回 None。"""
    try:
        _, cls, stem = parse_sample_id(sample_id)
    except StoreError:
        return None
    return cls, stem


def rewrite_journal_model_id(journal_path: str | Path, model_id: int) -> int:
    """把 journal 中全部批次 ids 的模型前缀改为 ``model_id``（“备份至”整目录复制后调用）。

    Returns:
        改写的 id 数。journal 不存在/损坏时不改动，返回 0；非法 id 会被丢弃。
    """
    path = Path(journal_path)
    try:
        data = read_json(path, default=None)
    except ValueError:
        log.error("journal.json 损坏，跳过前缀改写", extra={"path": str(path)})
        return 0
    if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
        return 0
    changed = 0
    for b in data["batches"]:
        if not isinstance(b, dict):
            continue
        new_ids: list[str] = []
        for sid in b.get("ids", []) if isinstance(b.get("ids"), list) else []:
            key = _id_key(sid)
            if key is None:
                changed += 1
                continue
            new = f"{model_id}/{key[0]}/{key[1]}"
            if new != sid:
                changed += 1
            new_ids.append(new)
        b["ids"] = new_ids
    if changed:
        atomic_write_json(path, data)
    return changed


def _parse_stem(stem: str) -> tuple[int | None, str | None, int | None]:
    """从文件名解析 ``(camera_id, 时间戳串, seq)``；不匹配时全为 None。"""
    m = _STEM_RE.match(stem)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_thumbnail(crop_rgba: np.ndarray, size: int) -> np.ndarray:
    """RGBA 小图 → BGR 缩略图（长边恰为 ``size``），背景用 alpha 合成到黑色。"""
    h, w = crop_rgba.shape[:2]
    rgb = crop_rgba[..., :3].astype(np.float32)
    alpha = crop_rgba[..., 3:4].astype(np.float32) / 255.0
    comp = np.clip(rgb * alpha, 0, 255).astype(np.uint8)
    if h >= w:
        new_h, new_w = size, max(1, round(w * size / h))
    else:
        new_w, new_h = size, max(1, round(h * size / w))
    interp = cv2.INTER_AREA if (new_h < h or new_w < w) else cv2.INTER_LINEAR
    resized = cv2.resize(comp, (new_w, new_h), interpolation=interp)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


def _encode(ext: str, image: np.ndarray, params: list[int] | None = None) -> bytes:
    ok, buf = cv2.imencode(ext, image, params or [])
    if not ok:
        raise StoreError("INTERNAL", f"图像编码失败: {ext}", {"ext": ext})
    return buf.tobytes()


# --------------------------------------------------------------------------------------
# SampleRepo
# --------------------------------------------------------------------------------------


class SampleRepo:
    """某个模型目录下的样本库。线程安全（一把 RLock）。"""

    def __init__(
        self,
        model_dir: Path,
        class_names: list[str],
        *,
        thumb_size: int = 160,
        min_free_mb: int = 0,
    ):
        """
        Args:
            model_dir: ``{data_dir}/models/{id}``，目录名必须是模型编号。
            class_names: 允许写入的类别名（顺序即 ``counts()`` 的输出顺序）。
            thumb_size: 缩略图长边像素。
            min_free_mb: >0 时写入前检查剩余磁盘空间，不足则拒绝。
        """
        self.model_dir = Path(model_dir)
        if not self.model_dir.name.isdigit():
            raise StoreError(
                "VALIDATION",
                f"模型目录名必须是编号: {self.model_dir.name!r}",
                {"dir": str(model_dir)},
            )
        if thumb_size < 16 or thumb_size > 1024:
            raise StoreError("VALIDATION", "thumb_size 应在 16~1024", {"thumb_size": thumb_size})
        if min_free_mb < 0:
            raise StoreError("VALIDATION", "min_free_mb 不能为负", {"min_free_mb": min_free_mb})
        self.model_id = int(self.model_dir.name)
        self.class_names = list(class_names)
        self.thumb_size = int(thumb_size)
        self.min_free_mb = int(min_free_mb)
        self.samples_dir = self.model_dir / "samples"
        self.thumbs_dir = self.model_dir / "thumbs"
        self.trash_dir = self.samples_dir / TRASH_DIR
        self.thumb_trash_dir = self.thumbs_dir / TRASH_DIR
        self.journal_path = self.trash_dir / JOURNAL_NAME
        self._lock = threading.RLock()
        self._next_seq = self._scan_max_seq() + 1

    # ---------------------------------------------------------------- 路径
    def class_dir(self, cls: str, deleted: bool = False) -> Path:
        """png/json 所在目录。"""
        return (self.trash_dir if deleted else self.samples_dir) / cls

    def thumb_dir(self, cls: str, deleted: bool = False) -> Path:
        """jpg 所在目录。"""
        return (self.thumb_trash_dir if deleted else self.thumbs_dir) / cls

    def _paths(self, cls: str, stem: str, deleted: bool) -> tuple[Path, Path, Path]:
        d = self.class_dir(cls, deleted)
        return d / f"{stem}.png", d / f"{stem}.json", self.thumb_dir(cls, deleted) / f"{stem}.jpg"

    # ---------------------------------------------------------------- 类别
    def set_class_names(self, class_names: list[str]) -> None:
        """配置热更新后同步类别名（不移动文件；改名请用 ``rename_class``）。"""
        for c in class_names:
            validate_class_name(c)
        with self._lock:
            self.class_names = list(class_names)

    def orphan_classes(self) -> list[str]:
        """磁盘上存在但不在 ``class_names`` 中的类别目录（改名/配置变更遗留），升序。"""
        found: set[str] = set()
        with self._lock:
            for d in (self.samples_dir, self.trash_dir):
                if not d.is_dir():
                    continue
                for cls_dir in d.iterdir():
                    if cls_dir.is_dir() and not cls_dir.name.startswith("."):
                        found.add(cls_dir.name)
        return sorted(found - set(self.class_names))

    def _check_cls(self, cls: str, *, for_add: bool = False) -> None:
        """类别名校验。``for_add`` 时必须在 ``class_names`` 中；其余操作还接受磁盘上已存在的历史类别目录。"""
        if isinstance(cls, str) and cls in self.class_names:
            return
        if (
            not for_add
            and isinstance(cls, str)
            and cls
            and not cls.startswith(".")
            and _SEGMENT_RE.match(cls)
            and (self.class_dir(cls, False).is_dir() or self.class_dir(cls, True).is_dir())
        ):
            return
        raise StoreError(
            "VALIDATION",
            f"未知类别: {cls!r}",
            {"cls": cls, "class_names": list(self.class_names)},
        )

    def _check_owner(self, sample_id: str) -> tuple[str, str]:
        mid, cls, stem = parse_sample_id(sample_id)
        if mid != self.model_id:
            raise StoreError(
                "VALIDATION",
                f"样本 {sample_id} 不属于模型 {self.model_id}",
                {"sample_id": sample_id, "model_id": self.model_id},
            )
        return cls, stem

    # ---------------------------------------------------------------- seq
    def _scan_max_seq(self) -> int:
        best = 0
        for d in (self.samples_dir, self.trash_dir):
            if not d.is_dir():
                continue
            for cls_dir in d.iterdir():
                if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                    continue
                for p in cls_dir.glob("*.png"):
                    _, _, seq = _parse_stem(p.stem)
                    if seq is not None and seq > best:
                        best = seq
        return best

    def refresh_seq(self) -> int:
        """目录内容被外部整体替换（如“备份至”覆盖了本模型）后重扫 seq；计数器只增不减。

        Returns:
            下一个将分配的 seq。
        """
        with self._lock:
            self._next_seq = max(self._next_seq, self._scan_max_seq() + 1)
            return self._next_seq

    # ---------------------------------------------------------------- journal
    def _load_journal(self) -> dict[str, Any]:
        try:
            data = read_json(self.journal_path, default=None)
        except ValueError:
            log.error("journal.json 损坏，按空处理", extra={"path": str(self.journal_path)})
            data = None
        if not isinstance(data, dict):
            data = {}
        nxt = data.get("next")
        batches = data.get("batches")
        return {
            "next": nxt if isinstance(nxt, int) and nxt >= 1 else 1,
            "batches": (
                [b for b in batches if isinstance(b, dict)] if isinstance(batches, list) else []
            ),
        }

    def _save_journal(self, journal: dict[str, Any]) -> None:
        atomic_write_json(self.journal_path, journal)

    @staticmethod
    def _batch_keys(batch: dict[str, Any]) -> list[tuple[str, str]]:
        ids = batch.get("ids")
        if not isinstance(ids, list):
            return []
        return [k for k in map(_id_key, ids) if k is not None]

    def _drop_from_journal(self, journal: dict[str, Any], keys: set[tuple[str, str]]) -> None:
        """从各批次移除 ``(cls, stem)`` 命中的 id（按类别+文件名匹配，不看模型前缀）；空批次删除。"""
        kept = []
        for b in journal["batches"]:
            ids = b.get("ids") if isinstance(b.get("ids"), list) else []
            b_ids = [i for i in ids if _id_key(i) is not None and _id_key(i) not in keys]
            if b_ids:
                b["ids"] = b_ids
                kept.append(b)
        journal["batches"] = kept

    # ---------------------------------------------------------------- 写入
    def add(self, cls: str, crop_rgba: np.ndarray, meta: dict[str, Any] | None = None) -> Sample:
        """写入一张样本（png + json + jpg，均原子写）。

        Args:
            cls: 类别名，须在 ``class_names`` 中。
            crop_rgba: ``(h, w, 4) uint8``，alpha 即 mask。
            meta: ``camera_id``（int 或 numpy 整数）、``bbox``([x,y,w,h])、``area_px``、``batch``
                及任意附加字段（附加字段落在 json 的 ``extra`` 下）。

        Raises:
            StoreError: VALIDATION（类别/数组/元数据非法）、CONFLICT（磁盘空间不足）、
                INTERNAL（文件写入失败，三件套已清理）。
        """
        self._check_cls(cls, for_add=True)
        if not isinstance(crop_rgba, np.ndarray) or crop_rgba.ndim != 3 or crop_rgba.shape[2] != 4:
            raise StoreError(
                "VALIDATION",
                "crop_rgba 必须是 (h, w, 4) 数组",
                {"shape": list(getattr(crop_rgba, "shape", ()))},
            )
        if crop_rgba.dtype != np.uint8:
            raise StoreError(
                "VALIDATION", "crop_rgba 必须是 uint8", {"dtype": str(crop_rgba.dtype)}
            )
        h, w = crop_rgba.shape[:2]
        if h == 0 or w == 0:
            raise StoreError("VALIDATION", "crop_rgba 不能为空图", {"shape": [h, w]})
        meta = dict(meta or {})

        camera_id = meta.pop("camera_id", 0)
        if (
            isinstance(camera_id, bool)
            or not isinstance(camera_id, (int, np.integer))
            or camera_id < 0
        ):
            raise StoreError("VALIDATION", "camera_id 必须是非负整数", {"camera_id": camera_id})
        camera_id = int(camera_id)
        bbox = meta.pop("bbox", None)
        if bbox is None:
            bbox = [0, 0, int(w), int(h)]
        try:
            bbox = [int(v) for v in bbox]
        except (TypeError, ValueError):
            bbox = []
        if len(bbox) != 4:
            raise StoreError("VALIDATION", "bbox 必须是 4 个整数 [x, y, w, h]", {"bbox": bbox})
        area_px = meta.pop("area_px", meta.pop("area", None))
        if area_px is None:
            area_px = int(np.count_nonzero(crop_rgba[..., 3]))
        try:
            area_px = int(area_px)
        except (TypeError, ValueError):
            raise StoreError("VALIDATION", "area_px 必须是整数", {"area_px": area_px}) from None
        batch = meta.pop("batch", None)
        if batch is not None and not isinstance(batch, str):
            batch = str(batch)

        if self.min_free_mb > 0:
            free = disk_free_mb(self.model_dir)
            if free < self.min_free_mb:
                raise StoreError(
                    "CONFLICT",
                    f"磁盘空间不足：剩余 {free:.0f} MB，低于阈值 {self.min_free_mb} MB",
                    {"free_mb": round(free, 1), "min_free_mb": self.min_free_mb},
                )

        # seq 与时间戳在同一临界区内取得：seq 顺序 == 时间顺序，不会出现“seq 小、时间戳大”的文件名
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            now = datetime.now().astimezone()
        created_at = now.isoformat(timespec="seconds")
        stem = f"cam{camera_id}_{now:%Y%m%d_%H%M%S}_{seq:06d}"
        png, js, jpg = self._paths(cls, stem, deleted=False)

        # 不存 sample_id / cls：由路径推导，备份至 / 类别改名后无需重写
        record = {
            "camera_id": camera_id,
            "bbox": bbox,
            "area_px": area_px,
            "created_at": created_at,
            "seq": seq,
            "batch": batch,
            "extra": meta,
        }
        png_bytes = _encode(".png", cv2.cvtColor(crop_rgba, cv2.COLOR_RGBA2BGRA))
        jpg_bytes = _encode(
            ".jpg",
            make_thumbnail(crop_rgba, self.thumb_size),
            [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_JPEG_QUALITY],
        )
        # 先写 json 与缩略图，最后写 png：png 存在 ⇔ 样本完整
        try:
            atomic_write_json(js, record)
            atomic_write_bytes(jpg, jpg_bytes, fsync=False)
            atomic_write_bytes(png, png_bytes)
        except OSError as e:
            self._unlink_all(png, js, jpg)
            raise StoreError(
                "INTERNAL",
                f"样本写入失败: {e}",
                {"cls": cls, "stem": stem, "errno": e.errno},
            ) from e
        except BaseException:
            self._unlink_all(png, js, jpg)
            raise
        log.debug(
            "样本写入", extra={"model_id": self.model_id, "cls": cls, "stem": stem, "area": area_px}
        )
        return Sample(
            model_id=self.model_id,
            cls=cls,
            stem=stem,
            path=png,
            json_path=js,
            thumb_path=jpg,
            created_at=created_at,
            camera_id=camera_id,
            bbox=bbox,
            area_px=area_px,
            deleted=False,
            batch=batch,
            seq=seq,
        )

    @staticmethod
    def _unlink_all(*paths: Path) -> None:
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                log.exception("清理样本残片失败", extra={"path": str(p)})

    # ---------------------------------------------------------------- 读取
    def _load(self, cls: str, stem: str, deleted: bool) -> Sample:
        png, js, jpg = self._paths(cls, stem, deleted)
        try:
            data = read_json(js, default=None)
        except ValueError:
            log.warning("样本 json 损坏", extra={"path": str(js)})
            data = None
        if not isinstance(data, dict):
            data = {}
        cam_f, ts_f, seq_f = _parse_stem(stem)
        created_at = data.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            if ts_f:
                created_at = datetime.strptime(ts_f, "%Y%m%d_%H%M%S").astimezone().isoformat()
            else:
                try:
                    created_at = (
                        datetime.fromtimestamp(png.stat().st_mtime).astimezone().isoformat()
                    )
                except OSError:
                    created_at = ""
        camera_id = data.get("camera_id", cam_f if cam_f is not None else 0)
        bbox = data.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        seq = data.get("seq", seq_f if seq_f is not None else 0)
        batch = data.get("batch")
        return Sample(
            model_id=self.model_id,
            cls=cls,
            stem=stem,
            path=png,
            json_path=js,
            thumb_path=jpg,
            created_at=str(created_at),
            camera_id=int(camera_id) if isinstance(camera_id, int) else 0,
            bbox=[int(v) for v in bbox],
            area_px=int(data.get("area_px", 0) or 0),
            deleted=deleted,
            batch=str(batch) if batch is not None else None,
            seq=int(seq) if isinstance(seq, int) else 0,
        )

    def _scan(self, cls: str, include_deleted: bool) -> list[tuple[str, bool]]:
        """返回 ``[(stem, deleted)]``，按 seq 升序（最早在前）。

        同一 stem 同时出现在正常目录与 .trash 时以正常目录为准，只出现一次。
        文件名不合约定的样本 seq 视为 -1（排在最前，再按时间戳串、文件名）。
        """
        found: dict[str, tuple[tuple[int, str, str], bool]] = {}
        sources = [(self.class_dir(cls, False), False)]
        if include_deleted:
            sources.append((self.class_dir(cls, True), True))
        for d, deleted in sources:
            if not d.is_dir():
                continue
            for p in d.glob("*.png"):
                if p.stem in found:
                    continue
                _, ts, seq = _parse_stem(p.stem)
                found[p.stem] = ((seq if seq is not None else -1, ts or "", p.stem), deleted)
        ordered = sorted(found.items(), key=lambda kv: kv[1][0])
        return [(stem, deleted) for stem, (_, deleted) in ordered]

    def list(
        self,
        cls: str,
        include_deleted: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[int, list[Sample]]:
        """分页列出某类样本，按 seq 升序（最早写入在前）。

        Returns:
            ``(total, items)``；``total`` 是过滤后（是否含已删除）的总数。
        """
        self._check_cls(cls)
        if not isinstance(offset, int) or offset < 0:
            raise StoreError("VALIDATION", "offset 必须 ≥ 0", {"offset": offset})
        if not isinstance(limit, int) or limit < 1 or limit > MAX_LIST_LIMIT:
            raise StoreError("VALIDATION", f"limit 应在 1~{MAX_LIST_LIMIT}", {"limit": limit})
        with self._lock:
            scanned = self._scan(cls, include_deleted)
            page = scanned[offset : offset + limit]
            return len(scanned), [self._load(cls, stem, deleted) for stem, deleted in page]

    def get(self, sample_id: str) -> Sample:
        """按 id 取样本（含 .trash 中的）；不存在 → NOT_FOUND。"""
        cls, stem = self._check_owner(sample_id)
        with self._lock:
            for deleted in (False, True):
                png = self._paths(cls, stem, deleted)[0]
                if png.is_file():
                    return self._load(cls, stem, deleted)
        raise StoreError("NOT_FOUND", f"样本不存在: {sample_id}", {"sample_id": sample_id})

    def _count_dir(self, d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for _ in d.glob("*.png"))

    def count(self, cls: str, include_deleted: bool = False) -> int:
        """某类样本数。"""
        self._check_cls(cls)
        with self._lock:
            n = self._count_dir(self.class_dir(cls, False))
            if include_deleted:
                n += self._count_dir(self.class_dir(cls, True))
            return n

    def counts(self, include_deleted: bool = False) -> dict[str, int]:
        """全部类别的样本数（按 ``class_names`` 顺序，所有类别都出现）。"""
        with self._lock:
            return {c: self.count(c, include_deleted) for c in self.class_names}

    def iter_active(self, cls: str) -> Iterator[Path]:
        """按写入顺序迭代某类有效样本的 png 路径（训练用）。"""
        self._check_cls(cls)
        with self._lock:
            stems = [stem for stem, _ in self._scan(cls, include_deleted=False)]
        for stem in stems:
            yield self.class_dir(cls, False) / f"{stem}.png"

    # ---------------------------------------------------------------- 移动
    def _move_triplet(self, cls: str, stem: str, *, to_trash: bool) -> None:
        """在正常目录与 .trash 之间移动 png/json/jpg（jpg/json 缺失时忽略）。

        任一步失败则把已移动的文件挪回原位，再抛 ``StoreError(INTERNAL)``。
        """
        src = self._paths(cls, stem, deleted=not to_trash)
        dst = self._paths(cls, stem, deleted=to_trash)
        moved: list[tuple[Path, Path]] = []
        try:
            # png 最后移，保证“png 在哪，样本就在哪”
            for s, d in (src[1], dst[1]), (src[2], dst[2]), (src[0], dst[0]):
                if not s.exists():
                    continue
                ensure_dir(d.parent)
                if d.exists():
                    log.warning("目标位置已有同名文件，将被覆盖", extra={"path": str(d)})
                os.replace(s, d)
                moved.append((d, s))
                if to_trash:
                    os.utime(d, None)  # mtime = 删除时刻，purge 据此判过期
        except OSError as e:
            for d, s in reversed(moved):
                try:
                    os.replace(d, s)
                except OSError:
                    log.exception("样本移动回滚失败", extra={"path": str(d)})
            raise StoreError(
                "INTERNAL",
                f"样本移动失败: {e}",
                {"cls": cls, "stem": stem, "to_trash": to_trash, "errno": e.errno},
            ) from e

    def soft_delete(self, ids: list[str]) -> str | None:
        """把一批同类样本移入 .trash，作为一个可撤销批次（ids 去重）。

        Args:
            ids: 本模型的 sample_id 列表，必须属于同一类别。

        Returns:
            批次 id（``"b-17"``）；若全部已删除、无需动作则返回 None（不产生批次）。

        Raises:
            StoreError: VALIDATION（空列表/跨类别/不属于本模型）、NOT_FOUND（任一 id 不存在）、
                INTERNAL（移动失败，已回滚）。
        """
        if not ids:
            raise StoreError("VALIDATION", "ids 不能为空", {})
        ids = list(dict.fromkeys(ids))
        parsed = [self._check_owner(sid) for sid in ids]
        classes = {cls for cls, _ in parsed}
        if len(classes) != 1:
            raise StoreError(
                "VALIDATION", "一次删除只能包含同一类别的样本", {"classes": sorted(classes)}
            )
        cls = parsed[0][0]
        self._check_cls(cls)
        with self._lock:
            todo: list[str] = []
            for sid, (_, stem) in zip(ids, parsed, strict=True):
                active, trashed = (self._paths(cls, stem, d)[0] for d in (False, True))
                if active.is_file():
                    todo.append(stem)
                elif trashed.is_file():
                    continue  # 已删除，跳过
                else:
                    raise StoreError("NOT_FOUND", f"样本不存在: {sid}", {"sample_id": sid})
            if not todo:
                return None
            journal = self._load_journal()
            batch_id = f"b-{journal['next']}"
            moved_ids: list[str] = []
            for stem in todo:
                self._move_triplet(cls, stem, to_trash=True)
                moved_ids.append(f"{self.model_id}/{cls}/{stem}")
            journal["next"] += 1
            journal["batches"].append(
                {"batch": batch_id, "cls": cls, "ids": moved_ids, "ts": _now_iso()}
            )
            self._save_journal(journal)
        log.info(
            "样本软删除",
            extra={"model_id": self.model_id, "cls": cls, "n": len(moved_ids), "batch": batch_id},
        )
        return batch_id

    def _restore_stem(self, cls: str, stem: str) -> bool:
        if self._paths(cls, stem, deleted=True)[0].is_file():
            self._move_triplet(cls, stem, to_trash=False)
            return True
        return False

    def restore(self, sample_id: str) -> bool:
        """恢复单张已软删除的样本；返回是否真的做了恢复（样本本来就有效 → False）。

        样本同时存在于正常目录与 .trash 时以正常目录为准，顺手清掉 .trash 中的孤儿副本。

        Raises:
            StoreError: NOT_FOUND（正常目录与 .trash 中都没有）、INTERNAL（移动失败）。
        """
        cls, stem = self._check_owner(sample_id)
        with self._lock:
            trash_paths = self._paths(cls, stem, deleted=True)
            if self._paths(cls, stem, deleted=False)[0].is_file():
                if trash_paths[0].exists():
                    log.warning(
                        "样本同时存在于正常目录与 .trash，清理 .trash 副本",
                        extra={"sample_id": sample_id},
                    )
                    self._unlink_all(*trash_paths)
                    journal = self._load_journal()
                    self._drop_from_journal(journal, {(cls, stem)})
                    self._save_journal(journal)
                return False
            if not self._restore_stem(cls, stem):
                raise StoreError("NOT_FOUND", f"样本不存在: {sample_id}", {"sample_id": sample_id})
            journal = self._load_journal()
            self._drop_from_journal(journal, {(cls, stem)})
            self._save_journal(journal)
        return True

    def undo_last(self, cls: str) -> int:
        """撤销该类最近一次删除批次（整批恢复），返回恢复张数。

        弹出的批次若一张都恢复不了（文件已被外部清理 / purge 竞争），视为僵尸批次丢弃，
        继续尝试更早的同类批次；全部为空时清掉这些僵尸批次并抛 NOT_FOUND。

        Raises:
            StoreError: NOT_FOUND（该类无可撤销批次）、INTERNAL（移动失败）。
        """
        self._check_cls(cls)
        with self._lock:
            journal = self._load_journal()
            restored = 0
            batch: dict[str, Any] | None = None
            zombies: list[Any] = []
            while True:
                idx = None
                for i in range(len(journal["batches"]) - 1, -1, -1):
                    if journal["batches"][i].get("cls") == cls:
                        idx = i
                        break
                if idx is None:
                    break
                batch = journal["batches"].pop(idx)
                for b_cls, stem in self._batch_keys(batch):
                    if b_cls == cls and self._restore_stem(cls, stem):
                        restored += 1
                if restored:
                    break
                zombies.append(batch.get("batch"))
            if zombies:
                log.warning(
                    "丢弃无文件可恢复的删除批次",
                    extra={"model_id": self.model_id, "cls": cls, "batches": zombies},
                )
                self._save_journal(journal)
            if restored == 0:
                raise StoreError("NOT_FOUND", f"类别 {cls} 没有可撤销的删除", {"cls": cls})
            self._save_journal(journal)
        log.info(
            "撤销删除",
            extra={
                "model_id": self.model_id,
                "cls": cls,
                "batch": batch.get("batch") if batch else None,
                "n": restored,
            },
        )
        return restored

    def clear(self, cls: str) -> tuple[int, str | None]:
        """整类软删除为一个批次。返回 ``(删除张数, 批次 id)``；该类为空时 ``(0, None)``。"""
        self._check_cls(cls)
        with self._lock:
            stems = [stem for stem, _ in self._scan(cls, include_deleted=False)]
            if not stems:
                return 0, None
            ids = [f"{self.model_id}/{cls}/{stem}" for stem in stems]
            batch_id = self.soft_delete(ids)
            return len(ids), batch_id

    def purge(self, older_than_s: float) -> int:
        """真删 .trash 中删除时刻早于 ``older_than_s`` 秒前的样本，并从 journal 移除对应 id。

        遍历磁盘上全部 .trash 类别目录（含不在 ``class_names`` 中的历史类别）。

        Returns:
            删除的样本数。
        """
        if older_than_s < 0:
            raise StoreError("VALIDATION", "older_than_s 不能为负", {"older_than_s": older_than_s})
        cutoff = time.time() - older_than_s
        removed: set[tuple[str, str]] = set()
        with self._lock:
            if self.trash_dir.is_dir():
                for cls_dir in self.trash_dir.iterdir():
                    if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                        continue
                    cls = cls_dir.name
                    for png in cls_dir.glob("*.png"):
                        try:
                            if png.stat().st_mtime > cutoff:
                                continue
                        except OSError:
                            continue
                        self._unlink_all(*self._paths(cls, png.stem, deleted=True))
                        removed.add((cls, png.stem))
            if removed:
                journal = self._load_journal()
                self._drop_from_journal(journal, removed)
                self._save_journal(journal)
        if removed:
            log.info("清理过期样本", extra={"model_id": self.model_id, "n": len(removed)})
        return len(removed)

    # ---------------------------------------------------------------- 类别改名
    def rename_class(self, old: str, new: str) -> int:
        """类别改名迁移：``samples/{old}``、``samples/.trash/{old}``、``thumbs/{old}``、
        ``thumbs/.trash/{old}`` 改名为 ``new``，并重写 journal 中的 cls 与 ids；``class_names`` 同步。

        Returns:
            迁移的样本数（含 .trash 中的）。

        Raises:
            StoreError: VALIDATION（名字非法/相同）、CONFLICT（``new`` 的目录已存在且非空）、
                INTERNAL（目录移动失败，已回滚）。
        """
        validate_class_name(old)
        validate_class_name(new)
        if old == new:
            raise StoreError("VALIDATION", "新旧类别名相同", {"cls": old})
        with self._lock:
            pairs = [(self.class_dir(old, d), self.class_dir(new, d)) for d in (False, True)]
            pairs += [(self.thumb_dir(old, d), self.thumb_dir(new, d)) for d in (False, True)]
            for _, dst in pairs:
                if dst.is_dir() and any(dst.iterdir()):
                    raise StoreError(
                        "CONFLICT",
                        f"类别 {new!r} 的目录已存在且非空",
                        {"old": old, "new": new, "dir": str(dst)},
                    )
            n = self._count_dir(self.class_dir(old, False)) + self._count_dir(
                self.class_dir(old, True)
            )
            moved: list[tuple[Path, Path]] = []
            try:
                for src, dst in pairs:
                    if dst.is_dir():
                        dst.rmdir()  # 空目录（scaffold 预建的）
                    if src.is_dir():
                        ensure_dir(dst.parent)
                        os.replace(src, dst)
                        moved.append((dst, src))
                    else:
                        ensure_dir(dst)
                journal = self._load_journal()
                changed = False
                for b in journal["batches"]:
                    if b.get("cls") == old:
                        b["cls"] = new
                        b["ids"] = [
                            f"{self.model_id}/{new}/{stem}" for _, stem in self._batch_keys(b)
                        ]
                        changed = True
                if changed:
                    self._save_journal(journal)
            except OSError as e:
                for dst, src in reversed(moved):
                    try:
                        os.replace(dst, src)
                    except OSError:
                        log.exception("类别目录改名回滚失败", extra={"path": str(dst)})
                raise StoreError(
                    "INTERNAL", f"类别目录改名失败: {e}", {"old": old, "new": new}
                ) from e
            self.class_names = [new if c == old else c for c in self.class_names]
        log.info("类别改名", extra={"model_id": self.model_id, "old": old, "new": new, "n": n})
        return n

    # ---------------------------------------------------------------- 统计 / 清理
    def disk_usage(self) -> int:
        """样本与缩略图占用字节数。"""
        total = 0
        for root in (self.samples_dir, self.thumbs_dir):
            if not root.exists():
                continue
            for dirpath, _, files in os.walk(root):
                for f in files:
                    try:
                        total += (Path(dirpath) / f).stat().st_size
                    except OSError:
                        continue
        return total

    def remove_all(self) -> None:
        """删除整个样本与缩略图目录（模型删除时由 ModelStore 调用）。"""
        with self._lock:
            for d in (self.samples_dir, self.thumbs_dir):
                shutil.rmtree(d, ignore_errors=True)
