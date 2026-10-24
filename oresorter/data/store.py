"""模型目录管理：新建 / 删除 / 备份至 / 元数据 / 列表 / 类别改名（docs/01 §8、docs/02 §3.4、docs/03 §8）。

目录布局（``root`` = ``{data_dir}/models``）::

    {root}/{id}/meta.json
    {root}/{id}/samples/{cls}/            样本（见 samples.py）
    {root}/{id}/samples/.trash/{cls}/
    {root}/{id}/thumbs/{cls}/
    {root}/{id}/train/                    best.pt  model.onnx  model.plan  metrics.json  train.log  progress.json

约定：
- "模型已创建" ⇔ ``meta.json`` 存在；``list()`` 对未创建编号返回占位项（created_at=None）。
  ``backup`` 的冲突判定同样以 meta.json 为准：残留的半成品目录（有目录无 meta.json）视为未创建，
  会被直接覆盖；``delete`` 允许清理这类残留目录（目录存在即可删）。
- ``meta.json`` 损坏（非法 JSON / 字段类型非法）时 ``describe``/``list``/``update_meta`` 降级为默认值
  （name=模型{id}，created_at 取文件 mtime，记录 error 日志；``update_meta`` 会把修复后的内容写回），
  保证 GET /models 永远能返回 1..max_models；``meta()`` 本身仍抛 VALIDATION 供显式检查。
- 当前部署的模型不可删除：``delete(id, deployed_id)`` 相等 → CONFLICT（上层也可自行判断）。
- ``backup``：整目录复制到 ``root`` 下的临时目录，再 ``atomic_replace_dir`` 换到目标，任何时刻目标
  要么旧要么新。复制在锁外进行（大目录耗时几十秒，期间 list / 其他模型采样不受阻塞），锁只包住
  前置检查与最终替换（替换前重新检查冲突）。复制忽略 ``*.tmp``（fsutil 原子写的临时文件）、容忍
  复制途中消失的文件（源模型正在采样/删除）；复制后改写 meta.id 与 journal 中 sample_id 的模型前缀。
  文件系统错误统一抛 ``StoreError(INTERNAL)``。上层 job 仍应在备份期间禁止对源模型采样，
  否则备份只是“某一时刻附近”的快照。
- ``rename_class(old, new)``：全部已创建模型的类别目录、journal、meta.classes / sample_counts 一起迁移，
  供 PUT /config/classes 改名时调用；``set_class_names`` 用于新增/删减类别后同步。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from oresorter.data.samples import (
    JOURNAL_NAME,
    TRASH_DIR,
    SampleRepo,
    StoreError,
    rewrite_journal_model_id,
    validate_class_name,
)
from oresorter.fsutil import atomic_replace_dir, atomic_write_json, ensure_dir, read_json
from oresorter.log import get_logger

__all__ = ["ModelMeta", "ModelStore", "StoreError"]

log = get_logger(__name__)

META_NAME = "meta.json"
TRAIN_DIR = "train"
BEST_PT = "best.pt"
ONNX_NAME = "model.onnx"
ENGINE_NAME = "model.plan"

# update_meta 允许修改的字段：HMI 可编辑项 + 训练写入项（id / created_at 不可改）
PATCHABLE_META_FIELDS = frozenset(
    {
        "name",
        "description",
        "trained_at",
        "algo",
        "image_size",
        "classes",
        "sample_counts",
        "metrics_summary",
        "version",
    }
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _validate_class_names(class_names: Any) -> list[str]:
    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise StoreError("VALIDATION", "class_names 不能为空", {})
    names = list(class_names)
    for c in names:
        validate_class_name(c)
    if len(set(names)) != len(names):
        raise StoreError("VALIDATION", "class_names 不能重复", {"class_names": names})
    return names


def _copy_tolerant(src: str, dst: str) -> None:
    """``copytree`` 用的复制函数：源文件在列目录后被移走（并发采样写入 / 软删除）时跳过并告警。"""
    try:
        shutil.copy2(src, dst)
    except FileNotFoundError:
        log.warning("备份时源文件已消失，跳过", extra={"path": src})


class ModelMeta(BaseModel):
    """``meta.json``。全部字段有默认值，读旧文件缺字段时自动补默认；未知字段忽略。"""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    id: int = 0
    name: str = ""
    description: str = ""
    created_at: str | None = None
    trained_at: str | None = None
    algo: str | None = None
    image_size: int | None = None
    classes: list[str] = []
    """训练时的类别顺序（即模型输出维度顺序）。"""
    sample_counts: dict[str, int] = {}
    metrics_summary: dict[str, Any] | None = None
    version: int = 0

    @classmethod
    def load(cls, path: str | Path) -> ModelMeta:
        """读取 meta.json；文件不存在时返回全默认。损坏/字段类型非法 → StoreError(VALIDATION)。"""
        try:
            data = read_json(path, default={})
        except ValueError as e:
            raise StoreError("VALIDATION", "meta.json 不是合法 JSON", {"path": str(path)}) from e
        if not isinstance(data, dict):
            raise StoreError("VALIDATION", "meta.json 顶层应为对象", {"path": str(path)})
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise StoreError(
                "VALIDATION",
                "meta.json 字段非法",
                {"path": str(path), "errors": e.errors(include_url=False)},
            ) from e

    @classmethod
    def load_lenient(cls, path: str | Path, model_id: int, class_names: list[str]) -> ModelMeta:
        """读取 meta.json；损坏时记录 error 日志并降级为默认值（created_at 取文件 mtime）。"""
        try:
            return cls.load(path)
        except StoreError as e:
            log.error("meta.json 损坏，按默认值处理", extra={"id": model_id, "detail": e.detail})
            try:
                created = (
                    datetime.fromtimestamp(Path(path).stat().st_mtime)
                    .astimezone()
                    .isoformat(timespec="seconds")
                )
            except OSError:
                created = None
            return cls(
                id=model_id,
                name=f"模型{model_id}",
                created_at=created,
                sample_counts=dict.fromkeys(class_names, 0),
            )

    def save(self, path: str | Path) -> None:
        """原子写回。"""
        atomic_write_json(path, self.model_dump(mode="json"))


class ModelStore:
    """``{data_dir}/models`` 下全部模型目录的管理器。线程安全。"""

    def __init__(
        self,
        root: Path,
        max_models: int,
        class_names: list[str],
        *,
        thumb_size: int = 160,
        min_free_mb: int = 0,
    ):
        """
        Args:
            root: ``{data_dir}/models``。
            max_models: 模型编号上限（编号 1..max_models）。
            class_names: 类别名列表（决定目录结构与列表顺序）。
            thumb_size / min_free_mb: 透传给 ``SampleRepo``。
        """
        if isinstance(max_models, bool) or not isinstance(max_models, int) or max_models < 1:
            raise StoreError("VALIDATION", "max_models 必须 ≥ 1", {"max_models": max_models})
        self.root = Path(root)
        self.max_models = int(max_models)
        self.class_names = _validate_class_names(class_names)
        self.thumb_size = thumb_size
        self.min_free_mb = min_free_mb
        self._lock = threading.RLock()
        self._repos: dict[int, SampleRepo] = {}

    # ---------------------------------------------------------------- 基础
    def ensure_root(self) -> Path:
        """确保 ``root`` 存在。"""
        return ensure_dir(self.root)

    def _check_id(self, model_id: int) -> int:
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise StoreError("VALIDATION", "模型编号必须是整数", {"id": model_id})
        if model_id < 1 or model_id > self.max_models:
            raise StoreError(
                "VALIDATION",
                f"模型编号 {model_id} 超出范围 1~{self.max_models}",
                {"id": model_id, "max_models": self.max_models},
            )
        return model_id

    def model_dir(self, model_id: int) -> Path:
        """``{root}/{id}``（不检查存在性）。"""
        return self.root / str(self._check_id(model_id))

    def meta_path(self, model_id: int) -> Path:
        return self.model_dir(model_id) / META_NAME

    def train_dir(self, model_id: int) -> Path:
        return self.model_dir(model_id) / TRAIN_DIR

    def exists(self, model_id: int) -> bool:
        """模型是否已创建（以 meta.json 存在为准）。"""
        return self.meta_path(model_id).is_file()

    def _require(self, model_id: int) -> Path:
        d = self.model_dir(model_id)
        if not (d / META_NAME).is_file():
            raise StoreError("NOT_FOUND", f"模型 {model_id} 不存在", {"id": model_id})
        return d

    def _scaffold(self, d: Path) -> None:
        ensure_dir(d / TRAIN_DIR)
        for cls in self.class_names:
            ensure_dir(d / "samples" / cls)
            ensure_dir(d / "samples" / TRASH_DIR / cls)
            ensure_dir(d / "thumbs" / cls)

    # ---------------------------------------------------------------- 创建 / 元数据
    def create(self, model_id: int, name: str = "", description: str = "") -> ModelMeta:
        """新建模型目录结构并写 meta.json。已存在 → CONFLICT；name 为空时用 ``模型{id}``。"""
        self._check_id(model_id)
        if not isinstance(name, str) or not isinstance(description, str):
            raise StoreError("VALIDATION", "name / description 必须是字符串", {"id": model_id})
        name = name.strip() or f"模型{model_id}"
        with self._lock:
            d = self.model_dir(model_id)
            if (d / META_NAME).exists():
                raise StoreError("CONFLICT", f"模型 {model_id} 已存在", {"id": model_id})
            self._scaffold(d)
            meta = ModelMeta(
                id=model_id,
                name=name,
                description=description,
                created_at=_now_iso(),
                sample_counts=dict.fromkeys(self.class_names, 0),
            )
            meta.save(d / META_NAME)
            self._repos.pop(model_id, None)
        log.info("新建模型", extra={"id": model_id, "model_name": name})
        return meta

    def meta(self, model_id: int) -> ModelMeta:
        """读取 meta.json；不存在 → NOT_FOUND；损坏 → VALIDATION。目录名与 meta.id 不一致时以目录名为准。"""
        d = self._require(model_id)
        meta = ModelMeta.load(d / META_NAME)
        if meta.id != model_id:
            log.warning(
                "meta.id 与目录不一致，已按目录修正", extra={"dir": model_id, "meta": meta.id}
            )
            meta.id = model_id
        return meta

    def _meta_lenient(self, model_id: int) -> ModelMeta:
        """同 ``meta()``，但损坏的 meta.json 降级为默认值（不存在仍抛 NOT_FOUND）。"""
        d = self._require(model_id)
        meta = ModelMeta.load_lenient(d / META_NAME, model_id, self.class_names)
        meta.id = model_id
        return meta

    def update_meta(self, model_id: int, **patch: Any) -> ModelMeta:
        """部分更新 meta.json（严格类型校验：``"160"`` 不会被当成 160）。

        允许 ``PATCHABLE_META_FIELDS``；未知字段 / 类型非法 / 空名 → VALIDATION。
        原 meta.json 损坏时以默认值为基础写回（修复），并记录 error 日志。
        """
        unknown = sorted(set(patch) - PATCHABLE_META_FIELDS)
        if unknown:
            raise StoreError(
                "VALIDATION",
                f"不允许修改的字段: {', '.join(unknown)}",
                {"unknown": unknown, "allowed": sorted(PATCHABLE_META_FIELDS)},
            )
        with self._lock:
            d = self._require(model_id)
            current = self._meta_lenient(model_id).model_dump()
            current.update(patch)
            current["id"] = model_id
            try:
                meta = ModelMeta.model_validate(current, strict=True)
            except ValidationError as e:
                raise StoreError(
                    "VALIDATION", "元数据字段非法", {"errors": e.errors(include_url=False)}
                ) from e
            if not meta.name.strip():
                raise StoreError("VALIDATION", "模型名称不能为空", {"name": meta.name})
            meta.save(d / META_NAME)
        return meta

    # ---------------------------------------------------------------- 样本库
    def samples(self, model_id: int) -> SampleRepo:
        """该模型的 ``SampleRepo``（缓存实例）。模型不存在 → NOT_FOUND。"""
        with self._lock:
            repo = self._repos.get(model_id)
            if repo is None:
                d = self._require(model_id)
                repo = SampleRepo(
                    d,
                    self.class_names,
                    thumb_size=self.thumb_size,
                    min_free_mb=self.min_free_mb,
                )
                self._repos[model_id] = repo
            return repo

    # ---------------------------------------------------------------- 列表
    def _placeholder(self, model_id: int) -> dict[str, Any]:
        return {
            "id": model_id,
            "name": f"模型{model_id}",
            "description": "",
            "created_at": None,
            "trained_at": None,
            "deployed": False,
            "algo": None,
            "image_size": None,
            "classes": [{"name": c, "sample_count": 0, "enabled": False} for c in self.class_names],
            "metrics_summary": None,
        }

    def describe(self, model_id: int, deployed_id: int | None = None) -> dict[str, Any]:
        """docs/03 ``GET /models`` 的单项。未创建的编号返回占位项（与 ``list()`` 一致）。

        meta.json 损坏时降级为默认名称/字段（created_at 取文件 mtime），样本计数仍实时统计。
        """
        self._check_id(model_id)
        if not self.exists(model_id):
            return self._placeholder(model_id)
        try:
            meta = self._meta_lenient(model_id)
            counts = self.samples(model_id).counts()
        except StoreError as e:
            if e.code != "NOT_FOUND":  # 列表期间被删除 → 占位项
                raise
            return self._placeholder(model_id)
        return {
            "id": model_id,
            "name": meta.name,
            "description": meta.description,
            "created_at": meta.created_at,
            "trained_at": meta.trained_at,
            "deployed": deployed_id == model_id,
            "algo": meta.algo,
            "image_size": meta.image_size,
            "classes": [
                {"name": c, "sample_count": counts.get(c, 0), "enabled": counts.get(c, 0) > 0}
                for c in self.class_names
            ],
            "metrics_summary": meta.metrics_summary,
        }

    def list(self, deployed_id: int | None = None) -> list[dict[str, Any]]:
        """1..max_models 全部编号（含未创建的占位项），供 HMI 固定显示。单个模型损坏不影响整体。"""
        return [self.describe(i, deployed_id) for i in range(1, self.max_models + 1)]

    def created_ids(self) -> list[int]:
        """已创建的模型编号（升序）。"""
        return [i for i in range(1, self.max_models + 1) if self.exists(i)]

    # ---------------------------------------------------------------- 训练产物
    def trained(self, model_id: int) -> bool:
        """``train/best.pt`` 是否存在。"""
        return (self.train_dir(model_id) / BEST_PT).is_file()

    def has_onnx(self, model_id: int) -> bool:
        return (self.train_dir(model_id) / ONNX_NAME).is_file()

    def has_engine(self, model_id: int) -> bool:
        return (self.train_dir(model_id) / ENGINE_NAME).is_file()

    # ---------------------------------------------------------------- 删除 / 备份
    def delete(self, model_id: int, deployed_id: int | None = None) -> None:
        """整目录删除（目录存在即可删，因此也能清理无 meta.json 的残留目录）。

        不存在 → NOT_FOUND；``model_id == deployed_id`` → CONFLICT。
        """
        self._check_id(model_id)
        if deployed_id is not None and deployed_id == model_id:
            raise StoreError("CONFLICT", f"模型 {model_id} 正在使用，不能删除", {"id": model_id})
        with self._lock:
            d = self.model_dir(model_id)
            if not d.exists():
                raise StoreError("NOT_FOUND", f"模型 {model_id} 不存在", {"id": model_id})
            self._repos.pop(model_id, None)
            try:
                shutil.rmtree(d)
            except OSError as e:
                raise StoreError("INTERNAL", f"删除模型目录失败: {e}", {"id": model_id}) from e
        log.info("删除模型", extra={"id": model_id})

    def backup(self, src: int, dst: int, overwrite: bool = False) -> ModelMeta:
        """“备份至”：把 ``src`` 整目录（样本、缩略图、训练产物）复制为 ``dst``。

        复制到 ``root`` 下的临时目录（锁外进行），再在锁内原子替换到目标；失败时目标目录保持原样、
        清理临时目录。复制后 ``dst`` 的 meta.id 改为 ``dst``、journal 中 sample_id 前缀改为 ``dst``，
        名称与其余字段保持。

        Raises:
            StoreError: VALIDATION（src == dst / 编号越界）、NOT_FOUND（src 不存在或复制期间被删除）、
                CONFLICT（dst 已存在且 overwrite=False，替换前会再检查一次）、INTERNAL（复制/替换失败）。
        """
        self._check_id(src)
        self._check_id(dst)
        if src == dst:
            raise StoreError("VALIDATION", "源模型与目标模型不能相同", {"src": src, "dst": dst})
        with self._lock:
            src_dir = self._require(src)
            dst_dir = self.model_dir(dst)
            if self.exists(dst) and not overwrite:
                raise StoreError(
                    "CONFLICT", f"目标模型 {dst} 已存在，需确认覆盖", {"src": src, "dst": dst}
                )
            self.ensure_root()
            tmp = Path(tempfile.mkdtemp(prefix=f".{dst}.backup.", dir=str(self.root)))
        try:
            try:
                # mkdtemp 已建目录，copytree 需要目标不存在 → 用 dirs_exist_ok 复制到其下。
                # 忽略 fsutil 原子写的 *.tmp；列目录后被移走的文件跳过（并发采样 / 软删除）。
                shutil.copytree(
                    src_dir,
                    tmp,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.tmp"),
                    copy_function=_copy_tolerant,
                )
            except (shutil.Error, OSError) as e:
                raise StoreError(
                    "INTERNAL", f"复制模型目录失败: {e}", {"src": src, "dst": dst}
                ) from e
            if not (tmp / META_NAME).is_file():
                raise StoreError("NOT_FOUND", f"模型 {src} 在备份期间被删除", {"id": src})
            meta = ModelMeta.load_lenient(tmp / META_NAME, src, self.class_names)
            meta.id = dst
            meta.save(tmp / META_NAME)
            rewrite_journal_model_id(tmp / "samples" / TRASH_DIR / JOURNAL_NAME, dst)
            with self._lock:
                if self.exists(dst) and not overwrite:
                    raise StoreError(
                        "CONFLICT",
                        f"目标模型 {dst} 在备份期间被创建，需确认覆盖",
                        {"src": src, "dst": dst},
                    )
                try:
                    atomic_replace_dir(tmp, dst_dir)
                except OSError as e:
                    raise StoreError(
                        "INTERNAL", f"替换目标目录失败: {e}", {"src": src, "dst": dst}
                    ) from e
                repo = self._repos.get(dst)
                if repo is not None:
                    repo.refresh_seq()  # 目录内容已整体替换，seq 计数器只增不减
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        log.info("备份模型", extra={"src": src, "dst": dst, "overwrite": overwrite})
        return meta

    # ---------------------------------------------------------------- 类别
    def set_class_names(self, class_names: list[str]) -> None:
        """同步配置中的类别名（新增类别会为已创建模型补建目录；改名请用 ``rename_class``，
        否则旧类别样本成为孤儿，只能经 ``SampleRepo.orphan_classes()`` 发现）。"""
        names = _validate_class_names(class_names)
        with self._lock:
            self.class_names = names
            for mid in self.created_ids():
                self._scaffold(self.model_dir(mid))
            for repo in self._repos.values():
                repo.set_class_names(names)

    def _rename_in_meta(self, model_id: int, old: str, new: str) -> None:
        meta = self._meta_lenient(model_id)
        changed = False
        if old in meta.classes:
            meta.classes = [new if c == old else c for c in meta.classes]
            changed = True
        if old in meta.sample_counts:
            meta.sample_counts = {new if k == old else k: v for k, v in meta.sample_counts.items()}
            changed = True
        if changed:
            meta.save(self.meta_path(model_id))

    def rename_class(self, old: str, new: str) -> dict[int, int]:
        """类别改名迁移：全部已创建模型的目录、journal、meta.classes / sample_counts，并更新 ``class_names``。

        供配置热更新（PUT /config/classes 改名）调用。任一模型失败时回滚已迁移的模型后再抛出。

        Returns:
            ``{model_id: 迁移样本数}``。

        Raises:
            StoreError: VALIDATION（old 不在 class_names / new 非法 / 相同）、
                CONFLICT（new 已是类别名或某模型下目标目录非空）、INTERNAL（目录移动失败）。
        """
        validate_class_name(new)
        if old not in self.class_names:
            raise StoreError(
                "VALIDATION", f"未知类别: {old!r}", {"cls": old, "class_names": self.class_names}
            )
        if new == old:
            raise StoreError("VALIDATION", "新旧类别名相同", {"cls": old})
        if new in self.class_names:
            raise StoreError("CONFLICT", f"类别 {new!r} 已存在", {"cls": new})
        result: dict[int, int] = {}
        with self._lock:
            done: list[int] = []
            try:
                for mid in self.created_ids():
                    result[mid] = self.samples(mid).rename_class(old, new)
                    done.append(mid)
                    self._rename_in_meta(mid, old, new)
            except StoreError:
                for mid in reversed(done):
                    try:
                        self.samples(mid).rename_class(new, old)
                        self._rename_in_meta(mid, new, old)
                    except StoreError:
                        log.exception("类别改名回滚失败", extra={"id": mid})
                raise
            self.class_names = [new if c == old else c for c in self.class_names]
            for repo in self._repos.values():
                repo.set_class_names(self.class_names)
        log.info("类别改名", extra={"old": old, "new": new, "models": result})
        return result

    # ---------------------------------------------------------------- 统计
    def disk_usage(self, model_id: int) -> int:
        """模型目录占用字节数。不存在 → NOT_FOUND。"""
        d = self._require(model_id)
        total = 0
        for dirpath, _, files in os.walk(d):
            for f in files:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    continue
        return total
