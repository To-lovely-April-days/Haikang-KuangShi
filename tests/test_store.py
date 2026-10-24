"""ModelStore：创建 / 列表 / 元数据 / 删除 / 备份至（原子性）。"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from oresorter.data.samples import StoreError
from oresorter.data.store import ModelMeta, ModelStore
from oresorter.fsutil import read_json

CLASSES = ["A", "B", "C"]


@pytest.fixture
def store(tmp_path: Path) -> ModelStore:
    s = ModelStore(tmp_path / "data" / "models", max_models=3, class_names=CLASSES)
    s.ensure_root()
    return s


def _crop(h: int = 20, w: int = 30) -> np.ndarray:
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = (200, 40, 40)
    img[2:-2, 2:-2, 3] = 255
    return img


def _err(code: str) -> _ErrCtx:
    return _ErrCtx(code)


class _ErrCtx:
    """断言抛出指定 code 的 StoreError。"""

    def __init__(self, code: str):
        self.code = code
        self._ctx = pytest.raises(StoreError)

    def __enter__(self):
        return self._ctx.__enter__()

    def __exit__(self, *exc):
        result = self._ctx.__exit__(*exc)
        if result:
            assert self._ctx.excinfo.value.code == self.code, self._ctx.excinfo.value.to_dict()
        return result


# ---------------------------------------------------------------- create


def test_create_builds_layout_and_meta(store: ModelStore):
    meta = store.create(1, "矿A", "说明")
    d = store.model_dir(1)
    assert (d / "meta.json").is_file()
    for cls in CLASSES:
        assert (d / "samples" / cls).is_dir()
        assert (d / "samples" / ".trash" / cls).is_dir()
        assert (d / "thumbs" / cls).is_dir()
    assert (d / "train").is_dir()
    assert meta.id == 1 and meta.name == "矿A" and meta.description == "说明"
    assert meta.created_at is not None and meta.trained_at is None
    assert meta.sample_counts == {"A": 0, "B": 0, "C": 0}
    on_disk = read_json(d / "meta.json")
    assert on_disk["id"] == 1 and on_disk["name"] == "矿A" and on_disk["version"] == 0
    assert store.exists(1) and not store.exists(2)


def test_create_default_name(store: ModelStore):
    assert store.create(2, "").name == "模型2"
    assert store.create(3, "   ").name == "模型3"


def test_create_duplicate_conflict(store: ModelStore):
    store.create(1, "x")
    with _err("CONFLICT"):
        store.create(1, "y")


@pytest.mark.parametrize("bad", [0, 4, -1, "1", True])
def test_id_out_of_range_validation(store: ModelStore, bad):
    with _err("VALIDATION"):
        store.create(bad, "x")
    with _err("VALIDATION"):
        store.model_dir(bad)


def test_store_error_shape():
    e = StoreError("NOT_FOUND", "没有", {"id": 9})
    assert e.to_dict() == {"code": "NOT_FOUND", "message": "没有", "detail": {"id": 9}}
    assert str(e) == "没有"


# ---------------------------------------------------------------- list / describe


def test_list_returns_all_ids_with_placeholders(store: ModelStore):
    store.create(2, "second", "d")
    store.samples(2).add("B", _crop(), {"camera_id": 1})
    items = store.list(deployed_id=2)
    assert [it["id"] for it in items] == [1, 2, 3]

    empty = items[0]
    assert empty["created_at"] is None and empty["deployed"] is False
    assert empty["name"] == "模型1" and empty["description"] == ""
    assert empty["algo"] is None and empty["image_size"] is None
    assert empty["metrics_summary"] is None and empty["trained_at"] is None
    assert empty["classes"] == [{"name": c, "sample_count": 0, "enabled": False} for c in CLASSES]

    created = items[1]
    assert created["name"] == "second" and created["description"] == "d"
    assert created["created_at"] is not None and created["deployed"] is True
    assert created["classes"] == [
        {"name": "A", "sample_count": 0, "enabled": False},
        {"name": "B", "sample_count": 1, "enabled": True},
        {"name": "C", "sample_count": 0, "enabled": False},
    ]
    assert set(created) == {
        "id",
        "name",
        "description",
        "created_at",
        "trained_at",
        "deployed",
        "algo",
        "image_size",
        "classes",
        "metrics_summary",
    }


def test_describe(store: ModelStore):
    store.create(1, "a")
    assert store.describe(1, deployed_id=None)["deployed"] is False
    assert store.describe(1, deployed_id=1) == store.list(deployed_id=1)[0]
    assert store.describe(3)["created_at"] is None
    with _err("VALIDATION"):
        store.describe(99)
    assert store.created_ids() == [1]


# ---------------------------------------------------------------- meta


def test_meta_not_found(store: ModelStore):
    with _err("NOT_FOUND"):
        store.meta(1)
    with _err("NOT_FOUND"):
        store.samples(1)


def test_update_meta_ok_and_invalid(store: ModelStore):
    store.create(1, "a")
    m = store.update_meta(
        1,
        description="新说明",
        trained_at="2026-10-24T10:00:00+11:00",
        algo="cnn-resnet18",
        image_size=160,
        classes=["A", "B"],
        sample_counts={"A": 10, "B": 5},
        metrics_summary={"val_accuracy": 0.97, "macro_f1": 0.96},
        version=1,
    )
    assert m.description == "新说明" and m.algo == "cnn-resnet18" and m.version == 1
    again = store.meta(1)
    assert again.classes == ["A", "B"] and again.metrics_summary["macro_f1"] == 0.96
    assert again.name == "a" and again.created_at is not None

    with _err("VALIDATION"):
        store.update_meta(1, id=2)
    with _err("VALIDATION"):
        store.update_meta(1, nope=1)
    with _err("VALIDATION"):
        store.update_meta(1, image_size="big")
    with _err("VALIDATION"):
        store.update_meta(1, name="  ")
    assert store.meta(1).image_size == 160, "失败的更新不应落盘"
    with _err("NOT_FOUND"):
        store.update_meta(2, name="x")


def test_meta_backward_compatible_and_extra_ignored(tmp_path: Path):
    p = tmp_path / "meta.json"
    p.write_text('{"id": 3, "name": "老", "future_field": 1}', encoding="utf-8")
    m = ModelMeta.load(p)
    assert m.id == 3 and m.name == "老" and m.classes == [] and m.version == 0
    assert m.metrics_summary is None and m.sample_counts == {}
    assert not hasattr(m, "future_field")
    assert ModelMeta.load(tmp_path / "missing.json") == ModelMeta()
    p.write_text("{not json", encoding="utf-8")
    with _err("VALIDATION"):
        ModelMeta.load(p)


def test_meta_id_fixed_from_dir(store: ModelStore):
    store.create(1, "a")
    read = read_json(store.meta_path(1))
    read["id"] = 7
    store.meta_path(1).write_text(json.dumps(read), encoding="utf-8")
    assert store.meta(1).id == 1


# ---------------------------------------------------------------- artifacts


def test_trained_has_onnx_has_engine(store: ModelStore):
    store.create(1, "a")
    assert not store.trained(1) and not store.has_onnx(1) and not store.has_engine(1)
    (store.train_dir(1) / "best.pt").write_bytes(b"pt")
    assert store.trained(1) and not store.has_onnx(1)
    (store.train_dir(1) / "model.onnx").write_bytes(b"onnx")
    (store.train_dir(1) / "model.plan").write_bytes(b"plan")
    assert store.has_onnx(1) and store.has_engine(1)
    assert not store.trained(2)


# ---------------------------------------------------------------- delete


def test_delete(store: ModelStore):
    store.create(1, "a")
    store.samples(1).add("A", _crop(), {"camera_id": 1})
    with _err("CONFLICT"):
        store.delete(1, deployed_id=1)
    assert store.exists(1)
    store.delete(1, deployed_id=2)
    assert not store.model_dir(1).exists()
    assert not store.exists(1)
    with _err("NOT_FOUND"):
        store.delete(1)
    # 删除后重新创建应是全新的样本库
    store.create(1, "b")
    assert store.samples(1).counts() == {"A": 0, "B": 0, "C": 0}


# ---------------------------------------------------------------- backup


def _fill(store: ModelStore, mid: int) -> list[str]:
    repo = store.samples(mid)
    ids = [
        repo.add("A", _crop(), {"camera_id": 1, "bbox": [1, 2, 3, 4]}).sample_id for _ in range(3)
    ]
    (store.train_dir(mid) / "best.pt").write_bytes(b"weights")
    (store.train_dir(mid) / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")
    return ids


def test_backup_copies_everything_and_rewrites_id(store: ModelStore):
    store.create(1, "源模型", "desc")
    ids = _fill(store, 1)
    store.update_meta(1, algo="cnn-resnet18", version=3)
    meta = store.backup(1, 2)
    assert meta.id == 2 and meta.name == "源模型" and meta.version == 3
    assert store.exists(2)
    m2 = store.meta(2)
    assert m2.id == 2 and m2.name == "源模型" and m2.description == "desc"
    assert m2.algo == "cnn-resnet18"
    assert store.trained(2)
    assert (store.train_dir(2) / "metrics.json").read_text(encoding="utf-8") == '{"accuracy": 0.9}'
    repo2 = store.samples(2)
    assert repo2.counts() == {"A": 3, "B": 0, "C": 0}
    total, items = repo2.list("A")
    assert total == 3
    assert [s.sample_id for s in items] == [i.replace("1/", "2/", 1) for i in ids]
    assert all(s.thumb_path.is_file() and s.json_path.is_file() for s in items)
    # 源不受影响
    assert store.meta(1).id == 1 and store.samples(1).count("A") == 3
    # 无残留临时目录 / .old
    assert sorted(p.name for p in store.root.iterdir()) == ["1", "2"]


def test_backup_conflict_and_overwrite(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    store.create(2, "b")
    store.samples(2).add("C", _crop(), {"camera_id": 2})
    with _err("CONFLICT"):
        store.backup(1, 2)
    assert store.meta(2).name == "b" and store.samples(2).count("C") == 1
    store.backup(1, 2, overwrite=True)
    assert store.meta(2).name == "a"
    assert store.samples(2).counts() == {"A": 3, "B": 0, "C": 0}
    assert not (store.root / "2.old").exists()


def test_backup_validation_and_not_found(store: ModelStore):
    store.create(1, "a")
    with _err("VALIDATION"):
        store.backup(1, 1)
    with _err("VALIDATION"):
        store.backup(1, 9)
    with _err("NOT_FOUND"):
        store.backup(2, 3)


def test_backup_atomic_on_copy_failure(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    store.create(2, "keep")
    store.samples(2).add("B", _crop(), {"camera_id": 1})
    before = sorted(str(p.relative_to(store.root)) for p in store.model_dir(2).rglob("*"))

    with (
        mock.patch.object(shutil, "copytree", side_effect=OSError("disk gone")),
        _err("INTERNAL"),
    ):
        store.backup(1, 2, overwrite=True)

    after = sorted(str(p.relative_to(store.root)) for p in store.model_dir(2).rglob("*"))
    assert after == before, "失败后目标目录内容应原样保留"
    assert store.meta(2).name == "keep"
    assert sorted(p.name for p in store.root.iterdir()) == ["1", "2"], "不应残留临时目录"


def test_backup_atomic_on_replace_failure(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    store.create(2, "keep")
    with (
        mock.patch(
            "oresorter.data.store.atomic_replace_dir", side_effect=OSError("replace failed")
        ),
        _err("INTERNAL"),
    ):
        store.backup(1, 2, overwrite=True)
    assert store.meta(2).name == "keep"
    assert sorted(p.name for p in store.root.iterdir()) == ["1", "2"]


# ---------------------------------------------------------------- disk usage


def test_disk_usage(store: ModelStore):
    store.create(1, "a")
    base = store.disk_usage(1)
    assert base > 0  # meta.json
    _fill(store, 1)
    assert store.disk_usage(1) > base
    with _err("NOT_FOUND"):
        store.disk_usage(2)


def test_samples_repo_cached(store: ModelStore):
    store.create(1, "a")
    assert store.samples(1) is store.samples(1)


# ---------------------------------------------------------------- 审查修复：backup 的 journal / 并发 / 锁 / 残留目录


def test_backup_rewrites_journal_ids(store: ModelStore):
    store.create(1, "a")
    r1 = store.samples(1)
    ids = [r1.add("A", _crop(), {"camera_id": 1}).sample_id for _ in range(2)]
    r1.soft_delete(ids)
    store.backup(1, 2)
    r2 = store.samples(2)
    journal = read_json(r2.journal_path)
    new_ids = [i.replace("1/", "2/", 1) for i in ids]
    assert journal["batches"][0]["ids"] == new_ids, "journal 中 sample_id 前缀应改为目标编号"
    assert read_json(r1.journal_path)["batches"][0]["ids"] == ids, "源 journal 不变"
    assert r2.restore(new_ids[0]) is True
    assert read_json(r2.journal_path)["batches"][0]["ids"] == [new_ids[1]]
    assert r2.undo_last("A") == 1
    assert read_json(r2.journal_path)["batches"] == []
    assert r2.count("A") == 2
    assert r2.get(new_ids[0]).sample_id == new_ids[0]
    assert r1.count("A", include_deleted=True) == 2 and r1.count("A") == 0


def test_backup_ignores_tmp_and_vanished_files(store: ModelStore):
    store.create(1, "a")
    ids = _fill(store, 1)
    a_dir = store.samples(1).class_dir("A")
    (a_dir / ".cam1_x.png.abc123.tmp").write_bytes(b"half")
    (store.train_dir(1) / ".best.pt.xyz.tmp").write_bytes(b"half")
    vanish = store.samples(1).get(ids[0]).json_path
    real = shutil.copy2

    def flaky(src, dst, **kw):
        if str(src) == str(vanish):
            raise FileNotFoundError(2, "gone", str(src))
        return real(src, dst, **kw)

    with mock.patch.object(shutil, "copy2", side_effect=flaky):
        meta = store.backup(1, 2)
    assert meta.id == 2
    assert not list(store.model_dir(2).rglob("*.tmp")), "fsutil 的临时文件不应被复制"
    assert not list(store.model_dir(2).rglob(".*.tmp"))
    r2 = store.samples(2)
    assert r2.count("A") == 3, "消失的文件只是跳过，备份照常完成"
    assert not (r2.class_dir("A") / vanish.name).exists()
    assert store.trained(2)


def test_backup_tolerates_concurrent_writes(store: ModelStore):
    store.create(1, "a")
    repo = store.samples(1)
    for _ in range(20):
        repo.add("A", _crop(), {"camera_id": 1})
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        try:
            while not stop.is_set():
                repo.add("B", _crop(), {"camera_id": 1})
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    try:
        time.sleep(0.05)
        meta = store.backup(1, 2)
    finally:
        stop.set()
        t.join(timeout=30)
    assert not errors, errors
    assert meta.id == 2
    assert not list(store.model_dir(2).rglob("*.tmp"))
    r2 = store.samples(2)
    assert r2.count("A") == 20
    assert repo.count("B") > 0 and r2.count("B") <= repo.count("B")
    for s in r2.list("B", limit=1000)[1]:
        assert s.path.stat().st_size > 0, "不应复制到半成品 png"
    assert sorted(p.name for p in store.root.iterdir()) == ["1", "2"]


def test_backup_does_not_block_list(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    store.create(3, "c")
    real = shutil.copytree
    started = threading.Event()
    src_dir = store.model_dir(1)

    def slow(src, *a, **kw):
        if Path(src) == src_dir:  # copytree 会递归调用自身（走到 mock），只在顶层拖慢
            started.set()
            time.sleep(1.0)
        return real(src, *a, **kw)

    result: dict = {}

    def run():
        with mock.patch.object(shutil, "copytree", side_effect=slow):
            result["meta"] = store.backup(1, 2)

    t = threading.Thread(target=run)
    t.start()
    assert started.wait(5)
    time.sleep(0.1)
    t0 = time.perf_counter()
    items = store.list(deployed_id=1)
    store.samples(3).add("A", _crop(), {"camera_id": 1})
    store.update_meta(3, description="during backup")
    elapsed = time.perf_counter() - t0
    t.join(timeout=30)
    assert elapsed < 0.5, f"备份复制期间 list/其他模型写入不应被阻塞，耗时 {elapsed:.2f}s"
    assert [it["id"] for it in items] == [1, 2, 3]
    assert result["meta"].id == 2 and store.exists(2)
    assert store.samples(2).count("A") == 3


def test_backup_conflict_if_dst_created_during_copy(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    real = shutil.copytree
    src_dir = store.model_dir(1)

    def sneaky(src, *a, **kw):
        out = real(src, *a, **kw)
        if Path(src) == src_dir:
            store.create(2, "sneaky")  # 复制期间别人创建了目标
        return out

    with mock.patch.object(shutil, "copytree", side_effect=sneaky), _err("CONFLICT"):
        store.backup(1, 2)
    assert store.meta(2).name == "sneaky", "后创建的目标不应被覆盖"
    assert sorted(p.name for p in store.root.iterdir()) == ["1", "2"], "临时目录已清理"
    store.backup(1, 2, overwrite=True)
    assert store.meta(2).name == "a"


def test_backup_src_deleted_during_copy(store: ModelStore):
    store.create(1, "a")
    real = shutil.copytree
    src_dir = store.model_dir(1)

    def deleting(src, dst, *a, **kw):
        out = real(src, dst, *a, **kw)
        if Path(src) == src_dir:  # 复制刚结束源就被删掉，且 meta.json 没来得及复制
            shutil.rmtree(src_dir)
            (Path(dst) / "meta.json").unlink()
        return out

    with mock.patch.object(shutil, "copytree", side_effect=deleting), _err("NOT_FOUND"):
        store.backup(1, 2)
    assert not store.exists(2)
    assert sorted(p.name for p in store.root.iterdir()) == []


def test_backup_residual_dir_is_not_a_conflict(store: ModelStore):
    store.create(1, "a")
    _fill(store, 1)
    residual = store.root / "2" / "samples" / "A"
    residual.mkdir(parents=True)
    (residual / "junk.png").write_bytes(b"x")
    assert not store.exists(2)
    assert store.list()[1]["created_at"] is None
    store.backup(1, 2)  # 不应 CONFLICT
    assert store.meta(2).name == "a"
    assert not (residual / "junk.png").exists(), "残留目录被整体替换"
    assert not (store.root / "2.old").exists()
    # delete 允许清理残留目录
    (store.root / "3" / "train").mkdir(parents=True)
    assert not store.exists(3)
    store.delete(3)
    assert not (store.root / "3").exists()


def test_backup_refreshes_cached_repo_seq(store: ModelStore):
    store.create(1, "a")
    r1 = store.samples(1)
    for _ in range(5):
        r1.add("A", _crop(), {"camera_id": 1})
    store.create(2, "b")
    r2 = store.samples(2)
    r2.add("B", _crop(), {"camera_id": 1})  # seq 1
    store.backup(1, 2, overwrite=True)
    assert store.samples(2) is r2, "缓存实例保留"
    s = r2.add("A", _crop(), {"camera_id": 1})
    assert s.seq == 6, "覆盖后 seq 从目标新内容的最大值继续，不与复制来的文件冲突"
    assert r2.count("A") == 6 and r2.count("B") == 0


# ---------------------------------------------------------------- 审查修复：损坏 meta / 严格校验


def test_list_survives_corrupt_meta(store: ModelStore):
    store.create(1, "a")
    store.create(2, "b")
    store.samples(1).add("A", _crop(), {"camera_id": 1})
    store.meta_path(1).write_text("{broken", encoding="utf-8")
    items = store.list(deployed_id=2)
    assert [it["id"] for it in items] == [1, 2, 3]
    broken = items[0]
    assert broken["name"] == "模型1" and broken["created_at"] is not None, "损坏模型仍显示为已创建"
    assert broken["classes"][0] == {"name": "A", "sample_count": 1, "enabled": True}
    assert items[1]["name"] == "b" and items[1]["deployed"] is True
    with _err("VALIDATION"):
        store.meta(1)
    # 字段类型非法同样降级
    store.meta_path(1).write_text('{"id": 1, "name": 123}', encoding="utf-8")
    assert store.describe(1)["name"] == "模型1"
    # update_meta 以默认值为基础写回 → 修复
    m = store.update_meta(1, name="修好了", description="d")
    assert m.name == "修好了" and m.created_at is not None
    assert store.meta(1).name == "修好了" and store.meta(1).sample_counts == {
        "A": 0,
        "B": 0,
        "C": 0,
    }
    assert store.describe(1)["name"] == "修好了"


def test_update_meta_is_strict(store: ModelStore):
    store.create(1, "a")
    store.update_meta(1, image_size=160, version=1, sample_counts={"A": 3})
    with _err("VALIDATION"):
        store.update_meta(1, image_size="160")
    with _err("VALIDATION"):
        store.update_meta(1, version=2.0)
    with _err("VALIDATION"):
        store.update_meta(1, sample_counts={"A": "3"})
    with _err("VALIDATION"):
        store.update_meta(1, classes=[1, 2])
    with _err("VALIDATION"):
        store.update_meta(1, version=True)
    m = store.meta(1)
    assert m.image_size == 160 and m.version == 1 and m.sample_counts == {"A": 3}


# ---------------------------------------------------------------- 审查修复：类别改名 / 类别同步


def test_rename_class_across_models(store: ModelStore):
    store.create(1, "a")
    store.create(2, "b")
    r1 = store.samples(1)
    ids = [r1.add("A", _crop(), {"camera_id": 1}).sample_id for _ in range(2)]
    r1.soft_delete(ids[:1])
    store.samples(2).add("B", _crop(), {"camera_id": 1})
    store.update_meta(1, classes=["A", "B"], sample_counts={"A": 2, "B": 0, "C": 0})

    assert store.rename_class("A", "X") == {1: 2, 2: 0}
    assert store.class_names == ["X", "B", "C"]
    assert r1.class_names == ["X", "B", "C"] and store.samples(2).class_names == ["X", "B", "C"]
    assert store.samples(1).counts() == {"X": 1, "B": 0, "C": 0}
    assert store.samples(2).counts() == {"X": 0, "B": 1, "C": 0}
    m1 = store.meta(1)
    assert m1.classes == ["X", "B"] and m1.sample_counts == {"X": 2, "B": 0, "C": 0}
    assert store.meta(2).sample_counts == {"X": 0, "B": 0, "C": 0}
    items = store.list()
    assert [c["name"] for c in items[0]["classes"]] == ["X", "B", "C"]
    assert [c["name"] for c in items[2]["classes"]] == ["X", "B", "C"], "占位项也用新类别名"
    assert store.samples(1).undo_last("X") == 1
    assert not (store.model_dir(1) / "samples" / "A").exists()
    assert (store.model_dir(2) / "samples" / "X").is_dir()
    # 新建模型按新类别 scaffold
    store.create(3, "c")
    assert (store.model_dir(3) / "samples" / "X").is_dir()
    assert not (store.model_dir(3) / "samples" / "A").exists()

    with _err("VALIDATION"):
        store.rename_class("A", "Q")
    with _err("VALIDATION"):
        store.rename_class("X", "X")
    with _err("VALIDATION"):
        store.rename_class("X", "a/b")
    with _err("CONFLICT"):
        store.rename_class("X", "B")
    assert store.class_names == ["X", "B", "C"]


def test_rename_class_rolls_back_all_models_on_conflict(store: ModelStore):
    store.create(1, "a")
    store.create(2, "b")
    a = store.samples(1).add("A", _crop(), {"camera_id": 1})
    # 模型 2 下已有非空的 samples/X → 该模型 CONFLICT，模型 1 需回滚
    junk = store.model_dir(2) / "samples" / "X"
    junk.mkdir(parents=True)
    (junk / "junk.png").write_bytes(b"x")
    with _err("CONFLICT"):
        store.rename_class("A", "X")
    assert store.class_names == CLASSES
    assert a.path.is_file() and a.thumb_path.is_file()
    assert store.samples(1).counts() == {"A": 1, "B": 0, "C": 0}
    assert not (store.model_dir(1) / "samples" / "X").exists()
    assert (junk / "junk.png").is_file()


def test_set_class_names(store: ModelStore):
    store.create(1, "a")
    store.samples(1).add("A", _crop(), {"camera_id": 1})
    store.set_class_names(["A", "B", "C", "D"])
    assert store.class_names == ["A", "B", "C", "D"]
    for sub in ("samples/D", "samples/.trash/D", "thumbs/D"):
        assert (store.model_dir(1) / sub).is_dir()
    assert store.samples(1).counts() == {"A": 1, "B": 0, "C": 0, "D": 0}
    assert store.describe(1)["classes"][3] == {"name": "D", "sample_count": 0, "enabled": False}
    store.samples(1).add("D", _crop(), {"camera_id": 1})
    with _err("VALIDATION"):
        store.set_class_names([])
    with _err("VALIDATION"):
        store.set_class_names(["A", "A"])
    with _err("VALIDATION"):
        store.set_class_names(["A", ".x"])
    assert store.class_names == ["A", "B", "C", "D"]
    with _err("VALIDATION"):
        ModelStore(store.root, 3, ["A", "A"])
    with _err("VALIDATION"):
        ModelStore(store.root, 0, CLASSES)
