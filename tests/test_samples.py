"""SampleRepo：写入 / 列表 / 软删除 / 撤销 / 清空 / 恢复 / 过期清理 / 并发 / 持久化。"""

from __future__ import annotations

import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from oresorter.data.samples import (
    MAX_LIST_LIMIT,
    Sample,
    SampleRepo,
    StoreError,
    make_thumbnail,
    parse_sample_id,
    rewrite_journal_model_id,
    validate_class_name,
)
from oresorter.fsutil import read_json

CLASSES = ["A", "B", "C"]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "models" / "1"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def repo(model_dir: Path) -> SampleRepo:
    return SampleRepo(model_dir, CLASSES)


def _crop(h: int = 24, w: int = 40, color=(200, 40, 40)) -> np.ndarray:
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = color
    img[3:-3, 3:-3, 3] = 255
    return img


def _raises(code: str):
    class _Ctx:
        def __enter__(self):
            self._ctx = pytest.raises(StoreError)
            return self._ctx.__enter__()

        def __exit__(self, *exc):
            r = self._ctx.__exit__(*exc)
            if r:
                assert self._ctx.excinfo.value.code == code, self._ctx.excinfo.value.to_dict()
            return r

    return _Ctx()


# ---------------------------------------------------------------- add / list


def test_add_writes_triplet_and_roundtrips(repo: SampleRepo, model_dir: Path):
    crop = _crop()
    s = repo.add(
        "A", crop, {"camera_id": 1, "bbox": [10, 20, 40, 24], "area_px": 500, "batch": "j-1"}
    )
    assert s.sample_id.startswith("1/A/cam1_")
    assert s.path == model_dir / "samples" / "A" / f"{s.stem}.png"
    assert s.json_path == model_dir / "samples" / "A" / f"{s.stem}.json"
    assert s.thumb_path == model_dir / "thumbs" / "A" / f"{s.stem}.jpg"
    assert s.path.is_file() and s.json_path.is_file() and s.thumb_path.is_file()
    assert not list(s.path.parent.glob(".*.tmp"))

    # PNG 为 RGBA（cv2 读出 BGRA），alpha 即 mask
    bgra = cv2.imread(str(s.path), cv2.IMREAD_UNCHANGED)
    assert bgra.shape == (24, 40, 4)
    assert np.array_equal(bgra[..., 3], crop[..., 3])
    assert np.array_equal(bgra[..., 2], crop[..., 0]), "R 通道应落在 BGRA 的第 2 位"

    js = read_json(s.json_path)
    assert js["camera_id"] == 1 and js["bbox"] == [10, 20, 40, 24] and js["area_px"] == 500
    assert js["batch"] == "j-1" and js["seq"] == s.seq and js["created_at"] == s.created_at
    assert (
        "sample_id" not in js and "cls" not in js
    ), "可由路径推导的字段不入 json（备份/改名无需重写）"

    d = s.to_dict()
    assert d == {
        "sample_id": s.sample_id,
        "thumb_url": f"/static/models/1/thumbs/A/{s.stem}.jpg",
        "image_url": f"/static/models/1/samples/A/{s.stem}.png",
        "created_at": s.created_at,
        "camera_id": 1,
        "bbox": [10, 20, 40, 24],
        "area_px": 500,
        "deleted": False,
    }
    # created_at 为 ISO 8601 带时区
    assert "T" in s.created_at and ("+" in s.created_at or s.created_at.endswith("Z"))


def test_add_defaults_and_extra_meta(repo: SampleRepo):
    crop = _crop()
    s = repo.add("B", crop, {"camera_id": 2, "note": "x", "job": 5})
    assert s.bbox == [0, 0, 40, 24]
    assert s.area_px == int(np.count_nonzero(crop[..., 3]))
    js = read_json(s.json_path)
    assert js["extra"] == {"note": "x", "job": 5}
    s2 = repo.add("B", crop, None)
    assert s2.camera_id == 0 and s2.stem.startswith("cam0_")


def test_add_validation(repo: SampleRepo):
    with _raises("VALIDATION"):
        repo.add("Z", _crop(), {})
    with _raises("VALIDATION"):
        repo.add("A", np.zeros((10, 10, 3), np.uint8), {})
    with _raises("VALIDATION"):
        repo.add("A", np.zeros((10, 10, 4), np.float32), {})
    with _raises("VALIDATION"):
        repo.add("A", np.zeros((0, 10, 4), np.uint8), {})
    with _raises("VALIDATION"):
        repo.add("A", _crop(), {"bbox": [1, 2, 3]})
    with _raises("VALIDATION"):
        repo.add("A", _crop(), {"camera_id": -1})
    assert repo.counts() == {"A": 0, "B": 0, "C": 0}


def test_list_order_total_and_paging(repo: SampleRepo):
    ids = [repo.add("A", _crop(), {"camera_id": 1}).sample_id for _ in range(5)]
    repo.add("B", _crop(), {"camera_id": 1})
    total, items = repo.list("A")
    assert total == 5
    assert [s.sample_id for s in items] == ids, "最早写入的在前"
    assert all(isinstance(s, Sample) and not s.deleted for s in items)
    total, page = repo.list("A", offset=3, limit=10)
    assert total == 5 and [s.sample_id for s in page] == ids[3:]
    total, page = repo.list("A", offset=1, limit=2)
    assert total == 5 and [s.sample_id for s in page] == ids[1:3]
    total, page = repo.list("A", offset=50)
    assert total == 5 and page == []
    with _raises("VALIDATION"):
        repo.list("A", offset=-1)
    with _raises("VALIDATION"):
        repo.list("A", limit=0)
    with _raises("VALIDATION"):
        repo.list("A", limit=MAX_LIST_LIMIT + 1)
    with _raises("VALIDATION"):
        repo.list("Z")


def test_counts_all_classes_and_iter_active(repo: SampleRepo):
    assert repo.counts() == {"A": 0, "B": 0, "C": 0}
    assert list(repo.counts()) == CLASSES
    a = [repo.add("A", _crop(), {"camera_id": 1}) for _ in range(3)]
    repo.add("C", _crop(), {"camera_id": 1})
    assert repo.counts() == {"A": 3, "B": 0, "C": 1}
    assert repo.count("A") == 3 and repo.count("B") == 0
    assert list(repo.iter_active("A")) == [s.path for s in a]
    repo.soft_delete([a[0].sample_id])
    assert repo.counts() == {"A": 2, "B": 0, "C": 1}
    assert repo.counts(include_deleted=True) == {"A": 3, "B": 0, "C": 1}
    assert repo.count("A", include_deleted=True) == 3
    assert list(repo.iter_active("A")) == [s.path for s in a[1:]]


# ---------------------------------------------------------------- 全链路：删除 / 撤销 / 清空 / 恢复


def test_delete_undo_clear_restore_chain(repo: SampleRepo, model_dir: Path):
    ss = [repo.add("A", _crop(), {"camera_id": 1}) for _ in range(4)]
    ids = [s.sample_id for s in ss]

    # 软删除两张
    batch = repo.soft_delete(ids[:2])
    assert batch == "b-1"
    total, items = repo.list("A")
    assert total == 2 and [s.sample_id for s in items] == ids[2:]
    total, items = repo.list("A", include_deleted=True)
    assert total == 4 and [s.sample_id for s in items] == ids, "含已删除时仍按时间升序"
    assert [s.deleted for s in items] == [True, True, False, False]
    trash_png = model_dir / "samples" / ".trash" / "A" / f"{ss[0].stem}.png"
    trash_jpg = model_dir / "thumbs" / ".trash" / "A" / f"{ss[0].stem}.jpg"
    assert trash_png.is_file() and trash_png.with_suffix(".json").is_file() and trash_jpg.is_file()
    assert not ss[0].path.exists() and not ss[0].thumb_path.exists()
    d = items[0].to_dict()
    assert d["deleted"] is True
    assert d["image_url"] == f"/static/models/1/samples/.trash/A/{ss[0].stem}.png"
    assert d["thumb_url"] == f"/static/models/1/thumbs/.trash/A/{ss[0].stem}.jpg"
    journal = read_json(model_dir / "samples" / ".trash" / "journal.json")
    assert journal["next"] == 2
    assert journal["batches"][0]["batch"] == "b-1" and journal["batches"][0]["cls"] == "A"
    assert journal["batches"][0]["ids"] == ids[:2] and "ts" in journal["batches"][0]

    # 再删一张已删除的 → 跳过，无批次
    assert repo.soft_delete([ids[0]]) is None
    assert read_json(model_dir / "samples" / ".trash" / "journal.json")["next"] == 2

    # get 能取到 .trash 中的
    g = repo.get(ids[0])
    assert g.deleted is True and g.path == trash_png

    # 撤销 → 两张回来
    assert repo.undo_last("A") == 2
    total, items = repo.list("A")
    assert total == 4 and [s.sample_id for s in items] == ids
    assert ss[0].path.is_file() and ss[0].thumb_path.is_file() and ss[0].json_path.is_file()
    assert not trash_png.exists() and not trash_jpg.exists()
    with _raises("NOT_FOUND"):
        repo.undo_last("A")

    # 清空 → 一个批次
    n, batch = repo.clear("A")
    assert n == 4 and batch == "b-2"
    assert repo.count("A") == 0 and repo.count("A", include_deleted=True) == 4
    assert repo.clear("A") == (0, None)
    assert repo.clear("B") == (0, None)
    with _raises("NOT_FOUND"):
        repo.undo_last("B")

    # 撤销清空 → 全部回来
    assert repo.undo_last("A") == 4
    assert repo.count("A") == 4

    # 单张恢复：有效样本 → False；删除后 → True，且 journal 中移除
    assert repo.restore(ids[3]) is False
    b3 = repo.soft_delete([ids[3], ids[2]])
    assert b3 == "b-3"
    assert repo.restore(ids[3]) is True
    assert ss[3].path.is_file()
    journal = read_json(model_dir / "samples" / ".trash" / "journal.json")
    assert journal["batches"][-1]["ids"] == [ids[2]]
    assert repo.undo_last("A") == 1
    assert repo.count("A") == 4
    assert read_json(model_dir / "samples" / ".trash" / "journal.json")["batches"] == []


def test_soft_delete_errors(repo: SampleRepo):
    a = repo.add("A", _crop(), {"camera_id": 1})
    b = repo.add("B", _crop(), {"camera_id": 1})
    with _raises("VALIDATION"):
        repo.soft_delete([])
    with _raises("VALIDATION"):
        repo.soft_delete([a.sample_id, b.sample_id])
    with _raises("VALIDATION"):
        repo.soft_delete(["2/A/" + a.stem])
    with _raises("NOT_FOUND"):
        repo.soft_delete([a.sample_id, "1/A/cam1_20260101_000000_999999"])
    assert repo.count("A") == 1, "任一 id 不存在时整批不应执行"
    with _raises("NOT_FOUND"):
        repo.restore("1/A/cam1_20260101_000000_999999")
    with _raises("VALIDATION"):
        repo.undo_last("Z")


def test_get_not_found_and_parse(repo: SampleRepo):
    with _raises("NOT_FOUND"):
        repo.get("1/A/cam1_20260101_000000_000001")
    with _raises("VALIDATION"):
        repo.get("2/A/cam1_20260101_000000_000001")
    s = repo.add("C", _crop(), {"camera_id": 3})
    got = repo.get(s.sample_id)
    assert got.sample_id == s.sample_id and got.camera_id == 3 and not got.deleted


@pytest.mark.parametrize(
    "sid, expected",
    [
        ("1/A/cam1_20261024_164012_000123", (1, "A", "cam1_20261024_164012_000123")),
        ("12/矿石/x", (12, "矿石", "x")),
    ],
)
def test_parse_sample_id_ok(sid, expected):
    assert parse_sample_id(sid) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "1/A",
        "1/A/b/c",
        "0/A/x",
        "x/A/x",
        "1//x",
        "1/A/",
        "1/.trash/x",
        "1/A/.hidden",
        "1/A/x.png",
        "1/A\\B/x",
        None,
        12,
    ],
)
def test_parse_sample_id_invalid(bad):
    with _raises("VALIDATION"):
        parse_sample_id(bad)


# ---------------------------------------------------------------- 持久化


def test_journal_and_seq_persist_across_instances(model_dir: Path):
    r1 = SampleRepo(model_dir, CLASSES)
    ss = [r1.add("A", _crop(), {"camera_id": 1}) for _ in range(3)]
    ids = [s.sample_id for s in ss]
    r1.soft_delete(ids[:2])
    max_seq = max(s.seq for s in ss)

    r2 = SampleRepo(model_dir, CLASSES)
    new = r2.add("A", _crop(), {"camera_id": 1})
    assert new.seq > max_seq, "seq 计数器不回退（含 .trash 中的文件）"
    assert new.sample_id not in ids
    assert len({*ids, new.sample_id}) == 4
    assert r2.undo_last("A") == 2
    assert r2.count("A") == 4
    batch = r2.soft_delete([new.sample_id])
    assert batch == "b-2", "批次编号从 journal 续号"
    with _raises("NOT_FOUND"):
        SampleRepo(model_dir, CLASSES).undo_last("B")
    assert SampleRepo(model_dir, CLASSES).undo_last("A") == 1


def test_corrupt_journal_treated_as_empty(repo: SampleRepo, model_dir: Path):
    s = repo.add("A", _crop(), {"camera_id": 1})
    repo.soft_delete([s.sample_id])
    (model_dir / "samples" / ".trash" / "journal.json").write_text("{broken", encoding="utf-8")
    with _raises("NOT_FOUND"):
        repo.undo_last("A")
    assert repo.restore(s.sample_id) is True


def test_sample_without_json_still_listed(repo: SampleRepo):
    s = repo.add("A", _crop(), {"camera_id": 7})
    s.json_path.unlink()
    total, items = repo.list("A")
    assert total == 1
    got = items[0]
    assert got.camera_id == 7 and got.seq == s.seq and got.bbox == [0, 0, 0, 0]
    assert got.created_at


# ---------------------------------------------------------------- 缩略图


def test_thumbnail_long_edge(repo: SampleRepo):
    wide = repo.add("A", _crop(h=30, w=90), {"camera_id": 1})
    tall = repo.add("A", _crop(h=400, w=100), {"camera_id": 1})
    img_w = cv2.imread(str(wide.thumb_path))
    img_t = cv2.imread(str(tall.thumb_path))
    assert img_w.shape[:2] == (53, 160)
    assert img_t.shape[:2] == (160, 40)
    assert max(img_w.shape[:2]) == 160 and max(img_t.shape[:2]) == 160


def test_thumbnail_alpha_composited_on_black():
    crop = _crop(h=64, w=64, color=(255, 255, 255))
    thumb = make_thumbnail(crop, 32)
    assert thumb.shape == (32, 32, 3)
    assert thumb[0, 0].tolist() == [0, 0, 0], "mask 外应为黑色"
    assert thumb[16, 16].tolist() == [255, 255, 255]


def test_custom_thumb_size(model_dir: Path):
    r = SampleRepo(model_dir, CLASSES, thumb_size=64)
    s = r.add("A", _crop(h=20, w=80), {"camera_id": 1})
    assert max(cv2.imread(str(s.thumb_path)).shape[:2]) == 64
    with _raises("VALIDATION"):
        SampleRepo(model_dir, CLASSES, thumb_size=8)


# ---------------------------------------------------------------- 并发


def test_concurrent_add_unique_names(repo: SampleRepo):
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(cam: int):
        try:
            for _ in range(10):
                s = repo.add("A", _crop(), {"camera_id": cam})
                with lock:
                    results.append(s.sample_id)
        except BaseException as e:
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(i % 3,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert len(results) == 200
    assert len(set(results)) == 200, "同秒并发写入文件名必须唯一"
    assert repo.count("A") == 200
    total, items = repo.list("A", limit=MAX_LIST_LIMIT)
    assert total == 200 and len(items) == 200
    seqs = [s.seq for s in items]
    assert seqs == sorted(seqs) and len(set(seqs)) == 200


# ---------------------------------------------------------------- purge


def test_purge(repo: SampleRepo, model_dir: Path):
    ss = [repo.add("A", _crop(), {"camera_id": 1}) for _ in range(3)]
    keep = repo.add("B", _crop(), {"camera_id": 1})
    repo.soft_delete([s.sample_id for s in ss[:2]])
    repo.soft_delete([keep.sample_id])
    assert repo.purge(3600) == 0, "未过期不删"
    assert repo.count("A", include_deleted=True) == 3

    # 把 A 类回收站文件的删除时刻改到过去
    old = time.time() - 7200
    for s in ss[:2]:
        for p in (
            model_dir / "samples" / ".trash" / "A" / f"{s.stem}.png",
            model_dir / "samples" / ".trash" / "A" / f"{s.stem}.json",
            model_dir / "thumbs" / ".trash" / "A" / f"{s.stem}.jpg",
        ):
            os.utime(p, (old, old))
    assert repo.purge(3600) == 2
    assert repo.count("A", include_deleted=True) == 1 and repo.count("A") == 1
    assert not (model_dir / "thumbs" / ".trash" / "A" / f"{ss[0].stem}.jpg").exists()
    with _raises("NOT_FOUND"):
        repo.get(ss[0].sample_id)
    journal = read_json(model_dir / "samples" / ".trash" / "journal.json")
    assert [b["cls"] for b in journal["batches"]] == ["B"], "空批次应被移除"
    with _raises("NOT_FOUND"):
        repo.undo_last("A")
    assert repo.undo_last("B") == 1
    assert repo.purge(0) == 0
    repo.soft_delete([ss[2].sample_id])
    assert repo.purge(0) == 1, "older_than_s=0 清掉全部回收站"
    assert repo.count("A", include_deleted=True) == 0
    with _raises("VALIDATION"):
        repo.purge(-1)


# ---------------------------------------------------------------- 磁盘空间


def test_min_free_mb_conflict(model_dir: Path):
    r = SampleRepo(model_dir, CLASSES, min_free_mb=10**9)  # 1 PB，必然不足
    with _raises("CONFLICT") as ei:
        r.add("A", _crop(), {"camera_id": 1})
    assert "磁盘空间不足" in str(ei.value)
    assert r.count("A") == 0
    ok = SampleRepo(model_dir, CLASSES, min_free_mb=1)
    ok.add("A", _crop(), {"camera_id": 1})
    assert ok.count("A") == 1


def test_repo_requires_numeric_dir(tmp_path: Path):
    with _raises("VALIDATION"):
        SampleRepo(tmp_path / "notanumber", CLASSES)
    with _raises("VALIDATION"):
        SampleRepo(tmp_path / "3", CLASSES, min_free_mb=-1)


def test_disk_usage_and_remove_all(repo: SampleRepo, model_dir: Path):
    assert repo.disk_usage() == 0
    repo.add("A", _crop(), {"camera_id": 1})
    assert repo.disk_usage() > 0
    repo.remove_all()
    assert not (model_dir / "samples").exists() and not (model_dir / "thumbs").exists()


# ---------------------------------------------------------------- 审查修复：去重 / numpy / 僵尸批次 / 同名冲突


def test_soft_delete_dedupes_ids(repo: SampleRepo):
    s = repo.add("A", _crop(), {"camera_id": 1})
    t = repo.add("A", _crop(), {"camera_id": 1})
    batch = repo.soft_delete([s.sample_id, s.sample_id, t.sample_id, s.sample_id])
    journal = read_json(repo.journal_path)
    assert journal["batches"][0]["batch"] == batch
    assert journal["batches"][0]["ids"] == [s.sample_id, t.sample_id], "重复 id 只记一次"
    assert repo.undo_last("A") == 2


def test_add_accepts_numpy_camera_id(repo: SampleRepo):
    s = repo.add("A", _crop(), {"camera_id": np.int64(3), "bbox": np.array([1, 2, 3, 4])})
    assert s.stem.startswith("cam3_") and s.camera_id == 3 and type(s.camera_id) is int
    js = read_json(s.json_path)
    assert js["camera_id"] == 3 and js["bbox"] == [1, 2, 3, 4]
    with _raises("VALIDATION"):
        repo.add("A", _crop(), {"camera_id": np.int64(-1)})
    with _raises("VALIDATION"):
        repo.add("A", _crop(), {"camera_id": True})
    with _raises("VALIDATION"):
        repo.add("A", _crop(), {"camera_id": 1.5})


def _trash_triplet(repo: SampleRepo, s: Sample) -> list[Path]:
    return [
        repo.class_dir(s.cls, True) / f"{s.stem}.png",
        repo.class_dir(s.cls, True) / f"{s.stem}.json",
        repo.thumb_dir(s.cls, True) / f"{s.stem}.jpg",
    ]


def test_undo_last_skips_zombie_batches(repo: SampleRepo):
    a = repo.add("A", _crop(), {"camera_id": 1})
    b = repo.add("A", _crop(), {"camera_id": 1})
    repo.soft_delete([a.sample_id])  # b-1
    repo.soft_delete([b.sample_id])  # b-2
    for p in _trash_triplet(repo, b):  # 外部清理掉 b-2 的文件 → 僵尸批次
        p.unlink()
    assert repo.undo_last("A") == 1, "僵尸批次跳过，继续恢复更早的 b-1"
    assert a.path.is_file()
    assert read_json(repo.journal_path)["batches"] == [], "僵尸批次应被清掉"
    with _raises("NOT_FOUND"):
        repo.undo_last("A")

    c = repo.add("A", _crop(), {"camera_id": 1})
    repo.soft_delete([c.sample_id])
    for p in _trash_triplet(repo, c):
        p.unlink()
    with _raises("NOT_FOUND"):
        repo.undo_last("A")
    assert read_json(repo.journal_path)["batches"] == [], "只剩僵尸批次时也要清掉并报 NOT_FOUND"


def test_duplicate_stem_in_active_and_trash(repo: SampleRepo):
    s = repo.add("A", _crop(), {"camera_id": 1})
    trash_png = repo.class_dir("A", True) / s.path.name
    trash_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(s.path, trash_png)
    total, items = repo.list("A", include_deleted=True)
    assert total == 1 and items[0].deleted is False, "同名以正常目录为准，只列一条"
    assert repo.count("A", include_deleted=True) == 2, "count 只数文件，不去重（文档约定）"
    assert repo.restore(s.sample_id) is False
    assert not trash_png.exists(), "restore 顺手清理 .trash 孤儿"
    assert s.path.is_file()
    # 再制造一份孤儿，soft_delete 直接覆盖它（不报错）
    shutil.copy(s.path, trash_png)
    assert repo.soft_delete([s.sample_id]) == "b-1"
    assert trash_png.is_file() and not s.path.exists()
    assert repo.undo_last("A") == 1


# ---------------------------------------------------------------- 审查修复：seq 与时间戳 / 排序


def test_list_order_follows_seq_even_if_clock_jumps_back(model_dir: Path):
    base = datetime(2026, 1, 1, 0, 0, 5).astimezone()
    times = iter([base, base - timedelta(seconds=2), base - timedelta(seconds=1)])

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(times)

    r = SampleRepo(model_dir, CLASSES)
    with mock.patch("oresorter.data.samples.datetime", FakeDatetime):
        ss = [r.add("A", _crop(), {"camera_id": 1}) for _ in range(3)]
    stamps = [s.stem.split("_")[1] + s.stem.split("_")[2] for s in ss]
    assert stamps == ["20260101000005", "20260101000003", "20260101000004"], "时钟确实回跳了"
    assert [s.seq for s in ss] == [1, 2, 3]
    total, items = r.list("A")
    assert total == 3
    assert [s.seq for s in items] == [1, 2, 3], "排序以 seq 为准（写入顺序），不受时钟回跳影响"
    assert [s.sample_id for s in items] == [s.sample_id for s in ss]
    assert [p.name for p in r.iter_active("A")] == [s.path.name for s in ss]


def test_refresh_seq_never_goes_backwards(repo: SampleRepo, model_dir: Path):
    s = repo.add("A", _crop(), {"camera_id": 1})
    assert repo.refresh_seq() == s.seq + 1
    # 外部放入更大 seq 的文件（如“备份至”覆盖了目录）
    (model_dir / "samples" / "A" / "cam1_20260101_000000_000900.png").write_bytes(b"x")
    assert repo.refresh_seq() == 901
    assert repo.add("A", _crop(), {"camera_id": 1}).seq == 901
    s.path.unlink()
    assert repo.refresh_seq() == 902, "只增不减"


# ---------------------------------------------------------------- 审查修复：失败路径


def test_add_cleans_up_when_png_write_fails(repo: SampleRepo, model_dir: Path):
    from oresorter.data import samples as mod

    real = mod.atomic_write_bytes

    def flaky(path, data, **kw):
        if str(path).endswith(".png"):
            raise OSError(28, "No space left on device")
        return real(path, data, **kw)

    with mock.patch.object(mod, "atomic_write_bytes", side_effect=flaky), _raises("INTERNAL") as ei:
        repo.add("A", _crop(), {"camera_id": 1})
    assert ei.value.detail["errno"] == 28
    for d in (model_dir / "samples" / "A", model_dir / "thumbs" / "A"):
        assert not any(d.iterdir()), f"{d} 应无残留（json/jpg/tmp）"
    assert repo.count("A") == 0
    s = repo.add("A", _crop(), {"camera_id": 1})
    assert s.seq == 2 and s.path.is_file(), "失败后 seq 不复用，再写正常"


def test_move_triplet_rolls_back_on_failure(repo: SampleRepo):
    s = repo.add("A", _crop(), {"camera_id": 1})
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 3:  # json、jpg 已移，png 失败
            raise OSError(5, "I/O error")
        return real(src, dst)

    with mock.patch("oresorter.data.samples.os.replace", side_effect=flaky), _raises("INTERNAL"):
        repo.soft_delete([s.sample_id])
    assert calls["n"] == 5, "两次回滚"
    assert s.path.is_file() and s.json_path.is_file() and s.thumb_path.is_file()
    assert not any(_trash_triplet(repo, s)[i].exists() for i in range(3))
    assert not repo.journal_path.exists() or read_json(repo.journal_path)["batches"] == []
    assert repo.soft_delete([s.sample_id]) == "b-1", "回滚后可正常删除"


# ---------------------------------------------------------------- 审查修复：journal 按 (cls, stem) 匹配 / 前缀改写


def test_journal_matching_ignores_model_prefix(model_dir: Path, tmp_path: Path):
    r1 = SampleRepo(model_dir, CLASSES)
    ss = [r1.add("A", _crop(), {"camera_id": 1}) for _ in range(2)]
    r1.soft_delete([s.sample_id for s in ss])
    # 整目录复制到编号 2（模拟“备份至”未改写 journal 的情形）
    d2 = tmp_path / "data" / "models" / "2"
    shutil.copytree(model_dir, d2)
    r2 = SampleRepo(d2, CLASSES)
    new_ids = [s.sample_id.replace("1/", "2/", 1) for s in ss]
    assert r2.restore(new_ids[0]) is True
    assert read_json(r2.journal_path)["batches"][0]["ids"] == [
        ss[1].sample_id
    ], "按 (cls, stem) 移除"
    assert r2.undo_last("A") == 1
    assert read_json(r2.journal_path)["batches"] == []
    # 改写前缀
    d3 = tmp_path / "data" / "models" / "3"
    shutil.copytree(model_dir, d3)
    assert rewrite_journal_model_id(d3 / "samples" / ".trash" / "journal.json", 3) == 2
    assert read_json(d3 / "samples" / ".trash" / "journal.json")["batches"][0]["ids"] == [
        s.sample_id.replace("1/", "3/", 1) for s in ss
    ]
    assert rewrite_journal_model_id(d3 / "samples" / ".trash" / "journal.json", 3) == 0, "幂等"
    assert rewrite_journal_model_id(tmp_path / "missing.json", 3) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    assert rewrite_journal_model_id(bad, 3) == 0 and bad.read_text(encoding="utf-8") == "{broken"


# ---------------------------------------------------------------- 审查修复：类别改名 / 历史类别


def test_rename_class_moves_dirs_and_rewrites_journal(repo: SampleRepo, model_dir: Path):
    a = [repo.add("A", _crop(), {"camera_id": 1}) for _ in range(3)]
    repo.add("B", _crop(), {"camera_id": 1})
    repo.soft_delete([a[0].sample_id])
    assert repo.rename_class("A", "X") == 3
    assert repo.class_names == ["X", "B", "C"]
    assert not (model_dir / "samples" / "A").exists()
    assert not (model_dir / "samples" / ".trash" / "A").exists()
    assert (
        not (model_dir / "thumbs" / "A").exists()
        and not (model_dir / "thumbs" / ".trash" / "A").exists()
    )
    assert repo.counts() == {"X": 2, "B": 1, "C": 0}
    total, items = repo.list("X", include_deleted=True)
    assert total == 3 and [s.stem for s in items] == [s.stem for s in a]
    assert items[0].deleted and items[0].to_dict()["image_url"].endswith(
        f"/.trash/X/{a[0].stem}.png"
    )
    journal = read_json(repo.journal_path)
    assert journal["batches"][0]["cls"] == "X"
    assert journal["batches"][0]["ids"] == [f"1/X/{a[0].stem}"]
    assert repo.undo_last("X") == 1
    assert repo.count("X") == 3
    assert repo.orphan_classes() == []
    with _raises("VALIDATION"):
        repo.rename_class("X", "X")
    with _raises("VALIDATION"):
        repo.rename_class("X", ".hidden")
    with _raises("VALIDATION"):
        repo.rename_class("X", "a/b")
    # 目标非空 → CONFLICT，且不动任何目录
    with _raises("CONFLICT"):
        repo.rename_class("X", "B")
    assert repo.counts() == {"X": 3, "B": 1, "C": 0}
    # 目标是空目录（scaffold 预建）→ 允许
    (model_dir / "samples" / "C").mkdir(exist_ok=True)
    assert repo.rename_class("X", "Y") == 3
    assert repo.counts() == {"Y": 3, "B": 1, "C": 0}
    # 重新构造的 repo 带新类别名也能看到数据
    assert SampleRepo(model_dir, ["Y", "B", "C"]).count("Y") == 3


def test_rename_class_rolls_back_on_failure(repo: SampleRepo, model_dir: Path):
    a = repo.add("A", _crop(), {"camera_id": 1})
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # samples/A 已改名，thumbs/A 失败
            raise OSError(5, "I/O error")
        return real(src, dst)

    with mock.patch("oresorter.data.samples.os.replace", side_effect=flaky), _raises("INTERNAL"):
        repo.rename_class("A", "X")
    assert calls["n"] == 3, "一次回滚"
    assert a.path.is_file() and a.thumb_path.is_file()
    assert repo.class_names == CLASSES and repo.count("A") == 1
    assert not (model_dir / "samples" / "X").exists()


def test_orphan_class_dirs_are_visible_and_deletable(model_dir: Path):
    r = SampleRepo(model_dir, CLASSES)
    ss = [r.add("A", _crop(), {"camera_id": 1}) for _ in range(2)]
    r.soft_delete([ss[0].sample_id])
    # 配置改名但没迁移 → A 成为历史类别目录
    r2 = SampleRepo(model_dir, ["X", "B", "C"])
    assert r2.counts() == {"X": 0, "B": 0, "C": 0}
    assert r2.orphan_classes() == ["A"]
    with _raises("VALIDATION"):
        r2.add("A", _crop(), {"camera_id": 1})
    total, _items = r2.list("A", include_deleted=True)
    assert total == 2 and r2.count("A") == 1
    assert r2.undo_last("A") == 1
    assert r2.clear("A") == (2, "b-2")
    assert r2.purge(0) == 2
    assert r2.count("A", include_deleted=True) == 0
    with _raises("VALIDATION"):
        r2.list("Z")
    with _raises("VALIDATION"):
        r2.list(".trash")


def test_set_class_names_and_validate_class_name(repo: SampleRepo):
    repo.set_class_names(["A", "B", "C", "D"])
    assert repo.counts() == {"A": 0, "B": 0, "C": 0, "D": 0}
    repo.add("D", _crop(), {"camera_id": 1})
    assert repo.count("D") == 1
    with _raises("VALIDATION"):
        repo.set_class_names(["A", "x/y"])
    for bad in ["", ".a", "a/b", "a\\b", None, 1]:
        with _raises("VALIDATION"):
            validate_class_name(bad)
    validate_class_name("矿石-A")
