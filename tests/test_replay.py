"""``cli replay``（``oresorter.vision.replay``）：合成大图 → 分割 →（可选）分类 → 标注图 + replay_report.json。

覆盖：整图路径（报告结构、召回）、按块路径与整图一致（颗粒数 ±1、bbox 对齐）、有模型时每颗带 cls/conf、
``--learn-bg`` 列背景、非法参数返回 1 不抛堆栈、单张坏图 / 分类器或分割层抛任意异常不拖垮整次回放、
同名不同扩展名图片的标注图不互相覆盖、输出目录不得落在图片目录内、labels.json 缺键 / 值非列表告警并计数、
边界颗粒标记、``oresorter.cli.main`` 入口。模型用 2 类 × 8 张 128 px 快速训 1 个 epoch（约 1 s）。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from oresorter import cli
from oresorter.ml import train
from oresorter.vision import replay, synthetic

REPO = Path(__file__).resolve().parent.parent
CLASSES = ["A", "B"]
WIDTH, HEIGHT = 1024, 600
N_FRAMES, N_PARTICLES = 2, 8
# 合成生成器不避免颗粒重叠；重叠颗粒会被分割成一个连通域而拉低“召回”。seed 13 的两张图颗粒互不重叠
# （整图 / 按块 / 列背景三种路径召回均为 1.0），用它验证召回指标本身。
FRAMES_SEED = 13


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frames_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``synthetic.make_frames`` 的大图目录（含 labels.json）。"""
    root = tmp_path_factory.mktemp("frames")
    return synthetic.make_frames(
        root,
        N_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        rng=np.random.default_rng(FRAMES_SEED),
        n_particles=N_PARTICLES,
        classes=CLASSES,
    )


@pytest.fixture(scope="module")
def cfg_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """临时 yaml：继承仓库 config/default.yaml，color 背景，data_dir 指向临时目录。"""
    root = tmp_path_factory.mktemp("cfg")
    data = {
        "base": str(REPO / "config" / "default.yaml"),
        "system": {"data_dir": str(root / "data"), "mcu_kind": "mock"},
        "capture": {"bg_mode": "color", "bg_color": [0, 0, 255]},
    }
    p = root / "replay.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """快速训一个 2 类 128 px 模型（1 epoch、不加载预训练权重），产物 best.pt + model.onnx。"""
    mdir = tmp_path_factory.mktemp("model") / "models" / "1"
    synthetic.make_sample_set(mdir, CLASSES, 8, rng=np.random.default_rng(1))
    cfg = train.TrainConfig(
        image_size=128,
        max_epochs=1,
        head_epochs=0,
        batch_size=8,
        pretrained=False,
        device="cpu",
        min_samples_per_class=5,
        num_workers=0,
        seed=0,
    )
    result = train.run(mdir, None, cfg, class_names=CLASSES)
    assert result["artifacts"]["onnx"] is True
    return mdir


def make_args(images: Path, config: Path, **over) -> argparse.Namespace:
    """与 ``cli.build_parser`` 的 replay 子命令同名的 Namespace。"""
    base = {
        "images": images,
        "config": config,
        "model_dir": None,
        "chunk_lines": None,
        "out": None,
        "learn_bg": 0,
    }
    base.update(over)
    return argparse.Namespace(**base)


def run_and_read(args: argparse.Namespace) -> tuple[int, dict]:
    rc = replay.run(args)
    report_path = Path(args.out) / replay.REPORT_FILE
    assert report_path.is_file(), "无论成败都应写出报告"
    return rc, json.loads(report_path.read_text(encoding="utf-8"))


def _labels(frames: Path) -> dict:
    return json.loads((frames / "labels.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# 整图路径
# --------------------------------------------------------------------------------------


def test_whole_image_report_overlay_and_recall(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    out = tmp_path / "out"
    rc, rep = run_and_read(make_args(frames_dir, cfg_yaml, out=out))
    assert rc == 0

    # 标注图：每张一张 jpg，尺寸与原图一致，且确实画了东西
    for i in range(N_FRAMES):
        overlay = out / f"{i:04d}_overlay.jpg"
        assert overlay.is_file()
        ov = cv2.imread(str(overlay))
        src = cv2.imread(str(frames_dir / f"{i:04d}.png"))
        assert ov.shape == src.shape == (HEIGHT, WIDTH, 3)
        assert np.abs(ov.astype(int) - src.astype(int)).max() > 100, "overlay 应画有白色 bbox"
    assert not list(out.glob(".*.tmp"))

    # 报告结构
    assert rep["version"] == replay.REPORT_VERSION
    assert rep["images_dir"] == str(frames_dir) and rep["out_dir"] == str(out)
    cfg = rep["config"]
    assert cfg["mode"] == "whole" and cfg["chunk_lines"] is None and cfg["bg"] == "color"
    assert cfg["overlap_lines"] == 256 and cfg["capture"]["bg_mode"] == "color"
    assert cfg["model_dir"] is None and cfg["backend"] is None and cfg["classes"] is None
    assert cfg["path"] == str(cfg_yaml) and cfg["learn_bg"] == 0
    assert cfg["bg_info"] == {"frames_used": 0, "width": None, "mean_rgb": [0, 0, 255]}

    assert len(rep["images"]) == N_FRAMES
    labels = _labels(frames_dir)
    for i, img in enumerate(rep["images"]):
        assert img["name"] == f"{i:04d}.png" and img["overlay"] == f"{i:04d}_overlay.jpg"
        assert img["width"] == WIDTH and img["height"] == HEIGHT
        assert img["count"] == len(img["particles"]) > 0
        assert img["per_class"] == {} and img["stitch"] is None
        assert img["ms"] >= img["segment_ms"] >= 0 and img["infer_ms"] == 0
        assert img["area_min"] <= img["area_mean"] <= img["area_max"]
        for p in img["particles"]:
            x, y, w, h = p["bbox"]
            assert 0 <= x < x + w <= WIDTH and 0 <= y < y + h <= HEIGHT
            assert p["area_px"] > 0 and p["cls"] is None and p["conf"] is None
            assert p["trailing_line"] == y + h - 1, "整图回放：尾沿行号 = 图内尾沿行"
            assert p["truncated"] is False
            assert p["touches_top"] is (y == 0) and p["touches_bottom"] is (y + h == HEIGHT)
        assert img["edge"] == sum(
            1 for p in img["particles"] if p["touches_top"] or p["touches_bottom"]
        )
        lab = img["labels"]
        assert lab["n_labels"] == len(labels[img["name"]]) == N_PARTICLES
        assert lab["recall"] >= 0.9 and lab["false_positives"] == 0
        assert lab["cls_accuracy"] is None and lab["cls_evaluated"] == 0

    totals = rep["totals"]
    assert totals["images"] == totals["images_ok"] == N_FRAMES and totals["images_failed"] == 0
    assert totals["particles"] == sum(img["count"] for img in rep["images"])
    assert totals["edge"] == sum(img["edge"] for img in rep["images"])
    assert totals["labels"]["n_labels"] == N_FRAMES * N_PARTICLES
    assert totals["labels"]["recall"] >= 0.9
    assert totals["labels"]["images_with_labels"] == N_FRAMES
    assert totals["labels"]["images_without_labels"] == 0
    assert totals["per_class"] == {}
    timing = rep["timing_ms"]
    assert 0 < timing["p50"] <= timing["p99"] <= timing["max"]
    assert timing["total"] == pytest.approx(sum(img["ms"] for img in rep["images"]), abs=0.1)
    assert rep["elapsed_s"] > 0


def test_detections_align_with_label_boxes(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    """分割 bbox 与合成标注 bbox 的 IoU 很高（形态学膨胀只扩 1~2 px）。"""
    rc, rep = run_and_read(make_args(frames_dir, cfg_yaml, out=tmp_path / "out"))
    assert rc == 0
    labels = _labels(frames_dir)
    for img in rep["images"]:
        for lab in labels[img["name"]]:
            best = max(replay.iou_xywh(lab["bbox"], p["bbox"]) for p in img["particles"])
            assert best >= 0.8, (lab, best)


def test_default_out_dir_is_sibling_replay_out(frames_dir: Path, cfg_yaml: Path):
    rc = replay.run(make_args(frames_dir, cfg_yaml))
    assert rc == 0
    default_out = frames_dir.resolve().parent / "replay_out"
    assert (default_out / replay.REPORT_FILE).is_file()
    assert (default_out / "0000_overlay.jpg").is_file()
    shutil.rmtree(default_out)


# --------------------------------------------------------------------------------------
# 按块路径
# --------------------------------------------------------------------------------------


def test_chunked_matches_whole_image(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    """chunk_lines=256 按块回放：每张颗粒数与整图一致（±1），bbox 换算回大图后一一对应。"""
    rc_w, whole = run_and_read(make_args(frames_dir, cfg_yaml, out=tmp_path / "whole"))
    rc_c, chunked = run_and_read(
        make_args(frames_dir, cfg_yaml, out=tmp_path / "chunk", chunk_lines=256)
    )
    assert rc_w == 0 and rc_c == 0
    assert chunked["config"]["mode"] == "chunked" and chunked["config"]["chunk_lines"] == 256
    assert abs(chunked["totals"]["particles"] - whole["totals"]["particles"]) <= 1
    n_chunks = -(-HEIGHT // 256)
    for w_img, c_img in zip(whole["images"], chunked["images"], strict=True):
        assert abs(c_img["count"] - w_img["count"]) <= 1
        st = c_img["stitch"]
        assert st["frames"] == n_chunks + 1, "N 块 + 1 帧冲洗帧"
        assert st["dropped"] == 0 and st["resets"] == 0 and st["lost"] == 0
        assert st["selected"] == c_img["count"]
        assert st["deferred"] > 0, "600 行切 256 行一定有颗粒跨块"
        # 每颗整图颗粒都能在按块结果里找到几乎同一个框；尾沿行号是大图行号
        for wp in w_img["particles"]:
            best = max(replay.iou_xywh(wp["bbox"], cp["bbox"]) for cp in c_img["particles"])
            assert best >= 0.9, (wp["bbox"], best)
        for cp in c_img["particles"]:
            x, y, w, h = cp["bbox"]
            assert 0 <= x < x + w <= WIDTH and 0 <= y < y + h <= HEIGHT
            assert cp["trailing_line"] == y + h - 1
            assert cp["truncated"] is False
        assert c_img["labels"]["recall"] >= 0.9 and c_img["labels"]["false_positives"] == 0
        # 按尾沿行排序
        trailing = [cp["trailing_line"] for cp in c_img["particles"]]
        assert trailing == sorted(trailing)


def test_chunked_flush_handles_particles_touching_last_row(cfg_yaml: Path, tmp_path: Path):
    """最后一块里触底的颗粒不能因为“没有下一块”而丢失（末尾冲洗帧）。"""
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (0, 0, 255)
    cv2.rectangle(img, (50, 40), (110, 100), (200, 40, 40), -1)  # 完整颗粒
    cv2.rectangle(img, (200, 250), (260, 299), (200, 40, 40), -1)  # 触底
    cv2.rectangle(img, (300, 0), (360, 40), (200, 40, 40), -1)  # 触顶
    d = tmp_path / "frames"
    d.mkdir()
    cv2.imwrite(str(d / "belt.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    rc_w, whole = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "w"))
    rc_c, chunked = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "c", chunk_lines=128))
    assert rc_w == 0 and rc_c == 0
    assert whole["totals"]["particles"] == chunked["totals"]["particles"] == 3
    boxes = sorted(p["bbox"] for p in chunked["images"][0]["particles"])
    assert boxes == sorted(p["bbox"] for p in whole["images"][0]["particles"])
    assert max(p["trailing_line"] for p in chunked["images"][0]["particles"]) == 299
    assert (tmp_path / "c" / "belt_overlay.jpg").is_file()
    # 顶到图片上下边界的颗粒在两种模式下都按大图边界标记（便于使用者过滤跨帧颗粒），但不算截断
    for rep in (whole, chunked):
        parts = rep["images"][0]["particles"]
        # 膨胀使 bbox 比画的矩形宽 1~2 px，按最近的 x 找回三颗颗粒
        by_x = {x0: min(parts, key=lambda p: abs(p["bbox"][0] - x0)) for x0 in (50, 200, 300)}
        assert all(abs(by_x[x0]["bbox"][0] - x0) <= 3 for x0 in by_x)
        assert by_x[50]["touches_top"] is False and by_x[50]["touches_bottom"] is False
        assert by_x[200]["touches_bottom"] is True and by_x[200]["touches_top"] is False
        assert by_x[300]["touches_top"] is True and by_x[300]["touches_bottom"] is False
        assert all(p["truncated"] is False for p in by_x.values())
        assert rep["images"][0]["edge"] == 2 and rep["totals"]["edge"] == 2


# --------------------------------------------------------------------------------------
# 有模型
# --------------------------------------------------------------------------------------


def test_with_model_every_particle_has_cls_and_conf(
    frames_dir: Path, cfg_yaml: Path, model_dir: Path, tmp_path: Path
):
    out = tmp_path / "out"
    rc, rep = run_and_read(make_args(frames_dir, cfg_yaml, out=out, model_dir=model_dir))
    assert rc == 0
    cfg = rep["config"]
    assert cfg["model_dir"] == str(model_dir)
    assert cfg["backend"] == "onnx", "开发机 prefer=(onnx, torch) 应用 ONNX"
    assert cfg["classes"] == CLASSES and cfg["image_size"] == 128
    total_per_class: dict[str, int] = {}
    for img in rep["images"]:
        assert img["count"] > 0 and img["infer_ms"] > 0
        assert sum(img["per_class"].values()) == img["count"]
        for p in img["particles"]:
            assert p["cls"] in CLASSES
            assert isinstance(p["conf"], float) and 0.0 <= p["conf"] <= 1.0
        for k, v in img["per_class"].items():
            total_per_class[k] = total_per_class.get(k, 0) + v
        lab = img["labels"]
        assert lab["cls_evaluated"] == lab["matched"] > 0
        assert 0 <= lab["cls_correct"] <= lab["cls_evaluated"]
        assert isinstance(lab["cls_accuracy"], float) and 0.0 <= lab["cls_accuracy"] <= 1.0
    assert rep["totals"]["per_class"] == total_per_class
    assert isinstance(rep["totals"]["labels"]["cls_accuracy"], float)
    # 标注图上带类别文字（比无模型的 overlay 多出的字符像素）
    plain = tmp_path / "plain"
    assert replay.run(make_args(frames_dir, cfg_yaml, out=plain)) == 0
    a = cv2.imread(str(out / "0000_overlay.jpg")).astype(int)
    b = cv2.imread(str(plain / "0000_overlay.jpg")).astype(int)
    assert (np.abs(a - b).max(axis=2) > 60).sum() > 50


def test_with_model_chunked_classifies_global_coordinates(
    frames_dir: Path, cfg_yaml: Path, model_dir: Path, tmp_path: Path
):
    """按块 + 模型：切图来自处理图（含重叠区），分类正常，坐标为大图坐标。"""
    rc, rep = run_and_read(
        make_args(frames_dir, cfg_yaml, out=tmp_path / "out", model_dir=model_dir, chunk_lines=300)
    )
    assert rc == 0 and rep["config"]["mode"] == "chunked"
    for img in rep["images"]:
        assert img["stitch"]["frames"] == 3
        for p in img["particles"]:
            assert p["cls"] in CLASSES and p["conf"] is not None
            assert p["bbox"][1] + p["bbox"][3] <= HEIGHT


# --------------------------------------------------------------------------------------
# 列背景
# --------------------------------------------------------------------------------------


def test_learn_bg_uses_column_background(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    rc_w, whole = run_and_read(make_args(frames_dir, cfg_yaml, out=tmp_path / "w"))
    rc_c, col = run_and_read(make_args(frames_dir, cfg_yaml, out=tmp_path / "c", learn_bg=1))
    assert rc_w == 0 and rc_c == 0
    assert col["config"]["bg"] == "column" and col["config"]["learn_bg"] == 1
    info = col["config"]["bg_info"]
    assert info["frames_used"] == 1 and info["width"] == WIDTH
    # 从带颗粒的图按列取中值学到的背景与蓝底几乎一致，颗粒数应与颜色背景相同
    assert np.abs(np.array(info["mean_rgb"]) - np.array([0, 0, 255])).max() < 10
    assert abs(col["totals"]["particles"] - whole["totals"]["particles"]) <= 1
    assert col["totals"]["labels"]["recall"] >= 0.9


def test_column_mode_without_learn_bg_falls_back_to_color(
    frames_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    data = {
        "base": str(REPO / "config" / "default.yaml"),
        "system": {"data_dir": str(tmp_path / "data"), "mcu_kind": "mock"},
        "capture": {"bg_mode": "column"},
    }
    p = tmp_path / "col.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with caplog.at_level("INFO", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(make_args(frames_dir, p, out=tmp_path / "out"))
    assert rc == 0 and rep["config"]["bg"] == "color" and rep["totals"]["particles"] > 0
    assert any("回放改用固定颜色背景" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------------------
# 错误路径
# --------------------------------------------------------------------------------------


def test_invalid_arguments_return_1_without_traceback(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    out = tmp_path / "out"
    cases = {
        "images 不存在": make_args(tmp_path / "missing", cfg_yaml, out=out),
        "images 是文件": make_args(cfg_yaml, cfg_yaml, out=out),
        "chunk_lines 过小": make_args(frames_dir, cfg_yaml, out=out, chunk_lines=127),
        "chunk_lines 过大": make_args(frames_dir, cfg_yaml, out=out, chunk_lines=4097),
        "chunk_lines 非整数": make_args(frames_dir, cfg_yaml, out=out, chunk_lines="abc"),
        "learn_bg 为负": make_args(frames_dir, cfg_yaml, out=out, learn_bg=-1),
        "learn_bg 大于图片数": make_args(frames_dir, cfg_yaml, out=out, learn_bg=N_FRAMES + 1),
        "模型目录不存在": make_args(frames_dir, cfg_yaml, out=out, model_dir=tmp_path / "nomodel"),
        "配置不存在": make_args(frames_dir, tmp_path / "nope.yaml", out=out),
    }
    empty_model = tmp_path / "empty_model"
    empty_model.mkdir()
    cases["模型目录无产物"] = make_args(frames_dir, cfg_yaml, out=out, model_dir=empty_model)
    no_images = tmp_path / "no_images"
    no_images.mkdir()
    (no_images / "notes.txt").write_text("x", encoding="utf-8")
    cases["目录里没有图片"] = make_args(no_images, cfg_yaml, out=out)
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        yaml.safe_dump(
            {"base": str(REPO / "config" / "default.yaml"), "capture": {"bg_threshold": 999}}
        ),
        encoding="utf-8",
    )
    cases["配置越界"] = make_args(frames_dir, bad_cfg, out=out)

    for name, args in cases.items():
        caplog.clear()
        with caplog.at_level("ERROR", logger="oresorter.vision.replay"):
            rc = replay.run(args)
        assert rc == 1, name
        assert any(r.levelname == "ERROR" for r in caplog.records), name
    assert not (out / replay.REPORT_FILE).exists(), "参数错误时不应产生报告"

    # chunk_lines 边界值合法
    assert replay.run(make_args(frames_dir, cfg_yaml, out=tmp_path / "lo", chunk_lines=128)) == 0
    assert replay.run(make_args(frames_dir, cfg_yaml, out=tmp_path / "hi", chunk_lines=4096)) == 0


def test_unreadable_image_is_reported_not_fatal(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """目录里混入一张坏图：其余图片正常回放，报告记录错误，返回 1。"""
    d = tmp_path / "frames"
    d.mkdir()
    shutil.copy(frames_dir / "0000.png", d / "0000.png")
    (d / "0001.png").write_bytes(b"not a png")
    with caplog.at_level("ERROR", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "out"))
    assert rc == 1
    assert rep["totals"]["images"] == 2 and rep["totals"]["images_ok"] == 1
    assert rep["totals"]["images_failed"] == 1
    assert rep["images"][0]["count"] > 0 and rep["images"][0]["labels"] is None
    assert rep["images"][1] == {"name": "0001.png", "error": "无法读取图片"}
    assert rep["totals"]["labels"] is None, "没有 labels.json 时不报召回"
    assert (tmp_path / "out" / "0000_overlay.jpg").is_file()
    assert any("无法读取图片" in r.getMessage() for r in caplog.records)


def test_corrupt_labels_json_is_ignored(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    d = tmp_path / "frames"
    shutil.copytree(frames_dir, d)
    (d / "labels.json").write_text("{broken", encoding="utf-8")
    rc, rep = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "out"))
    assert rc == 0 and rep["totals"]["labels"] is None
    assert all(img["labels"] is None for img in rep["images"])


@pytest.mark.parametrize("stage", ["classify", "segment"])
def test_per_image_exception_is_reported_not_fatal(
    frames_dir: Path,
    cfg_yaml: Path,
    model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stage: str,
):
    """分类器（onnxruntime/torch RuntimeError）或分割层（cv2.error）在第一张图上抛任意异常：
    只记为该图的 error 条目（带堆栈日志），第二张照常回放，报告写出，返回 1，不抛堆栈。"""
    calls = {"n": 0}
    if stage == "classify":
        orig_classify = replay.classify_particles

        def boom_classify(particles, classifier):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("onnxruntime boom")
            orig_classify(particles, classifier)

        monkeypatch.setattr(replay, "classify_particles", boom_classify)
        args = make_args(frames_dir, cfg_yaml, out=tmp_path / "out", model_dir=model_dir)
        exc_name = "RuntimeError"
    else:
        orig_segment = replay.segment_whole

        def boom_segment(image, bg, settings):
            calls["n"] += 1
            if calls["n"] == 1:
                raise cv2.error("cv boom")
            return orig_segment(image, bg, settings)

        monkeypatch.setattr(replay, "segment_whole", boom_segment)
        args = make_args(frames_dir, cfg_yaml, out=tmp_path / "out")
        exc_name = cv2.error.__name__

    with caplog.at_level("ERROR", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(args)
    assert rc == 1 and calls["n"] == 2
    t = rep["totals"]
    assert t["images"] == 2 and t["images_ok"] == 1 and t["images_failed"] == 1
    bad, good = rep["images"]
    assert bad == {"name": "0000.png", "error": f"{bad['error']}", "error_type": exc_name}
    assert "boom" in bad["error"]
    assert good["name"] == "0001.png" and good["count"] > 0 and good["labels"]["recall"] >= 0.9
    if stage == "classify":
        assert all(p["cls"] in CLASSES and p["conf"] is not None for p in good["particles"])
        assert rep["config"]["backend"] == "onnx"
    assert t["particles"] == good["count"]
    assert t["labels"]["images_with_labels"] == 1 and t["labels"]["images_without_labels"] == 0
    assert not (tmp_path / "out" / "0000_overlay.jpg").exists()
    assert (tmp_path / "out" / "0001_overlay.jpg").is_file()
    errs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errs) == 1 and errs[0].getMessage() == "回放失败"
    assert errs[0].exc_info is not None and "boom" in str(errs[0].exc_info[1]), "应带堆栈"
    assert getattr(errs[0], "type", None) == exc_name
    summary = replay.format_summary(rep)
    assert "错误: " in summary and "boom" in summary


def test_same_stem_different_ext_get_distinct_overlays(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """a.png 与 a.jpg 的标注图不能互相覆盖：文件名加入扩展名，报告里两条 overlay 不同。"""
    d = tmp_path / "frames"
    d.mkdir()
    shutil.copy(frames_dir / "0000.png", d / "a.png")
    img1 = cv2.imread(str(frames_dir / "0001.png"))
    assert cv2.imwrite(str(d / "a.jpg"), img1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    shutil.copy(frames_dir / "0000.png", d / "b.png")
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(make_args(d, cfg_yaml, out=out))
    assert rc == 0
    names = {img["name"]: img["overlay"] for img in rep["images"]}
    assert names == {
        "a.jpg": "a_jpg_overlay.jpg",
        "a.png": "a_png_overlay.jpg",
        "b.png": "b_overlay.jpg",
    }
    assert sorted(p.name for p in out.glob("*_overlay.jpg")) == sorted(names.values())
    assert not (out / "a_overlay.jpg").exists()
    assert any("同名不同扩展名" in r.getMessage() for r in caplog.records)
    # 纯函数：任意组合下标注图名唯一；无重名时保持 {stem}_overlay.jpg
    files = [Path("a.png"), Path("a.jpg"), Path("a_png.bmp"), Path("c.png")]
    mapping = replay.overlay_names(files)
    assert len(set(mapping.values())) == len(files)
    assert mapping[Path("c.png")] == "c_overlay.jpg"
    assert all(v.endswith(replay.OVERLAY_SUFFIX) for v in mapping.values())
    assert replay.overlay_names([Path("x.png"), Path("y.jpg")]) == {
        Path("x.png"): "x_overlay.jpg",
        Path("y.jpg"): "y_overlay.jpg",
    }


def test_out_inside_images_dir_is_rejected_and_old_overlays_skipped(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """--out 不能是图片目录本身或其子目录（返回 1）；目录里残留的 *_overlay.jpg 不会被当作输入。"""
    d = tmp_path / "frames"
    shutil.copytree(frames_dir, d)
    for out in (d, d / "sub", d / "a" / "b"):
        with caplog.at_level("ERROR", logger="oresorter.vision.replay"):
            assert replay.run(make_args(d, cfg_yaml, out=out)) == 1
        assert not out.exists() or not (out / replay.REPORT_FILE).exists()
    with pytest.raises(replay.ReplayError, match="输出目录"):
        replay.parse_options(make_args(d, cfg_yaml, out=d))
    with pytest.raises(replay.ReplayError, match="输出目录"):
        replay.parse_options(make_args(d, cfg_yaml, out=d / "sub"))
    # 图片目录在输出目录里则允许（标注图不会落进图片目录）
    assert replay.parse_options(make_args(d, cfg_yaml, out=tmp_path)).out == tmp_path

    # 残留的旧标注图：跳过并告警，不进入报告，也不影响召回统计
    shutil.copy(frames_dir / "0000.png", d / "0000_overlay.jpg")
    caplog.clear()
    with caplog.at_level("WARNING", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "out"))
    assert rc == 0
    assert [img["name"] for img in rep["images"]] == ["0000.png", "0001.png"]
    assert rep["totals"]["labels"]["n_labels"] == N_FRAMES * N_PARTICLES
    assert rep["totals"]["labels"]["images_without_labels"] == 0
    assert any("已跳过" in r.getMessage() for r in caplog.records)
    assert replay.list_images(d) == [d / "0000.png", d / "0001.png"]


def test_default_out_dir_avoids_images_dir_named_replay_out(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path
):
    d = tmp_path / "replay_out"
    d.mkdir()
    shutil.copy(frames_dir / "0000.png", d / "0000.png")
    opts = replay.parse_options(make_args(d, cfg_yaml))
    assert opts.out == tmp_path / "replay_out_replay_out"
    assert replay.default_out_dir(frames_dir) == frames_dir.resolve().parent / "replay_out"


def test_labels_json_non_list_and_missing_keys_warn(
    frames_dir: Path, cfg_yaml: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """值不是列表的图按“无标注”处理（不算全部误检）、缺键的图告警并计数、多余的键告警。"""
    d = tmp_path / "frames"
    shutil.copytree(frames_dir, d)
    labels = _labels(frames_dir)
    bad = {"0000.png": labels["0000.png"], "0001.png": "oops", "ghost.png": []}
    (d / "labels.json").write_text(json.dumps(bad), encoding="utf-8")
    with caplog.at_level("WARNING", logger="oresorter.vision.replay"):
        parsed = replay.load_labels(d)
        rc, rep = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "out"))
    assert parsed is not None and set(parsed) == {"0000.png", "ghost.png"}
    assert rc == 0
    first, second = rep["images"]
    assert first["labels"]["n_labels"] == N_PARTICLES and first["labels"]["recall"] >= 0.9
    assert second["labels"] is None, "值不是列表 → 不参与统计，而不是全部计为误检"
    t = rep["totals"]["labels"]
    assert t["n_labels"] == N_PARTICLES and t["n_detections"] == first["count"]
    assert t["false_positives"] == first["labels"]["false_positives"] == 0
    assert t["images_with_labels"] == 1 and t["images_without_labels"] == 1
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("不是列表" in m for m in msgs)
    assert any("没有该图的标注" in m for m in msgs)
    assert any("不在目录里" in m for m in msgs)
    assert "1 张无标注未计入" in replay.format_summary(rep)

    # labels.json 只覆盖不存在的图：totals.labels 仍是字典（全部无标注），而不是 None
    (d / "labels.json").write_text(json.dumps({"ghost.png": []}), encoding="utf-8")
    rc, rep = run_and_read(make_args(d, cfg_yaml, out=tmp_path / "out2"))
    assert rc == 0
    t = rep["totals"]["labels"]
    assert t["n_labels"] == 0 and t["recall"] is None
    assert t["images_with_labels"] == 0 and t["images_without_labels"] == N_FRAMES


def test_chunked_with_zero_overlap_warns(
    frames_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    data = {
        "base": str(REPO / "config" / "default.yaml"),
        "system": {"data_dir": str(tmp_path / "data"), "mcu_kind": "mock"},
        "capture": {"bg_mode": "color", "bg_color": [0, 0, 255]},
        "pipeline": {"overlap_lines": 0},
    }
    p = tmp_path / "zero.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with caplog.at_level("WARNING", logger="oresorter.vision.replay"):
        rc, rep = run_and_read(make_args(frames_dir, p, out=tmp_path / "out", chunk_lines=256))
    assert rc == 0 and rep["config"]["overlap_lines"] == 0
    assert any("overlap_lines=0" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger="oresorter.vision.replay"):
        assert replay.run(make_args(frames_dir, p, out=tmp_path / "whole")) == 0
    assert not any("overlap_lines=0" in r.getMessage() for r in caplog.records), "整图模式不告警"


# --------------------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------------------


def test_match_labels_and_iou():
    assert replay.iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert replay.iou_xywh((0, 0, 10, 10), (10, 10, 5, 5)) == 0.0
    assert replay.iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    labels = [
        {"cls": "A", "bbox": [0, 0, 20, 20]},
        {"cls": "B", "bbox": [100, 100, 20, 20]},
        {"cls": "A", "bbox": [300, 300, 20, 20]},
    ]
    dets = [
        {"cls": "A", "bbox": [1, 1, 20, 20]},  # 匹配 0，类别对
        {"cls": "A", "bbox": [101, 101, 20, 20]},  # 匹配 1，类别错
        {"cls": None, "bbox": [200, 200, 20, 20]},  # 误检
        {"cls": "A", "bbox": [2, 2, 20, 20]},  # 与 0 重叠但 0 已被更高 IoU 占用 → 误检
    ]
    m = replay.match_labels(dets, labels)
    assert m["n_labels"] == 3 and m["n_detections"] == 4
    assert m["matched"] == 2 and m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["false_positives"] == 2
    assert m["cls_evaluated"] == 2 and m["cls_correct"] == 1 and m["cls_accuracy"] == 0.5
    # 无类别的检测：准确率 None
    m2 = replay.match_labels([{"cls": None, "bbox": [0, 0, 20, 20]}], labels[:1])
    assert m2["matched"] == 1 and m2["cls_accuracy"] is None and m2["cls_evaluated"] == 0
    # 没有标注：recall None
    assert replay.match_labels(dets, [])["recall"] is None


def test_split_frames_covers_image_with_global_line_numbers():
    img = np.zeros((1000, 64, 3), np.uint8)
    frames = list(replay.split_frames(img, 256))
    assert [f.seq for f in frames] == [0, 1, 2, 3]
    assert [f.line_start for f in frames] == [0, 256, 512, 768]
    assert [f.lines for f in frames] == [256, 256, 256, 232]
    assert frames[-1].line_end == 1000 and all(
        f.camera_id == replay.REPLAY_CAMERA_ID for f in frames
    )
    with pytest.raises(ValueError):
        list(replay.split_frames(img, 0))


def test_parse_options_messages(frames_dir: Path, cfg_yaml: Path, tmp_path: Path):
    opts = replay.parse_options(make_args(frames_dir, cfg_yaml))
    assert opts.out == frames_dir.resolve().parent / "replay_out"
    assert opts.chunk_lines is None and opts.learn_bg == 0 and opts.model_dir is None
    with pytest.raises(replay.ReplayError, match="128~4096"):
        replay.parse_options(make_args(frames_dir, cfg_yaml, chunk_lines=100))
    with pytest.raises(replay.ReplayError, match="不存在"):
        replay.parse_options(make_args(tmp_path / "x", cfg_yaml))
    with pytest.raises(replay.ReplayError, match="负"):
        replay.parse_options(make_args(frames_dir, cfg_yaml, learn_bg=-3))
    with pytest.raises(replay.ReplayError, match="images"):
        replay.parse_options(argparse.Namespace(config=cfg_yaml))


# --------------------------------------------------------------------------------------
# CLI 入口
# --------------------------------------------------------------------------------------


@pytest.fixture
def keep_root_logging():
    """``cli.main`` 会调 ``setup_logging`` 重置根 logger（级别、handler，并把一些库 logger 调到 WARNING）；
    测试结束后恢复，避免把 INFO 级别与绑定在 pytest 临时 stderr 上的 handler 泄漏给后面的测试文件。"""
    root = logging.getLogger()
    quiet = ("uvicorn.access", "PIL", "matplotlib", "httpx", "httpcore")
    saved_level, saved_handlers = root.level, list(root.handlers)
    saved_quiet = {name: logging.getLogger(name).level for name in quiet}
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
            h.close()
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(saved_level)
    for name, level in saved_quiet.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.usefixtures("keep_root_logging")
def test_cli_main_replay_entry(frames_dir: Path, cfg_yaml: Path, model_dir: Path, tmp_path: Path):
    """``python -m oresorter.cli replay ...``：_lazy 找到 oresorter.vision.replay.run 并透传参数。"""
    out = tmp_path / "cli_out"
    rc = cli.main(
        [
            "replay",
            "--images",
            str(frames_dir),
            "--config",
            str(cfg_yaml),
            "--model-dir",
            str(model_dir),
            "--chunk-lines",
            "512",
            "--out",
            str(out),
            "--learn-bg",
            "1",
        ]
    )
    assert rc == 0
    rep = json.loads((out / replay.REPORT_FILE).read_text(encoding="utf-8"))
    assert rep["config"]["chunk_lines"] == 512 and rep["config"]["bg"] == "column"
    assert rep["config"]["backend"] == "onnx" and rep["config"]["learn_bg"] == 1
    assert rep["totals"]["particles"] > 0
    assert all(p["cls"] in CLASSES for img in rep["images"] for p in img["particles"])
    # 非法参数经入口同样返回 1，不抛
    assert (
        cli.main(["replay", "--images", str(tmp_path / "missing"), "--config", str(cfg_yaml)]) == 1
    )
    assert (
        cli.main(
            [
                "replay",
                "--images",
                str(frames_dir),
                "--config",
                str(cfg_yaml),
                "--chunk-lines",
                "64",
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 1
    )
