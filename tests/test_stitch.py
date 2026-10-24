"""跨块拼接去重：每颗矿石恰好处理一次、触底延后、丢块/行号重置、超长截断、块高不足、告警与计数。

注意：不 import ``oresorter.vision.segment.segment``（由其他模块实现），这里用 ``toy_segment`` 代替，
只复用 ``Particle`` 数据类与 ``camera.mock.synth_block``。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from oresorter.camera.base import Frame
from oresorter.camera.mock import synth_block
from oresorter.vision.segment import Particle
from oresorter.vision.stitch import ChunkInfo, ChunkStitcher

BG = (0, 0, 255)
RED = (200, 40, 40)
WIDTH = 4096
CHUNK = 512
OVERLAP = 256
LOGGER = "oresorter.vision.stitch"

Box = tuple[int, int, int, int]


# ---------------- 工具 ----------------


def toy_segment(image: np.ndarray, bg_color: tuple[int, int, int] = BG) -> list[Particle]:
    """玩具分割：与背景颜色距离 > 60 的像素做阈值，8 连通域即一颗颗粒；不做面积过滤。"""
    diff = image.astype(np.int32) - np.array(bg_color, np.int32)
    mask = ((diff * diff).sum(axis=2) > 60 * 60).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[Particle] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        crop = np.empty((h, w, 4), np.uint8)
        crop[:, :, :3] = image[y : y + h, x : x + w]
        crop[:, :, 3] = np.where(labels[y : y + h, x : x + w] == i, 255, 0).astype(np.uint8)
        out.append(
            Particle(
                bbox=(x, y, w, h),
                centroid_x=float(cents[i][0]),
                centroid_y=float(cents[i][1]),
                area=area,
                trailing_y=y + h - 1,
                crop_rgba=crop,
            )
        )
    return out


def _intersects(a: Box, b: Box, gap: int) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw + gap and bx < ax + aw + gap and ay < by + bh + gap and by < ay + ah + gap


def make_belt(
    seed: int, *, width: int, height: int, n_particles: int, diameter_range: tuple[int, int]
) -> tuple[np.ndarray, list[Box]]:
    """用 synth_block 合成皮带图，把相互重叠（含 2 px 间隙）的颗粒涂回背景色，返回确定的真值 bbox。"""
    rng = np.random.default_rng(seed)
    img, labels = synth_block(
        rng,
        width,
        height,
        bg_color=BG,
        n_particles=n_particles,
        diameter_range=diameter_range,
    )
    boxes = [b for _, b in labels]
    bad: set[int] = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _intersects(boxes[i], boxes[j], gap=2):
                bad.add(i)
                bad.add(j)
    for i in bad:
        x, y, w, h = boxes[i]
        img[y : y + h, x : x + w] = np.array(BG, np.uint8)
    truth = [b for i, b in enumerate(boxes) if i not in bad]
    return img, truth


def blank_belt(height: int, width: int = 256) -> np.ndarray:
    img = np.empty((height, width, 3), np.uint8)
    img[:] = np.array(BG, np.uint8)
    return img


def paint_rect(img: np.ndarray, x: int, y: int, w: int, h: int) -> Box:
    """画一个纯色矩形颗粒，返回 bbox。"""
    img[y : y + h, x : x + w] = np.array(RED, np.uint8)
    return (x, y, w, h)


def make_frames(
    belt: np.ndarray, chunk: int, *, camera_id: int = 1, line_start0: int = 0
) -> list[Frame]:
    """按 chunk 行切块成 line_start 连续的 Frame 序列。"""
    assert belt.shape[0] % chunk == 0
    return [
        Frame(
            seq=i,
            line_start=line_start0 + i * chunk,
            image=belt[i * chunk : (i + 1) * chunk],
            ts=float(i),
            camera_id=camera_id,
        )
        for i in range(belt.shape[0] // chunk)
    ]


def global_box(p: Particle) -> Box:
    x, y, w, h = p.bbox
    return (x, p.extra["y_offset"] + y, w, h)


def expected_used(overlap: int, rows_since_reset: int) -> int:
    """连续推块时第 k 块的 overlap_used = min(overlap, 自起始/断块起已累计的行数)。"""
    return min(overlap, rows_since_reset)


def expected_counts(truth: list[Box], chunk: int, n_chunks: int, overlap: int) -> tuple[int, int]:
    """等高连续分块、颗粒不触皮带末行时的期望 (deferred, skipped_seen)：

    - 第 k-1 块触底 ⟺ bbox 含第 k 块首行的前一行 ``b-1``；
    - 第 k 块处理图覆盖 ``[b - used_k, b + chunk)``，其中 ``used_k = min(overlap, b)``；
      颗粒在其中可见且尾沿 ``≤ b-2``（上一块已处理）⟺ ``b - used_k ≤ max_y ≤ b-2``。
    """
    deferred = skipped = 0
    for _x, y, _w, h in truth:
        y1 = y + h - 1
        for k in range(1, n_chunks):
            b = k * chunk
            used = expected_used(overlap, b)
            if y <= b - 1 <= y1:
                deferred += 1
            if b - used <= y1 <= b - 2:
                skipped += 1
    return deferred, skipped


def stitch_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER]


def run_belt(
    belt: np.ndarray, chunk: int, overlap: int
) -> tuple[ChunkStitcher, list[list[Particle]]]:
    """整条皮带切块推入，返回 stitcher 与每块 select 结果；逐块断言 overlap_used / y_offset / extra。"""
    st = ChunkStitcher(overlap)
    per_chunk: list[list[Particle]] = []
    for f in make_frames(belt, chunk):
        got = st.process(f, toy_segment)
        assert st.overlap_used == expected_used(overlap, f.line_start)
        assert st.y_offset == f.line_start - st.overlap_used
        assert st.last_chunk is not None and st.last_chunk.seq == f.seq
        tls = [p.trailing_line for p in got]
        assert tls == sorted(tls)
        for p in got:
            assert p.extra["seq"] == f.seq
            assert p.extra["y_offset"] == st.y_offset
            assert p.touches_bottom is False
        per_chunk.append(got)
    return st, per_chunk


def assert_each_once(selected: list[Particle], truth: list[Box]) -> None:
    """全局质心去重后数量不变、全局 bbox 与真值一一对应、trailing_line == 全局 y+h-1。"""
    cents = {
        (round(p.centroid_x, 1), round(p.extra["y_offset"] + p.centroid_y, 1)) for p in selected
    }
    assert len(cents) == len(selected)
    assert sorted(global_box(p) for p in selected) == sorted(truth)
    for p in selected:
        _x, gy, _w, h = global_box(p)
        assert p.trailing_line == gy + h - 1


# ---------------- 合成皮带：每颗恰好处理一次 ----------------


def test_synthetic_belt_each_particle_processed_once(caplog: pytest.LogCaptureFixture):
    n_chunks = 6
    belt, truth = make_belt(
        1234,
        width=WIDTH,
        height=n_chunks * CHUNK,
        n_particles=60,
        diameter_range=(40, OVERLAP // 2),
    )
    assert len(truth) >= 25, "剔除重叠后颗粒太少，测试不够有力"
    assert all(h < OVERLAP for _, _, _, h in truth)
    assert all(y + h < belt.shape[0] for _, y, _, h in truth), "触皮带末行的颗粒会一直延后"
    # 真值里至少有颗粒跨越块边界，否则没测到去重
    exp_deferred, exp_skipped = expected_counts(truth, CHUNK, n_chunks, OVERLAP)
    assert exp_deferred >= 1 and exp_skipped >= 1

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        st, per_chunk = run_belt(belt, CHUNK, OVERLAP)
    assert stitch_records(caplog) == [], "连续无丢块、无超长颗粒时不应有任何告警"
    for i, got in enumerate(per_chunk):
        assert st.stats()["frames"] == n_chunks
        for p in got:
            assert p.truncated is False
        if i >= 1:
            assert all(p.extra["y_offset"] == i * CHUNK - OVERLAP for p in got)

    selected = [p for got in per_chunk for p in got]
    assert_each_once(selected, truth)
    assert st.stats() == {
        "frames": n_chunks,
        "dropped": 0,
        "resets": 0,
        "deferred": exp_deferred,
        "skipped_seen": exp_skipped,
        "truncated": 0,
        "selected": len(truth),
        "lost": 0,
    }


def test_synthetic_belt_chunk_shorter_than_overlap_each_particle_processed_once():
    """chunk=128 < overlap=256：高度在 (chunk, overlap] 之间的颗粒也必须完整处理且不截断。"""
    chunk, overlap, n_chunks = 128, 256, 16
    belt, truth = make_belt(
        1234, width=WIDTH, height=n_chunks * chunk, n_particles=40, diameter_range=(60, 200)
    )
    assert len(truth) >= 25
    assert all(h <= overlap for _, _, _, h in truth)
    assert sum(1 for _, _, _, h in truth if h > chunk) >= 5, "需要足够多比块高还长的颗粒"
    assert all(y + h < belt.shape[0] for _, y, _, h in truth)
    exp_deferred, exp_skipped = expected_counts(truth, chunk, n_chunks, overlap)

    st, per_chunk = run_belt(belt, chunk, overlap)
    selected = [p for got in per_chunk for p in got]
    assert all(p.truncated is False for p in selected)
    assert_each_once(selected, truth)
    s = st.stats()
    assert s["selected"] == len(truth) and s["truncated"] == 0 and s["lost"] == 0
    assert s["deferred"] == exp_deferred and s["skipped_seen"] == exp_skipped


def test_processed_image_is_tail_plus_chunk_and_tail_is_a_copy():
    belt = blank_belt(3 * CHUNK)
    paint_rect(belt, 10, 400, 30, 20)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    proc0, off0 = st.push(frames[0])
    assert off0 == 0 and proc0.shape[0] == CHUNK and st.overlap_used == 0
    expected_tail = frames[0].image[-OVERLAP:].copy()
    # 篡改上一块图像：stitcher 保存的是拷贝，处理图不应受影响
    frames[0].image[:] = 0
    proc1, off1 = st.push(frames[1])
    assert off1 == CHUNK - OVERLAP
    assert proc1.shape == (CHUNK + OVERLAP, belt.shape[1], 3)
    assert np.array_equal(proc1[:OVERLAP], expected_tail)
    assert np.array_equal(proc1[OVERLAP:], frames[1].image)


# ---------------- 跨边界单颗 ----------------


def test_particle_across_boundary_deferred_then_processed_once():
    belt = blank_belt(2 * CHUNK)
    box = paint_rect(belt, 100, 480, 41, 81)  # 行 480~560，跨 512 边界
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)

    got0 = st.process(frames[0], toy_segment)
    assert got0 == []
    assert st.stats()["deferred"] == 1

    got1 = st.process(frames[1], toy_segment)
    assert len(got1) == 1
    p = got1[0]
    assert global_box(p) == box
    assert p.trailing_line == 480 + 81 - 1 == 560
    assert p.truncated is False
    assert p.extra["seq"] == 1 and p.extra["y_offset"] == CHUNK - OVERLAP

    s = st.stats()
    assert s["selected"] == 1 and s["deferred"] == 1 and s["skipped_seen"] == 0


def test_particle_ending_on_last_row_of_chunk_is_deferred_once():
    """尾沿正好是块的最后一行（y+h == 块高）：算触底，延后，下一块在重叠区边缘恰好处理。"""
    belt = blank_belt(2 * CHUNK)
    box = paint_rect(belt, 20, 470, 30, 42)  # 行 470~511
    st = ChunkStitcher(OVERLAP)
    frames = make_frames(belt, CHUNK)
    assert st.process(frames[0], toy_segment) == []
    got = st.process(frames[1], toy_segment)
    assert [global_box(p) for p in got] == [box]
    assert got[0].trailing_line == 511
    assert st.stats()["deferred"] == 1 and st.stats()["selected"] == 1


# ---------------- 丢块 / 行号重置 / 形状变化 ----------------


def test_dropped_chunk_resets_overlap_then_recovers(caplog: pytest.LogCaptureFixture):
    belt = blank_belt(5 * CHUNK)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        st.push(frames[0])
        st.push(frames[1])
        assert st.overlap_used == OVERLAP
        assert stitch_records(caplog) == []

        proc3, off3 = st.push(frames[3])  # 跳过 seq=2
    assert st.overlap_used == 0
    assert off3 == frames[3].line_start == 3 * CHUNK
    assert proc3.shape[0] == CHUNK
    s = st.stats()
    assert s["dropped"] == 1 and s["resets"] == 1 and s["frames"] == 3
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    r = recs[0]
    assert (r.reason, r.seq, r.prev_seq, r.camera) == ("dropped", 3, 1, 1)
    assert (r.line_start, r.prev_line_end, r.dropped_total) == (3 * CHUNK, 2 * CHUNK, 1)

    proc4, off4 = st.push(frames[4])
    assert st.overlap_used == OVERLAP
    assert off4 == 4 * CHUNK - OVERLAP
    assert np.array_equal(proc4[:OVERLAP], frames[3].image[-OVERLAP:])
    s = st.stats()
    assert s["dropped"] == 1 and s["resets"] == 1 and s["frames"] == 4


def test_multiple_dropped_chunks_counted():
    belt = blank_belt(6 * CHUNK)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.push(frames[0])
    st.push(frames[4])  # 丢了 1,2,3
    assert st.stats()["dropped"] == 3 and st.stats()["resets"] == 1
    assert st.overlap_used == 0


def test_line_counter_reset_counts_reset_and_drops_overlap(caplog: pytest.LogCaptureFixture):
    belt = blank_belt(2 * CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.push(Frame(seq=0, line_start=0, image=belt[:CHUNK], ts=0.0, camera_id=1))
    st.push(Frame(seq=1, line_start=CHUNK, image=belt[CHUNK:], ts=1.0, camera_id=1))
    assert st.overlap_used == OVERLAP
    # reset_line_counter 之后：seq 连续，但 line_start 回到 0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _proc, off = st.push(Frame(seq=2, line_start=0, image=belt[:CHUNK], ts=2.0, camera_id=1))
    assert st.overlap_used == 0 and off == 0
    s = st.stats()
    assert s["resets"] == 1 and s["dropped"] == 0
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].reason == "line_jump"
    # 之后恢复
    _proc, off = st.push(Frame(seq=3, line_start=CHUNK, image=belt[CHUNK:], ts=3.0, camera_id=1))
    assert st.overlap_used == OVERLAP and off == CHUNK - OVERLAP
    assert st.stats()["resets"] == 1


def test_seq_rewind_is_a_reset_without_dropped(caplog: pytest.LogCaptureFixture):
    """同一 seq 重推 / seq 回退：断块、不计 dropped。"""
    belt = blank_belt(2 * CHUNK, 64)
    st = ChunkStitcher(OVERLAP)
    st.push(Frame(seq=0, line_start=0, image=belt[:CHUNK], ts=0.0))
    st.push(Frame(seq=1, line_start=CHUNK, image=belt[CHUNK:], ts=1.0))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _proc, off = st.push(Frame(seq=1, line_start=2 * CHUNK, image=belt[:CHUNK], ts=2.0))
    assert st.overlap_used == 0 and off == 2 * CHUNK
    s = st.stats()
    assert s["resets"] == 1 and s["dropped"] == 0 and s["frames"] == 3
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].reason == "seq_rewind"


def test_width_change_is_a_reset(caplog: pytest.LogCaptureFixture):
    """seq / line_start 都连续但图像宽度变了：无法拼接，断块处理，之后按新宽度恢复。"""
    st = ChunkStitcher(OVERLAP)
    st.push(Frame(seq=0, line_start=0, image=blank_belt(CHUNK, 64), ts=0.0))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        proc, off = st.push(Frame(seq=1, line_start=CHUNK, image=blank_belt(CHUNK, 128), ts=1.0))
    assert st.overlap_used == 0 and off == CHUNK and proc.shape == (CHUNK, 128, 3)
    s = st.stats()
    assert s["resets"] == 1 and s["dropped"] == 0
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].reason == "shape_changed"
    proc, off = st.push(Frame(seq=2, line_start=2 * CHUNK, image=blank_belt(CHUNK, 128), ts=2.0))
    assert st.overlap_used == OVERLAP and proc.shape == (CHUNK + OVERLAP, 128, 3)
    assert st.stats()["resets"] == 1


def test_first_frame_is_not_a_reset(caplog: pytest.LogCaptureFixture):
    belt = blank_belt(CHUNK)
    st = ChunkStitcher(OVERLAP)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        proc, off = st.push(make_frames(belt, CHUNK)[0])
    assert proc is not None and off == 0
    assert st.stats()["resets"] == 0 and st.stats()["frames"] == 1
    assert stitch_records(caplog) == []


def test_deferred_particle_lost_on_dropped_chunk_is_counted(caplog: pytest.LogCaptureFixture):
    """触底延后的颗粒紧接着遇到丢块：永远不会再被处理，计入 lost 并写进告警。"""
    belt = blank_belt(3 * CHUNK, 64)
    paint_rect(belt, 10, 480, 20, 60)  # 行 480~539，跨第 0/1 块边界
    st = ChunkStitcher(OVERLAP)
    assert st.process(Frame(seq=0, line_start=0, image=belt[:CHUNK], ts=0.0), toy_segment) == []
    assert st.stats()["deferred"] == 1 and st.stats()["lost"] == 0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        got = st.process(
            Frame(seq=2, line_start=2 * CHUNK, image=belt[2 * CHUNK :], ts=2.0), toy_segment
        )
    assert got == []
    s = st.stats()
    assert s["lost"] == 1 and s["deferred"] == 1 and s["selected"] == 0
    assert s["dropped"] == 1 and s["resets"] == 1
    recs = stitch_records(caplog)
    assert len(recs) == 1 and (recs[0].lost, recs[0].lost_total) == (1, 1)


def test_lost_is_zero_when_nothing_was_deferred_or_previous_block_not_selected():
    belt = blank_belt(4 * CHUNK, 64)
    paint_rect(belt, 10, 480, 20, 60)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.push(frames[0])  # 没有 select → 不知道延后了什么，不能凭空计 lost
    st.push(frames[2])
    assert st.stats()["lost"] == 0 and st.stats()["resets"] == 1
    st.select([], CHUNK)  # 第 2 块无延后
    st.push(frames[3])
    st.push(Frame(seq=9, line_start=9 * CHUNK, image=frames[3].image, ts=9.0, camera_id=1))
    assert st.stats()["lost"] == 0 and st.stats()["resets"] == 2


# ---------------- 超长颗粒 / 截断判定 ----------------


def test_particle_longer_than_overlap_is_truncated_not_deferred_forever(
    caplog: pytest.LogCaptureFixture,
):
    belt = blank_belt(3 * CHUNK)
    paint_rect(belt, 50, 150, 100, 400)  # 行 150~549，高 400 > overlap 256
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert st.process(frames[0], toy_segment) == []  # 第 0 块触底 → 延后
        assert stitch_records(caplog) == []
        got1 = st.process(frames[1], toy_segment)
    assert len(got1) == 1
    p = got1[0]
    assert p.truncated is True and p.touches_top is True
    assert p.bbox[1] == 0  # 顶到处理图顶边
    assert p.trailing_line == 549  # 尾沿行号仍然正确
    assert global_box(p)[1] == CHUNK - OVERLAP  # 可见部分从重叠区首行开始
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.WARNING
    r = recs[0]
    assert (r.seq, r.camera, r.overlap_used, r.basis) == (1, 1, OVERLAP, "tracked")
    assert r.bbox == [50, 0, 100, 294]

    got2 = st.process(frames[2], toy_segment)
    assert got2 == []  # 第 2 块处理图从行 768 开始，它已完全不可见

    s = st.stats()
    assert s["selected"] == 1
    assert s["deferred"] == 1
    assert s["truncated"] == 1
    assert s["skipped_seen"] == 0


def test_long_particle_starting_inside_overlap_is_not_truncated():
    """高 400 但起点在重叠区内（行 300~699）：第 1 块处理图完整包含它，不截断。"""
    belt = blank_belt(3 * CHUNK)
    box = paint_rect(belt, 50, 300, 100, 400)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    assert st.process(frames[0], toy_segment) == []
    got1 = st.process(frames[1], toy_segment)
    assert [global_box(p) for p in got1] == [box]
    assert got1[0].truncated is False
    assert got1[0].trailing_line == 699
    assert st.process(frames[2], toy_segment) == []
    s = st.stats()
    assert s["selected"] == 1 and s["truncated"] == 0 and s["deferred"] == 1


def test_particle_exactly_overlap_tall_starting_at_overlap_top_is_not_truncated(
    caplog: pytest.LogCaptureFixture,
):
    """高度恰好 == overlap 且起点正好是重叠区首行（行 256~511）：完整可见，不得误报截断。"""
    belt = blank_belt(2 * CHUNK, 64)
    box = paint_rect(belt, 10, CHUNK - OVERLAP, 20, OVERLAP)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert st.process(frames[0], toy_segment) == []
        got = st.process(frames[1], toy_segment)
    assert [global_box(p) for p in got] == [box]
    p = got[0]
    assert p.bbox[1] == 0 and p.touches_top is True
    assert p.truncated is False
    assert p.trailing_line == CHUNK - 1
    assert st.stats()["truncated"] == 0 and st.stats()["selected"] == 1
    assert stitch_records(caplog) == []


def test_truncated_and_exactly_fitting_particles_in_same_block_are_told_apart():
    belt = blank_belt(2 * CHUNK, 256)
    long_box = paint_rect(belt, 10, 100, 20, 412)  # 行 100~511：越过重叠区顶边 → 截断
    fit_box = paint_rect(belt, 100, 256, 20, 256)  # 行 256~511：恰好填满重叠区 → 完整
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    assert st.process(frames[0], toy_segment) == []
    assert st.stats()["deferred"] == 2
    got = st.process(frames[1], toy_segment)
    by_x = {p.bbox[0]: p for p in got}
    assert set(by_x) == {10, 100}
    assert by_x[10].truncated is True
    assert global_box(by_x[10]) == (10, 256, 20, 256)  # 可见部分
    assert by_x[100].truncated is False and global_box(by_x[100]) == fit_box
    assert by_x[10].trailing_line == by_x[100].trailing_line == long_box[1] + long_box[3] - 1
    assert st.stats()["truncated"] == 1 and st.stats()["selected"] == 2


def test_top_touching_particle_without_previous_select_falls_back_to_heuristic(
    caplog: pytest.LogCaptureFixture,
):
    """上一块只 push 没 select：没有延后信息 → 顶到顶边即按截断处理（basis=untracked）。"""
    belt = blank_belt(2 * CHUNK, 64)
    paint_rect(belt, 10, CHUNK - OVERLAP, 20, OVERLAP)  # 实际完整，但无法得知
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.push(frames[0])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        got = st.process(frames[1], toy_segment)
    assert len(got) == 1 and got[0].truncated is True
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].basis == "untracked"
    assert st.stats()["truncated"] == 1


def test_top_touching_particle_with_no_matching_deferred_is_treated_as_truncated(
    caplog: pytest.LogCaptureFixture,
):
    """上一块 select 过但没有 x 重叠的延后颗粒（分割不一致）：保守按截断处理（basis=unmatched）。"""
    belt = blank_belt(2 * CHUNK, 64)
    paint_rect(belt, 10, CHUNK - OVERLAP, 20, OVERLAP)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.push(frames[0])
    st.select([], CHUNK)  # 假装上一块什么都没分割出来
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        got = st.process(frames[1], toy_segment)
    assert len(got) == 1 and got[0].truncated is True
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].basis == "unmatched"


def test_max_object_lines_marks_oversize(caplog: pytest.LogCaptureFixture):
    belt = blank_belt(CHUNK)
    paint_rect(belt, 10, 10, 20, 300)
    paint_rect(belt, 100, 10, 20, 50)
    st = ChunkStitcher(OVERLAP, max_object_lines=200)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        got = st.process(make_frames(belt, CHUNK)[0], toy_segment)
    flags = {p.bbox[0]: p.extra.get("oversize", False) for p in got}
    assert flags == {10: True, 100: False}
    recs = stitch_records(caplog)
    assert len(recs) == 1 and recs[0].max_object_lines == 200 and recs[0].bbox == [10, 10, 20, 300]


def test_max_object_lines_greater_than_overlap_warns_at_construction(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ChunkStitcher(OVERLAP, max_object_lines=OVERLAP)
        assert stitch_records(caplog) == []
        ChunkStitcher(OVERLAP, max_object_lines=OVERLAP + 1)
    recs = stitch_records(caplog)
    assert len(recs) == 1 and (recs[0].overlap, recs[0].max_object_lines) == (OVERLAP, OVERLAP + 1)


# ---------------- 块高 < overlap ----------------


def test_chunk_shorter_than_overlap_grows_to_full_overlap():
    """chunk=128 < overlap=256：重叠区由处理图尾部提供，第 1 块 128 行、之后稳定 256 行；
    比块高长但不超过 overlap 的颗粒（b，高 200）完整处理且不截断。"""
    chunk, overlap = 128, 256
    belt = blank_belt(10 * chunk)
    a = paint_rect(belt, 10, 20, 20, 41)  # 行 20~60，第 0 块内
    b = paint_rect(belt, 60, 100, 20, 200)  # 行 100~299，高 200：> chunk 但 ≤ overlap
    c = paint_rect(belt, 120, 500, 20, 61)  # 行 500~560，跨 512
    d = paint_rect(belt, 180, 1000, 20, 101)  # 行 1000~1100，跨 1024
    st, per_chunk = run_belt(belt, chunk, overlap)
    assert [expected_used(overlap, k * chunk) for k in range(10)] == [0, 128, *([256] * 8)]

    selected = [p for got in per_chunk for p in got]
    assert_each_once(selected, [a, b, c, d])
    assert all(p.truncated is False for p in selected)
    pb = next(p for p in selected if p.bbox[0] == 60)
    # 第 2 块处理图覆盖行 [0, 384)，b 完整可见（局部 y=100）
    assert pb.extra["seq"] == 2 and pb.extra["y_offset"] == 0 and pb.bbox == b
    assert pb.trailing_line == 299 and pb.touches_top is False
    assert [p.bbox[0] for p in per_chunk[0]] == [10]
    assert per_chunk[1] == []
    assert [p.bbox[0] for p in per_chunk[2]] == [60]
    assert [p.bbox[0] for p in per_chunk[4]] == [120]
    assert [p.bbox[0] for p in per_chunk[8]] == [180]

    s = st.stats()
    assert s["selected"] == 4 and s["truncated"] == 0
    # deferred：b 两次（边界 128、256），c 一次（512），d 一次（1024）
    assert s["deferred"] == 4
    # skipped_seen：a（第 1、2 块）、b（第 3、4 块）、c（第 5、6 块）、d（第 9 块）
    assert s["skipped_seen"] == 7
    assert s["dropped"] == 0 and s["resets"] == 0 and s["lost"] == 0


def test_chunk_shorter_than_overlap_processed_image_content():
    """chunk=128 / overlap=256：处理图 = 皮带上紧邻当前块之前的 overlap_used 行 + 当前块。"""
    chunk, overlap = 128, 256
    rng = np.random.default_rng(0)
    belt = rng.integers(0, 256, size=(6 * chunk, 32, 3), dtype=np.uint8)
    frames = make_frames(belt, chunk)
    st = ChunkStitcher(overlap)
    expected = [(0, 128), (0, 256), (0, 384), (128, 512), (256, 640), (384, 768)]
    for f, (lo, hi) in zip(frames, expected, strict=True):
        proc, off = st.push(f)
        assert off == lo and proc.shape[0] == hi - lo
        assert np.array_equal(proc, belt[lo:hi])
        assert st.overlap_used == f.line_start - lo


def test_short_middle_chunk_keeps_full_overlap():
    """块高不等（512、100、512）：短块之后重叠区仍是 256 行，高 221 (< overlap) 的颗粒完整处理。"""
    belt = blank_belt(512 + 100 + 512, 64)
    box = paint_rect(belt, 10, 400, 20, 221)  # 行 400~620
    frames = [
        Frame(seq=0, line_start=0, image=belt[:512], ts=0.0),
        Frame(seq=1, line_start=512, image=belt[512:612], ts=1.0),
        Frame(seq=2, line_start=612, image=belt[612:], ts=2.0),
    ]
    st = ChunkStitcher(OVERLAP)
    assert st.process(frames[0], toy_segment) == []
    assert st.process(frames[1], toy_segment) == []
    assert st.overlap_used == 256 and st.y_offset == 256
    got = st.process(frames[2], toy_segment)
    assert st.overlap_used == 256 and st.y_offset == 612 - 256
    assert [global_box(p) for p in got] == [box]
    assert got[0].truncated is False and got[0].trailing_line == 620
    s = st.stats()
    assert s["selected"] == 1 and s["deferred"] == 2 and s["truncated"] == 0


def test_short_chunk_then_normal_chunk_tail_grows():
    """块高变化：首块只有 100 行则 overlap_used=100；之后恢复 512 行块则 overlap_used=256。"""
    width = 64
    st = ChunkStitcher(OVERLAP)
    st.push(Frame(seq=0, line_start=0, image=blank_belt(100, width), ts=0.0))
    proc, off = st.push(Frame(seq=1, line_start=100, image=blank_belt(512, width), ts=1.0))
    assert st.overlap_used == 100 and off == 0 and proc.shape[0] == 612
    proc, off = st.push(Frame(seq=2, line_start=612, image=blank_belt(512, width), ts=2.0))
    assert st.overlap_used == OVERLAP and off == 612 - OVERLAP and proc.shape[0] == 768


# ---------------- select 细节与参数校验 ----------------


def _dummy_particle(x: int, y: int, w: int, h: int) -> Particle:
    return Particle(
        bbox=(x, y, w, h),
        centroid_x=x + w / 2,
        centroid_y=y + h / 2,
        area=w * h,
        trailing_y=y + h - 1,
        crop_rgba=np.zeros((h, w, 4), np.uint8),
    )


def test_select_sorts_by_trailing_line_and_fills_extra():
    st = ChunkStitcher(OVERLAP)
    frames = make_frames(blank_belt(2 * CHUNK, 64), CHUNK)
    st.push(frames[0])
    proc, off = st.push(frames[1])
    assert off == CHUNK - OVERLAP
    late = _dummy_particle(0, 600, 5, 10)  # max_y 609
    early = _dummy_particle(0, 300, 5, 10)  # max_y 309
    seen = _dummy_particle(0, 10, 5, 100)  # max_y 109 < 255 → 上一块已处理
    edge = _dummy_particle(0, 200, 5, 56)  # max_y 255 == overlap-1 → 保留
    bottom = _dummy_particle(0, 700, 5, 68)  # y+h == 768 → 触底
    got = st.select([late, bottom, early, seen, edge], proc.shape[0])
    assert got == [edge, early, late]
    assert [p.trailing_line for p in got] == [off + 255, off + 309, off + 609]
    for p in got:
        assert p.extra == {"seq": 1, "y_offset": off}
    assert bottom.touches_bottom is True and bottom.trailing_line == -1
    assert seen.trailing_line == -1 and seen.extra == {}
    s = st.stats()
    assert s["selected"] == 3 and s["skipped_seen"] == 1 and s["deferred"] == 1


def test_select_top_edge_without_overlap_is_not_truncated():
    """首块顶到顶边的颗粒：没有重叠区可参照，不标截断（只标 touches_top）。"""
    st = ChunkStitcher(OVERLAP)
    st.push(make_frames(blank_belt(CHUNK, 64), CHUNK)[0])
    p = _dummy_particle(0, 0, 5, 50)
    got = st.select([p], CHUNK)
    assert got == [p] and p.touches_top is True and p.truncated is False
    assert st.stats()["truncated"] == 0


def test_select_requires_push_and_matching_integer_height():
    st = ChunkStitcher(OVERLAP)
    with pytest.raises(RuntimeError):
        st.select([], CHUNK)
    st.push(make_frames(blank_belt(CHUNK, 64), CHUNK)[0])
    with pytest.raises(ValueError, match="image_h"):
        st.select([], CHUNK + 1)
    with pytest.raises(ValueError, match="image_h"):
        st.select([], float(CHUNK))  # type: ignore[arg-type]
    assert st.select([], CHUNK) == []


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        ChunkStitcher(-1)
    with pytest.raises(ValueError):
        ChunkStitcher(256.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChunkStitcher(256, max_object_lines=0)
    with pytest.raises(ValueError):
        ChunkStitcher(256, max_object_lines=1.5)  # type: ignore[arg-type]
    st = ChunkStitcher(256)
    bad_images = [
        np.zeros((0, 64, 3), np.uint8),  # 高 0
        np.zeros((64, 0, 3), np.uint8),  # 宽 0
        np.zeros((64, 64), np.uint8),  # 二维
        np.zeros((64, 64, 1), np.uint8),  # 单通道
        np.zeros((64, 64, 4), np.uint8),  # RGBA
        np.zeros((64, 64, 3), np.float32),  # 非 uint8
    ]
    for img in bad_images:
        with pytest.raises(ValueError):
            st.push(Frame(seq=0, line_start=0, image=img, ts=0.0))
    assert st.stats()["frames"] == 0 and st.last_chunk is None


def test_zero_overlap_never_stitches_but_stays_continuous():
    st = ChunkStitcher(0)
    frames = make_frames(blank_belt(3 * CHUNK, 64), CHUNK)
    for f in frames:
        proc, off = st.push(f)
        assert proc is f.image and off == f.line_start and st.overlap_used == 0
    assert st.stats() == {
        "frames": 3,
        "dropped": 0,
        "resets": 0,
        "deferred": 0,
        "skipped_seen": 0,
        "truncated": 0,
        "selected": 0,
        "lost": 0,
    }


def test_reset_clears_state_and_counters():
    belt = blank_belt(4 * CHUNK, 64)
    paint_rect(belt, 10, 480, 20, 60)
    frames = make_frames(belt, CHUNK)
    st = ChunkStitcher(OVERLAP)
    st.process(frames[0], toy_segment)
    st.push(frames[2])  # 丢块，上一块的延后颗粒丢失
    assert st.stats()["dropped"] == 1 and st.stats()["lost"] == 1
    st.reset()
    assert st.last_chunk is None and st.last_frame is None and st.overlap_used == 0
    assert all(v == 0 for v in st.stats().values())
    _proc, off = st.push(frames[3])
    assert off == frames[3].line_start and st.overlap_used == 0
    assert st.stats()["resets"] == 0 and st.stats()["frames"] == 1


def test_last_chunk_is_metadata_only_and_last_frame_is_alias():
    st = ChunkStitcher(OVERLAP)
    f = Frame(seq=5, line_start=2560, image=blank_belt(CHUNK, 64), ts=1.5, camera_id=2)
    st.push(f)
    info = st.last_chunk
    assert isinstance(info, ChunkInfo)
    assert st.last_frame is info
    assert (info.seq, info.line_start, info.lines, info.width, info.camera_id) == (
        5,
        2560,
        CHUNK,
        64,
        2,
    )
    assert info.line_end == f.line_end
    assert not hasattr(info, "image")
