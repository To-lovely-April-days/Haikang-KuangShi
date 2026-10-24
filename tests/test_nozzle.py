"""喷头映射：原始需求 §3 的 64 行表逐行校验、区间连续性、查询一致性、span、全局编号、to_dict。"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from oresorter.config import CameraConfig
from oresorter.vision.nozzle import NozzleMap

N = 64
Q = 10


def expected_widths(total: int, n: int) -> list[int]:
    """按原始需求 64 行表的归纳规则**独立**生成每个喷头的像素数。

    做法故意与实现不同：先每个喷头分 q 个，再把 r 个余数像素从两端交替分配——
    第 1 个给 1 号、第 2 个给 n 号、第 3 个给 2 号、第 4 个给 n−1 号……
    结果恰是"前 ⌈r/2⌉ 个与后 ⌊r/2⌋ 个各 q+1，中间各 q"。
    """
    q, r = divmod(total, n)
    widths = [q] * n
    front, back = 0, n - 1
    for k in range(r):
        if k % 2 == 0:
            widths[front] += 1
            front += 1
        else:
            widths[back] += 1
            back -= 1
    return widths


def plus_one_nozzles(m: NozzleMap) -> set[int]:
    """分到 q+1 像素的喷头编号集合（1 起）。"""
    return {i + 1 for i, w in enumerate(m.widths()) if w == m.quotient + 1}


# ----------------------------------------------------------------------------- 64 行表


@pytest.mark.parametrize("r", list(range(N)))
def test_table_row_matches_requirement(r: int) -> None:
    """n=64, q=10, start=0, end=640+r：逐喷头比对像素数。"""
    m = NozzleMap(0, N * Q + r, N)
    assert m.quotient == Q and m.remainder == r
    widths = m.widths()
    assert widths == expected_widths(N * Q + r, N), f"r={r} 像素分配与原始需求表不一致"
    assert sum(widths) == N * Q + r
    assert widths.count(Q + 1) == r and widths.count(Q) == N - r
    # 规则本身：前 ⌈r/2⌉ 与后 ⌊r/2⌋ 为 q+1
    head, tail = (r + 1) // 2, r // 2
    expected = set(range(1, head + 1)) | set(range(N - tail + 1, N + 1))
    assert plus_one_nozzles(m) == expected


@pytest.mark.parametrize(
    "r, expected",
    [
        (0, set()),
        (1, {1}),
        (2, {1, 64}),
        (3, {1, 2, 64}),
        (4, {1, 2, 63, 64}),
        (63, set(range(1, 33)) | set(range(34, 65))),
    ],
)
def test_documented_examples(r: int, expected: set[int]) -> None:
    """需求整理 §3 列出的校验样例。"""
    m = NozzleMap(0, N * Q + r, N)
    assert plus_one_nozzles(m) == expected
    if r == 63:
        assert m.widths()[32] == Q, "r=63 时 33 号喷头应为 q"


# ----------------------------------------------------------------------------- ranges / pixel_to_nozzle


def assert_ranges_consistent(m: NozzleMap) -> None:
    rs = m.ranges()
    assert len(rs) == m.n
    assert rs[0][0] == m.start_px
    assert rs[-1][1] == m.end_px - 1
    for (a0, b0), (a1, _b1) in pairwise(rs):
        assert a0 <= b0, "闭区间不能为空"
        assert a1 == b0 + 1, "相邻喷头必须无缝无重叠"
    widths = [b - a + 1 for a, b in rs]
    assert sum(widths) == m.valid_px
    assert set(widths) <= {m.quotient, m.quotient + 1}
    assert widths == m.widths()
    # 逐像素扫描：pixel_to_nozzle 与 ranges 一致
    for i, (a, b) in enumerate(rs, start=1):
        for x in range(a, b + 1):
            assert m.pixel_to_nozzle(x) == i
        assert m.range_of(i) == (a, b)
    assert m.pixel_to_nozzle(m.start_px - 1) is None
    assert m.pixel_to_nozzle(m.end_px) is None


@pytest.mark.parametrize(
    "start, end, n",
    [
        (0, 4096, 64),
        (0, 640, 64),
        (0, 703, 64),
        (100, 3000, 64),
        (100, 3000, 32),
        (7, 1000, 8),
        (0, 64, 64),  # 每喷头 1 像素
        (5, 6, 1),  # 单喷头、单像素
        (3, 1000, 1),
        (0, 4096, 1024),
    ],
)
def test_ranges_and_lookup_consistent(start: int, end: int, n: int) -> None:
    m = NozzleMap(start, end, n)
    assert (m.start_px, m.end_px, m.n) == (start, end, n)
    assert divmod(end - start, n) == (m.quotient, m.remainder)
    assert m.widths() == expected_widths(end - start, n)
    assert_ranges_consistent(m)


def test_nonzero_start() -> None:
    m = NozzleMap(100, 3000, 64)
    assert (m.quotient, m.remainder) == (45, 20)
    assert m.pixel_to_nozzle(99) is None
    assert m.pixel_to_nozzle(100) == 1
    assert m.pixel_to_nozzle(145) == 1  # 1 号 46 像素：100~145
    assert m.pixel_to_nozzle(146) == 2
    assert m.pixel_to_nozzle(2999) == 64
    assert m.pixel_to_nozzle(3000) is None
    assert plus_one_nozzles(m) == set(range(1, 11)) | set(range(55, 65))


def test_small_n() -> None:
    m32 = NozzleMap(0, 1000, 32)  # 1000 = 32·31 + 8
    assert (m32.quotient, m32.remainder) == (31, 8)
    assert plus_one_nozzles(m32) == {1, 2, 3, 4, 29, 30, 31, 32}
    m8 = NozzleMap(0, 101, 8)  # 101 = 8·12 + 5
    assert (m8.quotient, m8.remainder) == (12, 5)
    assert m8.widths() == [13, 13, 13, 12, 12, 12, 13, 13]


def test_pixel_lookup_out_of_range_and_types() -> None:
    m = NozzleMap(0, 640, 64)
    assert m.pixel_to_nozzle(-1) is None
    assert m.pixel_to_nozzle(10**9) is None
    assert m.pixel_to_nozzle(np.int64(15)) == 2  # numpy 整数可用
    with pytest.raises(TypeError):
        m.pixel_to_nozzle(12.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        m.pixel_to_nozzle("3")  # type: ignore[arg-type]


def test_ranges_returns_fresh_list() -> None:
    m = NozzleMap(0, 640, 64)
    rs = m.ranges()
    rs.clear()
    assert len(m.ranges()) == 64


# ----------------------------------------------------------------------------- span


def test_span_basic() -> None:
    m = NozzleMap(0, 640, 64)  # 每喷头 10 像素
    assert m.span(0, 9) == (1, 1)
    assert m.span(0, 10) == (1, 2)
    assert m.span(15, 15) == (2, 2)
    assert m.span(15, 37) == (2, 4)
    assert m.span(630, 639) == (64, 64)
    assert m.span(0, 639) == (1, 64)


def test_span_expand_clipped_to_1_n() -> None:
    m = NozzleMap(0, 640, 64)
    assert m.span(15, 37, expand=1) == (1, 5)
    assert m.span(15, 37, expand=2) == (1, 6)
    assert m.span(300, 305, expand=3) == (28, 34)
    assert m.span(630, 639, expand=5) == (59, 64)
    assert m.span(0, 639, expand=10) == (1, 64)


def test_span_partial_and_no_intersection() -> None:
    m = NozzleMap(100, 3000, 64)
    assert m.span(0, 99) is None
    assert m.span(3000, 4095) is None
    assert m.span(-50, -1) is None
    # 部分相交：裁到有效区
    assert m.span(50, 120) == (1, 1)
    assert m.span(2990, 3500) == (64, 64)
    assert m.span(0, 4095) == (1, 64)
    # 不相交时 expand 也不能把它救回来
    assert m.span(0, 99, expand=3) is None


def test_span_reversed_is_swapped() -> None:
    m = NozzleMap(0, 640, 64)
    assert m.span(37, 15) == m.span(15, 37) == (2, 4)
    assert m.span(639, 0) == (1, 64)


def test_span_invalid_inputs() -> None:
    m = NozzleMap(0, 640, 64)
    with pytest.raises(ValueError):
        m.span(0, 10, expand=-1)
    with pytest.raises(TypeError):
        m.span(0.0, 10)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- 全局编号


def test_to_global_and_global_span() -> None:
    m0 = NozzleMap(0, 640, 64, camera_index=0)
    m2 = NozzleMap(0, 640, 64, camera_index=2)
    assert m0.to_global(1) == 1 and m0.to_global(64) == 64
    assert m2.to_global(1) == 129 and m2.to_global(64) == 192
    assert m2.global_span(15, 37) == (130, 132)
    assert m2.global_span(0, 639, expand=3) == (129, 192)
    assert m2.global_span(640, 700) is None
    # HMI 孔位上限 768 = 12 × 64：最后一台相机的最后一个喷头
    m11 = NozzleMap(0, 4096, 64, camera_index=11)
    assert m11.to_global(64) == 768
    for bad in (0, 65, -1):
        with pytest.raises(ValueError):
            m0.to_global(bad)


# ----------------------------------------------------------------------------- 非法输入


@pytest.mark.parametrize(
    "start, end, n, kw",
    [
        (100, 100, 64, {}),  # start == end
        (200, 100, 64, {}),  # start > end
        (0, 63, 64, {}),  # 有效像素 < n
        (0, 640, 0, {}),  # n < 1
        (0, 640, -3, {}),
        (0, 640, 64, {"camera_index": -1}),
    ],
)
def test_constructor_rejects(start: int, end: int, n: int, kw: dict) -> None:
    with pytest.raises(ValueError) as ei:
        NozzleMap(start, end, n, **kw)
    assert str(ei.value), "错误必须带说明"


def test_constructor_rejects_non_integers() -> None:
    with pytest.raises(TypeError):
        NozzleMap(0.0, 640, 64)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NozzleMap(0, 640, 64.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NozzleMap(0, 640, 64).range_of(65)


# ----------------------------------------------------------------------------- to_dict / from_camera


def test_to_dict_matches_docs03() -> None:
    m = NozzleMap(0, 4096, 64)
    d = m.to_dict()
    assert set(d) == {"start_px", "end_px", "nozzles", "quotient", "remainder", "ranges"}
    assert d["start_px"] == 0 and d["end_px"] == 4096
    assert d["nozzles"] == 64 and d["quotient"] == 64 and d["remainder"] == 0
    assert len(d["ranges"]) == 64
    assert d["ranges"][0] == {"nozzle": 1, "px_from": 0, "px_to": 63}
    assert d["ranges"][1] == {"nozzle": 2, "px_from": 64, "px_to": 127}
    assert d["ranges"][-1] == {"nozzle": 64, "px_from": 4032, "px_to": 4095}
    assert [r["nozzle"] for r in d["ranges"]] == list(range(1, 65))
    # 可直接 JSON 序列化（API 原样返回）
    assert json.loads(json.dumps(d)) == d


def test_to_dict_with_remainder() -> None:
    d = NozzleMap(100, 3000, 64).to_dict()
    assert d["quotient"] == 45 and d["remainder"] == 20
    assert d["ranges"][0] == {"nozzle": 1, "px_from": 100, "px_to": 145}
    assert d["ranges"][-1] == {"nozzle": 64, "px_from": 2954, "px_to": 2999}


def test_from_camera(tmp_path: Path) -> None:
    cam = CameraConfig(id=1, start_px=100, end_px=3000, nozzles=32)
    m = NozzleMap.from_camera(cam, camera_index=1)
    assert m == NozzleMap(100, 3000, 32, camera_index=1)
    assert (m.start_px, m.end_px, m.n, m.camera_index) == (100, 3000, 32, 1)
    assert m.to_global(1) == 33

    from tests.conftest import make_cfg

    cfg = make_cfg(tmp_path)
    cam0 = cfg.cameras[0]
    m0 = NozzleMap.from_camera(cam0)
    assert (m0.start_px, m0.end_px, m0.n) == (cam0.start_px, cam0.end_px, cam0.nozzles)
    assert m0.camera_index == 0
    assert_ranges_consistent(m0)


def test_equality_and_repr() -> None:
    a = NozzleMap(0, 640, 64)
    b = NozzleMap(0, 640, 64)
    c = NozzleMap(0, 640, 64, camera_index=1)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert "q=10" in repr(a) and "r=0" in repr(a)
