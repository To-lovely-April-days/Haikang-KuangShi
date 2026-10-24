"""Mock 相机：块形状与行号连续、丢块注入、掉线、快照、参数。"""

from __future__ import annotations

import time

import numpy as np
import pytest

from oresorter.camera import CameraDisconnected, build_camera
from oresorter.camera.mock import CLASS_PALETTE, MockCamera, synth_block
from tests.conftest import make_cfg


def _cam(tmp_path, **mock_over) -> MockCamera:
    cfg = make_cfg(tmp_path)
    cam_cfg = cfg.cameras[0].model_copy(
        update={"mock": cfg.cameras[0].mock.model_copy(update=mock_over)}
    )
    cam = build_camera(cam_cfg)
    assert isinstance(cam, MockCamera)
    return cam


def test_stream_contiguous_lines(tmp_path):
    cam = _cam(tmp_path)
    cam.open()
    cam.start_stream(256)
    frames = [cam.get_frame(2.0) for _ in range(6)]
    cam.stop_stream()
    cam.close()
    assert all(f is not None for f in frames)
    for i, f in enumerate(frames):
        assert f.image.shape == (256, 4096, 3) and f.image.dtype == np.uint8
        assert f.seq == i
        assert f.line_start == i * 256
        assert f.camera_id == 1
    assert cam.stats.chunks_total == 6


def test_drop_injection_advances_line_counter(tmp_path):
    cam = _cam(tmp_path)
    cam.open()
    cam.inject_drop(2)
    cam.start_stream(128)
    f = cam.get_frame(2.0)
    cam.stop_stream()
    cam.close()
    assert f.seq == 2 and f.line_start == 256
    assert cam.stats.dropped_chunks == 2


def test_offline_raises_and_recovers(tmp_path):
    cam = _cam(tmp_path)
    cam.open()
    cam.start_stream(128)
    assert cam.get_frame(2.0) is not None
    cam.set_offline(True)
    assert not cam.connected
    with pytest.raises(CameraDisconnected):
        cam.get_frame(0.1)
    cam.set_offline(False)
    assert cam.connected
    assert cam.get_frame(2.0) is not None
    assert cam.stats.reconnects == 1
    cam.close()


def test_reset_line_counter(tmp_path):
    cam = _cam(tmp_path)
    cam.open()
    cam.start_stream(128)
    for _ in range(3):
        cam.get_frame(2.0)
    cam.reset_line_counter()
    f = cam.get_frame(2.0)
    assert f.line_start < 3 * 128
    cam.close()


def test_snapshot_and_params(tmp_path):
    cam = _cam(tmp_path)
    cam.open()
    img = cam.snapshot(300)
    assert img.shape == (300, 4096, 3)
    p = cam.get_params()
    p.width = 2048
    p.exposure_us = 77
    applied = cam.set_params(p)
    assert applied.width == 2048 and applied.exposure_us == 77
    assert cam.snapshot(64).shape == (64, 2048, 3)
    cam.awb_once()
    assert cam.get_params().awb_mode == "off"
    cam.close()
    with pytest.raises(CameraDisconnected):
        cam.get_params()


def test_replay_directory(tmp_path):
    import cv2

    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img, _ = synth_block(rng, 1024, 700, n_particles=3)
        cv2.imwrite(str(d / f"{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cfg = make_cfg(tmp_path)
    cam_cfg = cfg.cameras[0].model_copy(
        update={
            "width": 1024,
            "mock": cfg.cameras[0].mock.model_copy(update={"source_dir": str(d)}),
        }
    )
    cam = MockCamera(cam_cfg)
    cam.open()
    cam.start_stream(500)
    frames = [cam.get_frame(2.0) for _ in range(4)]  # 2000 行 > 1400 行素材，应循环
    cam.close()
    assert all(f.image.shape == (500, 1024, 3) for f in frames)


def test_speed_pacing(tmp_path):
    cam = _cam(tmp_path, speed=1.0)
    cam.cfg.__dict__  # noqa: B018
    cam.open()
    cam.start_stream(512)  # 9 kHz → 每块 ≈ 56.9 ms
    t0 = time.monotonic()
    for _ in range(5):
        assert cam.get_frame(2.0) is not None
    dt = time.monotonic() - t0
    cam.close()
    assert dt >= 0.2, f"5 块应至少耗时 ~0.23 s，实际 {dt:.3f}s"


def test_synth_block_labels():
    rng = np.random.default_rng(1)
    img, labels = synth_block(rng, 800, 600, n_particles=5)
    assert img.shape == (800 * 0 + 600, 800, 3)
    assert 1 <= len(labels) <= 5
    for cls, (x, y, w, h) in labels:
        assert cls in CLASS_PALETTE and w > 0 and h > 0 and 0 <= x < 800 and 0 <= y < 600
