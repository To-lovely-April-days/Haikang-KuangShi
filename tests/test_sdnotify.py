"""sd_notify：无 socket 时空操作；有 socket 时收到 READY/WATCHDOG；健康检查失败不喂狗。"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import time

from oresorter.sdnotify import SdNotify, WatchdogThread, watchdog_interval_s


def test_disabled_without_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    sd = SdNotify()
    assert not sd.enabled and sd.ready() is False and sd.watchdog() is False


def test_notify_and_watchdog(monkeypatch):
    # AF_UNIX 路径上限 104 字节（macOS），pytest 的 tmp_path 太深，用短临时目录
    tmpdir = tempfile.mkdtemp(prefix="sdn")
    path = os.path.join(tmpdir, "n.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(path)
    srv.settimeout(2.0)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    monkeypatch.setenv("WATCHDOG_USEC", "200000")  # 0.2 s → 喂狗间隔 max(0.5, 0.1)=0.5
    assert watchdog_interval_s() == 0.5
    sd = SdNotify()
    assert sd.enabled and sd.ready()
    assert srv.recv(64) == b"READY=1"
    healthy = {"ok": True}
    wd = WatchdogThread(sd, 0.05, healthy=lambda: healthy["ok"])
    wd.start()
    assert srv.recv(64) == b"WATCHDOG=1"
    healthy["ok"] = False
    time.sleep(0.2)
    skipped_before = wd.skipped
    assert skipped_before >= 1
    wd.stop()
    assert not wd.is_alive()
    sd.close()
    srv.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
