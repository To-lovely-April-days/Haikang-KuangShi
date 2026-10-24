"""systemd ``sd_notify`` 极简实现（无第三方依赖）：READY / STATUS / WATCHDOG。

不在 systemd 下运行（无 ``NOTIFY_SOCKET``）时所有调用都是空操作，开发机无感。
"""

from __future__ import annotations

import os
import socket
import threading
from collections.abc import Callable

from oresorter.log import get_logger

log = get_logger(__name__)


class SdNotify:
    def __init__(self, address: str | None = None):
        self.address = address if address is not None else os.environ.get("NOTIFY_SOCKET")
        self._sock: socket.socket | None = None
        if self.address:
            addr = self.address
            if addr.startswith("@"):  # 抽象命名空间
                addr = "\0" + addr[1:]
            try:
                self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._sock.settimeout(0.5)
                self._sock.connect(addr)
            except OSError as exc:
                log.warning("无法连接 NOTIFY_SOCKET，sd_notify 关闭", extra={"err": str(exc)})
                self._sock = None

    @property
    def enabled(self) -> bool:
        return self._sock is not None

    def notify(self, message: str) -> bool:
        if self._sock is None:
            return False
        try:
            self._sock.send(message.encode("utf-8"))
            return True
        except OSError as exc:
            log.warning("sd_notify 发送失败", extra={"err": str(exc), "notify_msg": message[:40]})
            return False

    def ready(self) -> bool:
        return self.notify("READY=1")

    def status(self, text: str) -> bool:
        return self.notify(f"STATUS={text}")

    def stopping(self) -> bool:
        return self.notify("STOPPING=1")

    def watchdog(self) -> bool:
        return self.notify("WATCHDOG=1")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


def watchdog_interval_s(default: float | None = None) -> float | None:
    """由 ``WATCHDOG_USEC`` 推出喂狗间隔（周期的一半）；未设置返回 default。"""
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return default
    try:
        return max(0.5, int(usec) / 1_000_000 / 2)
    except ValueError:
        return default


class WatchdogThread(threading.Thread):
    """周期喂狗；``healthy()`` 返回 False 时停止喂狗，让 systemd 重启服务。"""

    def __init__(self, sd: SdNotify, interval_s: float, healthy: Callable[[], bool] | None = None):
        super().__init__(name="sd-watchdog", daemon=True)
        self.sd = sd
        self.interval_s = interval_s
        self.healthy = healthy or (lambda: True)
        self._stop_evt = threading.Event()
        self.fed = 0
        self.skipped = 0

    def run(self) -> None:
        while not self._stop_evt.wait(self.interval_s):
            try:
                ok = bool(self.healthy())
            except Exception:
                log.exception("健康检查异常，本轮不喂狗")
                ok = False
            if ok:
                self.sd.watchdog()
                self.fed += 1
            else:
                self.skipped += 1
                log.error("健康检查失败，不喂狗（systemd 将在 WatchdogSec 后重启服务）")

    def stop(self) -> None:
        self._stop_evt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=2.0)
