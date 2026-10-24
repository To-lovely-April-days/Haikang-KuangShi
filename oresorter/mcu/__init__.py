"""下位机层：协议、链路抽象、TCP 实现与 Mock。"""

from oresorter.mcu.link import (
    Ack,
    EjectItem,
    LinkEvent,
    McuDisconnected,
    McuError,
    McuLink,
    McuNack,
    McuStatus,
    McuTimeout,
)

__all__ = [
    "Ack",
    "EjectItem",
    "LinkEvent",
    "McuDisconnected",
    "McuError",
    "McuLink",
    "McuNack",
    "McuStatus",
    "McuTimeout",
    "build_mcu_link",
]


def build_mcu_link(cfg):
    """按 ``cfg.system.mcu_kind`` 构造链路（mock / tcp）。"""
    if cfg.system.mcu_kind == "mock":
        from oresorter.mcu.mock import MockMcuLink

        return MockMcuLink(cfg)
    if cfg.system.mcu_kind == "tcp":
        from oresorter.mcu.link import TcpMcuLink

        return TcpMcuLink(cfg)
    raise McuError(f"未知下位机链路类型: {cfg.system.mcu_kind}")
