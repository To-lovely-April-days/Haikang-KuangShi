"""相机层：``Camera`` 抽象、``Frame`` 数据结构、Mock 与海康实现。"""

from oresorter.camera.base import (
    Camera,
    CameraDisconnected,
    CameraError,
    CameraInfo,
    CameraParams,
    CameraStats,
    CameraTimeout,
    Frame,
)

__all__ = [
    "Camera",
    "CameraDisconnected",
    "CameraError",
    "CameraInfo",
    "CameraParams",
    "CameraStats",
    "CameraTimeout",
    "Frame",
    "build_camera",
]


def build_camera(cfg):
    """按 ``cfg.kind`` 构造相机实例（mock / hik）。"""
    if cfg.kind == "mock":
        from oresorter.camera.mock import MockCamera

        return MockCamera(cfg)
    if cfg.kind == "hik":
        from oresorter.camera.hik import HikCamera

        return HikCamera(cfg)
    raise CameraError(f"未知相机类型: {cfg.kind}")
