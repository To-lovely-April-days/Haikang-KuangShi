"""海康 MVS SDK 线阵相机封装（P3 真机联调时实现）。

当前为占位：无 SDK 时给出清晰错误，不影响 Mock 路径。
SDK 位置见 CLAUDE.md「Jetson 环境提示」。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from oresorter.camera.base import Camera, CameraError

_SDK_HINT = (
    "未找到海康 MVS SDK 的 Python 封装（MvCameraControl_class）。"
    "请安装 MVS_STD_GML_V2.x_aarch64.deb 到 /opt/MVS，并设置 "
    "MVCAM_COMMON_RUNENV=/opt/MVS/lib、LD_LIBRARY_PATH 追加 /opt/MVS/lib/aarch64，"
    "或把 /opt/MVS/Samples/aarch64/Python/MvImport 加入 PYTHONPATH / 复制到 oresorter/camera/hik_sdk/。"
)


def _import_sdk():
    """尝试导入 MvImport；失败抛 CameraError 并给出安装提示。"""
    candidates = [
        Path(__file__).parent / "hik_sdk",
        Path(os.environ.get("MVS_PYTHON_DIR", "/opt/MVS/Samples/aarch64/Python/MvImport")),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.append(str(c))
    try:
        from MvCameraControl_class import MvCamera  # type: ignore[import-not-found]
    except Exception as exc:
        raise CameraError(_SDK_HINT) from exc
    return MvCamera


class HikCamera(Camera):
    """海康 MV-CL043-A1GC 封装。P3 实现；现在仅校验 SDK 可用性。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._MvCamera = _import_sdk()
        raise CameraError("HikCamera 尚未实现（P3 真机联调阶段）。开发机请使用 kind: mock。")

    # 以下仅为满足抽象类（P3 实现）
    def open(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError

    @property
    def connected(self) -> bool:  # pragma: no cover
        return False

    @property
    def streaming(self) -> bool:  # pragma: no cover
        return False

    def get_params(self):  # pragma: no cover
        raise NotImplementedError

    def set_params(self, p):  # pragma: no cover
        raise NotImplementedError

    def awb_once(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def snapshot(self, height: int):  # pragma: no cover
        raise NotImplementedError

    def start_stream(self, chunk_lines: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def stop_stream(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_frame(self, timeout: float):  # pragma: no cover
        raise NotImplementedError

    def reset_line_counter(self) -> None:  # pragma: no cover
        raise NotImplementedError
