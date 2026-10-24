"""相机与喷阀检测：/cameras/*、/capture/preview、/valve/*（docs/03 §6、§7、§9）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request

from oresorter.api.routes_params import strip_request_id
from oresorter.api.routes_system import get_runtime
from oresorter.api.schemas import (
    AcceptedOut,
    AppliedOut,
    CameraOut,
    CameraParamsApplyOut,
    CameraParamsOut,
    CapturePreviewIn,
    CapturePreviewOut,
    JobIdOut,
    NozzleMapOut,
    RequestIdBody,
    SnapshotIn,
    SnapshotOut,
    StoppedOut,
    ValveResetOut,
    ValveTestCycleIn,
    ValveTestSingleIn,
)

router = APIRouter(tags=["camera"])


@router.get("/cameras", response_model=list[CameraOut])
def list_cameras(request: Request) -> list[dict[str, Any]]:
    """全部相机的静态信息与连接状态。"""
    return get_runtime(request).list_cameras()


@router.get("/cameras/{camera_id}/params", response_model=CameraParamsOut)
def camera_params(camera_id: int, request: Request) -> dict[str, Any]:
    """从相机实时读取参数 + 算力盒侧起止像素 / 块高 + ``limits``；相机离线 → 503。"""
    return get_runtime(request).camera_params(camera_id)


@router.put("/cameras/{camera_id}/params", response_model=CameraParamsApplyOut)
def put_camera_params(
    camera_id: int,
    request: Request,
    patch: dict[str, Any] = Body(..., description="可编辑字段的部分或全部"),
) -> dict[str, Any]:
    """校验持久化 → 下发相机；``start_px / end_px`` 立即影响喷头映射；相机离线时保存并在 warnings 说明。"""
    return get_runtime(request).update_camera_params(camera_id, strip_request_id(patch))


@router.post("/cameras/{camera_id}/awb-once", response_model=AppliedOut)
def awb_once(camera_id: int, request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """做一次白平衡并锁定为 ``off``。"""
    return get_runtime(request).awb_once(camera_id)


@router.get("/cameras/{camera_id}/nozzle-map", response_model=NozzleMapOut)
def nozzle_map(camera_id: int, request: Request) -> dict[str, Any]:
    """像素 → 喷头映射表。"""
    return get_runtime(request).nozzle_map(camera_id)


@router.post(
    "/cameras/{camera_id}/snapshot", response_model=SnapshotOut, response_model_exclude_unset=True
)
def snapshot(camera_id: int, request: Request, body: SnapshotIn | None = None) -> dict[str, Any]:
    """单次采集，写入 ``/static/snapshots/``；``overlay=true`` 叠画分割框。"""
    svc = get_runtime(request)
    body = body or SnapshotIn()
    height = body.height if body.height is not None else svc.cfg.camera(camera_id).height
    return svc.snapshot_image(camera_id, height, body.max_width, body.overlay)


@router.post("/capture/preview", response_model=CapturePreviewOut)
def capture_preview(request: Request, body: CapturePreviewIn) -> dict[str, Any]:
    """用（临时覆盖的）采集设置分割一帧并叠画，返回颗粒列表与叠画图 URL。"""
    return get_runtime(request).capture_preview(body.camera_id, body.settings)


# ---- 喷阀检测（running / paused / sampling → 409）----


@router.post("/valve/test/single", response_model=AcceptedOut)
def valve_test_single(request: Request, body: ValveTestSingleIn) -> dict[str, Any]:
    """单阀检测；孔位范围见 ``limits.valve_test.hole``。"""
    return get_runtime(request).valve_test_single(body.hole)


@router.post("/valve/test/cycle", response_model=JobIdOut)
def valve_test_cycle(request: Request, body: ValveTestCycleIn) -> dict[str, Any]:
    """循环检测（后台任务，可取消 / ``/valve/test/stop``）。"""
    return get_runtime(request).valve_test_cycle(body.from_, body.to, body.interval_ms)


@router.post("/valve/test/stop", response_model=StoppedOut)
def valve_test_stop(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """停止喷阀检测。"""
    return get_runtime(request).valve_test_stop()


@router.post("/valve/reset", response_model=ValveResetOut)
def valve_reset(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """喷阀复位。"""
    return get_runtime(request).valve_reset()
