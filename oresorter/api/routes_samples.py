"""样本采集与管理：/models/{id}/classes/{cls}/*、/samples/*（docs/03 §9）。

``sample_id`` 形如 ``"{model_id}/{cls}/{stem}"``，含 ``/``：路由用 ``{sample_id:path}`` 接收，
URL 中既可写 ``%2F``（服务器解码后同样命中 path 参数），也可直接写 ``/``。
"""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, Query, Request

from oresorter.api.routes_system import get_runtime
from oresorter.api.schemas import (
    CaptureIn,
    ClearedOut,
    JobIdOut,
    RequestIdBody,
    RestoredOut,
    SampleDeletedOut,
    SampleListOut,
    SamplesDeletedOut,
    SamplesDeleteIn,
    UndoDeleteOut,
)

router = APIRouter(tags=["samples"])


def _sid(sample_id: str) -> str:
    """兼容客户端双重编码（``%252F``）：path 参数再做一次解码。"""
    return unquote(sample_id)


@router.post("/models/{model_id}/classes/{cls}/capture", response_model=JobIdOut)
def start_capture(model_id: int, cls: str, request: Request, body: CaptureIn) -> dict[str, Any]:
    """连续采集（后台任务）；前置 ``run_state ∈ {stopped, paused}``，否则 409。"""
    return get_runtime(request).start_capture(
        model_id,
        cls,
        body.camera_id,
        max_samples=body.max_samples,
        max_frames=body.max_frames,
        timeout_s=body.timeout_s,
    )


@router.get("/models/{model_id}/classes/{cls}/samples", response_model=SampleListOut)
def list_samples(
    model_id: int,
    cls: str,
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    include_deleted: bool = Query(False),
) -> dict[str, Any]:
    """分页样本列表（按写入顺序升序）。"""
    return get_runtime(request).list_samples(model_id, cls, offset, limit, include_deleted)


@router.post("/samples/delete", response_model=SamplesDeletedOut)
def delete_samples(request: Request, body: SamplesDeleteIn) -> dict[str, Any]:
    """批量软删除：跨类别的 ids 按类别拆成多个撤销批次。"""
    return get_runtime(request).delete_samples(body.ids)


@router.post("/samples/{sample_id:path}/restore", response_model=RestoredOut)
def restore_sample(
    sample_id: str, request: Request, body: RequestIdBody | None = None
) -> dict[str, Any]:
    """恢复单个已软删除样本。"""
    return get_runtime(request).restore_sample(_sid(sample_id))


@router.delete("/samples/{sample_id:path}", response_model=SampleDeletedOut)
def delete_sample(sample_id: str, request: Request) -> dict[str, Any]:
    """软删除单个样本，返回可撤销批次号。"""
    return get_runtime(request).delete_sample(_sid(sample_id))


@router.post("/models/{model_id}/classes/{cls}/undo-delete", response_model=UndoDeleteOut)
def undo_delete(
    model_id: int, cls: str, request: Request, body: RequestIdBody | None = None
) -> dict[str, Any]:
    """撤销该类最近一次删除批次；无可撤销 → 404。"""
    return get_runtime(request).undo_delete(model_id, cls)


@router.delete("/models/{model_id}/classes/{cls}/samples", response_model=ClearedOut)
def clear_samples(model_id: int, cls: str, request: Request) -> dict[str, Any]:
    """清空该类（软删除为一个批次，可撤销一次）。"""
    return get_runtime(request).clear_samples(model_id, cls)
