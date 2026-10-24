"""模型（方案）与任务：/models/*、/jobs/*（docs/03 §8、§10）。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request, status

from oresorter.api.routes_system import get_runtime
from oresorter.api.schemas import (
    BackupIn,
    CancelledOut,
    JobIdOut,
    JobOut,
    ModelCreateIn,
    ModelDeletedOut,
    ModelOut,
    ModelPatchIn,
    RequestIdBody,
    TrainIn,
)

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelOut])
def list_models(request: Request) -> list[dict[str, Any]]:
    """1..``system.max_models`` 全部编号（未创建的 ``created_at`` 为 null）。"""
    return get_runtime(request).list_models()


@router.get("/models/current", response_model=ModelOut | None)
def current_model(request: Request) -> dict[str, Any] | None:
    """当前部署模型；无则 ``null``。"""
    return get_runtime(request).current_model()


@router.post("/models/{model_id}", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
def create_model(
    model_id: int, request: Request, body: ModelCreateIn | None = None
) -> dict[str, Any]:
    """新建模型目录结构；已存在 → 409。"""
    body = body or ModelCreateIn()
    return get_runtime(request).create_model(model_id, body.name, body.description)


@router.patch("/models/{model_id}", response_model=ModelOut)
def patch_model(model_id: int, request: Request, body: ModelPatchIn) -> dict[str, Any]:
    """修改名称 / 说明。"""
    patch = body.model_dump(exclude_none=True, exclude={"request_id"})
    return get_runtime(request).patch_model(model_id, patch)


@router.delete("/models/{model_id}", response_model=ModelDeletedOut)
def delete_model(model_id: int, request: Request) -> dict[str, Any]:
    """删除全部参数与图片；当前部署 / 有任务在用 → 409。"""
    get_runtime(request).delete_model(model_id)
    return {"deleted": True, "id": model_id}


@router.post("/models/{model_id}/backup", response_model=JobIdOut)
def backup_model(model_id: int, request: Request, body: BackupIn) -> dict[str, Any]:
    """备份至另一个编号（整目录复制，后台任务）。"""
    return get_runtime(request).backup_model(model_id, body.target_id, body.overwrite)


@router.post("/models/{model_id}/deploy", response_model=JobIdOut)
def deploy_model(
    model_id: int, request: Request, body: RequestIdBody | None = None
) -> dict[str, Any]:
    """部署（后台任务）；未训练 → 409。"""
    return get_runtime(request).deploy_model(model_id)


@router.get("/models/{model_id}/metrics")
def model_metrics(model_id: int, request: Request) -> dict[str, Any]:
    """``metrics.json`` 全文（混淆矩阵、每类指标、训练曲线）。"""
    return get_runtime(request).model_metrics(model_id)


@router.post(
    "/models/{model_id}/train", response_model=JobIdOut, status_code=status.HTTP_202_ACCEPTED
)
def train_model(model_id: int, request: Request, body: TrainIn | None = None) -> dict[str, Any]:
    """提交训练任务；已有训练 → 409；样本不足 → 400 ``VALIDATION``（``detail.insufficient``）。"""
    body = body or TrainIn()
    return get_runtime(request).train_model(model_id, body.classes)


# ---- 任务 ----


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    request: Request, active: bool = Query(False, description="只列排队 / 运行中的任务")
) -> list[dict[str, Any]]:
    """任务列表（含最近结束的历史）。"""
    return get_runtime(request).list_jobs(active)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    """单个任务；不存在 → 404。"""
    return get_runtime(request).get_job(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=CancelledOut)
def cancel_job(job_id: str, request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """请求取消；不存在 / 已结束 → ``cancelled=false``。"""
    return get_runtime(request).cancel_job(job_id)
