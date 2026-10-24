"""参数与类别配置：GET /params、PUT /params/{section}、GET|PUT /config/classes（docs/03 §5、§8）。

``PUT /params/{section}`` 的请求体是该分组的部分字段（自由对象），范围校验由 ``ConfigService`` 完成；
``section`` 的合法取值（:data:`PARAM_SECTIONS`）写进 OpenAPI 的 ``enum`` 供 HMI 生成客户端，但路径参数
本身按 ``str`` 接收：未知分组由 service 以 404 ``NOT_FOUND`` 拒绝（而不是 FastAPI 的 400 校验错误）。
"""

from __future__ import annotations

from typing import Any, get_args

from fastapi import APIRouter, Body, Path, Request

from oresorter.api.idempotency import REQUEST_ID_FIELD
from oresorter.api.routes_system import get_runtime
from oresorter.api.schemas import (
    ClassesIn,
    ClassesOut,
    ParamsApplyOut,
    ParamSection,
    ParamsOut,
)
from oresorter.config import ConfigError

router = APIRouter(tags=["params"])

PARAM_SECTIONS: tuple[str, ...] = get_args(ParamSection)
"""``PUT /params/{section}`` 的合法分组（docs/03 §5）。"""


def strip_request_id(patch: Any) -> dict[str, Any]:
    """自由对象请求体：剔除幂等键 ``request_id``（它不是业务字段），非对象 → 400 VALIDATION。"""
    if not isinstance(patch, dict):
        raise ConfigError("VALIDATION", "请求体必须是 JSON 对象", {"type": type(patch).__name__})
    return {k: v for k, v in patch.items() if k != REQUEST_ID_FIELD}


@router.get("/params", response_model=ParamsOut)
def get_params(request: Request) -> dict[str, Any]:
    """六个参数分组 + ``limits``（HMI 用它做输入范围）。"""
    return get_runtime(request).get_params()


@router.put("/params/{section}", response_model=ParamsApplyOut)
def put_params(
    request: Request,
    section: str = Path(
        ...,
        description="参数分组：" + " | ".join(PARAM_SECTIONS) + "；其它 → 404 NOT_FOUND",
        json_schema_extra={"enum": list(PARAM_SECTIONS)},
    ),
    patch: dict[str, Any] = Body(..., description="该分组的部分或全部字段"),
) -> dict[str, Any]:
    """部分更新 ``valve | clean | feed | outputs | blow | capture``；越界 → 400 ``PARAM_OUT_OF_RANGE``
    且不做部分应用；``mcu_ack`` 为 null 表示该分组不下发下位机。"""
    return get_runtime(request).update_params(section, strip_request_id(patch))


@router.get("/config/classes", response_model=ClassesOut)
def get_classes(request: Request) -> dict[str, Any]:
    """类别名称与模型数量上限。"""
    return get_runtime(request).get_classes()


@router.put("/config/classes", response_model=ClassesOut)
def put_classes(request: Request, body: ClassesIn) -> dict[str, Any]:
    """整体替换类别名（需停止状态且所有模型未训练，否则 409）。"""
    return get_runtime(request).set_classes(body.class_names)
