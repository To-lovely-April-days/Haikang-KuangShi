"""请求/响应模型（与 docs/03 一一对应；字段名就是接口字段名）。P2 补齐全部接口。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error: ErrorDetail


class HealthOut(BaseModel):
    ok: bool = True
    version: str


class SavedOut(BaseModel):
    saved: bool = True
    backup_path: str | None = None


class RequestIdBody(BaseModel):
    """所有写操作可带的幂等键。"""

    request_id: str | None = Field(None, max_length=128)
