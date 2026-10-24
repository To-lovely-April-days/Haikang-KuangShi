"""运行控制：POST /run/start|pause|stop、GET /run/stats（docs/03 §4）。

业务规则（409 原因、MCU 下发、流水线启停）全部在 ``RuntimeService`` 中，这里只做转发。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from oresorter.api.routes_system import get_runtime
from oresorter.api.schemas import RequestIdBody, RunOut, StatsOut

router = APIRouter(tags=["run"])


@router.post("/run/start", response_model=RunOut)
def run_start(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """开始运行：前置条件不满足 → 409 ``CONFLICT``，``detail.reasons`` 给出原因枚举。"""
    return get_runtime(request).start()


@router.post("/run/pause", response_model=RunOut)
def run_pause(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """暂停（停吹气，继续分类统计）。"""
    return get_runtime(request).pause()


@router.post("/run/stop", response_model=RunOut)
def run_stop(request: Request, body: RequestIdBody | None = None) -> dict[str, Any]:
    """停止；``fault`` 态下返回 200 ``run_state=fault``；``sampling`` 态下取消采样。"""
    return get_runtime(request).stop()


@router.get("/run/stats", response_model=StatsOut)
def run_stats(request: Request) -> dict[str, Any]:
    """与 ``state.stats`` 同结构，另含 ``since_start`` 累计与各流水线统计。"""
    return get_runtime(request).run_stats()
