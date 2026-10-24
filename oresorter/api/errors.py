"""统一错误体 ``{"error": {"code", "message", "detail"}}`` 与异常映射。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from oresorter.config import ConfigError
from oresorter.data.samples import StoreError
from oresorter.log import get_logger
from oresorter.mcu.link import McuDisconnected, McuNack, McuTimeout
from oresorter.ml.infer import ClassifierUnavailable
from oresorter.ml.train import TrainError
from oresorter.runtime.jobs import JobConflict
from oresorter.runtime.state import IllegalTransition

log = get_logger(__name__)

# docs/03 §1 常用错误码 → HTTP 状态
CODE_STATUS: dict[str, int] = {
    "VALIDATION": 400,
    "PARAM_OUT_OF_RANGE": 400,
    "NOT_FOUND": 404,
    "CONFLICT": 409,
    "DEVICE_OFFLINE": 503,
    "MCU_NACK": 502,
    "MCU_TIMEOUT": 504,
    "INTERNAL": 500,
}


class ApiError(Exception):
    """业务错误。``code`` 取 docs/03 定义的值；``status`` 缺省按 CODE_STATUS。"""

    def __init__(
        self,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        status: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}
        self.status = status or CODE_STATUS.get(code, 500)

    def to_response(self) -> JSONResponse:
        return error_response(self.status, self.code, self.message, self.detail)


def error_response(
    status: int, code: str, message: str, detail: dict[str, Any] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "detail": detail or {}}},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return exc.to_response()

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, exc: ConfigError) -> JSONResponse:
        return error_response(CODE_STATUS.get(exc.code, 400), exc.code, exc.message, exc.detail)

    @app.exception_handler(StoreError)
    async def _store_error(_: Request, exc: StoreError) -> JSONResponse:
        return error_response(CODE_STATUS.get(exc.code, 400), exc.code, exc.message, exc.detail)

    @app.exception_handler(TrainError)
    async def _train_error(_: Request, exc: TrainError) -> JSONResponse:
        code = "CONFLICT" if exc.code == "CANCELLED" else exc.code
        return error_response(CODE_STATUS.get(code, 400), code, exc.message, exc.detail)

    @app.exception_handler(IllegalTransition)
    async def _illegal_transition(_: Request, exc: IllegalTransition) -> JSONResponse:
        return error_response(409, "CONFLICT", str(exc), exc.detail)

    @app.exception_handler(JobConflict)
    async def _job_conflict(_: Request, exc: JobConflict) -> JSONResponse:
        return error_response(409, "CONFLICT", str(exc), exc.detail)

    @app.exception_handler(ClassifierUnavailable)
    async def _classifier_unavailable(_: Request, exc: ClassifierUnavailable) -> JSONResponse:
        return error_response(409, "CONFLICT", str(exc), {})

    @app.exception_handler(McuNack)
    async def _mcu_nack(_: Request, exc: McuNack) -> JSONResponse:
        return error_response(502, "MCU_NACK", str(exc), {"result": exc.result, "cmd": exc.cmd})

    @app.exception_handler(McuTimeout)
    async def _mcu_timeout(_: Request, exc: McuTimeout) -> JSONResponse:
        return error_response(504, "MCU_TIMEOUT", str(exc), {})

    @app.exception_handler(McuDisconnected)
    async def _mcu_disconnected(_: Request, exc: McuDisconnected) -> JSONResponse:
        return error_response(503, "DEVICE_OFFLINE", str(exc), {"device": "mcu"})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = []
        for e in exc.errors():
            errors.append(
                {
                    "field": ".".join(str(x) for x in e.get("loc", ())),
                    "message": e.get("msg", ""),
                    "type": e.get("type", ""),
                }
            )
        msg = "; ".join(f"{e['field']}: {e['message']}" for e in errors) or "请求参数错误"
        return error_response(400, "VALIDATION", msg, {"errors": errors})

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = (
            "NOT_FOUND"
            if exc.status_code == 404
            else ("VALIDATION" if exc.status_code < 500 else "INTERNAL")
        )
        return error_response(exc.status_code, code, str(exc.detail), {})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("未处理异常", extra={"err": str(exc)})
        return error_response(500, "INTERNAL", f"内部错误: {type(exc).__name__}", {})
