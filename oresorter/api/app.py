"""FastAPI 应用工厂：路由挂载、静态文件、错误处理、幂等中间件、生命周期。

- ``create_app(ctx, manage_lifecycle=True)``：lifespan 内调用 ``ctx.start()/stop()``（周期广播、设备巡检
  等后台工作由 ``AppContext`` 负责，这里不重复）；
- ``app.state.ctx`` 组合根、``app.state.idempotency`` 幂等缓存（:class:`IdempotencyMiddleware` 只拦截
  ``API_PREFIX`` 下的写方法）；
- 按 ``ROUTE_MODULES`` 自动 include 各 ``routes_*.router`` 与 ``ws.router``，模块级 ``register(app, ctx)``
  存在则调用；``/static`` 挂到数据目录，但只暴露 docs/03 §12 承诺的只读路径（:class:`WhitelistStaticFiles`）。
"""

from __future__ import annotations

import asyncio
import importlib
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope

from oresorter import __version__
from oresorter.api.errors import install_error_handlers
from oresorter.api.idempotency import IdempotencyCache, IdempotencyMiddleware
from oresorter.log import get_logger
from oresorter.runtime.context import AppContext

log = get_logger(__name__)

API_PREFIX = "/api/v1"
IDEMPOTENCY_TTL_S = 60.0
"""写操作 ``request_id`` 幂等缓存有效期（docs/03 §1）。"""

STATIC_ALLOWED = re.compile(
    r"^(?:models/\d+/(?:thumbs|samples)/.+|snapshots/[^/]+|previews/[^/]+)$"
)
"""``/static`` 白名单（docs/03 §12）：``models/{id}/thumbs|samples/**``、``snapshots/*``、``previews/*``。

数据目录里的其它内容（``config.json`` / ``config.backup.json`` / ``state.json``、``models/{id}/best.pt`` /
``model.plan`` / ``train/``、``background/*.npz``、``logs/``）不是契约的一部分，不对外暴露。
"""


class WhitelistStaticFiles(StaticFiles):
    """只提供白名单路径的静态文件服务；其余路径一律 404 ``NOT_FOUND``（与不存在的文件无法区分）。"""

    def __init__(self, *args: Any, allowed: re.Pattern[str] = STATIC_ALLOWED, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.allowed = allowed

    async def get_response(self, path: str, scope: Scope) -> Response:
        """``path`` 是 Starlette 规范化后的相对路径（已去掉 ``..``）。"""
        if not self.allowed.match(path.replace(os.sep, "/")):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


# 按 docs/02 §2 的路由模块；不存在的模块（尚未实现）自动跳过
ROUTE_MODULES = (
    "oresorter.api.routes_system",
    "oresorter.api.routes_run",
    "oresorter.api.routes_params",
    "oresorter.api.routes_camera",
    "oresorter.api.routes_models",
    "oresorter.api.routes_samples",
    "oresorter.api.ws",
)


def create_app(ctx: AppContext, *, manage_lifecycle: bool = True) -> FastAPI:
    """创建应用。``manage_lifecycle=True`` 时在启动/关闭事件里调用 ``ctx.start()/stop()``。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manage_lifecycle:
            await asyncio.to_thread(ctx.start)
        hooks = getattr(app.state, "startup_hooks", [])
        for h in hooks:
            r = h(app)
            if asyncio.iscoroutine(r):
                await r
        try:
            yield
        finally:
            for h in getattr(app.state, "shutdown_hooks", []):
                try:
                    r = h(app)
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    log.exception("关闭钩子异常")
            if manage_lifecycle:
                await asyncio.to_thread(ctx.stop)

    app = FastAPI(
        title="oresorter — 矿石智能分选机算力盒后端",
        version=__version__,
        description="HMI ↔ 算力盒 REST + WebSocket 契约，见 docs/03_前端接口清单.md",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.state.ctx = ctx
    app.state.startup_hooks = []
    app.state.shutdown_hooks = []
    app.state.idempotency = IdempotencyCache(ttl_s=IDEMPOTENCY_TTL_S)
    if ctx.cfg.system.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(ctx.cfg.system.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Idempotent-Replay"],
        )
    install_error_handlers(app)
    app.add_middleware(IdempotencyMiddleware, cache=app.state.idempotency, prefix=API_PREFIX)

    api = APIRouter(prefix=API_PREFIX)
    for mod_name in ROUTE_MODULES:
        try:
            mod: Any = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name == mod_name:
                continue
            raise
        router = getattr(mod, "router", None)
        if router is not None:
            api.include_router(router)
        register = getattr(mod, "register", None)
        if callable(register):
            register(app, ctx)
    app.include_router(api)

    app.mount(
        "/static",
        WhitelistStaticFiles(directory=str(ctx.data_dir), check_dir=True),
        name="static",
    )
    return app
