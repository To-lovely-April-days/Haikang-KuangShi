"""FastAPI 应用工厂：路由挂载、静态文件、错误处理、生命周期。"""

from __future__ import annotations

import asyncio
import importlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from oresorter import __version__
from oresorter.api.errors import install_error_handlers
from oresorter.log import get_logger
from oresorter.runtime.context import AppContext

log = get_logger(__name__)

API_PREFIX = "/api/v1"

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
    install_error_handlers(app)

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

    app.mount("/static", StaticFiles(directory=str(ctx.data_dir), check_dir=True), name="static")
    return app
