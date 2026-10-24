"""写操作幂等（docs/03 §1、docs/01 §7.1）：``request_id`` 60 s 内重复提交直接返回首次结果。

实现方式：**纯 ASGI 中间件**（:class:`IdempotencyMiddleware`），只拦截 ``/api/v1`` 下的写方法
（POST / PUT / PATCH / DELETE）：

1. 读完整请求体，取 ``X-Request-Id`` 头或 JSON 体顶层 ``request_id`` 字段作为幂等键
   ``(method, path, request_id)``；没有幂等键的请求原样放行（请求体通过重放 ``receive`` 交给下游）；
2. 命中缓存 → 直接回放首次的 ``(status, headers, body)``，并加响应头 ``X-Idempotent-Replay: 1``；
3. 未命中 → 执行下游应用并把响应整体捕获后缓存（错误响应同样缓存：重试拿到的是首次的结论，
   不会重复执行一次已经失败/成功的操作）；
4. 并发同键 → 后到者等待先到者完成（``asyncio.Event``），然后返回同一结果；等待有上限，
   超时后重新检查缓存（不会无限阻塞）；
5. 过期条目在每次访问时惰性清理，并有总条数上限。

选中间件而不是依赖项的原因：FastAPI 的 dependency 拿不到路由返回的响应，而幂等需要缓存完整响应。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from oresorter.log import get_logger

log = get_logger(__name__)

T = TypeVar("T")

DEFAULT_TTL_S = 60.0
"""幂等键有效期（docs/03 §1）。"""
DEFAULT_MAX_ENTRIES = 2000
INFLIGHT_WAIT_S = 30.0
"""等待同键请求完成的单次上限；超时后回到循环重新检查缓存。"""
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
REQUEST_ID_HEADER = b"x-request-id"
REQUEST_ID_FIELD = "request_id"
REPLAY_HEADER = (b"x-idempotent-replay", b"1")
MAX_REQUEST_ID_LEN = 128


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class IdempotencyCache(Generic[T]):
    """``(method, path, request_id)`` → 首次结果 的 TTL 缓存，附带同键并发合并。

    仅在一个事件循环内使用（ASGI 应用），因此不需要锁；``await`` 之间的字典操作是原子的。

    Args:
        ttl_s: 条目有效期（秒）。
        max_entries: 条数上限，超出时淘汰最早过期的条目。
        clock: 单调时钟（测试可注入）。
    """

    def __init__(
        self,
        ttl_s: float = DEFAULT_TTL_S,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ):
        if ttl_s <= 0:
            raise ValueError(f"ttl_s 必须 > 0，收到 {ttl_s}")
        if max_entries < 1:
            raise ValueError(f"max_entries 必须 ≥ 1，收到 {max_entries}")
        self.ttl_s = float(ttl_s)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._entries: dict[Hashable, _Entry[T]] = {}
        self._inflight: dict[Hashable, asyncio.Event] = {}
        self.hits = 0
        self.misses = 0

    # ---- 基本操作 ----
    def __len__(self) -> int:
        return len(self._entries)

    def purge_expired(self) -> int:
        """删除过期条目，返回删除数。"""
        now = self._clock()
        dead = [k for k, e in self._entries.items() if e.expires_at <= now]
        for k in dead:
            del self._entries[k]
        return len(dead)

    def get(self, key: Hashable) -> T | None:
        """命中返回首次结果，否则 None（过期视为未命中）。"""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            del self._entries[key]
            return None
        return entry.value

    def put(self, key: Hashable, value: T) -> None:
        """写入（覆盖）一个条目；超出上限时淘汰最早过期的。"""
        self.purge_expired()
        if key not in self._entries and len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].expires_at)
            del self._entries[oldest]
        self._entries[key] = _Entry(value, self._clock() + self.ttl_s)

    def in_flight(self, key: Hashable) -> bool:
        return key in self._inflight

    # ---- 合并执行 ----
    async def run(self, key: Hashable, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """命中则返回缓存；否则执行 ``fn`` 并缓存。返回 ``(结果, 是否回放)``。

        同键并发：后到者等待先到者完成后返回同一结果；若先到者抛异常（没有产生结果），
        等待者醒来后自己执行一次。
        """
        while True:
            cached = self.get(key)
            if cached is not None:
                self.hits += 1
                return cached, True
            event = self._inflight.get(key)
            if event is None:
                break
            try:
                await asyncio.wait_for(event.wait(), INFLIGHT_WAIT_S)
            except asyncio.TimeoutError:
                log.warning("等待同键请求超时，重新检查缓存", extra={"key": repr(key)})
        event = asyncio.Event()
        self._inflight[key] = event
        self.misses += 1
        try:
            value = await fn()
            self.put(key, value)
            return value, False
        finally:
            self._inflight.pop(key, None)
            event.set()


# --------------------------------------------------------------------------------------
# ASGI 中间件
# --------------------------------------------------------------------------------------


@dataclass
class CachedResponse:
    """被缓存的完整 HTTP 响应。"""

    status: int
    headers: list[tuple[bytes, bytes]]
    body: bytes = b""
    chunks: list[bytes] = field(default_factory=list, repr=False)


def extract_request_id(headers: Iterable[tuple[bytes, bytes]], body: bytes) -> str | None:
    """取幂等键：``X-Request-Id`` 头优先，其次 JSON 体顶层 ``request_id``；空串 / 非字符串 / 过长视为无。"""
    for name, value in headers:
        if name.lower() == REQUEST_ID_HEADER:
            rid = value.decode("latin-1").strip()
            if rid:
                return rid[:MAX_REQUEST_ID_LEN]
    if not body:
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    if isinstance(doc, dict):
        rid = doc.get(REQUEST_ID_FIELD)
        if isinstance(rid, str) and rid.strip():
            return rid.strip()[:MAX_REQUEST_ID_LEN]
    return None


class IdempotencyMiddleware:
    """见模块说明。``app.add_middleware(IdempotencyMiddleware, cache=..., prefix="/api/v1")``。"""

    def __init__(self, app: Any, cache: IdempotencyCache[CachedResponse], prefix: str = ""):
        self.app = app
        self.cache = cache
        self.prefix = prefix

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") not in WRITE_METHODS
            or not str(scope.get("path", "")).startswith(self.prefix)
        ):
            await self.app(scope, receive, send)
            return

        body, disconnected = await _read_body(receive)
        if disconnected:
            return
        replay = _make_replay_receive(body, receive)
        request_id = extract_request_id(scope.get("headers", ()), body)
        if request_id is None:
            await self.app(scope, replay, send)
            return

        key = (scope["method"], scope["path"], request_id)

        async def execute() -> CachedResponse:
            captured = CachedResponse(status=500, headers=[])
            started = False

            async def capture_send(message: dict[str, Any]) -> None:
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                    captured.status = int(message["status"])
                    captured.headers = [(bytes(k), bytes(v)) for k, v in message.get("headers", [])]
                elif message["type"] == "http.response.body":
                    captured.chunks.append(bytes(message.get("body", b"")))

            await self.app(scope, replay, capture_send)
            if not started:
                raise RuntimeError("下游应用未产生响应")
            captured.body = b"".join(captured.chunks)
            captured.chunks = []
            return captured

        response, replayed = await self.cache.run(key, execute)
        headers = [(k, v) for k, v in response.headers if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(response.body)).encode()))
        if replayed:
            headers.append(REPLAY_HEADER)
            log.info(
                "幂等回放",
                extra={"method": key[0], "path": key[1], "request_id": request_id},
            )
        await send({"type": "http.response.start", "status": response.status, "headers": headers})
        await send({"type": "http.response.body", "body": response.body, "more_body": False})


async def _read_body(receive: Any) -> tuple[bytes, bool]:
    """读完整请求体；客户端中途断开返回 ``(b"", True)``。"""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"", True
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks), False


def _make_replay_receive(body: bytes, receive: Any) -> Any:
    """把已缓冲的请求体重放给下游：首次返回整段 body，之后转交原 ``receive``（等待断开消息）。"""
    delivered = False

    async def replay() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay
