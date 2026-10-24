"""日志配置：标准 ``logging``，带结构化字段（``extra``）。

用法::

    from oresorter.log import get_logger
    log = get_logger(__name__)
    log.info("块处理完成", extra={"seq": 12, "particles": 7, "ms": 31.2})

文本格式会把 extra 字段追加为 ``key=value``；JSON 格式整条记录为一行 JSON（journald/采集友好）。
禁止在业务代码中使用 ``print``。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# LogRecord 自带的属性集合，用于区分调用方传入的 extra 字段
_STD_ATTRS = frozenset(
    logging.LogRecord("x", logging.INFO, "x", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "taskName", "color_message"}  # color_message 来自 uvicorn
)


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS and not k.startswith("_")
    }


class KVFormatter(logging.Formatter):
    """文本格式：``时间 级别 logger: 消息 key=value ...``。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extra_fields(record)
        if extras:
            kv = " ".join(f"{k}={_fmt_value(v)}" for k, v in extras.items())
            return f"{base} {kv}"
        return base


class JsonFormatter(logging.Formatter):
    """JSON 单行格式。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    s = str(v)
    if " " in s:
        return json.dumps(s, ensure_ascii=False)
    return s


def setup_logging(
    level: str | int = "INFO",
    *,
    json_format: bool = False,
    file: str | Path | None = None,
    quiet_libs: bool = True,
) -> None:
    """配置根 logger。重复调用会先清掉已有 handler（便于测试）。

    Args:
        level: 日志级别名或数值。
        json_format: True 时输出 JSON 行。
        file: 可选日志文件路径（追加写）。
        quiet_libs: 把 uvicorn.access / PIL 等噪音库调到 WARNING。
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level if isinstance(level, int) else level.upper())

    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = KVFormatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if quiet_libs:
        for name in ("uvicorn.access", "PIL", "matplotlib", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger（薄封装，统一入口便于以后替换）。"""
    return logging.getLogger(name)
