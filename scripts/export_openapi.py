"""导出 OpenAPI 契约到 docs/openapi.json 与前端版 docs/06_openapi_hmi.json；``--check`` 比对是否最新（CI 用）。

前端版由完整版派生：去掉接口级与模型级 ``description``（含后端实现备注）、去掉实际不会出现的 ``422``
响应及其模型，改写 ``info``。字段级 ``description`` 保留。发给 HMI 前端的只有前端版。

用法：
    .venv/bin/python scripts/export_openapi.py            # 写两个文件
    .venv/bin/python scripts/export_openapi.py --check    # 任一不一致则退出码 1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oresorter.api.app import create_app
from oresorter.config import ConfigService, load
from oresorter.runtime.context import AppContext

HMI_INFO = {
    "title": "矿石分选机 HMI 接口",
    "description": "面向 HMI 前端的接口定义（v1）。字段含义、错误码与 WebSocket 消息格式见《HMI 接口调用手册》。",
}
_VALIDATION_SCHEMAS = ("HTTPValidationError", "ValidationError")


def build_spec() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load(ROOT / "config" / "dev.yaml")
        cfg.system.data_dir = Path(tmp) / "data"
        cfg.system.mcu_kind = "mock"
        ctx = AppContext(ConfigService(cfg, None))
        app = create_app(ctx, manage_lifecycle=False)
        return app.openapi()


def sanitize_for_hmi(spec: dict) -> dict:
    """完整契约 → 前端版：去掉后端备注与 422，保留路径 / 参数 / 模型 / 字段说明。"""
    out = copy.deepcopy(spec)
    out["info"] = {**out.get("info", {}), **HMI_INFO}
    for ops in out.get("paths", {}).values():
        for op in ops.values():
            op.pop("description", None)
            op.get("responses", {}).pop("422", None)
    schemas = out.get("components", {}).get("schemas", {})
    for name in _VALIDATION_SCHEMAS:
        schemas.pop(name, None)
    for schema in schemas.values():
        schema.pop("description", None)
    return out


def _dump(spec: dict) -> str:
    return json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "openapi.json")
    ap.add_argument("--hmi-out", type=Path, default=ROOT / "docs" / "06_openapi_hmi.json")
    ap.add_argument("--check", action="store_true", help="只比对不写入")
    args = ap.parse_args(argv)
    spec = build_spec()
    n_paths = len(spec.get("paths", {}))
    targets = [(args.out, _dump(spec)), (args.hmi_out, _dump(sanitize_for_hmi(spec)))]
    if args.check:
        stale = [
            p
            for p, text in targets
            if (p.read_text(encoding="utf-8") if p.exists() else "") != text
        ]
        if stale:
            names = ", ".join(str(p) for p in stale)
            print(f"{names} 已过期：请运行 scripts/export_openapi.py 重新导出", file=sys.stderr)
            return 1
        print(f"{args.out} 与 {args.hmi_out} 均与代码一致（{n_paths} 个路径）")
        return 0
    for path, text in targets:
        path.write_text(text, encoding="utf-8")
        print(f"已写入 {path}（{n_paths} 个路径）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
