"""导出 OpenAPI 契约到 docs/openapi.json；``--check`` 模式比对仓库中的文件是否最新（CI 用）。

用法：
    .venv/bin/python scripts/export_openapi.py            # 写 docs/openapi.json
    .venv/bin/python scripts/export_openapi.py --check    # 不一致则退出码 1
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oresorter.api.app import create_app
from oresorter.config import ConfigService, load
from oresorter.runtime.context import AppContext


def build_spec() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load(ROOT / "config" / "dev.yaml")
        cfg.system.data_dir = Path(tmp) / "data"
        cfg.system.mcu_kind = "mock"
        ctx = AppContext(ConfigService(cfg, None))
        app = create_app(ctx, manage_lifecycle=False)
        return app.openapi()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "openapi.json")
    ap.add_argument("--check", action="store_true", help="只比对不写入")
    args = ap.parse_args(argv)
    spec = build_spec()
    text = json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} 已过期：请运行 scripts/export_openapi.py 重新导出", file=sys.stderr)
            return 1
        print(f"{args.out} 与代码一致（{len(spec.get('paths', {}))} 个路径）")
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"已写入 {args.out}（{len(spec.get('paths', {}))} 个路径）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
