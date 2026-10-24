"""命令行入口：``python -m oresorter.cli <子命令>``。

子命令：
- ``serve``        启动服务（API + 运行时）
- ``eval``         对样本目录做交叉验证，输出混淆矩阵（P1，``oresorter.ml.evaluate``）
- ``replay``       回放大图目录走一遍 分割→分类（P1，``oresorter.vision.replay``）
- ``build-engine`` 由 model.onnx 生成 TensorRT 引擎（P3，``oresorter.ml.export``）
- ``learn-bg``     用空皮带帧学习逐列背景并保存（P1，``oresorter.vision.background``）
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from oresorter import __version__
from oresorter.log import get_logger, setup_logging

log = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "dev.yaml"


def _lazy(module: str, func: str) -> Callable[[argparse.Namespace], int]:
    """延迟导入子命令实现；模块尚未实现时给出明确提示而不是堆栈。"""

    def runner(args: argparse.Namespace) -> int:
        try:
            mod = importlib.import_module(module)
            fn = getattr(mod, func)
        except (ModuleNotFoundError, AttributeError) as exc:
            log.error("子命令尚未实现", extra={"module": module, "func": func, "err": str(exc)})
            return 2
        return int(fn(args) or 0)

    return runner


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from oresorter.api.app import create_app
    from oresorter.runtime.context import AppContext

    ctx = AppContext.from_yaml(args.config)
    cfg = ctx.cfg
    setup_logging(cfg.system.log_level, json_format=cfg.system.log_json)
    host = args.host or cfg.system.api_host
    port = args.port or cfg.system.api_port
    log.info(
        "启动服务",
        extra={
            "version": __version__,
            "config": str(args.config),
            "host": host,
            "port": port,
            "data_dir": str(ctx.data_dir),
        },
    )
    app = create_app(ctx)

    # systemd Type=notify：启动完成发 READY，之后周期喂狗（不在 systemd 下时全部为空操作）
    from oresorter.sdnotify import SdNotify, WatchdogThread, watchdog_interval_s

    sd = SdNotify()
    wd_interval = watchdog_interval_s()
    watchdog = (
        WatchdogThread(sd, wd_interval, healthy=lambda: _healthy(ctx))
        if (sd.enabled and wd_interval)
        else None
    )

    def _on_startup(_app) -> None:
        sd.ready()
        sd.status("running")
        if watchdog is not None:
            watchdog.start()

    def _on_shutdown(_app) -> None:
        sd.stopping()
        if watchdog is not None:
            watchdog.stop()
        sd.close()

    app.state.startup_hooks.append(_on_startup)
    app.state.shutdown_hooks.append(_on_shutdown)
    uvicorn.run(
        app, host=host, port=port, log_config=None, access_log=False, timeout_graceful_shutdown=5
    )
    return 0


def _healthy(ctx) -> bool:
    """喂狗前的健康检查：流水线对象若存在且声明不健康则不喂狗。设备掉线不算（有重连）。"""
    pipeline = getattr(ctx, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "healthy"):
        try:
            return bool(pipeline.healthy())
        except Exception:
            return False
    return True


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oresorter", description="矿石智能分选机算力盒后端")
    p.add_argument("--version", action="version", version=f"oresorter {__version__}")
    p.add_argument("--log-level", default=None, help="覆盖日志级别（DEBUG/INFO/...）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="启动 API 服务与运行时")
    s.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="yaml 配置（默认 config/dev.yaml）"
    )
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    e = sub.add_parser("eval", help="对样本目录做交叉验证（每个子目录一类）")
    e.add_argument("--folder", type=Path, required=True, help="样本根目录，子目录名即类别")
    e.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    e.add_argument("--image-size", type=int, default=None, help="覆盖 ml.image_size")
    e.add_argument("--backbone", default=None, help="覆盖 ml.backbone")
    e.add_argument("--folds", type=int, default=5, help="折数（样本 < 50 自动留一法）")
    e.add_argument("--epochs", type=int, default=None, help="覆盖 ml.max_epochs（快速试跑用）")
    e.add_argument(
        "--out", type=Path, default=None, help="eval_report.json 输出路径（默认写到 --folder 下）"
    )
    e.add_argument("--bg-color", default="0,0,255", help="无 alpha 图片自动抠 mask 用的背景 RGB")
    e.set_defaults(func=_lazy("oresorter.ml.evaluate", "run"))

    r = sub.add_parser("replay", help="回放大图目录：分割 → 分类，输出标注图与统计")
    r.add_argument("--images", type=Path, required=True, help="大图目录")
    r.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    r.add_argument(
        "--model-dir", type=Path, default=None, help="已训练模型目录；省略则只分割不分类"
    )
    r.add_argument("--chunk-lines", type=int, default=None, help="按块回放（省略=整图）")
    r.add_argument("--out", type=Path, default=None, help="输出目录（标注图 + replay_report.json）")
    r.add_argument(
        "--learn-bg", type=int, default=0, help="先用前 N 张图学习背景（0 = 用 color 模式）"
    )
    r.set_defaults(func=_lazy("oresorter.vision.replay", "run"))

    b = sub.add_parser("build-engine", help="ONNX → TensorRT 引擎（仅 Jetson）")
    b.add_argument("--model-dir", type=Path, required=True)
    b.add_argument(
        "--no-fp16", dest="fp16", action="store_false", default=True, help="关闭 FP16（默认开启）"
    )
    b.add_argument("--timeout", type=int, default=600, help="trtexec 超时秒数")
    b.add_argument(
        "--trtexec",
        default=None,
        help="显式 trtexec 路径（默认 $TRTEXEC / /usr/src/tensorrt/bin / PATH）",
    )
    b.add_argument("--max-batch", type=int, default=128)
    b.add_argument("--opt-batch", type=int, default=None, help="默认 min(32, max-batch)")
    b.set_defaults(func=_lazy("oresorter.ml.export", "build_engine_cli"))

    g = sub.add_parser("learn-bg", help="用空皮带帧学习逐列背景并保存")
    g.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    g.add_argument("--camera", type=int, default=1)
    g.add_argument("--frames", type=int, default=None, help="覆盖 capture.bg_learn_frames")
    g.add_argument("--images", type=Path, default=None, help="从目录学习而不是相机")
    g.set_defaults(func=_lazy("oresorter.vision.background", "learn_bg_cli"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level or "INFO")
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log.info("已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
