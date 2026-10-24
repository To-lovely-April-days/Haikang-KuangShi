"""公共 fixture：临时数据目录、开发配置、AppContext、TestClient。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oresorter.config import AppConfig, ConfigService
from oresorter.runtime.context import AppContext

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "config"


def make_cfg(tmp_path: Path, **overrides) -> AppConfig:
    """基于 config/dev.yaml，但 data_dir 指向临时目录、Mock 相机不限速。"""
    from oresorter.config import load

    cfg = load(CONFIG_DIR / "dev.yaml")
    data = cfg.model_dump(mode="json")
    data["system"]["data_dir"] = str(tmp_path / "data")
    data["system"]["mcu_kind"] = "mock"
    for cam in data["cameras"]:
        cam["mock"]["speed"] = 0
        cam["mock"]["seed"] = 7
    for k, v in overrides.items():
        sec, _, key = k.partition(".")
        if key:
            data[sec][key] = v
        else:
            data[sec] = v
    return AppConfig.model_validate(data)


@pytest.fixture(autouse=True)
def _restore_logging():
    """每个测试后恢复根 logger 与常见库 logger 的级别/handler。

    ``oresorter.cli.main`` / ``oresorter.ml.train.main`` 会调 ``setup_logging`` 重置根 logger，
    不恢复会把 INFO 级别与绑定在旧 stderr 上的 handler 泄漏给后续测试文件。
    """
    import logging

    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    names = ("uvicorn.access", "PIL", "matplotlib", "httpx", "httpcore", "oresorter.runtime.jobs")
    saved_lib = {n: logging.getLogger(n).level for n in names}
    try:
        yield
    finally:
        root.setLevel(saved_level)
        for h in list(root.handlers):
            if h not in saved_handlers:
                root.removeHandler(h)
        for h in saved_handlers:
            if h not in root.handlers:
                root.addHandler(h)
        for n, lvl in saved_lib.items():
            logging.getLogger(n).setLevel(lvl)


@pytest.fixture
def tmp_data(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return make_cfg(tmp_path)


@pytest.fixture
def cfg_service(cfg: AppConfig) -> ConfigService:
    svc = ConfigService(cfg, cfg.system.data_dir / "config.json")
    svc.save_atomic()
    return svc


@pytest.fixture
def ctx(cfg_service: ConfigService):
    c = AppContext(cfg_service)
    c.start()
    try:
        yield c
    finally:
        c.stop()


@pytest.fixture
def client(cfg_service: ConfigService):
    from fastapi.testclient import TestClient

    from oresorter.api.app import create_app

    c = AppContext(cfg_service)
    app = create_app(c)
    with TestClient(app) as tc:
        tc.ctx = c  # type: ignore[attr-defined]
        yield tc


@pytest.fixture
def yaml_file(tmp_path: Path):
    """写一个临时 yaml 的工厂。"""

    def _write(name: str, data: dict) -> Path:
        p = tmp_path / name
        p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return p

    return _write
