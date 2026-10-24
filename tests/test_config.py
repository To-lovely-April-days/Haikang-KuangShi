"""配置：范围校验、跨字段规则、分层加载、持久化覆盖、原子写、limits。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from oresorter.config import (
    AppConfig,
    ConfigError,
    ConfigService,
    build_limits,
    camera_limits,
    load,
)
from tests.conftest import CONFIG_DIR, make_cfg


def test_defaults_match_docs():
    cfg = AppConfig()
    assert cfg.valve.model_dump() == {"open_time": 255, "hold": 255, "reblow": 20, "delay": 376}
    assert cfg.clean.model_dump() == {"time": 20, "period": 200, "count": 5, "enabled": False}
    assert cfg.feed.speeds == [50] * 9
    assert cfg.system.class_names == list("ABCDEFG")
    assert cfg.system.max_models == 7
    assert cfg.ml.image_size == 160 and cfg.ml.backbone == "resnet18"
    assert cfg.cameras[0].nozzles == 64 and cfg.total_nozzles == 64


@pytest.mark.parametrize(
    "section,patch",
    [
        ("valve", {"open_time": 256}),
        ("valve", {"open_time": -1}),
        ("valve", {"reblow": 51}),
        ("valve", {"delay": 801}),
        ("clean", {"time": 201}),
        ("clean", {"count": 6}),
        ("feed", {"speeds": [50] * 8 + [101]}),
        ("feed", {"speeds": [50] * 8}),  # 长度不等于 feeder_count
        ("blow", {"min_confidence": 256}),
        ("blow", {"size_min": 200, "size_max": 100}),
        ("blow", {"classes": ["Z"]}),
        ("capture", {"bg_threshold": 300}),
        ("capture", {"min_area_px": 10, "max_area_px": 5}),
        ("capture", {"bg_mode": "magic"}),
        ("valve_test", {"hole": 65}),
        ("ml", {"image_size": 100}),
        ("valve", {"unknown_field": 1}),
    ],
)
def test_update_rejects_out_of_range(tmp_path: Path, section, patch):
    svc = ConfigService(make_cfg(tmp_path), tmp_path / "config.json")
    before = svc.config.model_dump()
    with pytest.raises(ConfigError) as ei:
        svc.update(section, patch)
    assert ei.value.code in {"PARAM_OUT_OF_RANGE", "VALIDATION"}
    assert svc.config.model_dump() == before, "失败不得部分应用"


def test_update_applies_and_persists(tmp_path: Path):
    path = tmp_path / "config.json"
    svc = ConfigService(make_cfg(tmp_path), path)
    seen = []
    svc.subscribe(lambda sec, val: seen.append(sec))
    applied, warnings = svc.update("valve", {"open_time": 100})
    assert applied.open_time == 100 and applied.hold == 255
    assert warnings == []
    assert seen == ["valve"]
    import json

    saved = json.loads(path.read_text())
    assert saved["valve"]["open_time"] == 100


def test_update_camera_editable_only(tmp_path: Path):
    svc = ConfigService(make_cfg(tmp_path), tmp_path / "config.json")
    cam, _ = svc.update_camera(1, {"start_px": 100, "end_px": 3000, "exposure_us": 80})
    assert (cam.start_px, cam.end_px, cam.exposure_us) == (100, 3000, 80)
    with pytest.raises(ConfigError):
        svc.update_camera(1, {"start_px": 3000, "end_px": 100})
    with pytest.raises(ConfigError) as ei:
        svc.update_camera(1, {"kind": "hik"})
    assert ei.value.code == "VALIDATION"
    with pytest.raises(ConfigError) as ei:
        svc.update_camera(9, {"start_px": 1})
    assert ei.value.code == "NOT_FOUND"


def test_camera_cross_rules():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"cameras": [{"id": 1, "start_px": 0, "end_px": 32}]}
        )  # 少于喷头数
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"cameras": [{"id": 1, "offset_x": 100, "width": 4096}]})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"cameras": [{"id": 1, "kind": "hik"}]})  # hik 缺 ip/serial
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"cameras": [{"id": 1}, {"id": 1}]})


def test_layered_yaml_loading():
    dev = load(CONFIG_DIR / "dev.yaml")
    assert dev.system.mcu_kind == "mock"
    assert dev.system.data_dir.is_absolute()
    assert dev.valve.open_time == 255  # 来自 default.yaml
    jet = load(CONFIG_DIR / "jetson.yaml")
    assert jet.cameras[0].kind == "hik" and jet.cameras[0].ip == "192.168.10.10"
    assert jet.system.mcu_kind == "tcp"


def test_yaml_base_cycle(yaml_file):
    a = yaml_file("a.yaml", {"base": "b.yaml"})
    yaml_file("b.yaml", {"base": "a.yaml"})
    with pytest.raises(ConfigError):
        load(a)


def test_from_yaml_overlays_persisted(tmp_path: Path, yaml_file):
    data_dir = tmp_path / "data"
    y = yaml_file(
        "x.yaml",
        {
            "system": {"data_dir": str(data_dir), "mcu_kind": "mock"},
            "cameras": [{"id": 1, "kind": "mock"}],
        },
    )
    svc = ConfigService.from_yaml(y)
    svc.update("valve", {"delay": 123})
    svc.update_camera(1, {"start_px": 50})
    # 重新启动：持久化的可编辑项覆盖 yaml
    svc2 = ConfigService.from_yaml(y)
    assert svc2.config.valve.delay == 123
    assert svc2.config.cameras[0].start_px == 50
    # 部署项仍以 yaml 为准
    assert svc2.config.system.mcu_kind == "mock"


def test_from_yaml_ignores_corrupt_persisted(tmp_path: Path, yaml_file):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text('{"valve": {"open_time": 9999}}')
    y = yaml_file("x.yaml", {"system": {"data_dir": str(data_dir), "mcu_kind": "mock"}})
    svc = ConfigService.from_yaml(y)
    assert svc.config.valve.open_time == 255


def test_limits_shape(tmp_path: Path):
    cfg = make_cfg(tmp_path)
    lim = build_limits(cfg)
    assert lim["valve"] == {
        "open_time": [0, 255],
        "hold": [0, 255],
        "reblow": [0, 50],
        "delay": [0, 800],
    }
    assert lim["clean"] == {"time": [0, 200], "period": [0, 200], "count": [0, 5]}
    assert lim["feed"]["speed"] == [0, 100]
    assert lim["valve_test"]["hole"] == [1, 64]
    cl = camera_limits()
    assert (
        cl["exposure_us"] == [3, 10000]
        and cl["px"] == [0, 4096]
        and cl["line_rate_hz"] == [100, 9000]
    )


def test_error_message_is_chinese_and_structured(tmp_path: Path):
    svc = ConfigService(make_cfg(tmp_path), tmp_path / "config.json")
    with pytest.raises(ConfigError) as ei:
        svc.update("valve", {"open_time": 300})
    err = ei.value.to_dict()
    assert err["code"] == "PARAM_OUT_OF_RANGE"
    assert "超出范围" in err["message"]
    assert err["detail"]["errors"][0]["field"] == "valve.open_time"
