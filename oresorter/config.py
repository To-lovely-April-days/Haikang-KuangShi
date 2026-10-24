"""配置模型（pydantic v2）、范围校验、分层加载与原子持久化。

分组与取值范围以 ``docs/01_技术选型与架构.md`` §9 为准。越界一律拒绝并给出原因，不静默截断。

加载规则：
1. ``config/*.yaml`` 是部署期配置（可用 ``base:`` 继承另一份 yaml，深合并）；
2. ``{data_dir}/config.json`` 是运行期持久化（HMI 修改过的参数）。启动时若存在，
   其中 **可编辑分组** 与 **相机可编辑字段** 覆盖 yaml 的值；网络、相机类型等部署项始终以 yaml 为准；
3. 每次 ``ConfigService.update`` 成功后原子写回 ``config.json``。
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from oresorter.fsutil import atomic_write_json, read_json

MAX_SENSOR_PX = 4096
DEFAULT_CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G"]


class ConfigError(Exception):
    """配置校验/持久化错误。``code`` 与 docs/03 的错误码一致。"""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class StrictModel(BaseModel):
    """禁止未知字段；赋值时也校验。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------------------
# 分组模型（docs/01 §9）
# --------------------------------------------------------------------------------------


class ValveParams(StrictModel):
    """喷阀参数 → 下位机。单位待定（docs/04 TBD）。"""

    open_time: int = Field(255, ge=0, le=255, description="开阀时间")
    hold: int = Field(255, ge=0, le=255, description="阀门维持")
    reblow: int = Field(20, ge=0, le=50, description="阀门重吹")
    delay: int = Field(376, ge=0, le=800, description="相机到喷阀的延时")


class ValveTestParams(StrictModel):
    """喷阀检测参数。``hole`` 上限 = 相机数 × 每相机喷头数，在 AppConfig 层校验。"""

    hole: int = Field(0, ge=0, description="喷阀检测用孔位（全局喷头编号，0 表示未选）")


class CleanParams(StrictModel):
    time: int = Field(20, ge=0, le=200, description="清灰时间")
    period: int = Field(200, ge=0, le=200, description="清灰周期")
    count: int = Field(5, ge=0, le=5, description="清灰次数")
    enabled: bool = Field(False, description="清灰使能")


FEED_SPEED_MIN = 0
FEED_SPEED_MAX = 100


class FeedParams(StrictModel):
    """给料速度 ×N（默认 9 路，范围 0~100 待确认）。"""

    speeds: list[int] = Field(default_factory=lambda: [50] * 9, description="各路给料速度")

    @model_validator(mode="after")
    def _check_range(self) -> FeedParams:
        for i, v in enumerate(self.speeds):
            if not (FEED_SPEED_MIN <= v <= FEED_SPEED_MAX):
                raise ValueError(f"speeds[{i}]={v} 超出范围 {FEED_SPEED_MIN}~{FEED_SPEED_MAX}")
        return self


class OutputsParams(StrictModel):
    conveyor: bool = Field(False, description="输送带")
    valve: bool = Field(False, description="喷阀气源/驱动")
    feeder: bool = Field(False, description="给料机")


class BlowParams(StrictModel):
    """吹气决策（后端执行）。"""

    classes: list[str] = Field(default_factory=list, description="需要吹气的类别名")
    min_confidence: int = Field(128, ge=0, le=255, description="置信度阈值（映射 0~1）")
    size_min: int = Field(0, ge=0, le=255, description="粒度下限（映射到面积）")
    size_max: int = Field(255, ge=0, le=255, description="粒度上限（映射到面积）")

    @model_validator(mode="after")
    def _check_size(self) -> BlowParams:
        if self.size_min > self.size_max:
            raise ValueError(f"size_min({self.size_min}) 不能大于 size_max({self.size_max})")
        return self


class CaptureSettings(StrictModel):
    """采集设置 = 分割参数（docs/01 §5.3）。"""

    bg_mode: Literal["column", "color"] = Field("column", description="背景模型：逐列 / 固定颜色")
    bg_color: list[int] = Field(
        default_factory=lambda: [0, 0, 255], description="color 模式背景 RGB"
    )
    bg_threshold: float = Field(40.0, ge=0, le=255, description="与背景的颜色距离阈值（Lab）")
    erode_px: int = Field(1, ge=0, le=50, description="腐蚀像素")
    dilate_px: int = Field(2, ge=0, le=50, description="膨胀像素")
    min_area_px: int = Field(400, ge=0, le=16_000_000, description="连通域最小面积")
    max_area_px: int = Field(2_000_000, ge=1, le=16_000_000, description="连通域最大面积")
    margin_px: int = Field(8, ge=0, le=200, description="裁切外扩")
    split_touching: bool = Field(False, description="分水岭拆粘连")
    bg_learn_frames: int = Field(20, ge=1, le=1000, description="学习背景用的空皮带帧数")

    @model_validator(mode="after")
    def _check(self) -> CaptureSettings:
        if len(self.bg_color) != 3 or any(not (0 <= c <= 255) for c in self.bg_color):
            raise ValueError("bg_color 必须是 3 个 0~255 的整数")
        if self.min_area_px > self.max_area_px:
            raise ValueError(
                f"min_area_px({self.min_area_px}) 不能大于 max_area_px({self.max_area_px})"
            )
        return self


class MlConfig(StrictModel):
    """训练/推理配置。超参属于固定配方，不暴露给操作工，但允许在 yaml 中覆盖以便测试。"""

    backbone: Literal["resnet18", "mobilenet_v3_large"] = "resnet18"
    image_size: Literal[128, 160, 224] = 160
    min_samples_per_class: int = Field(20, ge=1)
    max_train_minutes: int = Field(10, ge=1)
    max_epochs: int = Field(30, ge=1, le=1000)
    head_epochs: int = Field(3, ge=0, le=100, description="先只训分类头的 epoch 数")
    batch_size: int = Field(32, ge=1, le=512)
    patience: int = Field(5, ge=1, le=100, description="验证 macro-F1 连续多少轮不升则早停")
    val_ratio: float = Field(0.2, gt=0, lt=1)
    num_workers: int = Field(0, ge=0, le=16)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    pretrained: bool = Field(
        True, description="加载离线 ImageNet 权重；权重文件缺失时降级为随机初始化并告警"
    )
    engine_build_timeout_s: int = Field(600, ge=30, le=3600)


class PipelineConfig(StrictModel):
    """运行时流水线参数（docs/01 §5.2 / §5.4 / §7.3）。"""

    overlap_lines: int = Field(256, ge=0, le=2048, description="跨块处理时携带的上一块尾部行数")
    queue_size: int = Field(8, ge=1, le=256, description="采集→处理 环形队列容量（块）")
    nozzle_expand: int = Field(0, ge=0, le=8, description="喷头区间向两侧各扩展个数")
    infer_batch_max: int = Field(128, ge=1, le=512)
    frame_timeout_s: float = Field(2.0, gt=0, le=60, description="多久没收到块算相机超时")
    latency_warn_ms: float = Field(80.0, gt=0, description="块处理耗时超过则告警")


class MockCameraConfig(StrictModel):
    """Mock 相机：目录回放或合成图。"""

    source_dir: str | None = Field(None, description="回放大图目录（png/jpg/bmp）；空则合成")
    speed: float = Field(0.0, ge=0, le=1000, description="回放速度倍率，0 = 不限速（尽快）")
    seed: int = Field(0, ge=0)
    particles_per_chunk: float = Field(6.0, ge=0, description="合成模式下每块平均颗粒数")
    particle_px: list[int] = Field(
        default_factory=lambda: [40, 160], description="合成颗粒直径范围"
    )
    bg_color: list[int] = Field(
        default_factory=lambda: [0, 0, 255],
        description="合成背景 RGB（应与 capture.bg_color 一致）",
    )
    class_colors: dict[str, list[int]] | None = Field(
        None, description="合成模式下各类别颜色 RGB；None 用内置调色板"
    )
    queue_size: int = Field(64, ge=1, le=1024, description="SDK 侧缓冲块数，满则丢块")


class CameraConfig(StrictModel):
    """单台相机。前 4 个字段是部署项（yaml），其余可由 HMI 修改并持久化。"""

    id: int = Field(..., ge=1)
    name: str = "相机1"
    kind: Literal["mock", "hik"] = "mock"
    ip: str | None = None
    serial: str | None = None
    model: str = "MV-CL043-A1GC"
    nozzles: int = Field(64, ge=1, le=1024)
    mock: MockCameraConfig = Field(default_factory=MockCameraConfig)

    # --- HMI 可编辑（CAMERA_EDITABLE）---
    awb_mode: Literal["off", "once", "continuous"] = "once"
    exposure_us: float = Field(50.0, ge=3, le=10000)
    width: int = Field(MAX_SENSOR_PX, ge=64, le=MAX_SENSOR_PX)
    offset_x: int = Field(0, ge=0, le=MAX_SENSOR_PX - 64)
    height: int = Field(MAX_SENSOR_PX, ge=64, le=MAX_SENSOR_PX, description="采样帧高")
    runtime_chunk_lines: int = Field(512, ge=128, le=1024)
    start_px: int = Field(0, ge=0, le=MAX_SENSOR_PX)
    end_px: int = Field(MAX_SENSOR_PX, ge=0, le=MAX_SENSOR_PX)
    line_rate_hz: int = Field(9000, ge=100, le=9000)
    trigger: Literal["internal", "encoder"] = "internal"

    @model_validator(mode="after")
    def _check(self) -> CameraConfig:
        if self.offset_x + self.width > MAX_SENSOR_PX:
            raise ValueError(
                f"offset_x({self.offset_x}) + width({self.width}) 超过传感器宽度 {MAX_SENSOR_PX}"
            )
        if self.start_px >= self.end_px:
            raise ValueError(f"start_px({self.start_px}) 必须小于 end_px({self.end_px})")
        if self.end_px - self.start_px < self.nozzles:
            raise ValueError(
                f"有效像素 {self.end_px - self.start_px} 少于喷头数 {self.nozzles}，无法映射"
            )
        if self.kind == "hik" and not (self.ip or self.serial):
            raise ValueError("hik 相机必须提供 ip 或 serial")
        return self


class SystemConfig(StrictModel):
    max_models: int = Field(7, ge=1, le=64)
    class_names: list[str] = Field(default_factory=lambda: list(DEFAULT_CLASS_NAMES))
    feeder_count: int = Field(9, ge=1, le=32, description="给料机路数（高级设置页数值框个数）")
    data_dir: Path = Path("/data/oresorter")
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1, le=65535)
    mcu_kind: Literal["tcp", "mock"] = "tcp"
    mcu_host: str = "192.168.1.30"
    mcu_port: int = Field(5000, ge=1, le=65535)
    auto_run_on_boot: bool = True
    preview_max_fps: int = Field(10, ge=1, le=15)
    preview_jpeg_quality: int = Field(70, ge=10, le=100)
    min_free_disk_mb: int = Field(2048, ge=0, description="低于此值拒绝采样并告警")
    max_snapshots_kept: int = Field(50, ge=1, le=1000)
    log_level: str = "INFO"
    log_json: bool = False
    version: str = "1.0.0"

    @model_validator(mode="after")
    def _check(self) -> SystemConfig:
        names = self.class_names
        if not names:
            raise ValueError("class_names 不能为空")
        if len(set(names)) != len(names):
            raise ValueError("class_names 不能重复")
        for n in names:
            if not n or "/" in n or "\\" in n or n.startswith("."):
                raise ValueError(f"非法类别名: {n!r}")
        return self


class AppConfig(StrictModel):
    """全部配置。"""

    system: SystemConfig = Field(default_factory=SystemConfig)
    valve: ValveParams = Field(default_factory=ValveParams)
    valve_test: ValveTestParams = Field(default_factory=ValveTestParams)
    clean: CleanParams = Field(default_factory=CleanParams)
    feed: FeedParams = Field(default_factory=FeedParams)
    outputs: OutputsParams = Field(default_factory=OutputsParams)
    blow: BlowParams = Field(default_factory=BlowParams)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    ml: MlConfig = Field(default_factory=MlConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cameras: list[CameraConfig] = Field(default_factory=lambda: [CameraConfig(id=1)])

    @property
    def total_nozzles(self) -> int:
        return sum(c.nozzles for c in self.cameras)

    def camera(self, camera_id: int) -> CameraConfig:
        for c in self.cameras:
            if c.id == camera_id:
                return c
        raise ConfigError("NOT_FOUND", f"相机 {camera_id} 不存在", {"camera_id": camera_id})

    @model_validator(mode="after")
    def _cross_check(self) -> AppConfig:
        if not self.cameras:
            raise ValueError("至少配置一台相机")
        ids = [c.id for c in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"相机 id 重复: {ids}")
        unknown = [c for c in self.blow.classes if c not in self.system.class_names]
        if unknown:
            raise ValueError(f"blow.classes 含未定义类别: {unknown}")
        if self.valve_test.hole > self.total_nozzles:
            raise ValueError(
                f"valve_test.hole({self.valve_test.hole}) 超出范围 0~{self.total_nozzles}"
            )
        if len(self.feed.speeds) != self.system.feeder_count:
            raise ValueError(
                f"feed.speeds 长度 {len(self.feed.speeds)} 应等于 feeder_count {self.system.feeder_count}"
            )
        return self


# --------------------------------------------------------------------------------------
# 范围表（给 HMI 的 limits）
# --------------------------------------------------------------------------------------


def field_range(model: type[BaseModel], name: str) -> tuple[float | None, float | None]:
    """从 pydantic 字段元数据读出 (下限, 上限)。没有约束的一侧为 None。"""
    info = model.model_fields[name]
    lo: float | None = None
    hi: float | None = None
    for m in info.metadata:
        if isinstance(m, Ge):
            lo = m.ge  # type: ignore[assignment]
        elif isinstance(m, Gt):
            lo = m.gt  # type: ignore[assignment]
        elif isinstance(m, Le):
            hi = m.le  # type: ignore[assignment]
        elif isinstance(m, Lt):
            hi = m.lt  # type: ignore[assignment]
    return lo, hi


def model_limits(model: type[BaseModel], names: list[str] | None = None) -> dict[str, list[float]]:
    """返回 ``{字段: [lo, hi]}``，只含两侧都有界的数值字段。"""
    out: dict[str, list[float]] = {}
    for n in names or list(model.model_fields):
        lo, hi = field_range(model, n)
        if lo is not None and hi is not None:
            out[n] = [lo, hi]
    return out


def build_limits(cfg: AppConfig) -> dict[str, Any]:
    """GET /params 的 ``limits`` 段（docs/03 §5）。"""
    return {
        "valve": model_limits(ValveParams),
        "clean": model_limits(CleanParams, ["time", "period", "count"]),
        "feed": {"speed": [FEED_SPEED_MIN, FEED_SPEED_MAX], "count": cfg.system.feeder_count},
        "valve_test": {"hole": [0, cfg.total_nozzles]},
        "blow": model_limits(BlowParams),
        "capture": model_limits(CaptureSettings),
    }


def camera_limits() -> dict[str, list[float]]:
    """GET /cameras/{id}/params 的 ``limits`` 段。"""
    lim = model_limits(
        CameraConfig,
        ["exposure_us", "width", "height", "line_rate_hz", "runtime_chunk_lines", "offset_x"],
    )
    lim["px"] = [0, MAX_SENSOR_PX]
    return lim


# --------------------------------------------------------------------------------------
# 加载 / 合并
# --------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """递归合并，overlay 覆盖 base；列表整体替换。返回新字典。"""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_layered(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """读取 yaml；若含 ``base: <相对路径>``，先加载 base 再用本文件覆盖。"""
    p = Path(path).resolve()
    seen = _seen or set()
    if p in seen:
        raise ConfigError("VALIDATION", f"配置 base 循环引用: {p}")
    seen.add(p)
    if not p.exists():
        raise ConfigError("NOT_FOUND", f"配置文件不存在: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("VALIDATION", f"配置文件顶层必须是映射: {p}")
    base_ref = data.pop("base", None)
    if base_ref:
        base = load_yaml_layered(p.parent / base_ref, seen)
        data = deep_merge(base, data)
    return data


def validation_error_to_config_error(exc: ValidationError) -> ConfigError:
    """把 pydantic 错误翻译成带中文说明的 ConfigError。"""
    errors = exc.errors()
    items = []
    out_of_range = False
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", ()))
        typ = e.get("type", "")
        if typ in {"greater_than_equal", "less_than_equal", "greater_than", "less_than"}:
            out_of_range = True
        items.append(
            {"field": loc, "type": typ, "message": e.get("msg", ""), "input": _safe(e.get("input"))}
        )
    if out_of_range and len(items) == 1:
        it = items[0]
        ctx = errors[0].get("ctx", {})
        bound = ", ".join(f"{k}={v}" for k, v in ctx.items())
        msg = f"{it['field']} = {it['input']} 超出范围（{bound}）"
        return ConfigError("PARAM_OUT_OF_RANGE", msg, {"errors": items})
    msg = "; ".join(f"{it['field']}: {it['message']}" for it in items) or "配置校验失败"
    return ConfigError(
        "PARAM_OUT_OF_RANGE" if out_of_range else "VALIDATION", msg, {"errors": items}
    )


def _safe(v: Any) -> Any:
    try:
        import json

        json.dumps(v)
        return v
    except Exception:
        return str(v)


def parse_config(data: dict[str, Any]) -> AppConfig:
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise validation_error_to_config_error(exc) from None


def load(path: str | Path) -> AppConfig:
    """加载 yaml（含 base 继承）并校验。``data_dir`` 相对路径按 **当前工作目录** 解析为绝对路径。"""
    data = load_yaml_layered(path)
    cfg = parse_config(data)
    cfg.system.data_dir = Path(cfg.system.data_dir).expanduser().resolve()
    return cfg


# --------------------------------------------------------------------------------------
# ConfigService
# --------------------------------------------------------------------------------------

EDITABLE_SECTIONS: tuple[str, ...] = (
    "valve",
    "valve_test",
    "clean",
    "feed",
    "outputs",
    "blow",
    "capture",
    "ml",
    "pipeline",
)
MCU_SECTIONS: tuple[str, ...] = ("valve", "clean", "feed", "outputs")
CAMERA_EDITABLE: tuple[str, ...] = (
    "awb_mode",
    "exposure_us",
    "width",
    "offset_x",
    "height",
    "runtime_chunk_lines",
    "start_px",
    "end_px",
    "line_rate_hz",
    "trigger",
)

ChangeListener = Callable[[str, Any], None]


class ConfigService:
    """线程安全的配置持有者：读快照、分组更新、原子持久化、变更通知。

    ``update`` 的流程：复制 → 打补丁 → 整体重新校验（跨字段规则一起跑）→ 替换 → 落盘 → 通知。
    任何一步失败都不会改变当前配置。
    """

    def __init__(self, cfg: AppConfig, persist_path: str | Path | None = None):
        self._cfg = cfg
        self._lock = threading.RLock()
        self._listeners: list[ChangeListener] = []
        self.persist_path = Path(persist_path) if persist_path else None

    # ---- 构造 ----
    @classmethod
    def from_yaml(cls, yaml_path: str | Path, *, persist: bool = True) -> ConfigService:
        """从 yaml 启动；若 ``{data_dir}/config.json`` 存在则用其中的可编辑项覆盖，然后写回。"""
        cfg = load(yaml_path)
        persist_path = cfg.system.data_dir / "config.json" if persist else None
        if persist_path is not None:
            saved = read_json(persist_path, default=None)
            if isinstance(saved, dict):
                cfg = cls._overlay_persisted(cfg, saved)
        svc = cls(cfg, persist_path)
        if persist_path is not None:
            svc.save_atomic()
        return svc

    @staticmethod
    def _overlay_persisted(cfg: AppConfig, saved: dict[str, Any]) -> AppConfig:
        data = cfg.model_dump(mode="json")
        for sec in EDITABLE_SECTIONS:
            if isinstance(saved.get(sec), dict):
                data[sec] = deep_merge(data[sec], saved[sec])
        saved_cams = {c.get("id"): c for c in saved.get("cameras", []) if isinstance(c, dict)}
        for cam in data["cameras"]:
            s = saved_cams.get(cam["id"])
            if s:
                for k in CAMERA_EDITABLE:
                    if k in s:
                        cam[k] = s[k]
        try:
            merged = parse_config(data)
        except ConfigError:
            # 持久化文件损坏/不兼容：忽略它，用 yaml 启动（上层会记录告警）
            return cfg
        merged.system.data_dir = cfg.system.data_dir
        return merged

    # ---- 读取 ----
    @property
    def config(self) -> AppConfig:
        """当前配置快照（请勿原地修改；要改走 update）。"""
        with self._lock:
            return self._cfg

    def section(self, name: str) -> BaseModel:
        if name not in EDITABLE_SECTIONS:
            raise ConfigError("NOT_FOUND", f"未知配置分组: {name}", {"section": name})
        return getattr(self.config, name)

    def camera(self, camera_id: int) -> CameraConfig:
        return self.config.camera(camera_id)

    def limits(self) -> dict[str, Any]:
        return build_limits(self.config)

    # ---- 更新 ----
    def update(self, section: str, patch: dict[str, Any]) -> tuple[BaseModel, list[str]]:
        """部分更新一个分组。返回 (该分组新值, warnings)。失败抛 ConfigError 且不改任何状态。"""
        if section not in EDITABLE_SECTIONS:
            raise ConfigError("NOT_FOUND", f"未知配置分组: {section}", {"section": section})
        if not isinstance(patch, dict):
            raise ConfigError("VALIDATION", "补丁必须是对象", {"section": section})
        with self._lock:
            data = self._cfg.model_dump(mode="json")
            data[section] = deep_merge(data[section], patch)
            new_cfg = parse_config(data)
            new_cfg.system.data_dir = self._cfg.system.data_dir
            self._commit(new_cfg)
            applied = getattr(new_cfg, section)
        self._notify(section, applied)
        return applied, []

    def update_camera(
        self, camera_id: int, patch: dict[str, Any]
    ) -> tuple[CameraConfig, list[str]]:
        """更新某台相机的可编辑字段。"""
        bad = [k for k in patch if k not in CAMERA_EDITABLE]
        if bad:
            raise ConfigError(
                "VALIDATION",
                f"相机字段不可修改或不存在: {bad}",
                {"fields": bad, "editable": list(CAMERA_EDITABLE)},
            )
        with self._lock:
            self._cfg.camera(camera_id)  # 不存在则抛 NOT_FOUND
            data = self._cfg.model_dump(mode="json")
            for cam in data["cameras"]:
                if cam["id"] == camera_id:
                    cam.update(patch)
            new_cfg = parse_config(data)
            new_cfg.system.data_dir = self._cfg.system.data_dir
            self._commit(new_cfg)
            applied = new_cfg.camera(camera_id)
        self._notify(f"camera:{camera_id}", applied)
        return applied, []

    def replace(self, new_cfg: AppConfig) -> None:
        """整体替换（测试/恢复用）。"""
        with self._lock:
            self._commit(new_cfg)
        self._notify("*", new_cfg)

    def _commit(self, new_cfg: AppConfig) -> None:
        self._cfg = new_cfg
        if self.persist_path is not None:
            self.save_atomic()

    # ---- 持久化 ----
    def save_atomic(self, path: str | Path | None = None) -> Path:
        """把当前配置原子写为 JSON。返回写入路径。"""
        target = Path(path) if path else self.persist_path
        if target is None:
            raise ConfigError("INTERNAL", "未设置持久化路径")
        with self._lock:
            payload = self._cfg.model_dump(mode="json")
        atomic_write_json(target, payload)
        return target

    def export_backup(self) -> Path:
        """显式落盘并导出 ``config.backup.json``（POST /system/save）。"""
        if self.persist_path is None:
            raise ConfigError("INTERNAL", "未设置持久化路径")
        self.save_atomic()
        return self.save_atomic(self.persist_path.with_name("config.backup.json"))

    # ---- 通知 ----
    def subscribe(self, cb: ChangeListener) -> Callable[[], None]:
        """注册变更监听 ``cb(section, new_value)``；返回取消函数。"""
        with self._lock:
            self._listeners.append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._listeners:
                    self._listeners.remove(cb)

        return unsubscribe

    def _notify(self, section: str, value: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(section, value)
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "配置变更监听器异常", extra={"section": section}
                )
