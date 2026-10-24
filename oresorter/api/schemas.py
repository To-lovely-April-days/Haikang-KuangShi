"""请求/响应模型（与 docs/03 一一对应；字段名就是接口字段名）。

约定：

- 请求体只做 **类型与必填** 校验；取值范围来自配置（``ConfigService.limits()``），由
  ``RuntimeService`` / ``ConfigService`` 校验并以 ``PARAM_OUT_OF_RANGE`` 拒绝，这里不写死数值；
- 所有写操作请求体可带 ``request_id``（幂等键，见 ``oresorter.api.idempotency``）；
- 响应模型主要服务于 OpenAPI 文档：路由直接返回 service 的 dict，``extra="allow"`` 让 service
  附带的补充字段（如 ``warnings`` / ``since_start``）原样透出。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RunStateName = Literal["stopped", "running", "paused", "sampling", "fault"]
JobStatusName = Literal["queued", "running", "succeeded", "failed", "cancelled"]
JobTypeName = Literal["capture", "train", "deploy", "valve_test", "backup"]
ParamSection = Literal["valve", "clean", "feed", "outputs", "blow", "capture"]


class _In(BaseModel):
    """请求体基类：未知字段拒绝（400 VALIDATION），便于发现拼写错误。"""

    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    """响应体基类：允许 service 附加文档未列出的补充字段。"""

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorBody(BaseModel):
    error: ErrorDetail


class RequestIdBody(_In):
    """所有写操作可带的幂等键（也可放在 ``X-Request-Id`` 头）。"""

    request_id: str | None = Field(None, max_length=128)


class AcceptedOut(_Out):
    accepted: bool = True


# --------------------------------------------------------------------------------------
# 系统
# --------------------------------------------------------------------------------------


class HealthOut(BaseModel):
    ok: bool = True
    version: str


class SavedOut(_Out):
    saved: bool = True
    backup_path: str | None = None


class PowerOut(_Out):
    accepted: bool = True
    action: Literal["poweroff", "reboot"] | None = None


class FaultOut(_Out):
    code: str
    since: str
    detail: str = ""


class StateOut(_Out):
    """``GET /state`` 全量快照（docs/03 §3）。"""

    time: str
    uptime_total_s: int
    run_state: RunStateName
    deployed_model_id: int | None
    cameras: list[dict[str, Any]]
    mcu: dict[str, Any]
    faults: list[FaultOut]
    jobs: list[dict[str, Any]]
    stats: dict[str, Any]
    version: str
    pipeline: dict[str, Any] | None = None


class SystemInfoOut(_Out):
    version: str
    hostname: str
    platform: str | None = None
    python: str | None = None
    data_dir: str | None = None
    disk_total_mb: int
    disk_free_mb: int
    cpu_percent: float
    mem_percent: float
    temperatures_c: dict[str, float]
    gpu_percent: float | None
    ips: dict[str, list[str]]
    pid: int | None = None
    preview_max_fps: int | None = None


# --------------------------------------------------------------------------------------
# 运行控制
# --------------------------------------------------------------------------------------


class RunOut(_Out):
    run_state: RunStateName
    mcu_ack: bool | None


class StatsOut(_Out):
    """``state.stats`` / ``GET /run/stats``：速率为 int，延迟分位数为 float，累计值自本次 start 起。"""

    particles_per_min: int
    blown_per_min: int
    per_class_per_min: dict[str, int]
    chunk_latency_ms_p50: float
    chunk_latency_ms_p99: float
    chunk_latency_ms_max: float
    chunks_total: int
    dropped_chunks: int
    particles_total: int
    blown_total: int
    per_class_total: dict[str, int]
    since: str | None = None
    elapsed_s: float
    since_start: dict[str, Any] | None = None
    pipelines: dict[str, Any] | None = None


# --------------------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------------------


class ParamsOut(_Out):
    valve: dict[str, Any]
    clean: dict[str, Any]
    feed: dict[str, Any]
    outputs: dict[str, Any]
    blow: dict[str, Any]
    capture: dict[str, Any]
    limits: dict[str, Any]


class ParamsApplyOut(_Out):
    applied: dict[str, Any]
    mcu_ack: bool | None = Field(
        None, description="null=该分组不下发下位机；false=已持久化但未确认"
    )
    warnings: list[str] = Field(default_factory=list)


class ClassesOut(_Out):
    class_names: list[str]
    max_models: int


class ClassesIn(RequestIdBody):
    class_names: list[str] = Field(..., min_length=1)


# --------------------------------------------------------------------------------------
# 相机 / 喷阀检测
# --------------------------------------------------------------------------------------


class CameraOut(_Out):
    id: int
    name: str
    model: str | None = None
    serial: str | None = None
    ip: str | None = None
    connected: bool
    streaming: bool = False
    nozzles: int


class CameraParamsOut(_Out):
    awb_mode: Literal["off", "once", "continuous"]
    exposure_us: float
    width: int
    offset_x: int
    height: int
    runtime_chunk_lines: int
    start_px: int
    end_px: int
    line_rate_hz: int
    trigger: Literal["internal", "encoder"]
    nozzles: int | None = None
    limits: dict[str, Any] | None = None


class CameraParamsApplyOut(_Out):
    applied: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class AppliedOut(_Out):
    applied: bool = True


class NozzleRangeOut(BaseModel):
    nozzle: int
    px_from: int
    px_to: int


class NozzleMapOut(_Out):
    start_px: int
    end_px: int
    nozzles: int
    quotient: int
    remainder: int
    ranges: list[NozzleRangeOut]


class SnapshotIn(RequestIdBody):
    height: int | None = Field(None, description="采集行数；缺省取相机配置 height")
    max_width: int = Field(2000, description="缩放后最大宽度（像素）")
    overlay: bool = False


class SnapshotOut(_Out):
    image_url: str
    width: int
    height: int
    warnings: list[str] = Field(default_factory=list)
    count: int | None = None


class CapturePreviewIn(RequestIdBody):
    camera_id: int
    settings: dict[str, Any] | None = Field(None, description="临时覆盖的采集设置（不持久化）")


class ParticleOut(BaseModel):
    bbox: list[int]
    area_px: int


class CapturePreviewOut(_Out):
    image_url: str
    particles: list[ParticleOut]
    count: int
    width: int | None = None
    height: int | None = None
    warnings: list[str] = Field(default_factory=list)


class ValveTestSingleIn(RequestIdBody):
    hole: int


class ValveTestCycleIn(RequestIdBody):
    from_: int = Field(..., alias="from")
    to: int
    interval_ms: int = 200

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class JobIdOut(_Out):
    job_id: str


class StoppedOut(_Out):
    stopped: bool = True


class ValveResetOut(_Out):
    accepted: bool = True
    mcu_ack: bool | None = True


# --------------------------------------------------------------------------------------
# 模型 / 任务
# --------------------------------------------------------------------------------------


class ModelClassOut(BaseModel):
    name: str
    sample_count: int
    enabled: bool


class ModelOut(_Out):
    id: int
    name: str | None = None
    description: str | None = None
    created_at: str | None = None
    trained_at: str | None = None
    deployed: bool = False
    algo: str | None = None
    image_size: int | None = None
    classes: list[ModelClassOut] = Field(default_factory=list)
    metrics_summary: dict[str, Any] | None = None


class ModelCreateIn(RequestIdBody):
    name: str = ""
    description: str = ""


class ModelPatchIn(RequestIdBody):
    name: str | None = None
    description: str | None = None


class ModelDeletedOut(_Out):
    deleted: bool = True
    id: int


class BackupIn(RequestIdBody):
    target_id: int
    overwrite: bool = False


class TrainIn(RequestIdBody):
    classes: list[str] | None = Field(None, description="省略 = 该模型所有有样本的类别")


class JobOut(_Out):
    job_id: str
    type: JobTypeName
    model_id: int | None = None
    status: JobStatusName
    progress: float
    stage: str | None = None
    message: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class CancelledOut(_Out):
    cancelled: bool


# --------------------------------------------------------------------------------------
# 样本
# --------------------------------------------------------------------------------------


class CaptureIn(RequestIdBody):
    camera_id: int
    max_samples: int = 200
    max_frames: int | None = 50
    timeout_s: float = 120.0


class SampleOut(_Out):
    sample_id: str
    thumb_url: str
    image_url: str
    created_at: str
    camera_id: int
    bbox: list[int]
    area_px: int
    deleted: bool = False


class SampleListOut(_Out):
    total: int
    items: list[SampleOut]


class SampleDeletedOut(_Out):
    deleted: bool
    undo_batch: str | None = None


class SamplesDeleteIn(RequestIdBody):
    ids: list[str] = Field(..., min_length=1)


class SamplesDeletedOut(_Out):
    deleted: int
    undo_batches: list[str] = Field(default_factory=list)


class RestoredOut(_Out):
    restored: bool


class UndoDeleteOut(_Out):
    restored: int


class ClearedOut(_Out):
    deleted: int
    undo_batch: str | None = None
