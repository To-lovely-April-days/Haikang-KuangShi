# oresorter — 矿石智能分选机算力盒后端

运行在图为 TW-T201P（Jetson Orin Nano 8G，JetPack 6.x，Python 3.10）上的分选机"大脑"：
线阵相机采集 → 矿石分割切图 → CNN 分类 → 吹气决策与喷头映射 → 下发下位机；同时向 HMI 提供 REST + WebSocket 接口，管理样本库与模型训练/部署。

当前状态：**P0 工程骨架、P1 识别率验证工具链、P2 服务化已实现并通过测试**（Mock 相机 + Mock 下位机下全功能可跑）；P3 Jetson 真机联调（海康相机驱动、TensorRT 引擎、性能与 48 h 老化）待在算力盒上执行，脚本与检查清单已备好。

## 目录

```
backend/
├── CLAUDE.md                  # Claude Code 项目约束（先读）
├── README.md
├── pyproject.toml             # ruff / black / pytest 配置
├── requirements.txt           # 开发机依赖（宽松版本）
├── requirements-lock.txt      # 开发机精确锁定（uv pip freeze）
├── requirements-jetson.txt    # Jetson（PyTorch 走 NVIDIA 专用 wheel）
├── config/
│   ├── default.yaml           # 全部默认值（与 docs/01 §9 一致）
│   ├── dev.yaml               # 开发机：Mock 相机 + Mock 下位机，data_dir=./data
│   └── jetson.yaml            # 真机：海康相机 + 下位机 TCP，data_dir=/data/oresorter
├── deploy/                    # systemd 单元、锁频/网口调优、netplan 示例、sysctl、一键安装脚本
├── docs/                      # 需求整理、技术选型与架构、开发任务书、前端接口清单、下位机协议草案、openapi.json
├── oresorter/                 # 代码包（下表）
├── scripts/                   # make_synthetic（合成数据）、export_openapi、soak_test（老化）
└── tests/                     # pytest（全部可在无硬件的 Mac/PC 上跑）
```

| 包 | 职责 |
|---|---|
| `oresorter/config.py` | pydantic 配置模型、范围校验、yaml 分层加载、`config.json` 原子持久化、变更通知 |
| `oresorter/camera/` | `Camera` 抽象、`Frame`；`MockCamera`（目录回放 / 合成蓝底图、限速、丢块/掉线注入）；`HikCamera` 占位（P3） |
| `oresorter/vision/` | 逐列背景模型、分割（阈值/形态学/连通域/带 mask 裁切/拆粘连）、跨块拼接去重、训练推理共用预处理、像素→喷头映射、合成数据、回放 |
| `oresorter/data/` | 模型目录（新建/删除/备份至/元数据）与样本库（原子写、缩略图、软删除/撤销/清空、批次日志、清理） |
| `oresorter/ml/` | 骨干与离线权重、数据集与受限增广、一键训练配方（子进程入口）、指标、ONNX 导出与 TensorRT 引擎、Torch/ONNX/TRT 分类器、交叉验证评估 |
| `oresorter/mcu/` | docs/04 协议（CRC16、帧解析、负载打包）、`TcpMcuLink`（心跳/重连/ACK 重发）、进程内 `MockMcuLink` 与 TCP `MockMcuServer` |
| `oresorter/runtime/` | 状态机、任务系统、运行统计、吹气决策、三线程流水线、采集/训练/部署/备份任务、`RuntimeService` 门面、事件总线、`AppContext` 组合根 |
| `oresorter/api/` | FastAPI 应用工厂、统一错误体、docs/03 全部 REST 路由、`/ws/events` 与 `/ws/preview/{id}`、`request_id` 幂等 |
| `oresorter/cli.py` | `serve` / `eval` / `replay` / `build-engine` / `learn-bg` |
| `oresorter/sdnotify.py` | systemd READY / WATCHDOG 喂狗（非 systemd 环境自动空操作） |

## 开发机安装（Mac / PC）

需要 Python 3.10（与 Jetson 一致）。推荐用 [uv](https://docs.astral.sh/uv/) 管理解释器与依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # 一次性
uv python install 3.10
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt   # 或 requirements.txt 取最新兼容版本
```

不用 uv 也可以：`python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt`。

预训练权重（ResNet18 / MobileNetV3，torchvision 官方发布）已随仓库放在 `oresorter/ml/weights/`，运行时不联网。

## 常用命令

```bash
.venv/bin/python -m pytest -q                                   # 全部测试（约 1 分钟，无硬件）
.venv/bin/ruff check oresorter tests scripts && .venv/bin/black --check oresorter tests scripts

.venv/bin/python -m oresorter.cli serve --config config/dev.yaml  # 启动服务：Mock 相机 + Mock 下位机
#   http://127.0.0.1:8000/api/v1/docs       交互式接口文档（OpenAPI）
#   http://127.0.0.1:8000/api/v1/state      全量状态
#   ws://127.0.0.1:8000/api/v1/ws/events    事件流

.venv/bin/python scripts/make_synthetic.py --out /tmp/synth --classes A,B,C --per-class 50 --frames 5   # 合成样本与大图
.venv/bin/python -m oresorter.cli eval --folder /path/to/samples          # 子目录即类别（支持客户的 a/ b/ c/ 目录），交叉验证 + 混淆矩阵
.venv/bin/python -m oresorter.cli replay --images /tmp/synth/frames --model-dir ./data/models/1   # 回放大图：分割 → 分类，输出标注图与报告
.venv/bin/python -m oresorter.cli learn-bg --config config/dev.yaml --camera 1                    # 用空皮带帧学习逐列背景
.venv/bin/python -m oresorter.cli build-engine --model-dir /data/oresorter/models/1               # Jetson：ONNX → TensorRT 引擎
.venv/bin/python scripts/export_openapi.py --check                                                # 校验 docs/openapi.json 与代码一致
.venv/bin/python scripts/soak_test.py --duration 1h --interval 60                                  # Mock 下老化：RSS / 线程 / 延迟 / 丢块
```

## 配置

- `config/*.yaml` 是部署期配置，`dev.yaml` / `jetson.yaml` 通过 `base: default.yaml` 继承并覆盖；列表字段（`cameras`、`feed.speeds`）整体替换。
- 运行期由 HMI 修改的参数（喷阀、清灰、给料、输出、吹气、采集设置、相机可编辑项）原子写入 `{data_dir}/config.json`，重启后覆盖 yaml 中的同名项；网络、相机类型等部署项始终以 yaml 为准。
- 全部取值范围见 `docs/01_技术选型与架构.md` §9，越界请求会被拒绝并返回原因（`PARAM_OUT_OF_RANGE`）。
- 模型数量与类别名可配置（默认 7 个模型、类别 A~G），HMI 通过 `GET /api/v1/config/classes` 读取。

## 数据目录（`system.data_dir`）

```
config.json / config.backup.json   运行参数 / 显式保存的备份
state.json                         当前部署模型、累计运行时长、背景版本
background/cam1.npz                逐列背景模型
models/{id}/meta.json              名称、说明、类别、样本计数、训练信息、指标摘要
models/{id}/samples/{cls}/*.png    RGBA 样本（alpha 即 mask）+ 同名 .json 元数据
models/{id}/samples/.trash/        软删除（可撤销），journal.json 记录删除批次
models/{id}/thumbs/{cls}/*.jpg     缩略图（长边 160）
models/{id}/train/                 best.pt、model.onnx、model.plan、metrics.json、train.log、progress.json
snapshots/ previews/ logs/ captures/
```

## 测试与质量门禁

- `pytest -q` 必须全绿再提交；ruff + black 无告警。
- 必测项均已落地：喷头映射 64 行表逐行校验、协议编解码与 CRC、分块拼接去重、样本软删除/撤销/清空、配置范围校验、训练→导出→推理一致性（同一批图 PyTorch 与 ONNX top-1 一致、概率差 < 1e-2）、接口清单逐条集成测试、链路心跳超时/重连/ACK 重发。
- 需要真机的测试带 `@pytest.mark.jetson`（TensorRT、海康 SDK），在 Jetson 上用 `pytest -m jetson` 运行。

## Jetson 部署

```bash
sudo ./deploy/install_jetson.sh            # 同步到 /opt/oresorter、建 venv、装依赖（优先 wheels/ 离线）、建 /data/oresorter、装 systemd 单元并启动
journalctl -u oresorter -f                 # 日志
curl http://127.0.0.1:8000/api/v1/health
```

- PyTorch / torchvision 必须用 NVIDIA 为 JetPack 编译的 wheel（见 `docs/01` §10），放到 `backend/wheels/` 后再运行安装脚本。
- 海康 MVS SDK 安装到 `/opt/MVS`；环境变量与网口调优见 `CLAUDE.md`「Jetson 环境提示」、`deploy/tune_jetson.sh`、`deploy/netplan-example.yaml`。
- `oresorter.service` 为 `Type=notify` + `WatchdogSec=30`，主服务每 10 s 喂狗，流水线不健康时停止喂狗由 systemd 拉起。
- TensorRT 引擎（`model.plan`）绑定设备与 TensorRT 版本，在算力盒上由首次部署自动生成（或 `cli build-engine`），不要提交到仓库。

## 与其他同事的接口

- 前端（HMI）：`docs/03_前端接口清单.md` + `docs/openapi.json`（`/api/v1/docs` 可在线查看）。
- 下位机（MCU）：`docs/04_下位机协议草案.md`，§9 列出待确认项（单位、链路、STATUS 格式、位图映射等）。
- 需求侧待确认项汇总在 `docs/00_原始需求/需求整理.md` §5 与 `docs/01` §11。
