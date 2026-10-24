# 矿石智能分选机 — 算力盒后端（oresorter）

本仓库是矿石色选机（皮带式）的**算力盒端后端服务**，运行在图为 TW-T201P（NVIDIA Jetson Orin Nano 8G，JetPack 6.x，Ubuntu 22.04，Python 3.10）上。

职责：线阵相机采集 → 矿石分割切图 → CNN 分类 → 吹气决策与喷头映射 → 下发下位机主控；同时向 HMI（前端，由其他同事开发）提供 HTTP/WebSocket 接口，管理样本库与模型训练/部署。

不在本仓库范围内：HMI 界面、下位机固件、机械/气路。

## 必读文档（按顺序读完再动手）

1. `docs/01_技术选型与架构.md` — 已拍板的选型、网络拓扑、模块划分、识别管线、可靠性设计、数据目录、配置项与取值范围
2. `docs/02_开发任务书.md` — 分阶段任务、每个模块的职责/接口/测试要求、验收标准
3. `docs/03_前端接口清单.md` — 与 HMI 的接口契约（REST + WebSocket），改接口必须同步改这份文档
4. `docs/04_下位机协议草案.md` — 与下位机主控的帧格式与指令表（草案，待与 MCU 同事确认，TBD 项已标出）
5. `docs/00_原始需求/需求整理.md` — 客户流程文档与 7 个界面的功能整理（需求来源，有疑问以此为准）

## 硬约束（违反即为 bug）

- **Python 3.10 兼容**：Jetson 上就是 3.10，禁止 3.11+ 语法与标准库（`typing.Self`、`tomllib`、`except*`、`def f[T]()` 泛型语法等）。
- **运行时不依赖外网**：预训练权重、pip 依赖都要能离线安装（来源写在 docs/01 §10）。
- **所有硬件都在抽象接口后面**：相机（`oresorter/camera/base.py`）与下位机（`oresorter/mcu/link.py`）必须各有一个 Mock 实现，整套系统（含 API、训练、运行时流水线）可在 Mac/PC 上无硬件跑通。
- **任何对外 IO 必须有超时与重连**（网络、SDK 调用、子进程），不允许无限阻塞；主服务不能因为某个设备掉线而崩溃。
- **参数范围以 docs/01 §9 配置表为准**，越界必须拒绝并返回原因，不要静默截断。
- **训练与运行时推理共用同一个预处理函数**（`oresorter/vision/preprocess.py`），避免训练/推理不一致。
- **模型数量、类别名称可配置**（默认 7 个模型、类别 A~G），HMI 按配置显示。
- **后端是下位机的唯一主机**：HMI 的运行/暂停/停止与所有参数一律经后端转发，前端不直连下位机。
- 写文件要原子（写临时文件后 `os.replace`），配置与模型部署切换不能出现半成品。

## 开发约定

- 包名 `oresorter`，目录结构见 `docs/02_开发任务书.md` §2，不要自行发明别的布局。
- 配置与接口模型用 pydantic v2；Web 框架 FastAPI + uvicorn；训练 PyTorch；推理 TensorRT（PyTorch 兜底）。
- 代码英文命名，注释与文档中文；ruff + black；公共函数带类型标注与 docstring。
- 日志用 `logging`（结构化字段），不要 `print`。
- 测试用 pytest，`pytest -q` 必须全绿再提交。以下必须有单元测试：喷头映射（按原始需求那张 64 行表逐行校验）、协议编解码与 CRC、分块拼接去重、样本软删除/撤销/清空、配置范围校验、训练→导出→推理一致性（同一张图 PyTorch 与 ONNX/TensorRT 输出一致）。
- 一个模块一次提交，附带测试；提交信息中文即可。

## 常用命令

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # 开发机
pip install -r requirements-jetson.txt     # Jetson（PyTorch 走 NVIDIA 专用 wheel，见 docs/01 §10）
pytest -q
python -m oresorter.cli serve --config config/dev.yaml      # 启动服务（Mock 相机 + Mock 下位机）
python -m oresorter.cli eval --folder /path/to/samples      # 对 a/ b/ c/ 子目录做交叉验证，输出混淆矩阵
python -m oresorter.cli replay --images /path/to/frames     # 回放大图走一遍分割→分类
```

## Jetson 环境提示

- 海康 MVS SDK 安装在 `/opt/MVS`，Python 封装在 `/opt/MVS/Samples/aarch64/Python/MvImport`；运行前需要 `export MVCAM_COMMON_RUNENV=/opt/MVS/lib` 与 `LD_LIBRARY_PATH` 追加 `/opt/MVS/lib/aarch64`。
- 相机走 5G 网口直连（MTU 9000，静态 IP），HMI 与下位机走 1G 网口经工业交换机。
- 运行前 `sudo nvpmodel -m 0 && sudo jetson_clocks` 锁频；`trtexec` 在 `/usr/src/tensorrt/bin/`。
- TensorRT 引擎文件（`model.plan`）绑定设备与 TensorRT 版本，必须在算力盒上生成，不要提交到仓库。
