# Jetson 真机联调核对清单 —— 海康相机驱动（`oresorter/camera/hik.py`）

`HikCamera` 已按 MVS SDK（Linux aarch64，Python 封装 MvImport）的公开 API 预实现，并用假 SDK 模块做了单元测试
（`tests/test_camera_hik.py`）。但 **GenICam 节点名、枚举项符号/数值、SDK 行为细节** 只能在真机上核对。
请按下面顺序逐项打勾；发现与代码不符的地方，改 `hik.py` 顶部「GenICam 节点名与枚举项」常量区
（必要时同步 `tests/test_camera_hik.py` 的 `default_nodes()`），重跑 `pytest -q tests/test_camera_hik.py`。

相机数据手册（MV-CL043-A1GC）要点：4096 × 3 线 CMOS；像素格式 Mono8 / BayerRG8 / RGB8 / BGR8；
最大行频 9 kHz@RGB8（28 kHz@Bayer8/Mono8）；曝光 3 µs ~ 10 ms；I/O 为 Line 0/1/3/4；支持行触发、帧触发、行+帧触发。

## 0. 环境

- [ ] MVS SDK 已装：`dpkg -l | grep -i mvs`；`/opt/MVS/lib/aarch64` 与
      `/opt/MVS/Samples/aarch64/Python/MvImport/MvCameraControl_class.py` 存在。
- [ ] 环境变量（`deploy/oresorter.service` 已带）：`MVCAM_COMMON_RUNENV=/opt/MVS/lib`，
      `LD_LIBRARY_PATH` 含 `/opt/MVS/lib/aarch64`。
- [ ] 驱动能导入 SDK（`MVS_PYTHON_DIR` 默认即 MvImport 目录），并记录 SDK 版本：

```bash
MVS_PYTHON_DIR=/opt/MVS/Samples/aarch64/Python/MvImport .venv/bin/python -c "from oresorter.camera.hik import _import_sdk; sdk=_import_sdk(); print(sdk.__file__, hex(sdk.MvCamera().MV_CC_GetSDKVersion()))"
```

- [ ] 5G 网口：MTU 9000、静态 IP 192.168.10.1/24，相机 192.168.10.10；巨帧可达：`ping -M do -s 8972 192.168.10.10`。
- [ ] 收包缓冲：`deploy/sysctl-oresorter.conf` 已生效（`sysctl net.core.rmem_max`）。
- [ ] 先用 MVS 客户端（`/opt/MVS/bin/MVS.sh`）能枚举、独占打开、连续取图，再跑我们的驱动；**核对节点时打开属性树的「Guru」级别**。

## 1. 枚举与打开（`_open_device` / `_setup_transport`）

- [ ] 设备信息解析正确：`GET /api/v1/cameras` 的 `serial` / 日志「海康相机已打开」的 `device` 字段（ip、serial、model、name）与 MVS 客户端一致。
      涉及 `MV_CC_DEVICE_INFO.SpecialInfo.stGigEInfo` 的 `nCurrentIp`（主机序，高字节为第一段）、`chSerialNumber`、`chModelName`、`chUserDefinedName`。
- [ ] 按 `ip` 匹配与按 `serial` 匹配各试一次（`config/jetson.yaml` 的 `cameras[0].ip` / `serial`）。
- [ ] 独占打开（`MV_ACCESS_Exclusive`）：MVS 客户端同时占用时应报 `0x80000203 设备无访问权限`。
- [ ] SDK 版本是否有 `MvCamera.MV_CC_Initialize`（4.x 有；代码自动判断，只调一次）。
- [ ] `MV_CC_GetOptimalPacketSize` 返回值（MTU 9000 时约 8164）；`GevSCPSPacketSize` 写入成功（失败只告警「设置 GevSCPSPacketSize 失败」）。
- [ ] `MV_GIGE_SetResend(1, 10, 50)`、`MV_CC_SetImageNodeNum(16)` 返回 0（失败只记 DEBUG「SDK 调用失败（忽略）」）；
      观察 16 节点 × 6.3 MB（512 行块）的内存占用是否可接受，必要时改 `IMAGE_NODE_NUM`。
- [ ] `MV_CC_RegisterExceptionCallBack` 注册成功；拔网线后日志是否出现「SDK 异常回调上报设备断开」（`MV_EXCEPTION_DEV_DISCONNECT = 0x8001`）。

## 2. 节点名与枚举项（`hik.py` 顶部常量，逐项对照 MVS 属性树）

| 代码常量 | 预置值 | 现场核对内容 |
| --- | --- | --- |
| `NODE_PIXEL_FORMAT` / `PIXEL_FORMAT_RGB8` | `PixelFormat` = `RGB8Packed` | 符号名是 `RGB8Packed` 还是 `RGB8`；设置后取到的 `enPixelType` 应为 `0x02180014` |
| `NODE_WIDTH` / `NODE_OFFSET_X` | `Width` / `OffsetX` | 步进 `nInc`（代码向下对齐）、`OffsetX+Width ≤ 4096` 的拒绝码 |
| `NODE_HEIGHT` | `Height` | 线阵相机一帧行数的 min/max/inc；512 / 128~1024 可设；取流中是否允许改（代码默认停流后改） |
| `NODE_EXPOSURE_AUTO` | `ExposureAuto` = `Off` | 节点是否存在（可选，缺失只告警） |
| `NODE_EXPOSURE` | `ExposureTime` | 单位 µs；`fMax` 是否随行频变化（代码按相机 fMin/fMax 夹取；需 < 1/行频） |
| `NODE_AWB` | `BalanceWhiteAuto` | 枚举项 `Off/Once/Continuous` 的符号名与数值（兜底表 `ENUM_SYMBOLS`：0/1/2 = Off/Continuous/Once）；`Once` 是否要在取流中才执行（代码未开流时临时抓一帧）；执行后是否自动回 `Off` |
| `NODE_LINE_RATE_ENABLE` / `NODE_LINE_RATE` | `AcquisitionLineRateEnable` / `AcquisitionLineRate` | 节点名；编码器触发时 `AcquisitionLineRate` 是否可写（代码失败只记 INFO） |
| `NODE_TRIGGER_SELECTOR` / `TRIGGER_SELECTOR_LINE` | `TriggerSelector` = `LineStart` | 是否有 `LineStart` / `FrameBurstStart`；`TriggerMode` / `TriggerSource` 是否按 selector 分别保存 |
| `NODE_TRIGGER_MODE` | `TriggerMode` | `Off/On` 数值（兜底表 0/1）；内触发应为 `Off` |
| `NODE_TRIGGER_SOURCE` / `TRIGGER_SOURCE_ENCODER` | `TriggerSource` = `EncoderModuleOut` | 枚举项符号名与数值（兜底表 Line0~3=0~3、Counter0=4、EncoderModuleOut=6、Software=7、FrequencyConverter=8）；需要分频/倍频时改为 `FrequencyConverter` |
| `ENCODER_SETUP` | `EncoderSelector=Encoder0`、`EncoderSourceA=Line0`、`EncoderSourceB=Line1`、`EncoderTriggerMode=AnyDirection`、`EncoderCounterMode=IgnoreDirection` | 节点名与枚举项；**按实际接线改 A/B 相所接的 Line（手册 I/O 为 Line 0/1/3/4）**；缺失只告警 |
| （未下发） | `FrequencyConverterControl`：`InputSource`、`PreDivider`、`Multiplier`、`PostDivider` | 编码器每脉冲对应的行数 ≠ 1 时需要；先在 MVS 客户端设置并保存 UserSet，或告知后补进 `ENCODER_SETUP` |
| `MV_CC_GetEnumEntrySymbolic` | — | SDK 是否提供（代码优先用它读枚举符号名，否则查 `ENUM_SYMBOLS`） |
| `NODE_PACKET_SIZE` | `GevSCPSPacketSize` | 见 §1 |

## 3. 取流（`start_stream` / `_grab_loop`）

- [ ] `start_stream(512)` 后 `Height=512`；每帧 `nWidth=4096`、`nHeight=512`、`nFrameLen=512×4096×3=6291456`、`enPixelType=0x02180014`。
- [ ] `nFrameNum` 起始值（0 或 1）、是否每次 `StartGrabbing` 重置、正常运行下连续递增（代码只按断号 > 0 计丢块）。
- [ ] `nLostPacket` 长时间为 0（重传有效）；若持续非 0 调 `RESEND_MAX_PERCENT` / 网卡环形缓冲。
- [ ] 取帧超时的返回码是否为 `0x80000007`（`MV_E_NODATA`）或 `0x80000107`；出现其它码请记录（会按硬错误计数，连续 3 次判掉线）。
- [ ] 9 kHz 满行频连续 10 min：`GET /api/v1/state` 中 `dropped_chunks=0`、`chunks_per_s≈17.6`；取帧线程 CPU（memmove ≈ 110 MB/s）可接受。
- [ ] 内触发静默判定：`OFFLINE_AFTER_S=5 s`（低行频自动放宽到 4 个块周期）不会误判；编码器模式下皮带停止**不**应判掉线（代码只在 internal 下用静默判定）。
- [ ] 低行频（如 100 Hz，块周期 5 s）下 `get_frame` 超时 / 流水线 `frame_timeout_s` 的配合是否合理。

## 4. 掉线与重连（`_mark_offline` / `_reconnect_loop`）

- [ ] 拔网线：几秒内 `connected=false`、故障 `CAMERA_1_DISCONNECTED`；记录 `MV_CC_IsDeviceConnected` 变为 False 的延迟与异常回调是否先到。
- [ ] 插回：按 0.5 / 1 / 2 / 4 / 8 s 退避重连成功；参数恢复（曝光、ROI、触发、`PixelFormat`、`Height=512`）、自动重新开流、`stats.reconnects` +1、流水线继续收块。
- [ ] 相机断电重启（节点恢复出厂值）后同样恢复；重连后首帧 `nFrameNum` 重新计数不应被当作丢块。
- [ ] 重连成功后 `line_start` 接着原行号走（不重置）；是否需要在重连后强制操作员重新 `start`（SYNC）由运行时策略决定。
- [ ] `close()` 在 2 s 内返回（取帧线程在 `GetImageBuffer` 1 s 超时内退出）；`systemctl stop oresorter` 不卡。

## 5. 快照与白平衡

- [ ] 未开流 `snapshot(4096)`（HMI 采样页）单帧耗时 ≈ 0.46 s@9 kHz；完成后 `Height` 恢复为采样帧高、流已停。
- [ ] 开流中 `snapshot` 从流拼块（不打断流水线），`GET /api/v1/cameras/1/snapshot` 正常。
- [ ] 编码器模式皮带停止时 `snapshot` 在 `max(2 s, 3×height/行频)` 内超时并给出「请确认皮带在运行」提示。
- [ ] `POST /api/v1/cameras/1/awb-once`：`Once → Off` 后颜色稳定，配置 `awb_mode=off`；开机 `awb_mode: once` 时每次打开都会做一次白平衡（光源稳定后可改为 `off`）。

## 6. 发现差异时的处理

- 节点名 / 枚举符号名不同：改 `hik.py` 顶部常量；枚举数值不同且 SDK 没有 `GetEnumEntrySymbolic`：改 `ENUM_SYMBOLS`。
- 需要新增节点（如分频器）：加进 `ENCODER_SETUP`（可选节点，失败只告警），或在 MVS 客户端配置后保存到 `UserSet1` 并把 `UserSetDefault=UserSet1` 作为兜底。
- 改完重跑：`.venv/bin/python -m pytest -q tests/test_camera_hik.py` 与 `ruff check oresorter tests && black --check oresorter tests`。
- 把核对结果（SDK 版本、节点表截图、返回码）记到 `docs/reports/`，供后续多相机 / 换固件时参考。
