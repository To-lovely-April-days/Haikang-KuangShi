# oresorter — 矿石智能分选机算力盒后端

当前状态：**设计文档阶段**，代码按 `docs/02_开发任务书.md` 由 Claude Code 生成。

```
backend/
├── CLAUDE.md                      # Claude Code 项目约束（先读）
├── docs/
│   ├── 00_原始需求/需求整理.md     # 客户流程文档 + 7 个界面的功能整理
│   ├── 01_技术选型与架构.md        # 已拍板选型、拓扑、模块、识别管线、可靠性、数据目录、配置表
│   ├── 02_开发任务书.md            # 分阶段任务、目录结构、模块接口、验收标准、测试计划
│   ├── 03_前端接口清单.md          # HMI ↔ 算力盒 REST + WebSocket 契约（给前端同事）
│   └── 04_下位机协议草案.md        # 算力盒 ↔ 下位机帧格式与指令表（给 MCU 同事确认）
└── README.md
```

## 开工方式

1. 在 `backend/` 目录启动 Claude Code；
2. 第一条指令建议：`按 CLAUDE.md 和 docs/02_开发任务书.md 完成 P0 工程骨架，然后进入 P1；每个模块完成后跑 pytest 并汇报`；
3. 把 `docs/03` 发给前端同事、`docs/04` 发给下位机同事确认，确认结果回填到文档的 TBD 项。

## 环境

- 目标机：图为 TW-T201P（Jetson Orin Nano 8G，JetPack 6.x，Python 3.10）
- 开发机：Mac/PC，Python 3.10，Mock 相机 + Mock 下位机即可跑通全部功能
