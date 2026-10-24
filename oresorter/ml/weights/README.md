# 离线预训练权重

来源：torchvision 官方发布（https://download.pytorch.org/models/），运行时**不下载**。

| 文件 | 骨干 | SHA256 前缀 |
|---|---|---|
| `resnet18-f37072fd.pth` | `torchvision.models.resnet18` (IMAGENET1K_V1) | `f37072fd` |
| `mobilenet_v3_large-8738ca79.pth` | `torchvision.models.mobilenet_v3_large` (IMAGENET1K_V1) | `8738ca79` |

缺失时 `oresorter.ml.models.build_model(pretrained=True)` 会告警并降级为随机初始化（训练仍可进行，但收敛更慢、精度更低）。
