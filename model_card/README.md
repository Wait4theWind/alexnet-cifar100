---
license: mit
library_name: pytorch
tags:
  - image-classification
  - cifar100
  - alexnet
  - pytorch
datasets:
  - cifar100
metrics:
  - accuracy
---

# AlexNet · CIFAR-100

手写 AlexNet，适配 CIFAR-100 的 32x32 输入，在卷积层后加入 BatchNorm2d 与 ReLU，
并使用 Dropout 缩减全连接层容量。

## 模型说明

- 5 个卷积层：3→64→192→384→256→256，全部使用 `kernel_size=3, stride=1, padding=1`
- 每个卷积层后接 `BatchNorm2d` + `ReLU`
- 第 1、2、5 个卷积块后接 2x2 最大池化：32 → 16 → 8 → 4
- 展平维度：`256 * 4 * 4 = 4096`
- 分类层：4096 → 1024 → 512 → 100，两次 Dropout(0.5)
- 参数量：7,025,316（约 7.03 M）
- 输入：3 x 32 x 32，RGB，使用 CIFAR-100 均值 `(0.5071, 0.4867, 0.4408)` 与标准差 `(0.2675, 0.2565, 0.2761)` 归一化
- 输出：100 类 logits

## 训练配置

| 配置项 | 取值 |
| --- | --- |
| 数据集 | CIFAR-100 |
| epochs | 40 |
| batch size | 128 |
| 优化器 | SGD（momentum=0.9, weight_decay=5e-4） |
| 初始学习率 | 0.01 |
| 学习率调度 | StepLR（step_size=15, gamma=0.1） |
| 数据增强 | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Dropout | 0.5 |
| 随机种子 | 42 |

## 最终准确率

CIFAR-100 测试集 top-1 准确率：**63.01%**（第 36 个 epoch）。

## 仓库文件

- `best_model.pth`：最佳测试准确率对应的权重（包含 `model_state_dict`、`epoch`、`test_acc`）
- `model.py`：与训练时完全一致的模型定义
- `README.md`：本模型卡

## 使用方式

```python
import torch
from huggingface_hub import hf_hub_download
from model import AlexNetCIFAR100

repo_id = "YOUR_HF_USERNAME/alexnet-cifar100"
weights = hf_hub_download(repo_id=repo_id, filename="best_model.pth")
checkpoint = torch.load(weights, map_location="cpu")

model = AlexNetCIFAR100(num_classes=100)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

`model.py` 也可以从 Model 仓库或 GitHub 获取：

- GitHub：https://github.com/Wait4theWind/alexnet-cifar100
- Space 体验：https://huggingface.co/spaces/YOUR_HF_USERNAME/alexnet-cifar100-demo

## 已知局限

- 输入被固定为 32x32，真实照片缩小后细节损失严重，泛化能力有限
- CIFAR-100 每类仅 500 张训练图，类别间差异细微（如 oak_tree / maple_tree / willow_tree）
- 轻量化全连接层与较少训练轮次限制了最终准确率，预期明显低于 ResNet 等现代架构
- 模型不适用于安全关键场景，输出仅作演示用途
