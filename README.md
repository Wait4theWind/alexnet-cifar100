# AlexNet on CIFAR-100

[![Python syntax check](https://github.com/Wait4theWind/alexnet-cifar100/actions/workflows/python-app.yml/badge.svg)](https://github.com/Wait4theWind/alexnet-cifar100/actions/workflows/python-app.yml)

从零手写的 AlexNet，适配 CIFAR-100 的 32x32 输入，包含 BatchNorm、Dropout、
学习率衰减、最佳权重保存、训练曲线绘制、单图推理与 Hugging Face 部署脚本。

## 项目概览

- 模型：5 层卷积 + 3 层全连接，参数量约 7.03 M
- 数据集：CIFAR-100（50,000 张训练图 / 10,000 张测试图，100 类）
- 输入：3 x 32 x 32，使用 CIFAR-100 均值与标准差归一化
- 数据增强：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()`
- 训练：SGD + StepLR，最佳测试准确率权重保存到 Google Drive
- 部署：Hugging Face Model 仓库 + Gradio Space

最终测试集准确率：**63.01%**（第 36 个 epoch）

## 模型结构

| 阶段 | 配置 | 输出尺寸 |
| --- | --- | --- |
| Conv1 + BN + ReLU | 3→64, k=3, s=1, p=1 | 64 x 32 x 32 |
| MaxPool | k=2, s=2 | 64 x 16 x 16 |
| Conv2 + BN + ReLU | 64→192, k=3, s=1, p=1 | 192 x 16 x 16 |
| MaxPool | k=2, s=2 | 192 x 8 x 8 |
| Conv3 + BN + ReLU | 192→384, k=3, s=1, p=1 | 384 x 8 x 8 |
| Conv4 + BN + ReLU | 384→256, k=3, s=1, p=1 | 256 x 8 x 8 |
| Conv5 + BN + ReLU | 256→256, k=3, s=1, p=1 | 256 x 8 x 8 |
| MaxPool | k=2, s=2 | 256 x 4 x 4 |
| Flatten | 256 x 4 x 4 | 4096 |
| FC1 + ReLU + Dropout | 4096→1024 | 1024 |
| FC2 + ReLU + Dropout | 1024→512 | 512 |
| FC3 | 512→100 | 100 |

三次池化把特征图从 32x32 缩小到 4x4，因此展平维度为 `256 * 4 * 4 = 4096`。

## 目录结构

```
alexnet-cifar100/
├─ src/
│  ├─ model.py          # 模型定义与形状/参数量自测
│  ├─ train.py          # 训练、评估、调度器、Drive 保存、metrics.json
│  ├─ predict.py        # 加载权重，打印预测并保存正确/错误样例图
│  └─ plot_metrics.py   # 读取 metrics.json 绘制曲线
├─ space/               # 可直接推送到 Hugging Face Space 的文件
│  ├─ app.py
│  ├─ model.py
│  ├─ requirements.txt
│  └─ README.md
├─ model_card/README.md # Hugging Face Model 仓库的模型卡
├─ assets/              # 曲线图与预测样例图（训练后生成）
├─ .github/workflows/python-app.yml
├─ requirements.txt
└─ README.md
```

## 环境依赖

- Python 3.10+
- PyTorch / torchvision（GPU 训练，CPU 也可运行）
- matplotlib、Pillow

```bash
pip install -r requirements.txt
```

## 复现步骤

### 1. 检查模型

```bash
python src/model.py
```

预期输出：

```
特征展平后: torch.Size([2, 4096])
模型输出: torch.Size([2, 100])
参数量: 7025316
```

### 2. 训练

```bash
python src/train.py --epochs 40 --batch-size 128 --lr 0.01
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--epochs` | 40 | 训练轮数 |
| `--batch-size` | 128 | batch size |
| `--lr` | 0.01 | 初始学习率 |
| `--data-dir` | ./data | 数据集目录 |
| `--out-dir` | ./checkpoints | 权重和 metrics 的输出目录 |

每个 epoch 会写入 `metrics.json`，包含 `train_loss`、`test_acc` 等字段；
测试准确率创新高时保存 `best_model.pth`。

### 3. 在 Colab / 云端训练

推荐先在笔记本中挂载 Drive，再以显式输出目录运行：

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/Wait4theWind/alexnet-cifar100.git
%cd alexnet-cifar100
!pip install -r requirements.txt
!python src/train.py --epochs 40 --out-dir /content/drive/MyDrive/alexnet_cifar100
```

不传 `--out-dir` 时权重会保存到项目下的 `checkpoints/`，Colab 实例释放后会丢失。
所以云端训练建议显式传 Drive 路径。CIFAR-100 下载较慢时，可重试或把
`--data-dir` 指向 Drive 上的缓存目录。

### 4. 绘制曲线

```bash
python src/plot_metrics.py --metrics checkpoints/metrics.json
```

生成：

- `assets/loss_curve.png`
- `assets/accuracy_curve.png`
- `assets/training_curves.png`

### 5. 推理与样例图

```bash
python src/predict.py --weights checkpoints/best_model.pth --num-samples 8
```

脚本会打印每张图的真实标签、预测标签与置信度，并保存：

- `assets/prediction_correct_1.png`
- `assets/prediction_correct_2.png`
- `assets/prediction_wrong_1.png`

## 训练曲线

![loss](assets/loss_curve.png)

![accuracy](assets/accuracy_curve.png)

## 预测样例

| 预测正确 | 预测错误 |
| --- | --- |
| ![correct 1](assets/prediction_correct_1.png) | ![wrong 1](assets/prediction_wrong_1.png) |
| ![correct 2](assets/prediction_correct_2.png) | |

## Hugging Face

- Model 仓库：https://huggingface.co/bbbegemot/alexnet-cifar100
- Space 体验：https://huggingface.co/spaces/bbbegemot/alexnet-cifar100-demo

发布步骤：

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli upload bbbegemot/alexnet-cifar100 best_model.pth
huggingface-cli upload bbbegemot/alexnet-cifar100 src/model.py model.py
huggingface-cli upload bbbegemot/alexnet-cifar100 model_card/README.md README.md
huggingface-cli upload bbbegemot/alexnet-cifar100-demo space/ .
```

Space 的 `space/app.py` 中 `MODEL_REPO` 已配置为 `bbbegemot/alexnet-cifar100`。

## 已知局限

- 输入固定为 32x32，真实照片缩小后细节损失明显
- CIFAR-100 类别间差异细微，模型容量与训练轮次有限
- 预期准确率明显低于 ResNet / ViT 等现代架构
- 仅用于学习与演示，不适用于安全关键场景

## 参考

- [Dive into Deep Learning - 卷积神经网络](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)
- [Dive into Deep Learning - AlexNet](https://zh.d2l.ai/chapter_convolutional-modern/alexnet.html)
- [AlexNet 论文](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
