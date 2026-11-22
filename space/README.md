---
title: AlexNet CIFAR-100
sdk: gradio
app_file: app.py
pinned: false
---

# AlexNet · CIFAR-100 图片分类

手写 AlexNet（适配 32x32 输入，加入 BatchNorm 与 Dropout）在 CIFAR-100 上的推理演示。

- 上传任意图片，输出 Top-5 类别及置信度
- 输入会被 Resize 到 32x32，并使用 CIFAR-100 的均值/方差归一化
- 模型权重通过 `hf_hub_download` 从 Model 仓库加载

## 链接

- GitHub：https://github.com/Wait4theWind/alexnet-cifar100
- Model：https://huggingface.co/YOUR_HF_USERNAME/alexnet-cifar100
- Space：https://huggingface.co/spaces/YOUR_HF_USERNAME/alexnet-cifar100-demo

## 说明

模型只在 CIFAR-100 这类 32x32 小图上训练，对真实场景的高分辨率照片泛化能力有限。
