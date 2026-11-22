---
title: AlexNet CIFAR-100
sdk: static
pinned: false
---

# AlexNet · CIFAR-100

手写 AlexNet 在 CIFAR-100 上的图片分类演示。上传图片后，浏览器通过 ONNX Runtime Web 直接推理，输出 Top-5 类别和置信度。

- 测试集准确率：63.01%
- 输入尺寸：32 × 32
- 参数量：7,025,316
- 推理方式：ONNX Runtime Web（WASM），不需要 Python 后端

## 链接

- GitHub：https://github.com/Wait4theWind/alexnet-cifar100
- Model：https://huggingface.co/bbbegemot/alexnet-cifar100

## 文件说明

- `index.html`：页面结构
- `style.css`：样式
- `app.js`：图片预处理、ONNX 推理、Top-5 展示
- `classes.json`：CIFAR-100 类别名
- `alexnet_cifar100.onnx`：模型文件，需要单独上传

## 已知局限

模型只在 32 × 32 的 CIFAR-100 图片上训练，真实照片缩小后细节损失明显，预测结果仅用于演示。
