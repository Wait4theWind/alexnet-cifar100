"""手写 AlexNet，输入尺寸适配 CIFAR-100 的 32x32。

特征提取部分包含 5 个卷积层，每个卷积后接 BatchNorm2d + ReLU，
并在第 1、2、5 个卷积块后做 2x2 最大池化：
    32 -> 16 -> 8 -> 8 -> 8 -> 4
因此最终特征图为 256 x 4 x 4，展平后维度为 4096。
"""

import torch
import torch.nn as nn

FEATURE_FLAT_DIM = 256 * 4 * 4


class AlexNetCIFAR100(nn.Module):
    def __init__(self, num_classes: int = 100, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(FEATURE_FLAT_DIM, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    torch.manual_seed(0)
    model = AlexNetCIFAR100(num_classes=100)
    model.eval()

    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        flat = model.features(inputs)
        outputs = model(inputs)

    print(f"features output: {tuple(flat.shape)}")
    print(f"logits output:   {tuple(outputs.shape)}")
    assert flat.shape == (2, FEATURE_FLAT_DIM), flat.shape
    assert outputs.shape == (2, 100), outputs.shape

    total, trainable = count_parameters(model)
    print(f"total parameters:     {total:,}")
    print(f"trainable parameters: {trainable:,}")
    print("model self-test passed")


if __name__ == "__main__":
    main()
