"""加载训练好的权重，对 CIFAR-100 测试集图片做推理并保存样例图。"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AlexNetCIFAR100

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CIFAR-100 test samples")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="assets")
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def load_model(weights: str, device: torch.device) -> AlexNetCIFAR100:
    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "test_acc" in checkpoint:
            print(f"checkpoint test acc: {checkpoint['test_acc']:.4f}")
    model = AlexNetCIFAR100(num_classes=100)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_sample(
    image: torch.Tensor,
    target: int,
    pred: int,
    confidence: float,
    classes: list[str],
    path: Path,
) -> None:
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    display = (image.cpu() * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(3, 3.4))
    ax.imshow(display)
    ax.axis("off")
    ax.set_title(
        f"true: {classes[target]}\npred: {classes[pred]} ({confidence:.2%})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_set = datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    classes = test_set.classes

    model = load_model(args.weights, device)
    correct_samples: list[tuple[torch.Tensor, int, int, float]] = []
    wrong_samples: list[tuple[torch.Tensor, int, int, float]] = []
    printed = 0

    with torch.no_grad():
        for index in range(len(test_set)):
            image, target = test_set[index]
            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).squeeze(0)
            confidence, pred = probs.max(dim=0)
            pred = int(pred)
            confidence = float(confidence)
            is_correct = pred == target

            if printed < args.num_samples:
                mark = "OK" if is_correct else "WRONG"
                print(
                    f"sample {index:05d} | true: {classes[target]:<15} | "
                    f"pred: {classes[pred]:<15} | conf: {confidence:.4f} | {mark}"
                )
                printed += 1

            if is_correct and len(correct_samples) < 2:
                correct_samples.append((image, target, pred, confidence))
            if not is_correct and len(wrong_samples) < 1:
                wrong_samples.append((image, target, pred, confidence))

            if (
                printed >= args.num_samples
                and len(correct_samples) >= 2
                and len(wrong_samples) >= 1
            ):
                break

    for number, (image, target, pred, confidence) in enumerate(correct_samples, 1):
        save_sample(
            image,
            target,
            pred,
            confidence,
            classes,
            out_dir / f"prediction_correct_{number}.png",
        )
    for number, (image, target, pred, confidence) in enumerate(wrong_samples, 1):
        save_sample(
            image,
            target,
            pred,
            confidence,
            classes,
            out_dir / f"prediction_wrong_{number}.png",
        )

    if len(correct_samples) < 2 or not wrong_samples:
        print("[warn] did not find enough correct/wrong samples to save")


if __name__ == "__main__":
    main()
