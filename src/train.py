"""在 CIFAR-100 上训练 AlexNetCIFAR100。

Colab 下会自动挂载 Google Drive 并把权重与 metrics.json 保存到
MyDrive/alexnet_cifar100；非 Colab 环境则保存到 ./checkpoints。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AlexNetCIFAR100, count_parameters

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-100")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--step-size", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_out_dir(out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir)
    else:
        path = Path("./checkpoints")
        try:
            import google.colab  # noqa: F401

            from google.colab import drive

            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            path = Path("/content/drive/MyDrive/alexnet_cifar100")
        except Exception as exc:
            print(f"[info] Google Drive unavailable ({exc}); using ./checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_set = datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def save_metrics(path: Path, config: dict, history: list[dict]) -> None:
    payload = {"config": config, "history": history}
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = resolve_out_dir(args.out_dir)
    print(f"device: {device}")
    print(f"output dir: {out_dir.resolve()}")

    train_loader, test_loader = build_loaders(args)
    model = AlexNetCIFAR100(num_classes=100).to(device)
    total_params, _ = count_parameters(model)
    print(f"parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    metrics_path = out_dir / "metrics.json"
    best_acc = 0.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": current_lr,
        }
        history.append(record)
        save_metrics(metrics_path, vars(args), history)

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | "
            f"lr {current_lr:.6f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": best_acc,
                    "num_classes": 100,
                    "args": vars(args),
                },
                out_dir / "best_model.pth",
            )
            print(f"  -> new best model saved (test_acc={best_acc:.4f})")

    torch.save(model.state_dict(), out_dir / "last_model.pth")
    print(f"training finished, best test accuracy: {best_acc:.4f}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
