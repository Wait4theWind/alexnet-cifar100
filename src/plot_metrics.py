"""读取 metrics.json，绘制 Loss / Accuracy 曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics", type=str, default="checkpoints/metrics.json")
    parser.add_argument("--out-dir", type=str, default="assets")
    return parser.parse_args()


def load_history(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data["history"]
    return data


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(Path(args.metrics))

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row.get("test_loss") for row in history]
    train_acc = [row["train_acc"] for row in history]
    test_acc = [row["test_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    if all(value is not None for value in test_loss):
        plt.plot(epochs, test_loss, marker="s", label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("CIFAR-100 Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="train acc")
    plt.plot(epochs, test_acc, marker="s", label="test acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CIFAR-100 Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker="o", label="train")
    if all(value is not None for value in test_loss):
        plt.plot(epochs, test_loss, marker="s", label="test")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, marker="o", label="train")
    plt.plot(epochs, test_acc, marker="s", label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    print(f"saved curves to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
