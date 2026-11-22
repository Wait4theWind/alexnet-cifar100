import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import AlexNetCIFAR100

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    return parser.parse_args()


def train(model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in trainloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.shape[0]
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.shape[0]

    return total_loss / total, correct / total


@torch.no_grad()
def test(model, testloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in testloader:
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = criterion(y_hat, y)

        total_loss += loss.item() * y.shape[0]
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.shape[0]

    return total_loss / total, correct / total


def main():
    args = get_args()
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    trainset = datasets.CIFAR100(
        root=args.data_dir, train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = AlexNetCIFAR100().to(device)
    print("参数量:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = test(model, testloader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
        })

        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "history": history}, f, indent=2, ensure_ascii=False)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | lr {lr:.6f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": best_acc,
            }, os.path.join(args.out_dir, "best_model.pth"))
            print(f"  保存新的最好模型，test_acc={best_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pth"))
    print("训练结束，最高 test_acc:", best_acc)


if __name__ == "__main__":
    main()
