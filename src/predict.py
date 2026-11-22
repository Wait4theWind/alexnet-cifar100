import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import AlexNetCIFAR100

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="assets")
    parser.add_argument("--num-samples", type=int, default=8)
    return parser.parse_args()


def save_image(image, label, pred, conf, classes, path):
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    img = img.clip(0, 1)

    plt.figure(figsize=(3, 3.2))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"true: {classes[label]}\npred: {classes[pred]} ({conf:.2%})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("保存图片:", path)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    testset = datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    classes = testset.classes

    checkpoint = torch.load(args.weights, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    print("checkpoint test_acc:", checkpoint.get("test_acc"))

    model = AlexNetCIFAR100().to(device)
    model.load_state_dict(state)
    model.eval()

    correct_imgs = []
    wrong_imgs = []

    with torch.no_grad():
        for i in range(len(testset)):
            image, label = testset[i]
            output = model(image.unsqueeze(0).to(device))
            prob = torch.softmax(output, dim=1)
            conf, pred = prob.max(1)
            pred = pred.item()
            conf = conf.item()

            if i < args.num_samples:
                print(
                    f"第 {i} 张 | 真实: {classes[label]} | "
                    f"预测: {classes[pred]} | 置信度: {conf:.4f}"
                )

            if pred == label and len(correct_imgs) < 2:
                correct_imgs.append((image, label, pred, conf))
            if pred != label and len(wrong_imgs) < 1:
                wrong_imgs.append((image, label, pred, conf))

            if i + 1 >= args.num_samples and len(correct_imgs) == 2 and len(wrong_imgs) == 1:
                break

    for i, (image, label, pred, conf) in enumerate(correct_imgs, 1):
        path = os.path.join(args.out_dir, f"prediction_correct_{i}.png")
        save_image(image, label, pred, conf, classes, path)

    for i, (image, label, pred, conf) in enumerate(wrong_imgs, 1):
        path = os.path.join(args.out_dir, f"prediction_wrong_{i}.png")
        save_image(image, label, pred, conf, classes, path)


if __name__ == "__main__":
    main()
