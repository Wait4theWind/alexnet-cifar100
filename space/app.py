"""Hugging Face Space：AlexNet CIFAR-100 图片分类（Top-5）。"""

import os

import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision import transforms

from model import AlexNetCIFAR100

MODEL_REPO = os.environ.get("MODEL_REPO", "YOUR_HF_USERNAME/alexnet-cifar100")
WEIGHTS_FILE = os.environ.get("WEIGHTS_FILE", "best_model.pth")
GITHUB_URL = os.environ.get(
    "GITHUB_URL", "https://github.com/Wait4theWind/alexnet-cifar100"
)
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair",
    "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab",
    "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree",
    "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
    "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit",
    "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]
)


def load_model():
    weights_path = hf_hub_download(repo_id=MODEL_REPO, filename=WEIGHTS_FILE)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint
    best_acc = None
    best_epoch = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        best_acc = checkpoint.get("test_acc")
        best_epoch = checkpoint.get("epoch")

    model = AlexNetCIFAR100(num_classes=len(CLASS_NAMES))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, best_acc, best_epoch


model, best_acc, best_epoch = load_model()

if best_acc is not None:
    accuracy_text = f"{best_acc * 100:.2f}%"
    if best_epoch is not None:
        accuracy_text += f"（第 {best_epoch} 个 epoch）"
else:
    accuracy_text = "见 Model 卡"


def predict(image):
    if image is None:
        return {}
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = probs.topk(5)
    return {
        CLASS_NAMES[index]: float(prob)
        for prob, index in zip(top_probs.tolist(), top_indices.tolist())
    }


with gr.Blocks(title="AlexNet CIFAR-100") as demo:
    gr.Markdown(
        """
        # AlexNet · CIFAR-100 图片分类
        上传一张图片，模型会输出 100 个类别中置信度最高的 5 个。
        模型为手写 AlexNet（BatchNorm + Dropout），输入固定为 32x32。
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片")
            predict_button = gr.Button("开始预测", variant="primary")
            gr.Markdown(
                f"""
                **模型信息**
                - 数据集：CIFAR-100（100 类）
                - 测试集准确率：**{accuracy_text}**
                - 输入尺寸：32 x 32
                - 参数量：约 7.03 M

                **链接**
                - [GitHub 仓库]({GITHUB_URL})
                - [模型权重]({MODEL_URL})
                """
            )
        with gr.Column(scale=2):
            label_output = gr.Label(num_top_classes=5, label="Top-5 预测")

    predict_button.click(predict, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()
