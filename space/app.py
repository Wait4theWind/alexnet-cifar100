import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torchvision import transforms

from model import AlexNetCIFAR100

MODEL_REPO = "YOUR_HF_USERNAME/alexnet-cifar100"
GITHUB_URL = "https://github.com/Wait4theWind/alexnet-cifar100"

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

weights_path = hf_hub_download(MODEL_REPO, "best_model.pth")
checkpoint = torch.load(weights_path, map_location=device)

model = AlexNetCIFAR100(num_classes=100)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

best_acc = checkpoint.get("test_acc", 0.0)
best_epoch = checkpoint.get("epoch", "-")
acc_text = f"{best_acc * 100:.2f}%（第 {best_epoch} 个 epoch）"


def predict(image):
    if image is None:
        return {}

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)[0]

    top5_prob, top5_idx = prob.topk(5)

    result = {}
    for i in range(5):
        name = CLASS_NAMES[top5_idx[i]]
        result[name] = float(top5_prob[i])
    return result


with gr.Blocks(title="AlexNet CIFAR-100") as demo:
    gr.Markdown(
        """
        # AlexNet · CIFAR-100 图片分类
        上传一张图片，输出置信度最高的 5 个类别。
        模型是手写 AlexNet，输入会被缩放到 32x32。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片")
            button = gr.Button("预测", variant="primary")
            gr.Markdown(
                f"""
                **模型信息**
                - 数据集：CIFAR-100
                - 测试集准确率：{acc_text}
                - 输入尺寸：32 x 32
                - 参数量：7,025,316

                **链接**
                - [GitHub]({GITHUB_URL})
                - [模型权重](https://huggingface.co/{MODEL_REPO})
                """
            )
        with gr.Column(scale=2):
            label = gr.Label(num_top_classes=5, label="Top-5")

    button.click(predict, inputs=image_input, outputs=label)

if __name__ == "__main__":
    demo.launch()
