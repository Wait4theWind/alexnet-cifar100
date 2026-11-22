import argparse
import os

import torch

from model import AlexNetCIFAR100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--out", type=str, default="space_static/alexnet_cifar100.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)

    model = AlexNetCIFAR100()
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 32, 32)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print("saved:", args.out)
    print("size MB:", round(os.path.getsize(args.out) / 1024 / 1024, 2))

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        print("inputs:", [(item.name, item.shape) for item in session.get_inputs()])
        print("outputs:", [(item.name, item.shape) for item in session.get_outputs()])
    except ImportError:
        print("onnxruntime not installed, skip check")


if __name__ == "__main__":
    main()
