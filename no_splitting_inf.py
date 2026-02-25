import time
import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


def main():
    # -------------------------
    # Use normal CPU behavior
    # (remove this line if you want PyTorch to use all cores)
    # -------------------------
    # torch.set_num_threads(1)

    # Load pretrained model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights).eval()

    labels = weights.meta["categories"]
    preprocess = weights.transforms()

    # Load image
    image_path = "bear.jpeg"  # change if needed
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]

    # Warmup
    with torch.no_grad():
        _ = model(x)
        _ = model(x)

    # Timed run
    N = 20
    t0 = time.perf_counter()

    outputs = []
    with torch.no_grad():
        for _ in range(N):
            y = model(x)
            outputs.append(y)

    t1 = time.perf_counter()

    # Results
    logits = outputs[-1]
    pred = int(logits.argmax(1))

    print("pred idx:", pred)
    print("pred label:", labels[pred])
    print(f"Throughput: {N / (t1 - t0):.2f} images/sec")
    print(f"Avg latency per image: {(t1 - t0) * 1000 / N:.2f} ms")


if __name__ == "__main__":
    main()