import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from torchvision import transforms


def set_one_core_behavior():
    # Match TP setup for fair comparison
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def freeze_batchnorm_stats(model: nn.Module):
    """
    Freeze BatchNorm running stats for stability (and to match TP demo behavior).
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()


def main():
    set_one_core_behavior()

    # -------------------------
    # Match TP-stable setup
    # -------------------------
    num_classes = 10
    batch_size = 16
    steps = 20

    # Simple transform (stable for FakeData)
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Model: train from scratch + smaller head
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.train()

    freeze_batchnorm_stats(model)

    # Dataset
    dataset = FakeData(
        size=256,
        image_size=(3, 224, 224),
        num_classes=num_classes,
        transform=tfm,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Optimizer (lower LR to avoid NaNs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # -------------------------
    # Training loop (timed)
    # -------------------------
    step = 0
    t0 = time.perf_counter()

    for images, targets in loader:
        logits = model(images)

        # quick sanity checks (optional but helpful)
        if torch.isnan(logits).any():
            print("NaN detected in logits; stopping.", flush=True)
            break

        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Prevent gradient explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 5 == 0:
            print(f"step {step:02d}  loss={loss.item():.4f}", flush=True)

        step += 1
        if step >= steps:
            break

    t1 = time.perf_counter()

    # Use actual steps completed (in case it stopped early)
    total_time = t1 - t0
    total_images = step * batch_size

    print("\n--- Baseline Results (No Splitting) ---")
    print(f"Steps completed: {step}")
    print(f"Total time: {total_time:.2f} s")
    if step > 0:
        print(f"Throughput: {total_images / total_time:.2f} images/sec")
        print(f"Avg latency per batch: {total_time / step * 1000:.2f} ms")


if __name__ == "__main__":
    main()