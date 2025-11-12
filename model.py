import os
import json
import time
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

os.environ["TORCH_HOME"] = r"D:\akselerator\torch_cache"
# ================== Конфигурация ==================
DATA_ROOT = r"D:/akselerator/dataset/dataset"
OUTPUT_DIR = "./out_freshguard"

MODEL_NAME = "mobilenet_v3_small"
IMG_SIZE = 192
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42


# ==================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return train_tfms, val_tfms


def build_model(model_name: str, num_classes: int):
    model_name = model_name.lower()
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    crit = nn.CrossEntropyLoss()
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        loss = crit(logits, targets)
        loss_total += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)
    return loss_total / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, device, optimizer, scheduler=None, log_every=100):
    model.train()
    crit = nn.CrossEntropyLoss()
    correct, total, loss_total = 0, 0, 0.0
    start = time.time()

    for step, (imgs, targets) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = crit(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_total += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)

        if step % log_every == 0 or step == len(loader):
            cur_loss = loss_total / total
            cur_acc = correct / total
            print(f"    step {step:4d}/{len(loader)} "
                  f"| loss={cur_loss:.4f} acc={cur_acc:.4f}")

    dur = time.time() - start
    return loss_total / max(total, 1), correct / max(total, 1), dur


def save_label_map(path: Path, class_to_idx: dict):
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
                  f, ensure_ascii=False, indent=2)


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = os.path.join(DATA_ROOT, "train")
    test_dir = os.path.join(DATA_ROOT, "test")
    assert os.path.isdir(train_dir), f"train dir not found: {train_dir}"
    assert os.path.isdir(test_dir), f"test dir not found: {test_dir}"

    train_tfms, val_tfms = build_transforms(IMG_SIZE)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(test_dir, transform=val_tfms)

    classes = train_ds.classes
    print("Классы:", classes)
    save_label_map(out_dir / "labels.json", train_ds.class_to_idx)

    pin = (device.type == "cuda")
    num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )

    model = build_model(MODEL_NAME, num_classes=len(classes))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    best_path = out_dir / "best_model.pth"

    print("\n===== Старт обучения =====")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Эпоха {epoch}/{EPOCHS} (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
        tr_loss, tr_acc, tr_time = train_one_epoch(
            model, train_loader, device, optimizer, scheduler, log_every=max(1, len(train_loader) // 5)
        )
        va_loss, va_acc = evaluate(model, val_loader, device)

        train_losses.append(tr_loss);
        val_losses.append(va_loss)
        train_accs.append(tr_acc);
        val_accs.append(va_acc)

        print(f"Итог эпохи: train_loss={tr_loss:.4f} acc={tr_acc:.4f} "
              f"| val_loss={va_loss:.4f} acc={va_acc:.4f} "
              f"| time={tr_time:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "config": {
                    "model_name": MODEL_NAME,
                    "img_size": IMG_SIZE,
                }
            }, best_path)
            print(f"  -> saved best to: {best_path} (val_acc={best_acc:.4f})")

    print("\n=== Обучение завершено ===")
    print(f"Лучшая точность на валидации: {best_acc:.4f}")
    print(f"Модель сохранена в: {best_path}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('График потерь')
    plt.xlabel('Эпоха');
    plt.ylabel('Loss');
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('График точности')
    plt.xlabel('Эпоха');
    plt.ylabel('Accuracy');
    plt.legend()

    curves_path = out_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(curves_path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    print(f"Графики сохранены: {curves_path}")

    try:
        model.eval()
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        traced = torch.jit.trace(model, example)
        traced.save(str(out_dir / "model_ts.pt"))
        print("Сохранён TorchScript:", out_dir / "model_ts.pt")
    except Exception as e:
        print("Не удалось сохранить TorchScript:", e)


# ================== Инференс-утилита ==================
def load_model_for_inference(ckpt_path: str, model_name: str, num_classes: int, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt.get("classes", None)


@torch.no_grad()
def predict_image(model, img_path: str, img_size: int, device=None):
    device = device or next(model.parameters()).device
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ================== Точка входа ==================
if __name__ == "__main__":
    # Windows-safe multiprocessing
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
