import os
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ========= настройки  =========
CKPT_PATH = Path("./out_freshguard/best_model.pth")
IMAGES_DIR = Path("./real_photo")
OUTPUT_DIR = Path("./out_freshguard")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"
TOPK = 3


# =====================================================


def build_model(model_name: str, num_classes: int) -> nn.Module:
    mn = model_name.lower()
    if mn == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif mn == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif mn == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Не найден чекпоинт: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", "mobilenet_v3_small")
    img_size = int(cfg.get("img_size", 192))
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("В чекпоинте нет списка классов (ключ 'classes').")

    model = build_model(model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return model, classes, model_name, img_size


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


@torch.no_grad()
def predict_one(model: nn.Module,
                img_path: Path,
                tfm,
                device: torch.device) -> Tuple[int, List[float]]:
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.tensor(probs).argmax().item())
    return pred_idx, probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, classes, model_name, img_size = load_model_from_ckpt(CKPT_PATH, device)
    tfm = build_transform(img_size)
    print(f"Loaded model: {model_name} | img_size={img_size}")
    print(f"Classes: {classes}\n")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in exts]
    if not image_paths:
        print(f"В папке {IMAGES_DIR} не найдено изображений ({', '.join(exts)})")
        return

    rows = []
    for i, path in enumerate(sorted(image_paths), 1):
        pred_idx, probs = predict_one(model, path, tfm, device)
        topk = sorted(
            [(classes[j], float(probs[j])) for j in range(len(classes))],
            key=lambda x: x[1],
            reverse=True
        )[:TOPK]

        print(f"[{i:03d}] {path.name}")
        print(f"  → Предсказание: {classes[pred_idx]} ({probs[pred_idx] * 100:.2f}%)")
        print("  → Топ-варианты:")
        for name, pr in topk:
            print(f"     - {name:15s} : {pr * 100:6.2f}%")
        print()

        row = {
            "filename": str(path),
            "pred_class": classes[pred_idx],
            "pred_prob": probs[pred_idx]
        }
        for k, (name, pr) in enumerate(topk, 1):
            row[f"top{k}_class"] = name
            row[f"top{k}_prob"] = pr
        rows.append(row)

    fieldnames = ["filename", "pred_class", "pred_prob"] + \
                 sum(([f"top{k}_class", f"top{k}_prob"] for k in range(1, TOPK + 1)), [])
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Готово! Результаты сохранены в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
