# infer_freshguard.py
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch

from model import build_predictor_from_ckpt, TopKPostprocessor

CKPT_PATH = Path("./out_freshguard/best_model.pth")
IMAGES_DIR = Path("./real_photo")
OUTPUT_DIR = Path("./out_freshguard")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"
TOPK = 3


class CsvPredictionWriter:
    """GRASP Pure Fabrication: separate IO responsibility."""
    def write(self, out_path: Path, rows: List[dict], fieldnames: List[str]) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor, cfg = build_predictor_from_ckpt(
        CKPT_PATH,
        device=device,
        postprocessor=TopKPostprocessor(k=TOPK)
    )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in exts]
    if not image_paths:
        print(f"В папке {IMAGES_DIR} не найдено изображений")
        return

    rows = []
    for i, path in enumerate(sorted(image_paths), 1):
        img = Image.open(path).convert("RGB")
        topk = predictor.predict(img).items

        pred_class, pred_prob = topk[0]
        print(f"[{i:03d}] {path.name}")
        print(f"  → Предсказание: {pred_class} ({pred_prob * 100:.2f}%)")

        row = {
            "filename": str(path),
            "pred_class": pred_class,
            "pred_prob": pred_prob
        }
        for k, (name, pr) in enumerate(topk, 1):
            row[f"top{k}_class"] = name
            row[f"top{k}_prob"] = pr
        rows.append(row)

    fieldnames = ["filename", "pred_class", "pred_prob"] + \
                 sum(([f"top{k}_class", f"top{k}_prob"] for k in range(1, TOPK + 1)), [])
    CsvPredictionWriter().write(OUTPUT_CSV, rows, fieldnames)
    print(f"Готово! Результаты сохранены в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
