# model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Tuple, Dict, Any, Optional, Callable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# =========================
# DTO / Result objects
# =========================

@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    img_size: int
    classes: List[str]
    state_dict: Dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    pred: str
    prob_percent: float


@dataclass(frozen=True)
class TopKPrediction:
    items: List[Tuple[str, float]]  # (class_name, prob in [0..1])


# =========================
# Abstractions (SOLID: DIP/ISP)
# =========================

class ICheckpointRepository(Protocol):
    def load(self) -> ModelConfig: ...


class IModelFactory(Protocol):
    def create(self, model_name: str, num_classes: int) -> nn.Module: ...


class IPreprocessor(Protocol):
    def __call__(self, img: Image.Image) -> torch.Tensor: ...


class IClassifier(Protocol):
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor: ...


class IPostprocessor(Protocol):
    def __call__(self, probs: torch.Tensor, classes: List[str]) -> Any: ...


# =========================
# Infrastructure
# =========================

class TorchCheckpointRepository(ICheckpointRepository):
    """Repository + Adapter to the .pth checkpoint format."""
    def __init__(self, ckpt_path: Path, device: torch.device):
        self._ckpt_path = ckpt_path
        self._device = device

    def load(self) -> ModelConfig:
        ckpt = torch.load(self._ckpt_path, map_location=self._device)
        classes = ckpt["classes"]
        cfg = ckpt.get("config", {})
        model_name = cfg.get("model_name", "mobilenet_v3_small")
        img_size = int(cfg.get("img_size", 192))
        state_dict = ckpt["model_state"]
        return ModelConfig(
            model_name=str(model_name),
            img_size=img_size,
            classes=list(classes),
            state_dict=state_dict,
        )


class TorchvisionModelFactory(IModelFactory):
    """Factory Method: creates torchvision model by name."""
    def create(self, model_name: str, num_classes: int) -> nn.Module:
        mn = model_name.lower()
        if mn == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
            return model
        if mn == "resnet18":
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            return model
        if mn == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            return model
        raise ValueError(f"Unknown model_name: {model_name}")


class TorchImagePreprocessor(IPreprocessor):
    """Strategy: preprocessing pipeline (resize->tensor->normalize)."""
    def __init__(self, img_size: int):
        self._tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._tfm(img).unsqueeze(0)


class TorchSoftmaxClassifier(IClassifier):
    """Wraps torch model inference (single responsibility)."""
    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        return probs.cpu()


class Top1Postprocessor(IPostprocessor):
    """Strategy: returns exactly (pred, prob%) like your Flask now."""
    def __call__(self, probs: torch.Tensor, classes: List[str]) -> Prediction:
        idx = int(torch.argmax(probs).item())
        pred = classes[idx]
        prob_percent = round(float(probs[idx]) * 100, 2)
        return Prediction(pred=pred, prob_percent=prob_percent)


class TopKPostprocessor(IPostprocessor):
    """Strategy for batch/CSV: returns top-k list (prob in [0..1])."""
    def __init__(self, k: int = 3):
        self._k = k

    def __call__(self, probs: torch.Tensor, classes: List[str]) -> TopKPrediction:
        k = min(self._k, probs.numel())
        vals, idxs = torch.topk(probs, k=k)
        items = [(classes[int(i)], float(v)) for v, i in zip(vals, idxs)]
        return TopKPrediction(items=items)


# =========================
# Facade / Template Method
# =========================

class PredictorService:
    """
    Facade: one entry point for prediction, hides subsystem complexity. [web:72]
    Also acts as a Template Method: preprocess -> infer -> postprocess.
    """
    def __init__(
        self,
        preprocessor: IPreprocessor,
        classifier: IClassifier,
        postprocessor: IPostprocessor,
        classes: List[str],
    ):
        self._pre = preprocessor
        self._clf = classifier
        self._post = postprocessor
        self._classes = classes

    def predict(self, img: Image.Image):
        x = self._pre(img)
        probs = self._clf.predict_proba(x)
        return self._post(probs, self._classes)


# =========================
# Composition root (wiring)
# =========================

def build_predictor_from_ckpt(
    ckpt_path: Path,
    device: Optional[torch.device] = None,
    postprocessor: Optional[IPostprocessor] = None,
) -> Tuple[PredictorService, ModelConfig]:
    """
    Composition Root: creates concrete objects and wires dependencies (DIP).
    """
    device = device or torch.device("cpu")

    repo = TorchCheckpointRepository(ckpt_path, device=device)
    cfg = repo.load()

    factory = TorchvisionModelFactory()
    model = factory.create(cfg.model_name, num_classes=len(cfg.classes))
    model.load_state_dict(cfg.state_dict)
    model.to(device).eval()

    pre = TorchImagePreprocessor(cfg.img_size)
    clf = TorchSoftmaxClassifier(model, device=device)
    post = postprocessor or Top1Postprocessor()

    service = PredictorService(pre, clf, post, cfg.classes)
    return service, cfg
