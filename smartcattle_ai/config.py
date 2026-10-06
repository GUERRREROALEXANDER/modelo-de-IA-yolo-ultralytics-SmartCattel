"""Validated detection settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _default_weights() -> str:
    models = Path(__file__).resolve().parents[1] / "models"
    for filename in ("cattle_coco_yolo11n_best.pt", "cattle_yolo11n_best.pt"):
        weights = models / filename
        if weights.is_file():
            return str(weights)
    return "yolo11n.pt"


@dataclass(frozen=True)
class Settings:
    weights: str = field(default_factory=_default_weights)
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    class_names: tuple[str, ...] = ("cow", "person")
    device: str = field(default_factory=_default_device)
    person_weights: str = "yolo11n.pt"

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be greater than 0 and at most 1")
        if not 0 < self.iou <= 1:
            raise ValueError("iou must be greater than 0 and at most 1")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32 and at least 32")
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise ValueError("class_names must contain at least one nonempty name")
        if any(name not in {"cow", "person"} for name in self.class_names):
            raise ValueError("class_names may contain only cow and person")
        if not self.person_weights.strip():
            raise ValueError("person_weights must be nonempty")

    @classmethod
    def from_env(cls) -> "Settings":
        classes = os.getenv("SMARTCATTLE_CLASSES", "cow,person")
        return cls(
            weights=os.getenv("SMARTCATTLE_WEIGHTS", _default_weights()),
            confidence=float(os.getenv("SMARTCATTLE_CONF", "0.35")),
            iou=float(os.getenv("SMARTCATTLE_IOU", "0.5")),
            imgsz=int(os.getenv("SMARTCATTLE_IMGSZ", "640")),
            class_names=tuple(name.strip() for name in classes.split(",")),
            device=os.getenv("SMARTCATTLE_DEVICE", _default_device()),
            person_weights=os.getenv("SMARTCATTLE_PERSON_WEIGHTS", "yolo11n.pt"),
        )
