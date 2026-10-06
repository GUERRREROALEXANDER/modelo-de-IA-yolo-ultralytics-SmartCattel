"""Validated detection settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _default_weights() -> str:
    weights = Path(__file__).resolve().parents[1] / "models" / "cattle_yolo11n_best.pt"
    return str(weights) if weights.is_file() else "yolo11n.pt"


@dataclass(frozen=True)
class Settings:
    weights: str = field(default_factory=_default_weights)
    confidence: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    class_names: tuple[str, ...] = ("cow",)
    device: str = field(default_factory=_default_device)

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be greater than 0 and at most 1")
        if not 0 < self.iou <= 1:
            raise ValueError("iou must be greater than 0 and at most 1")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32 and at least 32")
        if not self.class_names or any(not name.strip() for name in self.class_names):
            raise ValueError("class_names must contain at least one nonempty name")

    @classmethod
    def from_env(cls) -> "Settings":
        classes = os.getenv("SMARTCATTLE_CLASSES", "cow")
        return cls(
            weights=os.getenv("SMARTCATTLE_WEIGHTS", _default_weights()),
            confidence=float(os.getenv("SMARTCATTLE_CONF", "0.35")),
            iou=float(os.getenv("SMARTCATTLE_IOU", "0.5")),
            imgsz=int(os.getenv("SMARTCATTLE_IMGSZ", "640")),
            class_names=tuple(name.strip() for name in classes.split(",")),
            device=os.getenv("SMARTCATTLE_DEVICE", _default_device()),
        )
