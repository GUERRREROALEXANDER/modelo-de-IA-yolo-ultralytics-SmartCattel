"""Thread-safe, lazily loaded cattle detector."""

from threading import Lock

import numpy as np

from .config import Settings
from .model_loader import load_model, resolve_class_ids


class Detector:
    def __init__(self, settings: Settings | None = None, model=None):
        self.settings = settings or Settings()
        self._model = model
        self._class_ids = None
        self._lock = Lock()

    def _prepare(self) -> None:
        if self._model is None:
            self._model = load_model(self.settings.weights)
        if self._class_ids is None:
            self._class_ids = resolve_class_ids(self._model.names, self.settings.class_names)

    @property
    def class_ids(self) -> list[int]:
        with self._lock:
            self._prepare()
            return self._class_ids.copy()

    def detect(self, frame: np.ndarray) -> list[dict]:
        with self._lock:
            self._prepare()
            result = self._model.predict(
                frame,
                conf=self.settings.confidence,
                iou=self.settings.iou,
                imgsz=self.settings.imgsz,
                classes=self._class_ids,
                device=self.settings.device,
                verbose=False,
            )[0]
            detections = []
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id not in self._class_ids:
                    continue
                detections.append({
                    "class": self._model.names[class_id],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                })
            return sorted(detections, key=lambda item: item["confidence"], reverse=True)
