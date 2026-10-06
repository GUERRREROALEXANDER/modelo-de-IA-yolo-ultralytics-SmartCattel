"""Thread-safe, lazily loaded cow and person detector."""

from pathlib import Path
from threading import Lock

import numpy as np

from .config import Settings
from .model_loader import load_model


class Detector:
    def __init__(self, settings: Settings | None = None, model=None, person_model=None):
        self.settings = settings or Settings()
        self._model = model
        self._person_model = person_model
        self._routes = None
        self._lock = Lock()

    def _prepare(self) -> None:
        if self._routes is not None:
            return
        if self._model is None:
            self._model = load_model(self.settings.weights)

        primary_names = set(self._model.names.values())
        missing = set(self.settings.class_names) - primary_names
        same_weights = Path(self.settings.weights).resolve() == Path(self.settings.person_weights).resolve()
        if missing and not same_weights and self._person_model is None:
            self._person_model = load_model(self.settings.person_weights)
        fallback = self._model if same_weights else self._person_model

        routes = {}
        for name in self.settings.class_names:
            selected = self._model if name in primary_names else fallback
            if selected is None:
                raise RuntimeError(f"Requested class {name!r} does not exist in the model")
            class_ids = [class_id for class_id, class_name in selected.names.items()
                         if class_name == name]
            if not class_ids:
                raise RuntimeError(f"Requested class {name!r} does not exist in the model")
            key = id(selected)
            if key not in routes:
                routes[key] = (selected, {})
            for class_id in class_ids:
                routes[key][1][class_id] = name
        self._routes = list(routes.values())

    @property
    def class_ids(self) -> list[int]:
        with self._lock:
            self._prepare()
            return list(dict.fromkeys(class_id for _, names in self._routes
                                      for class_id in names))

    def detect(self, frame: np.ndarray) -> list[dict]:
        with self._lock:
            self._prepare()
            detections = []
            for model, names in self._routes:
                result = model.predict(
                    frame,
                    conf=self.settings.confidence,
                    iou=self.settings.iou,
                    imgsz=self.settings.imgsz,
                    classes=list(names),
                    device=self.settings.device,
                    verbose=False,
                )[0]
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id not in names:
                        continue
                    detections.append({
                        "class": names[class_id],
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                    })
            return sorted(detections, key=lambda item: item["confidence"], reverse=True)
