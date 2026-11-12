"""Standing or lying, from how the animal looks: a small classifier run on the box of each cow.

The shape of a box alone cannot tell a cow lying side-on far from one standing in the distance, and it says nothing
about one lying facing the camera. See reports/posture_evaluation.json for what each approach measures on real cattle.
"""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "cattle_posture_cls.pt"
#: The crop keeps some of the ground around the animal: the legs and what they stand on are what tell the two apart.
CONTEXT = 0.10
INPUT_SIZE = 128
LYING_CLASS = "lying"
STANDING_CLASS = "standing"
#: Below this the classifier is guessing, so the box shape and the history decide instead.
MIN_CONFIDENCE = 0.7


def square_crop(frame: np.ndarray, bbox, size: int = INPUT_SIZE) -> np.ndarray:
    """The box with some context, padded to a square and resized. Padding keeps the proportions of the body."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    margin = CONTEXT * max(x2 - x1, y2 - y1)
    left, top = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    right, bottom = int(min(width, x2 + margin)), int(min(height, y2 + margin))
    crop = frame[top:max(bottom, top + 1), left:max(right, left + 1)]
    side = max(crop.shape[:2])
    pad_y, pad_x = side - crop.shape[0], side - crop.shape[1]
    square = cv2.copyMakeBorder(crop, pad_y // 2, pad_y - pad_y // 2, pad_x // 2, pad_x - pad_x // 2,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)


def as_infrared_night(image: np.ndarray, seed: int = 0) -> np.ndarray:
    """A picture as a camera in night mode would give it: grey, dim, low in contrast and noisy.

    A stand-in for real night footage, of which the labelled set has almost none.
    """
    rng = np.random.default_rng(seed)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dim = grey * rng.uniform(0.25, 0.55) + rng.uniform(2.0, 12.0)
    noisy = dim + rng.normal(0.0, rng.uniform(3.0, 10.0), grey.shape)
    return cv2.cvtColor(np.clip(noisy, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def load_default_classifier(device: str = "cpu") -> "PostureClassifier | None":
    """The trained classifier, or None on a copy without it: the box's shape and history still answer then."""
    return PostureClassifier(DEFAULT_WEIGHTS, device) if DEFAULT_WEIGHTS.is_file() else None


class PostureClassifier:
    """Loaded on the first crop, like the detector: a service with no camera must not pay for it."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str = "cpu", model=None):
        self.weights = str(weights)
        self.device = device
        self._model = model
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            from .model_loader import load_model

            self._model = load_model(self.weights)
        return self._model

    def predict(self, frame: np.ndarray, boxes: list) -> list[tuple[str, float]]:
        """(class, confidence) for each box, in order."""
        if not boxes:
            return []
        crops = [square_crop(frame, bbox) for bbox in boxes]
        with self._lock:
            model = self._load()
            results = model.predict(crops, imgsz=INPUT_SIZE, device=self.device, verbose=False)
        answers = []
        for result in results:
            top = int(result.probs.top1)
            answers.append((str(result.names[top]), float(result.probs.top1conf)))
        return answers
