from types import SimpleNamespace

import numpy as np
import pytest

from smartcattle_ai import Detector, Settings


class FakeBox:
    def __init__(self, confidence, class_id=19):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([[1.234, 2.345, 30.456, 40.567]])


class FakeModel:
    names = {0: "person", 19: "cow"}

    def __init__(self, confidences=()):
        self.confidences = confidences
        self.kwargs = None

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=[FakeBox(value) for value in self.confidences])]


@pytest.mark.parametrize("confidences", [(), (0.91,), (0.5, 0.95, 0.7)])
def test_detection_counts_and_sorting(confidences):
    model = FakeModel(confidences)
    detector = Detector(Settings(device="cpu"), model)
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(detections) == len(confidences)
    assert [item["confidence"] for item in detections] == sorted(confidences, reverse=True)
    assert all(item["class"] == "cow" for item in detections)
    assert all(item["bbox"] == [1.2, 2.3, 30.5, 40.6] for item in detections)
    assert detector.class_ids == [19]
    assert model.kwargs == {
        "conf": 0.35, "iou": 0.5, "imgsz": 640,
        "classes": [19], "device": "cpu", "verbose": False,
    }


def test_missing_class_lists_available_names():
    detector = Detector(Settings(class_names=("horse",), device="cpu"), FakeModel())
    with pytest.raises(RuntimeError, match="person, cow"):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
