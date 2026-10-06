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
    def __init__(self, names, boxes=()):
        self.names = names
        self.boxes = boxes
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=[FakeBox(confidence, class_id)
                                       for confidence, class_id in self.boxes])]


@pytest.mark.parametrize("confidences", [(), (0.91,), (0.5, 0.95, 0.7)])
def test_detection_counts_and_sorting(confidences):
    model = FakeModel({0: "person", 19: "cow"}, [(value, 19) for value in confidences])
    detector = Detector(Settings(class_names=("cow",), device="cpu"), model)
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(detections) == len(confidences)
    assert [item["confidence"] for item in detections] == sorted(confidences, reverse=True)
    assert all(item["class"] == "cow" for item in detections)
    assert all(item["bbox"] == [1.2, 2.3, 30.5, 40.6] for item in detections)
    assert detector.class_ids == [19]
    assert model.calls == [{
        "conf": 0.35, "iou": 0.5, "imgsz": 640,
        "classes": [19], "device": "cpu", "verbose": False,
    }]


def test_primary_cow_and_fallback_person_filter_other_classes():
    primary = FakeModel({0: "cow"}, [(0.8, 0)])
    person = FakeModel({0: "person", 19: "cow", 2: "car"},
                       [(0.95, 0), (0.99, 19), (0.75, 2)])
    detector = Detector(Settings(weights="cattle.pt", device="cpu"),
                        model=primary, person_model=person)
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert [(item["class"], item["confidence"]) for item in detections] == [
        ("person", 0.95), ("cow", 0.8),
    ]
    assert primary.calls[0]["classes"] == [0]
    assert person.calls[0]["classes"] == [0]


def test_same_weights_uses_one_model_and_one_prediction():
    model = FakeModel({0: "person", 19: "cow", 2: "car"},
                      [(0.9, 0), (0.8, 19), (0.7, 2)])
    detector = Detector(Settings(weights="yolo11n.pt", device="cpu"), model=model)
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert {item["class"] for item in detections} == {"cow", "person"}
    assert len(model.calls) == 1
    assert model.calls[0]["classes"] == [19, 0]


def test_coco_primary_never_loads_person_weights(monkeypatch):
    primary = FakeModel({0: "person", 19: "cow"}, [(0.9, 0), (0.8, 19)])
    loaded = []

    def fake_load(weights):
        loaded.append(weights)
        assert weights == "cattle_coco_yolo11n_best.pt"
        return primary

    monkeypatch.setattr("smartcattle_ai.detector.load_model", fake_load)
    detector = Detector(Settings(weights="cattle_coco_yolo11n_best.pt",
                                 person_weights="people.pt", device="cpu"))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(2):
        detections = detector.detect(frame)
        assert {item["class"] for item in detections} == {"cow", "person"}
    assert loaded == ["cattle_coco_yolo11n_best.pt"]
    assert detector._person_model is None
    assert len(primary.calls) == 2
    assert all(call["classes"] == [19, 0] for call in primary.calls)


def test_person_model_loads_lazily_and_is_cached(monkeypatch):
    primary = FakeModel({0: "cow"}, [(0.8, 0)])
    person = FakeModel({0: "person"}, [(0.9, 0)])
    loaded = []

    def fake_load(weights):
        loaded.append(weights)
        return person

    monkeypatch.setattr("smartcattle_ai.detector.load_model", fake_load)
    detector = Detector(Settings(weights="cattle.pt", person_weights="people.pt", device="cpu"),
                        model=primary)
    assert loaded == []
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detector.detect(frame)
    detector.detect(frame)
    assert loaded == ["people.pt"]
    assert len(primary.calls) == len(person.calls) == 2


def test_unavailable_requested_class_raises():
    primary = FakeModel({0: "cow"})
    fallback = FakeModel({19: "cow"})
    detector = Detector(Settings(weights="cattle.pt", device="cpu"),
                        model=primary, person_model=fallback)
    with pytest.raises(RuntimeError, match="person"):
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
