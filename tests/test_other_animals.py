"""Dogs, horses, sheep and bears: detected on request, never counted as cattle."""

import numpy as np
import pytest

from smartcattle_ai import Settings
from smartcattle_ai.backend_client import occupancy_report
from smartcattle_ai.config import ALLOWED_CLASSES
from smartcattle_ai.live import VisitorLog, classify_zone
from smartcattle_ai.media import draw_detections


def animal(cls, track_id=None, x=10):
    return {"class": cls, "confidence": 0.8, "bbox": [x, 10, x + 20, 30], "track_id": track_id}


def test_farm_animals_can_be_requested_but_cattle_and_people_stay_the_default(monkeypatch):
    assert Settings(device="cpu").class_names == ("cow", "person")
    monkeypatch.setenv("SMARTCATTLE_CLASSES", "cow,person,dog,horse")
    monkeypatch.setenv("SMARTCATTLE_DEVICE", "cpu")
    assert Settings.from_env().class_names == ("cow", "person", "dog", "horse")
    assert set(ALLOWED_CLASSES) == {"cow", "person", "dog", "horse", "sheep", "bear"}


def test_an_unknown_class_is_rejected_with_the_allowed_list():
    with pytest.raises(ValueError, match="dog"):
        Settings(class_names=("cat",), device="cpu")


def test_other_animals_never_count_as_cattle_or_escape():
    detections = classify_zone([animal("cow"), animal("dog"), animal("horse", x=200)], 100, 100, (0, 0, 1, 1))
    assert [item["inside_zone"] for item in detections if item["class"] != "cow"] == [None, None]
    report = occupancy_report(detections, 100, 100, "2026-11-01T00:00:00+00:00")
    assert (report["cattle_inside"], report["cattle_outside"]) == (1, 0)
    assert [box["label"] for box in report["detections"]] == ["cow"]


def test_a_dog_or_bear_is_announced_once_per_animal():
    log = VisitorLog()
    first = log.new([animal("dog", 1), animal("horse", 2), animal("cow", 3), animal("dog", None)])
    assert [(item["class"], item["track_id"]) for item in first] == [("dog", 1)]
    assert log.new([animal("dog", 1)]) == []
    assert [item["track_id"] for item in log.new([animal("dog", 1), animal("bear", 4)])] == [4]


def test_other_animals_have_their_own_color_and_count():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    as_cow = draw_detections(frame, [animal("cow")])
    as_horse = draw_detections(frame, [animal("horse")])
    assert not np.array_equal(as_cow, as_horse)
