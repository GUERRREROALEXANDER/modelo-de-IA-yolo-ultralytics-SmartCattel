from pathlib import Path

import pytest
from ultralytics.utils import ASSETS

from smartcattle_ai import Detector, Settings
from smartcattle_ai.media import read_image


@pytest.mark.real
def test_real_yolo_on_bus_image():
    frame = read_image(ASSETS / "bus.jpg")
    detector = Detector(Settings(device="cpu"))
    detections = detector.detect(frame)
    assert any(item["class"] == "person" for item in detections)
    assert all(item["class"] in {"cow", "person"} for item in detections)


@pytest.mark.real
def test_real_yolo_on_cattle_sample_if_available():
    samples = sorted(Path("data/samples").glob("*.jpg"))
    if not samples:
        pytest.skip("No cattle JPG samples are available in data/samples")
    detector = Detector(Settings(device="cpu"))
    assert any(detector.detect(read_image(path)) for path in samples)
