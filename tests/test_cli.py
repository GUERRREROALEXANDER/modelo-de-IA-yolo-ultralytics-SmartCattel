import numpy as np

import detect
from smartcattle_ai.media import save_image


class FakeDetector:
    def __init__(self, settings):
        self.settings = settings

    def detect(self, frame):
        return [{"class": "cow", "confidence": 0.91, "bbox": [1, 2, 30, 40]}]


def test_missing_source(capsys, tmp_path):
    assert detect.main(["--source", str(tmp_path / "missing.jpg")]) == 1
    assert "Source does not exist" in capsys.readouterr().err


def test_image_with_fake_detector(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(detect, "Detector", FakeDetector)
    source = tmp_path / "cow.jpg"
    save_image(source, np.zeros((64, 64, 3), dtype=np.uint8))
    assert detect.main(["--source", str(source), "--no-save"]) == 0
    output = capsys.readouterr().out
    assert "Cattle: 1" in output
    assert "cow  conf=0.91 bbox=[1, 2, 30, 40]" in output
