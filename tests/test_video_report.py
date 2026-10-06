"""Offline video report coverage using synthetic pixels and a fake detector."""

import cv2
import numpy as np
import pytest

from scripts import video_report
from smartcattle_ai.media import read_image


def test_video_report_table_and_artifacts(tmp_path, monkeypatch, capsys):
    source = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(6):
            writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
    finally:
        writer.release()
    second = tmp_path / "second.mp4"
    second.write_bytes(source.read_bytes())
    settings_seen = []

    class FakeDetector:
        def __init__(self, settings):
            settings_seen.append(settings)
            self.calls = 0

        def detect(self, frame):
            cows, persons = [(0, 1), (2, 0), (1, 2)][self.calls]
            self.calls += 1
            return [{"class": name, "confidence": 0.9, "bbox": [2, 2, 20, 20]}
                    for name, count in (("cow", cows), ("person", persons))
                    for _ in range(count)]

    monkeypatch.setattr(video_report, "ROOT", tmp_path)
    monkeypatch.setattr(video_report, "Detector", FakeDetector)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("smartcattle_ai.detector.load_model",
                        lambda _: pytest.fail("The report must not load YOLO in this test"))
    times = iter([0.0, 0.3, 1.0, 1.3, 2.0, 2.3, 3.0, 3.3])
    monkeypatch.setattr(video_report, "perf_counter", lambda: next(times))
    assert video_report.main(["--videos", str(source), "--videos", str(second),
                              "--weights", "first.pt", "--weights", "other.pt"]) == 0
    markdown = (tmp_path / "reports" / "video_report.md").read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "| Frames processed | Max cattle | Avg cattle | Max persons | Avg persons |" in markdown
    assert len(settings_seen) == 4
    assert all(settings.class_names == ("cow", "person") for settings in settings_seen)
    for weight in ("first", "other"):
        for video in (source, second):
            expected = f"| {weight}.pt | {video} | 3 | 2 | 1.00 | 2 | 1.00 | 2 | 100.00 |"
            assert expected in markdown
            assert expected in printed
            sheet = tmp_path / "reports" / "video_frames" / f"{weight}_{video.stem}.jpg"
            assert f"[Contact sheet](video_frames/{sheet.name})" in markdown
            image = read_image(sheet)
            assert image.shape == (1440, 640, 3)
            # First/middle/last frames remain temporally ordered in the contact sheet.
            brightness = [image[index * 480 + 400:index * 480 + 460, 500:600].mean()
                          for index in range(3)]
            assert brightness[0] < brightness[1] < brightness[2]
            assert (tmp_path / "outputs" / "videos" / weight / f"{video.stem}_annotated.mp4").is_file()


def test_format_table_escapes_pipes():
    table = video_report.format_table([{
        "weights": "weights|one.pt", "video": "video|one.mp4", "frames_processed": 0,
        "max_cattle": 0, "avg_cattle": 0, "max_persons": 0, "avg_persons": 0,
        "frames_with_cattle": 0, "mean_ms": 0,
    }])
    assert "weights\\|one.pt | video\\|one.mp4 | 0 | 0 | 0.00 | 0 | 0.00 | 0 | 0.00" in table
