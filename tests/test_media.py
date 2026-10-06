import cv2
import numpy as np
import pytest

from smartcattle_ai.media import (
    draw_detections, process_video, read_image, save_image,
)


class FakeDetector:
    def detect(self, frame):
        return [{"class": "cow", "confidence": 0.91, "bbox": [2, 2, 20, 20]}]


def test_image_path_with_spaces_and_accent(tmp_path):
    path = tmp_path / "imagen vaca á.jpg"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    save_image(path, image)
    assert read_image(path).shape == image.shape


def test_invalid_image(tmp_path):
    path = tmp_path / "invalid.jpg"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Not a valid image"):
        read_image(path)


def test_draw_does_not_change_frame():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    annotated = draw_detections(frame, FakeDetector().detect(frame))
    assert not np.array_equal(annotated, frame)
    assert not frame.any()


def test_process_video_with_stride(tmp_path):
    source = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (64, 64))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((64, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    result = process_video(FakeDetector(), source, tmp_path / "outputs", vid_stride=2)
    assert result["frames_read"] == 10
    assert result["frames_processed"] == 5
    assert [item["frame"] for item in result["frames"]] == [1, 3, 5, 7, 9]
    assert result["max_cattle"] == result["avg_cattle"] == 1
    assert result["frames_with_cattle"] == 5
    output = cv2.VideoCapture(result["output"])
    try:
        assert output.isOpened()
        assert int(output.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert output.read()[0]
    finally:
        output.release()
