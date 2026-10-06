"""Print detections from a camera index or video path."""

import json

import cv2


def run_stream(detector, source) -> None:
    if isinstance(source, str) and source.isdecimal():
        source = int(source)
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open stream: {source}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            print(json.dumps(detector.detect(frame)))
    finally:
        capture.release()
