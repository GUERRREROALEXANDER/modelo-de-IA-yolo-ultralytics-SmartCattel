"""Read, annotate, and process image and video files."""

from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def read_image(path) -> np.ndarray:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise ValueError(f"Not a valid image: {path}") from exc
    if frame is None:
        raise ValueError(f"Not a valid image: {path}")
    return frame


def draw_detections(frame: np.ndarray, detections: list[dict], zone=None) -> np.ndarray:
    annotated = frame.copy()
    if zone is not None:
        height, width = frame.shape[:2]
        left, top, right, bottom = zone
        cv2.rectangle(annotated, (round(left * width), round(top * height)),
                      (round(right * width), round(bottom * height)), (255, 0, 0), 2)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
        outside = detection.get("inside_zone") is False
        color = (0, 0, 255) if outside else (0, 165, 255) if detection["class"] == "person" else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{detection["class"]} {detection["confidence"]:.2f}' + (" OUT" if outside else "")
        cv2.putText(annotated, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cattle = sum(item["class"] == "cow" for item in detections)
    persons = sum(item["class"] == "person" for item in detections)
    cv2.putText(annotated, f"Cattle: {cattle} | Persons: {persons}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated


def save_image(path, frame: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, frame)
    if not ok:
        raise ValueError(f"Could not encode image: {target}")
    encoded.tofile(str(target))


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def process_image(detector, path, output_dir=None, zone=None) -> dict:
    source = Path(path)
    frame = read_image(source)
    height, width = frame.shape[:2]
    detections = detector.detect(frame)
    output = None
    if output_dir is not None:
        output = Path(output_dir) / f"{source.stem}_annotated{source.suffix}"
        save_image(output, draw_detections(frame, detections, zone))
    return {
        "source": str(source), "width": width, "height": height,
        "total_cattle": sum(item["class"] == "cow" for item in detections),
        "total_persons": sum(item["class"] == "person" for item in detections),
        "detections": detections,
        "output": str(output) if output else None,
    }


def process_video(detector, path, output_dir=None, vid_stride=1,
                  max_frames=None, on_frame=None, zone=None) -> dict:
    if vid_stride < 1:
        raise ValueError("vid_stride must be at least 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        frames_read = 0
        width = height = 0
        output = None
        while max_frames is None or frames_read < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            if (frames_read - 1) % vid_stride:
                continue
            height, width = frame.shape[:2]
            detections = detector.detect(frame)
            item = {"frame": frames_read, "time_s": (frames_read - 1) / fps,
                    "width": width, "height": height,
                    "total_cattle": sum(item["class"] == "cow" for item in detections),
                    "total_persons": sum(item["class"] == "person" for item in detections),
                    "detections": detections}
            frames.append(item)
            if on_frame is not None:
                on_frame(item)
            if output_dir is not None:
                if writer is None:
                    output = Path(output_dir) / f"{source.stem}_annotated.mp4"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / vid_stride, (width, height))
                    if not writer.isOpened():
                        raise ValueError(f"Could not write video: {output}")
                writer.write(draw_detections(frame, detections, zone))
        counts = [item["total_cattle"] for item in frames]
        person_counts = [item["total_persons"] for item in frames]
        return {
            "source": str(source), "frames_read": frames_read,
            "frames_processed": len(frames), "fps": fps,
            "width": width, "height": height,
            "total_cattle": sum(counts), "total_persons": sum(person_counts),
            "max_cattle": max(counts, default=0),
            "max_persons": max(person_counts, default=0),
            "avg_cattle": sum(counts) / len(counts) if counts else 0.0,
            "frames_with_cattle": sum(count > 0 for count in counts),
            "output": str(output) if output else None, "frames": frames,
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
