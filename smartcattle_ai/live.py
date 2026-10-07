"""Live camera service: YOLO on the camera stream, MJPEG and JSON over HTTP, status and events to the backend."""

import json
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event

import cv2

from .backend_client import BackendError
from .media import draw_detections
from .rules import evaluate

DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:4173", "http://127.0.0.1:4173")
_BACKEND_STATUS = {"online": "online", "error": "error"}


class LatestFrame:
    """Keeps only the newest camera frame, so detection never falls behind the stream."""

    def __init__(self):
        self._cond = Condition()
        self._frame = None

    def put(self, frame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify_all()

    def take(self, timeout: float):
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame


class LiveState:
    """Latest annotated JPEG and detections, shared by the worker and every HTTP client."""

    def __init__(self, camera_id: str, zone: tuple, public_url: str | None = None):
        self.camera_id = camera_id
        self.zone = zone
        # Base URL where browsers reach this server (e.g. a tunnel); reported to the backend while online.
        self.public_url = public_url
        self._cond = Condition()
        self._seq = 0
        self._jpeg = None
        self._detections = []
        self._captured_at = None
        self._size = None
        self._fps = None
        self._status = "connecting"
        self._error = None

    def set_camera_status(self, status: str, error: str | None = None) -> None:
        with self._cond:
            self._status = status
            self._error = error if status == "error" else None

    def publish(self, jpeg: bytes, detections: list[dict], width: int, height: int, fps: float | None) -> None:
        with self._cond:
            self._seq += 1
            self._jpeg = jpeg
            self._detections = detections
            self._captured_at = datetime.now(timezone.utc)
            self._size = (width, height)
            self._fps = fps
            self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout: float) -> tuple[int, bytes] | None:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq, timeout)
            if self._seq <= after_seq or self._jpeg is None:
                return None
            return self._seq, self._jpeg

    def latest_jpeg(self) -> bytes | None:
        with self._cond:
            return self._jpeg

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "camera_id": self.camera_id, "status": self._status, "error": self._error,
                "width": self._size[0] if self._size else None,
                "height": self._size[1] if self._size else None,
                "fps": self._fps, "zone": list(self.zone),
                "captured_at": self._captured_at.isoformat() if self._captured_at else None,
                "detections": self._detections,
            }


def classify_zone(detections: list[dict], width: int, height: int, zone: tuple) -> list[dict]:
    """Cows get inside_zone; persons get None because the safe zone only applies to cattle."""
    cows, _ = evaluate([item for item in detections if item["class"] == "cow"], width, height, zone)
    others = [{**item, "inside_zone": None} for item in detections if item["class"] != "cow"]
    return sorted(cows + others, key=lambda item: item["confidence"], reverse=True)


def detection_worker(detector, frames: LatestFrame, state: LiveState, stop: Event,
                     publisher=None, jpeg_quality: int = 80) -> None:
    fps = None
    last = None
    while not stop.is_set():
        frame = frames.take(timeout=1.0)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        detections = classify_zone(detector.detect(frame), width, height, state.zone)
        if publisher is not None:
            publisher.handle(detections, width, height, time.time())
        ok, encoded = cv2.imencode(".jpg", draw_detections(frame, detections, state.zone),
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        now = time.monotonic()
        if last is not None and now > last:
            current = 1 / (now - last)
            fps = round(current if fps is None else 0.8 * fps + 0.2 * current, 2)
        last = now
        if ok:
            state.publish(encoded.tobytes(), detections, width, height, fps)


def capture_worker(stream, frames: LatestFrame, stop: Event) -> None:
    for frame in stream.frames():
        frames.put(frame)
        if stop.is_set():
            break
    stream.release()


def status_report(state: LiveState) -> dict | None:
    """Backend camera status body, or None while still connecting."""
    snapshot = state.snapshot()
    status = _BACKEND_STATUS.get(snapshot["status"])
    if status is None:
        return None
    error = snapshot["error"]
    if error and "://" in error:  # The backend rejects anything that could carry stream credentials.
        error = "Stream error"
    report: dict = {"status": status, "observed_at": datetime.now(timezone.utc).isoformat()}
    if status == "error":
        report["error"] = (error or "Stream error")[:300]
    elif state.public_url:
        report["stream_url"] = state.public_url
    if snapshot["width"]:
        report["frame_width"], report["frame_height"] = snapshot["width"], snapshot["height"]
    if snapshot["fps"]:
        report["fps"] = min(snapshot["fps"], 240)
    return report


def status_worker(client, state: LiveState, stop: Event, interval_s: float) -> None:
    """Reports on every status change right away, and otherwise every interval_s as a heartbeat."""
    warned = False
    last_status = None
    last_sent = float("-inf")
    while not stop.is_set():
        report = status_report(state)
        if report is not None and (report["status"] != last_status or time.monotonic() - last_sent >= interval_s):
            # Recorded even on failure, so a down backend is retried on the heartbeat, not every second.
            last_status, last_sent = report["status"], time.monotonic()
            try:
                client.report_camera_status(state.camera_id, report)
                warned = False
            except BackendError as exc:
                if not warned:
                    print(f"Warning: could not report camera status: {exc}", file=sys.stderr)
                warned = True
        stop.wait(1.0)


def make_handler(state: LiveState, stop: Event, allowed_origins):
    origins = {origin.rstrip("/") for origin in allowed_origins}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _cors(self) -> None:
            origin = self.headers.get("Origin", "").rstrip("/")
            if origin in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, body: dict) -> None:
            self._send(status, json.dumps(body).encode("utf-8"), "application/json")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/status":
                self._json(200, state.snapshot())
            elif path == "/snapshot.jpg":
                jpeg = state.latest_jpeg()
                if jpeg is None:
                    self._json(503, {"detail": "No frame yet"})
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/video.mjpg":
                self._stream()
            else:
                self._json(404, {"detail": "Not found"})

        def _stream(self) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while not stop.is_set():
                    item = state.wait_frame(seq, timeout=1.0)
                    if item is None:
                        continue
                    seq, jpeg = item
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                     + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def make_server(state: LiveState, stop: Event, host: str, port: int, allowed_origins=DEFAULT_ORIGINS):
    server = ThreadingHTTPServer((host, port), make_handler(state, stop, allowed_origins))
    server.daemon_threads = True
    return server
