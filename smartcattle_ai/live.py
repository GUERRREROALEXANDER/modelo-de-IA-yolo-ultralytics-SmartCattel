"""Live camera service: YOLO on the camera stream, MJPEG and JSON over HTTP; status, head count,
escapes and safe zones exchanged with the backend."""

import json
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event

import cv2
import numpy as np

from .backend_client import BackendError, occupancy_report
from .media import draw_detections
from .rules import evaluate
from .zones import ZoneSet, as_zone_set

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

    def __init__(self, camera_id: str, zone, public_url: str | None = None):
        self.camera_id = camera_id
        # Replaced as a whole when the dashboard's zones arrive; readers take one reference, never half of one.
        self.zone = as_zone_set(zone)
        # Optional gate line counter (--gate-line); only the detection thread updates it.
        self.gate = None
        # Optional heat map; the detection thread adds to it, /heatmap.jpg reads a copy of its grid.
        self.heat = None
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
                "fps": self._fps,
                # "zone" is the first rectangle, kept for clients written before zones came from the backend.
                "zone": list(self.zone.describe()[0]["rect"]), "zones": self.zone.describe(),
                "zone_source": self.zone.source,
                "gate": self.gate.describe() if self.gate is not None else None,
                "heatmap_samples": self.heat.samples if self.heat is not None else None,
                "captured_at": self._captured_at.isoformat() if self._captured_at else None,
                "detections": self._detections,
            }


def classify_zone(detections: list[dict], width: int, height: int, zone) -> list[dict]:
    """Cows get inside_zone; persons get None because the safe zone only applies to cattle."""
    cows, _ = evaluate([item for item in detections if item["class"] == "cow"], width, height, zone)
    others = [{**item, "inside_zone": None} for item in detections if item["class"] != "cow"]
    return sorted(cows + others, key=lambda item: item["confidence"], reverse=True)


#: Animals worth a console warning when they show up: they can scatter or attack the herd.
VISITORS = frozenset({"dog", "bear"})


class VisitorLog:
    """Warns once per tracked dog or bear, not once per frame it stays in view."""

    def __init__(self, classes=VISITORS):
        self.classes = frozenset(classes)
        self._seen: set[int] = set()

    def new(self, detections: list[dict]) -> list[dict]:
        found = []
        for item in detections:
            track_id = item.get("track_id")
            if item["class"] in self.classes and track_id is not None and track_id not in self._seen:
                self._seen.add(track_id)
                found.append(item)
        return found


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def mark_intruders(night, detections: list[dict], now: datetime, camera_id: str) -> list[dict]:
    """During the night window, flags every tracked person with night=True and warns once per person."""
    for item in night.new_intruders(detections, now):
        print(f"Warning: camera {camera_id}, person #{item['track_id']} in view at night "
              f"({item['confidence']:.2f})", flush=True)
    if not night.is_night(now):
        return detections
    return [{**item, "night": True} if item["class"] == "person" and item.get("track_id") is not None else item
            for item in detections]


def mark_still(stillness, detections: list[dict], width: int, height: int, now: float,
               camera_id: str) -> list[dict]:
    """Adds still_for_s to tracked cows, and still=True once past the limit; warns once per still period.

    The backend has no event type for this yet, so the warning stays here: on the
    video, in /status and in the console.
    """
    detections = stillness.update(detections, width, height, now)
    for track_id in stillness.new_alerts(detections):
        minutes = round(stillness.still_after_s / 60)
        print(f"Warning: camera {camera_id}, cow #{track_id} has not moved for {minutes} min", flush=True)
    return [{**item, "still": True} if item.get("still_for_s", 0) >= stillness.still_after_s else item
            for item in detections]


def detection_worker(detector, frames: LatestFrame, state: LiveState, stop: Event,
                     publisher=None, jpeg_quality: int = 80, tracker=None, stillness=None,
                     visitors=None, night=None, clock=None) -> None:
    fps = None
    last = None
    while not stop.is_set():
        frame = frames.take(timeout=1.0)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        zone = state.zone
        detections = detector.detect(frame)
        if tracker is not None:
            detections = tracker.update(detections)
        detections = classify_zone(detections, width, height, zone)
        if stillness is not None:
            detections = mark_still(stillness, detections, width, height, time.time(), state.camera_id)
        if night is not None:
            detections = mark_intruders(night, detections, (clock or _utc_now)(), state.camera_id)
        if visitors is not None:
            for item in visitors.new(detections):
                print(f"Warning: camera {state.camera_id}, {item['class']} #{item['track_id']} in view "
                      f"({item['confidence']:.2f})", flush=True)
        if state.heat is not None:
            state.heat.add(detections, width, height)
        if state.gate is not None:
            for crossing in state.gate.update(detections, width, height, time.time()):
                print(f"Gate: cow #{crossing['track_id']} went {crossing['direction']} "
                      f"(in {state.gate.entered}, out {state.gate.exited})", flush=True)
        if publisher is not None:
            publisher.handle(detections, width, height, time.time())
        ok, encoded = cv2.imencode(".jpg", draw_detections(frame, detections, zone, state.gate),
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


def occupancy_worker(client, state: LiveState, stop: Event, interval_s: float) -> None:
    """Sends the latest head count, at most every interval_s and only for a frame not sent yet.

    A separate thread, so a slow backend never slows detection down. The backend
    keeps one sample per minute for history and pushes every report to the
    open dashboards, so a second is a good default.
    """
    warned = False
    last_sent = None
    while not stop.wait(interval_s):
        snapshot = state.snapshot()
        captured_at = snapshot["captured_at"]
        if snapshot["status"] != "online" or captured_at is None or captured_at == last_sent:
            continue
        report = occupancy_report(snapshot["detections"], snapshot["width"], snapshot["height"], captured_at)
        last_sent = captured_at
        try:
            client.report_occupancy(state.camera_id, report)
            warned = False
        except BackendError as exc:
            if not warned:
                print(f"Warning: could not report the head count: {exc}", file=sys.stderr)
            warned = True


def sync_zones(client, state: LiveState, fallback: ZoneSet, publisher=None) -> bool:
    """Loads the camera's zones from the backend once; True if they changed.

    No active zone in the dashboard means the configured one (--zone) applies:
    a camera with no zone at all would call every animal an escape.
    """
    zones = ZoneSet.from_backend(client.camera_zones(state.camera_id)) or fallback
    if zones == state.zone:
        return False
    state.zone = zones
    if publisher is not None:
        publisher.zone = zones
    names = ", ".join(area.name or "unnamed" for area in zones.areas)
    print(f"Safe zones from {zones.source}: {names}", flush=True)
    return True


def zone_sync_worker(client, state: LiveState, fallback: ZoneSet, stop: Event, interval_s: float,
                     publisher=None) -> None:
    """Polls the dashboard's zones, so drawing a new one applies within interval_s, without a restart."""
    warned = False
    while not stop.is_set():
        try:
            sync_zones(client, state, fallback, publisher)
            warned = False
        except (BackendError, KeyError, TypeError, ValueError) as exc:
            # The zone in use stays: a backend that is down must not erase it.
            if not warned:
                print(f"Warning: could not load safe zones: {exc}", file=sys.stderr)
            warned = True
        stop.wait(interval_s)


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
            elif path == "/heatmap.jpg":
                self._heatmap()
            else:
                self._json(404, {"detail": "Not found"})

        def _heatmap(self) -> None:
            """Where the cattle have stood since the service started, over the latest frame."""
            if state.heat is None:
                self._json(404, {"detail": "Heat map is off"})
                return
            jpeg = state.latest_jpeg()
            if jpeg is None:
                self._json(503, {"detail": "No frame yet"})
                return
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            ok, encoded = cv2.imencode(".jpg", state.heat.overlay(frame))
            if not ok:
                self._json(500, {"detail": "Could not encode the heat map"})
                return
            self._send(200, encoded.tobytes(), "image/jpeg")

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
