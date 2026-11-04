"""Live camera service: YOLO on the camera stream, MJPEG and JSON over HTTP; status, head count,
escapes and safe zones exchanged with the backend."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event, Lock
from urllib.parse import parse_qs

import cv2
import numpy as np

from .backend_client import BackendError, detection_log_report, occupancy_report
from .camera_manager import CameraLimitError, check_reachable_from_cloud, check_source
from .media import draw_detections
from .discovery import local_ipv4, no_answer, scan_cameras
from .limits import RateLimit, SessionCache, StreamSlots, client_address
from .pushed import PUSHED_SOURCE
from .rules import evaluate
from .stream import camera_url_from_env, normalize_source, redact_url
from .tracking import track_frame
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

    # Longer than any frame takes, even on a slow CPU with every camera waiting its turn for the model.
    STALLED_AFTER_S = 180.0

    def __init__(self, camera_id: str, zone, public_url: str | None = None):
        self.working_since: float | None = None
        self.camera_id = camera_id
        # Replaced as a whole when the dashboard's zones arrive; readers take one reference, never half of one.
        self.zone = as_zone_set(zone)
        # Optional gate line counter (--gate-line); only the detection thread updates it.
        self.gate = None
        # Optional posture estimate; only the detection thread uses it.
        self.posture = None
        # Optional sender of what must reach a person; only the detection thread uses it.
        self.anomalies = None
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

    def stalled(self, now: float | None = None) -> bool:
        """A frame has been in the detector for too long. Only a restart gets out of that."""
        started = self.working_since
        return started is not None and (time.monotonic() if now is None else now) - started > self.STALLED_AFTER_S

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
    failed_with = None
    while not stop.is_set():
        frame = frames.take(timeout=1.0)
        if frame is None:
            continue
        # A worker that dies leaves a frozen picture and a service that says it is fine: one bad frame must not do that.
        state.working_since = time.monotonic()
        try:
            fps, last = _process_frame(frame, detector, state, publisher, jpeg_quality, tracker, stillness, visitors,
                                       night, clock, fps, last)
            failed_with = None
        except Exception as exc:
            if type(exc) is not failed_with:  # Once per kind of failure, not once per frame.
                print(f"Warning: camera {state.camera_id}, frame skipped: {type(exc).__name__}", file=sys.stderr,
                      flush=True)
            failed_with = type(exc)
            stop.wait(1.0)
        finally:
            state.working_since = None


def _process_frame(frame, detector, state, publisher, jpeg_quality, tracker, stillness, visitors, night, clock,
                   fps, last):
    height, width = frame.shape[:2]
    zone = state.zone
    detections = track_frame(detector, tracker, frame) if tracker is not None else detector.detect(frame)
    detections = classify_zone(detections, width, height, zone)
    if state.posture is not None:
        detections = state.posture.update(detections, width, height, time.time(), frame)
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
    if state.anomalies is not None:
        state.anomalies.handle(detections, time.time())
    ok, encoded = cv2.imencode(".jpg", draw_detections(frame, detections, zone, state.gate),
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    now = time.monotonic()
    if last is not None and now > last:
        current = 1 / (now - last)
        fps = round(current if fps is None else 0.8 * fps + 0.2 * current, 2)
    last = now
    if ok:
        state.publish(encoded.tobytes(), detections, width, height, fps)
    return fps, last


def capture_worker(stream, frames: LatestFrame, stop: Event) -> None:
    while not stop.is_set():
        try:
            for frame in stream.frames():
                frames.put(frame)
                if stop.is_set():
                    break
            break
        except Exception as exc:  # The stream reconnects by itself; this is for what it did not foresee.
            print(f"Warning: camera capture restarted after {type(exc).__name__}", file=sys.stderr, flush=True)
            stream.release()
            stop.wait(1.0)
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


def source_worker(client, camera_id: str, source: str, stop: Event, retry_s: float = 30.0) -> None:
    """Records in the backend where this camera is reached, retrying while the backend is down.

    `source` must already be free of credentials. A backend older than this endpoint answers 404: the
    camera works the same without the record, so that is not retried.
    """
    warned = False
    while not stop.is_set():
        try:
            client.report_camera_source(camera_id, source)
            return
        except BackendError as exc:
            if not warned:
                print(f"Warning: could not record camera {camera_id} in the backend: {exc}", file=sys.stderr)
            warned = True
            if exc.status in (404, 405, 422):
                return
        stop.wait(retry_s)


def _service_url(state, port: int) -> str | None:
    """Where other devices reach this service: its public address, or else this PC on the network."""
    if state.public_url:
        return state.public_url
    address = local_ipv4()
    return f"http://{address}:{port}" if address else None


def lan_origins(address: str | None, ports=(5173, 4173)) -> list[str]:
    """The dashboard as other devices open it: this PC's address, on the Vite ports."""
    return [f"http://{address}:{port}" for port in ports] if address else []


#: The backend keeps the last report of each minute: this catches every minute, and a failed send gets a second chance.
DETECTION_LOG_INTERVAL_S = 30.0


def detection_log_worker(client, state: LiveState, stop: Event, interval_s: float = DETECTION_LOG_INTERVAL_S) -> None:
    """Feeds the daily report. Empty scenes too: they tell a quiet pasture from a camera that was not watching."""
    warned = False
    last_sent = None
    while not stop.wait(interval_s):
        snapshot = state.snapshot()
        captured_at = snapshot["captured_at"]
        if snapshot["status"] != "online" or captured_at is None or captured_at == last_sent:
            continue
        try:
            client.report_detection_log(state.camera_id, detection_log_report(snapshot["detections"], captured_at))
            last_sent = captured_at
            warned = False
        except BackendError as exc:
            if not warned:
                print(f"Warning: could not report the detection log: {exc}", file=sys.stderr)
            warned = True


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


MAX_CAMERA_BODY = 4096
# A phone's picture. It is decoded in memory, so it stays well under what one worth analysing weighs at most.
MAX_FRAME_BODY = 2 * 1024 * 1024
# A few a second, per address, which is more than the detector can use.
MAX_FRAMES_PER_MINUTE = 900
# Everything a dashboard does in a minute fits well under these: a preview per camera each second and /status polling.
MAX_REQUESTS_PER_MINUTE = 1200
# Each of these makes this service call the backend, or changes what it runs.
MAX_DASHBOARD_CALLS_PER_MINUTE = 30
MAX_STREAMS = 32
MAX_STREAMS_PER_ADDRESS = 12
MAX_CONNECTIONS = 256
# A client that stops reading or sending for this long gives its thread back.
SOCKET_TIMEOUT_S = 20
VIDEO_PATHS = frozenset({"/video.mjpg", "/snapshot.jpg", "/heatmap.jpg"})
_CAMERA_HOST = re.compile(r"[A-Za-z0-9.-]{1,253}")
_CAMERA_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_ADDED_CAMERA_PATH = re.compile(r"/cameras/([A-Za-z0-9_-]{1,64})(/[a-z.]+)")
_REMOVE_CAMERA_PATH = re.compile(r"/cameras/([A-Za-z0-9_-]{1,64})")
_FRAME_PATH = re.compile(r"/cameras/([A-Za-z0-9_-]{1,64})/frame")


def camera_source_from_request(body: dict, defaults=None) -> str:
    """The stream URL for a POST /camera body: a full URL or phone-app address, or an IP camera's parts.

    {"url": "192.168.1.20:8080"}  or  {"host": "192.168.1.30", "username": "admin", "password": "...",
    "port": 554, "path": "/cam/realmonitor?channel=1&subtype=1"}

    With only the host, the user, password, port and path come from `defaults` (this PC's .env, the
    CAMERA_* variables), so a person who already set the camera up here only has to type its IP.
    """
    defaults = os.environ if defaults is None else defaults
    url = body.get("url")
    if isinstance(url, str) and url.strip():
        return normalize_source(url)
    host = body.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Send the camera's IP address (host) or a stream url")

    def given(key: str) -> str:
        value = body.get(key)
        return value.strip() if isinstance(value, str) else ""

    password = body.get("password") if isinstance(body.get("password"), str) and body.get("password") else ""
    password = password or defaults.get("CAMERA_PASSWORD", "")
    if not password:
        raise ValueError("This camera needs its password: none is saved on this PC yet")
    port = body.get("port")
    env = {
        "CAMERA_HOST": host,
        "CAMERA_USERNAME": given("username") or defaults.get("CAMERA_USERNAME", "").strip() or "admin",
        "CAMERA_PASSWORD": password,
        "CAMERA_RTSP_PORT": str(port) if isinstance(port, int) and 0 < port < 65536 else defaults.get("CAMERA_RTSP_PORT", ""),
        "CAMERA_RTSP_PATH": given("path") or defaults.get("CAMERA_RTSP_PATH", ""),
    }
    return camera_url_from_env(env)


def new_camera_source(body: dict, defaults=None) -> str:
    """The stream URL for a POST /cameras body, which adds a camera instead of re-pointing this service's.

    Saved credentials apply only to the host they were saved for: an address typed into the dashboard must
    never receive another camera's password.
    """
    if body.get("browser") is True:
        # It sends its own pictures, so being in the cloud is no obstacle and there is no address to check.
        return PUSHED_SOURCE
    defaults = os.environ if defaults is None else defaults
    host = body.get("host")
    if isinstance(host, str) and host.strip():
        if not _CAMERA_HOST.fullmatch(host.strip()):
            raise ValueError("That is not a valid IP address or host name")
        if host.strip() != defaults.get("CAMERA_HOST", "").strip():
            defaults = {key: value for key, value in defaults.items()
                        if key not in ("CAMERA_USERNAME", "CAMERA_PASSWORD")}
    source = check_source(camera_source_from_request(body, defaults))
    return check_reachable_from_cloud(source) if defaults.get("SMARTCATTLE_CLOUD") == "1" else source


def make_handler(state: LiveState, stop: Event, allowed_origins, camera=None, manager=None, session_check=None,
                 camera_check=None):
    """session_check(authorization) guards the requests that change cameras or look around the network. It is
    set when this service can be reached from the internet, where Origin proves nothing: curl can forge it.
    camera_check(authorization, camera_id) says whether that person's account owns the camera; without one, nobody's
    does, because a valid session alone must not open another account's camera."""
    origins = {origin.rstrip("/") for origin in allowed_origins}
    public = session_check is not None
    session = SessionCache(session_check) if public else None

    def owns(key: str) -> bool:
        authorization, _, camera_id = key.rpartition("|")
        return camera_check is not None and camera_check(authorization, camera_id)

    ownership = SessionCache(owns) if public else None
    requests = RateLimit(MAX_REQUESTS_PER_MINUTE)
    dashboard_calls = RateLimit(MAX_DASHBOARD_CALLS_PER_MINUTE)
    pushed_frames = RateLimit(MAX_FRAMES_PER_MINUTE)
    streams = StreamSlots(MAX_STREAMS, MAX_STREAMS_PER_ADDRESS)

    class Handler(BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT_S

        def log_message(self, *args):
            pass

        def _address(self) -> str:
            return client_address(self.headers, self.client_address[0], public)

        def _refused_by(self, limit: RateLimit) -> bool:
            """True if the address is over this limit; it has then been answered. Only with a public address."""
            wait = limit.hit(self._address()) if public else 0
            if wait:
                self.send_response(429)
                self._cors()
                self.send_header("Retry-After", str(wait))
                self.send_header("Content-Length", "0")
                self.end_headers()
            return bool(wait)

        def _may_watch(self, camera_id: str) -> bool:
            """On the internet a picture is only for the account that owns the camera. An <img> cannot send headers,
            hence the query."""
            if not public:
                return True
            token = parse_qs(self.path.partition("?")[2]).get("token", [""])[0]
            authorization = f"Bearer {token}" if token else self.headers.get("Authorization", "")
            if ownership(f"{authorization}|{camera_id}"):
                return True
            self._json(403, {"detail": "Sign in to the account that owns this camera to see it"})
            return False

        def _owns_camera(self, camera_id: str | None) -> bool:
            """True if the signed-in person may run this camera; otherwise it has been answered."""
            if camera_id and ownership(f"{self.headers.get('Authorization', '')}|{camera_id}"):
                return True
            self._json(403, {"detail": "Save this camera in your account first"})
            return False

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

        def _dashboard_only(self) -> bool:
            """True if the request may go on; otherwise it has been answered. 403, not 401: a 401 would sign the
            person out of the dashboard, and here it may only mean this service cannot reach the backend."""
            if self.headers.get("Origin", "").rstrip("/") not in origins:
                self._json(403, {"detail": "Origin not allowed"})
                return False
            if self._refused_by(dashboard_calls):
                return False
            if public and not session(self.headers.get("Authorization", "")):
                self._json(403, {"detail": "Sign in to the dashboard to do this"})
                return False
            return True

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/health":
                # Not rate limited: Render restarts the service when this stops answering.
                hung = state.stalled() or (manager is not None and manager.any_stalled())
                self._json(503 if hung else 200, {"status": "stalled" if hung else "ok"})
                return
            if self._refused_by(requests):
                return
            # An added camera is served under /cameras/<id>/ with the same paths as the service's own.
            view, view_stop = state, stop
            added = _ADDED_CAMERA_PATH.fullmatch(path)
            if added:
                entry = manager.get(added.group(1)) if manager is not None else None
                if entry is None:
                    self._json(404, {"detail": "Not found"})
                    return
                view, view_stop, path = entry.state, entry.stop, added.group(2)
            if path in VIDEO_PATHS and not self._may_watch(view.camera_id):
                return
            if path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/status":
                # service_url lets the dashboard on this PC build a link a phone on the same wifi can open.
                self._json(200, {**view.snapshot(), "service_url": _service_url(state, self.server.server_address[1]),
                                 "lan_host": local_ipv4()})
            elif path == "/snapshot.jpg":
                jpeg = view.latest_jpeg()
                if jpeg is None:
                    self._json(503, {"detail": "No frame yet"})
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/video.mjpg":
                self._stream(view, view_stop)
            elif path == "/heatmap.jpg":
                self._heatmap(view)
            elif path == "/cameras/scan":
                # The scan reveals what is on the LAN, so like POST /camera it answers the dashboard only.
                if self._dashboard_only():
                    # In the cloud this would look around the server's own network, which is nobody's farm.
                    if os.getenv("SMARTCATTLE_CLOUD") == "1":
                        self._json(409, {"detail": "This service runs in the cloud and has no local network to search"})
                    else:
                        self._json(200, scan_cameras())
            else:
                self._json(404, {"detail": "Not found"})

        def do_OPTIONS(self):
            # Preflight for POST /camera with a JSON body.
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self):
            route = self.path.split("?", 1)[0]
            # Read even to refuse: on Windows, closing with the body unread cuts the connection, and the browser
            # sees a network failure in place of the answer.
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            pushed = _FRAME_PATH.fullmatch(route)
            sent = self.rfile.read(length) if 0 < length <= (MAX_FRAME_BODY if pushed else MAX_CAMERA_BODY) else b""
            if pushed:
                if length > MAX_FRAME_BODY:
                    self.rfile.read(min(length, 4 * MAX_FRAME_BODY))
                self._take_frame(pushed.group(1), sent)
                return
            if route not in ("/camera", "/cameras"):
                self._json(404, {"detail": "Not found"})
                return
            # Only the dashboard may point the service at a camera: the body carries its password.
            if not self._dashboard_only():
                return
            if (camera if route == "/camera" else manager) is None:
                self._json(503, {"detail": "This service cannot change cameras"})
                return
            if length <= 0 or length > MAX_CAMERA_BODY:
                self._json(400, {"detail": "Send a JSON body"})
                return
            try:
                body = json.loads(sent)
                if not isinstance(body, dict):
                    raise ValueError("Send a JSON object")
                source = new_camera_source(body) if route == "/cameras" else camera_source_from_request(body)
            except ValueError as error:  # json.JSONDecodeError is a ValueError
                self._json(422, {"detail": str(error)})
                return
            if route == "/cameras":
                camera_id = body.get("camera_id")
                if camera_id is not None and not (isinstance(camera_id, str) and _CAMERA_ID.fullmatch(camera_id)):
                    self._json(422, {"detail": "camera_id must be 1 to 64 letters, digits, '-' or '_'"})
                    return
                if public and not self._owns_camera(camera_id):
                    return
                problem = no_answer(source)
                if problem:
                    self._json(422, {"detail": problem})
                    return
                try:
                    entry = manager.add(source, camera_id)
                except CameraLimitError as error:
                    self._json(409, {"detail": str(error)})
                    return
                # Never echo the source: it holds the camera credentials.
                self._json(202, {"status": "connecting", "camera_id": entry.camera_id,
                                 "stream_url": manager.stream_url(entry.camera_id)})
                return
            camera.switch(source)
            state.set_camera_status("connecting")
            # Never echo the URL back: it holds the camera credentials.
            self._json(202, {"status": "connecting", "camera": redact_url(source)})

        def _take_frame(self, camera_id: str, jpeg: bytes) -> None:
            """A picture from a phone. Not one of the dashboard's calls: it comes several times a second."""
            if self.headers.get("Origin", "").rstrip("/") not in origins:
                self._json(403, {"detail": "Origin not allowed"})
                return
            if self._refused_by(pushed_frames) or (public and not self._owns_camera(camera_id)):
                return
            entry = manager.get(camera_id) if manager is not None else None
            push = getattr(entry.stream, "push", None) if entry is not None else None
            if push is None:
                self._json(404, {"detail": "This camera is not connected"})
                return
            try:
                push(jpeg)
            except ValueError as error:
                self._json(422, {"detail": str(error)})
                return
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_DELETE(self):
            added = _REMOVE_CAMERA_PATH.fullmatch(self.path.split("?", 1)[0])
            if not added:
                self._json(404, {"detail": "Not found"})
            elif not self._dashboard_only():
                return
            elif public and not self._owns_camera(added.group(1)):
                return
            elif manager is None or not manager.remove(added.group(1)):
                self._json(404, {"detail": "This camera was not added from the dashboard"})
            else:
                self._json(200, {"status": "removed"})

        def _heatmap(self, view: LiveState) -> None:
            """Where the cattle have stood since the service started, over the latest frame."""
            if view.heat is None:
                self._json(404, {"detail": "Heat map is off"})
                return
            jpeg = view.latest_jpeg()
            if jpeg is None:
                self._json(503, {"detail": "No frame yet"})
                return
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            ok, encoded = cv2.imencode(".jpg", view.heat.overlay(frame))
            if not ok:
                self._json(500, {"detail": "Could not encode the heat map"})
                return
            self._send(200, encoded.tobytes(), "image/jpeg")

        def _stream(self, view: LiveState, view_stop: Event) -> None:
            address = self._address()
            if not streams.take(address):
                self.send_response(503)
                self._cors()
                self.send_header("Retry-After", "30")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            try:
                self._send_frames(view, view_stop)
            finally:
                streams.give_back(address)

        def _send_frames(self, view: LiveState, view_stop: Event) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while not stop.is_set() and not view_stop.is_set():
                    item = view.wait_frame(seq, timeout=1.0)
                    if item is None:
                        continue
                    seq, jpeg = item
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                     + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                pass

    return Handler


def make_server(state: LiveState, stop: Event, host: str, port: int, allowed_origins=DEFAULT_ORIGINS, camera=None,
                manager=None, session_check=None, camera_check=None):
    """camera, when given, is the CameraStream that POST /camera re-points (see CameraStream.switch);
    manager is the CameraManager behind POST and DELETE /cameras."""
    server = BoundedHTTPServer((host, port), make_handler(state, stop, allowed_origins, camera, manager, session_check,
                                                                   camera_check))
    server.daemon_threads = True
    return server


class BoundedHTTPServer(ThreadingHTTPServer):
    """A thread per connection, up to a limit: past it a connection is closed, not given memory it would never free."""
    request_queue_size = 128
    block_on_close = False

    def __init__(self, *args, **kwargs):
        self._active = 0
        self._active_lock = Lock()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        with self._active_lock:
            full = self._active >= MAX_CONNECTIONS
            if not full:
                self._active += 1
        if full:
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_lock:
                self._active -= 1
