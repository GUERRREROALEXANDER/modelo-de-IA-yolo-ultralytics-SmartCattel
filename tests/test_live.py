"""Live service tests with fake detectors and a local HTTP server only."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from urllib.request import Request, urlopen

import numpy as np
import pytest

from smartcattle_ai.backend_client import BackendClient
from smartcattle_ai.live import (
    LatestFrame, LiveState, classify_zone, detection_worker, make_server, status_report, status_worker,
)

ZONE = (0.1, 0.1, 0.9, 0.9)


class FakeDetector:
    def detect(self, frame):
        return [{"class": "cow", "confidence": 0.9, "bbox": [0, 0, 4, 4]},
                {"class": "person", "confidence": 0.8, "bbox": [10, 10, 50, 50]}]


def test_classify_zone_marks_only_cattle():
    cow, person = classify_zone(FakeDetector().detect(None), 64, 64, ZONE)
    assert cow["inside_zone"] is False
    assert person["inside_zone"] is None


def test_latest_frame_keeps_only_newest():
    frames = LatestFrame()
    frames.put("old")
    frames.put("new")
    assert frames.take(0) == "new"
    assert frames.take(0) is None


def test_worker_publishes_annotated_jpeg_and_detections():
    frames, state, stop = LatestFrame(), LiveState("camera-01", ZONE), Event()
    frames.put(np.zeros((64, 64, 3), dtype=np.uint8))
    worker = Thread(target=detection_worker, args=(FakeDetector(), frames, state, stop))
    worker.start()
    try:
        seq, jpeg = state.wait_frame(0, timeout=5)
    finally:
        stop.set()
        worker.join()
    assert seq == 1 and jpeg.startswith(b"\xff\xd8")
    snapshot = state.snapshot()
    assert (snapshot["width"], snapshot["height"]) == (64, 64)
    assert [item["inside_zone"] for item in snapshot["detections"]] == [False, None]


def test_status_report_waits_for_connection_and_hides_urls():
    state = LiveState("camera-01", ZONE)
    assert status_report(state) is None
    state.set_camera_status("error", "Could not open rtsp://***:***@host/stream")
    report = status_report(state)
    assert report["status"] == "error" and report["error"] == "Stream error"
    state.set_camera_status("online")
    state.publish(b"jpeg", [], 640, 352, 11.0)
    report = status_report(state)
    assert "error" not in report
    assert (report["frame_width"], report["frame_height"], report["fps"]) == (640, 352, 11.0)


def test_status_worker_reports_changes_immediately():
    class Client:
        def __init__(self):
            self.sent = []

        def report_camera_status(self, camera_id, report):
            self.sent.append(report["status"])

    client, state, stop = Client(), LiveState("camera-01", ZONE), Event()
    state.set_camera_status("online")
    worker = Thread(target=status_worker, args=(client, state, stop, 3600))
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while client.sent != ["online"] and time.monotonic() < deadline:
            time.sleep(0.05)
        state.set_camera_status("error", "Stream read failed")
        while client.sent != ["online", "error"] and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
        worker.join()
    assert client.sent == ["online", "error"]


@pytest.fixture
def live_server():
    state, stop = LiveState("camera-01", ZONE), Event()
    server = make_server(state, stop, "127.0.0.1", 0, ["http://localhost:5173"])
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    stop.set()
    server.shutdown()
    server.server_close()


def test_http_status_cors_and_mjpeg(live_server):
    state, base = live_server
    state.set_camera_status("online")
    state.publish(b"\xff\xd8frame", [{"class": "cow", "confidence": 0.9, "bbox": [1, 2, 3, 4], "inside_zone": True}],
                  640, 352, 10.0)
    with urlopen(Request(base + "/status", headers={"Origin": "http://localhost:5173"}), timeout=5) as response:
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
        body = json.load(response)
    assert body["camera_id"] == "camera-01" and body["width"] == 640 and len(body["detections"]) == 1
    with urlopen(Request(base + "/status", headers={"Origin": "http://evil.example"}), timeout=5) as response:
        assert response.headers["Access-Control-Allow-Origin"] is None
    with urlopen(base + "/video.mjpg", timeout=5) as response:
        assert response.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        chunk = response.read(64)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")


def test_report_camera_status_uses_put():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_PUT(self):
            received["path"] = self.path
            received["key"] = self.headers.get("X-API-Key")
            received["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"id": "camera-01", "status": "online"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = BackendClient(f"http://127.0.0.1:{server.server_address[1]}", "test-key")
        report = {"status": "online", "observed_at": "2026-10-07T00:00:00+00:00"}
        assert client.report_camera_status("camera-01", report)["status"] == "online"
    finally:
        server.shutdown()
        server.server_close()
    assert received == {"path": "/api/ai/cameras/camera-01/status", "key": "test-key", "body": report}
