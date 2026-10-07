"""Backend integration tests using a local HTTP server only."""

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import URLError

import cv2
import numpy as np
import pytest

import detect
from smartcattle_ai.backend_client import (
    BackendClient, BackendError, EventThrottle, to_backend_event,
)
from smartcattle_ai.pipeline import EventPublisher, parse_zone


@pytest.fixture
def backend():
    events = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, body):
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ok"})
            else:
                self.respond(404, {"detail": "Not found"})

        def do_POST(self):
            if self.path != "/api/ai/events":
                self.respond(404, {"detail": "Not found"})
                return
            if self.headers.get("X-API-Key") != "test-key":
                self.respond(401, {"detail": "Invalid or missing API key"})
                return
            event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if set(event) != {"event_type", "camera_id", "detected_object", "confidence", "timestamp"}:
                self.respond(422, {"detail": "Extra or missing fields"})
                return
            if (event["event_type"] != "cattle_out_of_zone" or event["detected_object"] != "cow"
                    or not 1 <= len(event["camera_id"]) <= 64
                    or not 0 <= event["confidence"] <= 1
                    or datetime.fromisoformat(event["timestamp"]).tzinfo is None):
                self.respond(422, {"detail": "Invalid event"})
                return
            events.append(event)
            self.respond(201, {**event, "id": str(len(events)), "received_at": event["timestamp"]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", events
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def rule_event():
    return {"type": "cattle_outside_zone",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detection": {"class": "cow", "confidence": 0.91,
                          "bbox": [1, 2, 20, 30], "inside_zone": False}}


def test_mapping_and_authenticated_request(backend):
    url, events = backend
    payload = to_backend_event(rule_event(), "camera-01")
    assert set(payload) == {"event_type", "camera_id", "detected_object", "confidence", "timestamp"}
    client = BackendClient(url + "/", api_key="test-key")
    assert client.base_url == url
    assert client.health()
    assert client.send_event(payload)["id"] == "1"
    assert events == [payload]


def test_http_errors_include_status_and_detail(backend):
    url, _ = backend
    payload = to_backend_event(rule_event(), "camera-01")
    with pytest.raises(BackendError, match="Invalid or missing API key") as wrong_key:
        BackendClient(url, api_key="wrong").send_event(payload)
    assert wrong_key.value.status == 401
    with pytest.raises(BackendError, match="Extra or missing fields") as extra:
        BackendClient(url, api_key="test-key").send_event({**payload, "extra": True})
    assert extra.value.status == 422


def test_connection_error(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("smartcattle_ai.backend_client.urlopen", fail)
    with pytest.raises(BackendError, match="Could not connect"):
        BackendClient("http://127.0.0.1:8000").health()


def test_throttle_and_zone_validation():
    throttle = EventThrottle(1)
    assert throttle.allow("camera-01", 0)
    assert not throttle.allow("camera-01", 0.9)
    assert throttle.allow("camera-02", 0.9)
    assert throttle.allow("camera-01", 1)
    assert parse_zone("0.1,0.1,0.9,0.9") == (0.1, 0.1, 0.9, 0.9)
    for invalid in ("0,0,1", "a,0,1,1", "-0.1,0,1,1", "0,0,2,1",
                    "0.5,0,0.5,1", "0,1,1,0", "nan,0,1,1"):
        with pytest.raises(ValueError):
            parse_zone(invalid)


def test_publisher_warns_and_continues(capsys):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def send_event(self, payload):
            self.calls.append(payload)
            if len(self.calls) == 1:
                raise BackendError("temporary failure", 503)

    client = FakeClient()
    publisher = EventPublisher(client, "camera-01", parse_zone("0.1,0.1,0.9,0.9"),
                               EventThrottle(1))
    detections = [{"class": "cow", "confidence": 0.9, "bbox": [0, 0, 10, 20]}]
    assert publisher.handle(detections, 64, 64, 0) == []
    assert publisher.handle(detections, 64, 64, 0.5) == []
    assert len(publisher.handle(detections, 64, 64, 1)) == 1
    assert len(client.calls) == 2
    assert "temporary failure" in capsys.readouterr().err


def test_publisher_ignores_persons():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def send_event(self, payload):
            self.calls.append(payload)

    client = FakeClient()
    publisher = EventPublisher(client, "camera-01", parse_zone("0.1,0.1,0.9,0.9"),
                               EventThrottle(10))
    person = {"class": "person", "confidence": 0.9, "bbox": [0, 0, 10, 20]}
    cow = {"class": "cow", "confidence": 0.8, "bbox": [0, 0, 10, 20]}
    assert publisher.handle([person], 64, 64, 0) == []
    assert len(publisher.handle([person, cow], 64, 64, 0)) == 1
    assert [event["detected_object"] for event in client.calls] == ["cow"]


def test_detect_video_sends_three_throttled_events(backend, monkeypatch, tmp_path, capsys):
    url, events = backend

    class FakeDetector:
        def __init__(self, settings):
            pass

        def detect(self, frame):
            return [{"class": "cow", "confidence": 0.9, "bbox": [0, 0, 10, 20]}]

    source = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(detect, "Detector", FakeDetector)
    monkeypatch.setenv("SMARTCATTLE_API_KEY", "test-key")
    assert detect.main(["--source", str(source), "--no-save", "--backend-url", url,
                        "--event-cooldown", "1"]) == 0
    assert len(events) == 3
    assert "Backend events sent: 3" in capsys.readouterr().out
