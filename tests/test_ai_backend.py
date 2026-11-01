"""Head count, safe zones from the dashboard, phone and webcam sources: no model, no camera."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest

from smartcattle_ai.backend_client import (
    BackendClient, BackendError, EventThrottle, normalized_box, occupancy_report, safe_error,
)
from smartcattle_ai.live import LiveState, occupancy_worker, sync_zones, zone_sync_worker
from smartcattle_ai.pipeline import EventPublisher
from smartcattle_ai.stream import camera_url_from_env, is_device_index, normalize_source
from smartcattle_ai.zones import ZoneSet

FALLBACK = ZoneSet.from_rectangle((0.1, 0.1, 0.9, 0.9))
PEN = {"id": 3, "camera_id": "camera-01", "name": "Pen", "active": True, "outline": None,
       "x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0, "created_at": "2026-10-31T00:00:00Z"}


@pytest.fixture
def backend():
    """A stand-in for the backend that records what it receives and checks the key."""
    received = []
    queue = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_any(self):
            if self.headers.get("X-API-Key") != "test-key":
                return self.reply(401, {"detail": "Invalid or missing API key"})
            length = int(self.headers.get("Content-Length") or 0)
            received.append((self.command, self.path, json.loads(self.rfile.read(length)) if length else None))
            if self.path == "/api/ai/cameras/camera-01/zones":
                return self.reply(200, {"items": [PEN], "total": 1})
            if self.path == "/api/ai/video-jobs/claim":
                return self.reply(200, queue.pop(0)) if queue else self.reply(204)
            if self.path.endswith("/occupancy"):
                return self.reply(204)
            self.reply(404, {"detail": "Not found"})

        do_GET = do_PUT = do_POST = handle_any

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield BackendClient(f"http://127.0.0.1:{server.server_port}", "test-key"), received, queue
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_the_key_goes_on_reads_too(backend):
    client, received, _ = backend
    assert client.camera_zones("camera-01") == [PEN]
    assert received == [("GET", "/api/ai/cameras/camera-01/zones", None)]
    with pytest.raises(BackendError) as error:
        BackendClient(client.base_url, "wrong").camera_zones("camera-01")
    assert error.value.status == 401


def test_an_empty_queue_is_none_and_a_204_is_not_an_error(backend):
    client, received, queue = backend
    assert client.claim_video_job() is None
    queue.append({"job_id": "j1"})
    assert client.claim_video_job() == {"job_id": "j1"}
    client.report_occupancy("camera-01", {"cattle_inside": 0})
    assert received[-1][:2] == ("PUT", "/api/ai/cameras/camera-01/occupancy")


def test_the_occupancy_report_counts_cattle_and_normalizes_boxes():
    detections = [
        {"class": "cow", "confidence": 0.9, "bbox": [10, 20, 50, 60], "inside_zone": True, "track_id": 4},
        {"class": "cow", "confidence": 0.8, "bbox": [-5, 0, 300, 100], "inside_zone": False, "track_id": 5},
        {"class": "person", "confidence": 0.7, "bbox": [0, 0, 10, 10], "inside_zone": None},
    ]
    report = occupancy_report(detections, 200, 100, "2026-10-31T12:00:00+00:00")
    assert (report["cattle_inside"], report["cattle_outside"]) == (1, 1)
    assert report["observed_at"] == "2026-10-31T12:00:00+00:00"
    # Persons are not counted or boxed: inside_zone has no meaning for them, and the backend requires it.
    assert report["detections"] == [
        {"track_id": 4, "label": "cow", "confidence": 0.9, "bbox": [0.05, 0.2, 0.25, 0.6], "inside_zone": True},
        # Clipped to the frame.
        {"track_id": 5, "label": "cow", "confidence": 0.8, "bbox": [0.0, 0.0, 1.0, 1.0], "inside_zone": False},
    ]


def test_a_box_with_no_area_left_after_clipping_is_dropped():
    assert normalized_box([250, 0, 300, 10], 200, 100) is None
    assert normalized_box([10, 10, 10, 20], 200, 100) is None


def test_error_text_never_carries_a_url_and_fits_the_backend():
    assert safe_error("open rtsp://admin:secret@cam failed") == "Video processing failed"
    assert safe_error("  a\nb  ") == "a b"
    assert len(safe_error("x" * 400)) == 300
    assert safe_error("") == "Video processing failed"


def test_zones_drawn_in_the_dashboard_replace_the_configured_one(backend, capsys):
    client, _, _ = backend
    state = LiveState("camera-01", FALLBACK)
    publisher = EventPublisher(client, "camera-01", FALLBACK, EventThrottle(0))
    assert sync_zones(client, state, FALLBACK, publisher)
    assert state.zone.source == "backend" and publisher.zone is state.zone
    assert state.snapshot()["zones"] == [{"name": "Pen", "rect": [0.0, 0.0, 0.5, 1.0], "outline": None}]
    assert "Safe zones from backend: Pen" in capsys.readouterr().out
    # Nothing changed: nothing to announce.
    assert not sync_zones(client, state, FALLBACK, publisher)


def test_a_camera_without_zones_uses_the_configured_one():
    class NoZones:
        def camera_zones(self, camera_id):
            return []

    state = LiveState("camera-01", ZoneSet.from_rectangle((0, 0, 1, 1)))
    assert sync_zones(NoZones(), state, FALLBACK)
    assert state.zone == FALLBACK


def test_a_backend_that_is_down_does_not_erase_the_zone(capsys):
    class Down:
        def camera_zones(self, camera_id):
            stop.set()
            raise BackendError("Could not connect to backend")

    stop = Event()
    state = LiveState("camera-01", FALLBACK)
    zone_sync_worker(Down(), state, FALLBACK, stop, 0.01)
    assert state.zone == FALLBACK
    assert "could not load safe zones" in capsys.readouterr().err


def test_each_frame_is_counted_once():
    class Recorder:
        def __init__(self):
            self.reports = []

        def report_occupancy(self, camera_id, report):
            self.reports.append((camera_id, report))
            if len(self.reports) == 1:
                state.publish(b"jpeg", [], 64, 64, None)
            else:
                stop.set()

    stop = Event()
    state = LiveState("camera-01", FALLBACK)
    state.set_camera_status("online")
    state.publish(b"jpeg", [{"class": "cow", "confidence": 0.9, "bbox": [0, 0, 8, 8], "inside_zone": False}],
                  64, 64, None)
    recorder = Recorder()
    worker = Thread(target=occupancy_worker, args=(recorder, state, stop, 0.01))
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert [report["cattle_outside"] for _, report in recorder.reports] == [1, 0]


def test_no_count_while_the_camera_is_down():
    class Fails:
        def report_occupancy(self, camera_id, report):
            raise AssertionError("must not report")

    stop = Event()
    state = LiveState("camera-01", FALLBACK)
    state.publish(b"jpeg", [], 64, 64, None)
    state.set_camera_status("error", "Stream read failed")
    Thread(target=lambda: (stop.wait(0.1), stop.set())).start()
    occupancy_worker(Fails(), state, stop, 0.01)


def test_a_tracked_cow_is_reported_once_per_escape():
    class Recorder:
        def __init__(self):
            self.events = []

        def send_event(self, payload):
            self.events.append(payload)

    client = Recorder()
    publisher = EventPublisher(client, "camera-01", FALLBACK, EventThrottle(0))
    outside = {"class": "cow", "confidence": 0.9, "bbox": [0, 0, 1, 1], "inside_zone": False, "track_id": 7}
    inside = {**outside, "inside_zone": True}
    for second in range(5):
        publisher.handle([outside], 64, 64, second)
    assert len(client.events) == 1
    publisher.handle([inside], 64, 64, 5)
    publisher.handle([outside], 64, 64, 6)
    # Back inside and out again: a second escape.
    assert len(client.events) == 2
    # A different animal escaping at the same time is its own alert.
    publisher.handle([outside, {**outside, "track_id": 8}], 64, 64, 7)
    assert len(client.events) == 3


def test_a_failed_send_is_retried_on_a_later_frame(capsys):
    class Flaky:
        def __init__(self):
            self.calls = 0

        def send_event(self, payload):
            self.calls += 1
            if self.calls == 1:
                raise BackendError("temporary failure", 503)

    client = Flaky()
    publisher = EventPublisher(client, "camera-01", FALLBACK, EventThrottle(0))
    outside = {"class": "cow", "confidence": 0.9, "bbox": [0, 0, 1, 1], "inside_zone": False, "track_id": 7}
    assert publisher.handle([outside], 64, 64, 0) == []
    assert len(publisher.handle([outside], 64, 64, 1)) == 1
    assert "temporary failure" in capsys.readouterr().err


@pytest.mark.parametrize(("value", "expected"), [
    ("0", "0"),
    ("1", "1"),
    # What the IP Webcam app shows on the phone screen.
    ("192.168.1.20:8080", "http://192.168.1.20:8080/video"),
    # DroidCam's MJPEG feed, path included.
    ("192.168.1.20:4747/video", "http://192.168.1.20:4747/video"),
    ("  http://192.168.1.20:8080/video  ", "http://192.168.1.20:8080/video"),
    ("rtsp://user:pass@cam/stream", "rtsp://user:pass@cam/stream"),
])
def test_phone_and_webcam_sources(value, expected):
    assert normalize_source(value) == expected


def test_an_empty_source_is_rejected():
    with pytest.raises(ValueError):
        normalize_source("   ")


def test_device_indexes():
    assert is_device_index("0") and is_device_index(" 2 ")
    assert not is_device_index("192.168.1.20") and not is_device_index(0)


def test_camera_url_wins_over_the_rtsp_variables():
    env = {"CAMERA_URL": "192.168.1.20:8080", "CAMERA_RTSP_URL": "rtsp://cam/stream"}
    assert camera_url_from_env(env) == "http://192.168.1.20:8080/video"
    assert camera_url_from_env({"CAMERA_RTSP_URL": "rtsp://cam/stream"}) == "rtsp://cam/stream"
    assert camera_url_from_env({"CAMERA_URL": "1"}) == "1"
