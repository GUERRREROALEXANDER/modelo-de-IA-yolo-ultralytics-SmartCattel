"""POST /camera re-points the live service at another camera; GET /cameras/scan finds candidates on the LAN."""

import json
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smartcattle_ai.discovery import scan_cameras
from smartcattle_ai.live import LiveState, camera_source_from_request, make_server
from smartcattle_ai.stream import CameraStream

ZONE = (0.1, 0.1, 0.9, 0.9)
ORIGIN = "http://localhost:5173"


class FakeCamera:
    def __init__(self):
        self.urls = []

    def switch(self, url):
        self.urls.append(url)


@pytest.fixture
def server_with_camera():
    state, stop, camera = LiveState("camera-01", ZONE), Event(), FakeCamera()
    server = make_server(state, stop, "127.0.0.1", 0, [ORIGIN], camera=camera)
    Thread(target=server.serve_forever, daemon=True).start()
    yield state, camera, f"http://127.0.0.1:{server.server_address[1]}"
    stop.set()
    server.shutdown()
    server.server_close()


def post(base, body, origin=ORIGIN):
    data = json.dumps(body).encode()
    request = Request(base + "/camera", data=data, method="POST",
                      headers={"Content-Type": "application/json", "Origin": origin})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_builds_an_rtsp_url_from_the_ip_and_never_echoes_the_password(server_with_camera):
    state, camera, base = server_with_camera
    status, body = post(base, {"host": "192.168.100.30", "username": "admin", "password": "s3cret!"})
    assert status == 202 and body["status"] == "connecting"
    assert camera.urls == ["rtsp://admin:s3cret%21@192.168.100.30:554/cam/realmonitor?channel=1&subtype=1"]
    assert "s3cret" not in json.dumps(body)
    assert state.snapshot()["status"] == "connecting"


def test_accepts_a_phone_app_address(server_with_camera):
    _, camera, base = server_with_camera
    assert post(base, {"url": "192.168.1.20:8080"})[0] == 202
    assert camera.urls == ["http://192.168.1.20:8080/video"]


def test_rejects_other_origins_and_bad_bodies(server_with_camera):
    _, camera, base = server_with_camera
    assert post(base, {"host": "192.168.1.30", "password": "x"}, origin="http://evil.example")[0] == 403
    assert post(base, {"host": "192.168.1.30"})[0] == 422
    assert post(base, {})[0] == 422
    assert camera.urls == []


def test_source_defaults_to_admin_and_keeps_a_custom_port_and_path():
    url = camera_source_from_request({"host": "10.0.0.5", "password": "p", "port": 8554, "path": "/live"})
    assert url == "rtsp://admin:p@10.0.0.5:8554/live"


def test_stream_switch_reconnects_on_the_capture_thread():
    opened = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, _prop):
            return 0

        def read(self):
            return False, None

        def release(self):
            pass

    stream = CameraStream("rtsp://a", capture_factory=lambda url, *t: opened.append(url) or Capture(),
                          sleep=lambda _s: stream.stop())  # one pass of the loop, then stop
    assert stream.open()
    stream.switch("rtsp://b")
    assert stream.state == "connecting"
    list(stream.frames())
    assert opened[-1] == "rtsp://b"


def test_scan_reports_hosts_with_camera_ports():
    open_ports = {("192.168.100.30", 554), ("192.168.100.42", 8080)}
    result = scan_cameras("192.168.100.10", probe=lambda host, port, _t: (host, port) in open_ports)
    assert result["network"] == "192.168.100.0/24"
    assert [(camera["host"], camera["kinds"]) for camera in result["cameras"]] == [
        ("192.168.100.30", ["rtsp"]), ("192.168.100.42", ["ip-webcam"]),
    ]


def test_scan_never_probes_public_networks():
    calls = []
    assert scan_cameras("8.8.8.8", probe=lambda *args: calls.append(args) or False)["cameras"] == []
    assert calls == []
