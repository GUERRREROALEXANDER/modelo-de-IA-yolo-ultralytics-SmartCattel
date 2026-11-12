"""A phone's browser sends its own pictures: no address, no password, and nobody else's camera."""

import json
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from smartcattle_ai.camera_manager import CameraManager, ManagedCamera, check_source
from smartcattle_ai.live import MAX_FRAME_BODY, LiveState, make_server, new_camera_source
from smartcattle_ai.pushed import PUSHED_SOURCE, PushedStream

ZONE = (0.1, 0.1, 0.9, 0.9)
ORIGIN = "https://panel.example"
OWNER = "Bearer dueno"
OTHER = "Bearer otro"


def jpeg(width=64, height=48):
    return cv2.imencode(".jpg", np.full((height, width, 3), 120, dtype=np.uint8))[1].tobytes()


def fetch_next(stream):
    return next(stream.frames())


def test_a_picture_comes_out_as_a_frame_and_the_camera_goes_online():
    stream, states = PushedStream(), []
    stream.on_state_change(lambda state, error: states.append(state))
    stream.push(jpeg())
    assert fetch_next(stream).shape == (48, 64, 3)
    assert stream.frame_size == (64, 48) and states == ["online"]


def test_only_the_newest_picture_is_kept():
    stream = PushedStream()
    stream.push(jpeg(64, 48))
    stream.push(jpeg(32, 24))
    assert fetch_next(stream).shape == (24, 32, 3)


def test_what_is_not_a_picture_is_refused():
    stream = PushedStream()
    for body in (b"", b"not a picture"):
        with pytest.raises(ValueError, match="picture"):
            stream.push(body)


def test_a_phone_that_goes_quiet_is_reported_and_can_come_back():
    now = [0.0]
    stream, seen = PushedStream(quiet_after_s=10, clock=lambda: now[0]), []
    stream.on_state_change(lambda state, error: seen.append((state, error)))
    frames = stream.frames()
    stream.push(jpeg())
    next(frames)
    now[0] = 11.0

    worker = Thread(target=lambda: next(frames, None), daemon=True)
    worker.start()
    for _ in range(100):
        if seen[-1][0] == "error":
            break
        Event().wait(0.05)
    assert seen[-1][0] == "error" and "celular" in seen[-1][1]
    stream.push(jpeg())
    worker.join(5)
    assert stream.state == "online"
    stream.stop()


def test_stopping_ends_the_frames():
    stream = PushedStream()
    stream.stop()
    assert list(stream.frames()) == []


def test_it_needs_no_address_even_when_the_service_is_in_the_cloud():
    assert new_camera_source({"browser": True}, {"SMARTCATTLE_CLOUD": "1"}) == PUSHED_SOURCE
    assert check_source(PUSHED_SOURCE) == PUSHED_SOURCE


def test_a_url_cannot_pretend_to_be_one():
    assert new_camera_source({"browser": "yes", "url": "192.168.1.20:8080"}, {}) != PUSHED_SOURCE


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr("smartcattle_ai.live.no_answer", lambda source: None)
    own, stop = LiveState("camera-01", ZONE), Event()

    def start(camera_id, source):
        return ManagedCamera(camera_id, source, LiveState(camera_id, ZONE), PushedStream(), Event())

    manager = CameraManager(start, lambda camera_id: f"https://finca.example/cameras/{camera_id}")
    server = make_server(
        own, stop, "127.0.0.1", 0, [ORIGIN], manager=manager,
        session_check=lambda authorization: authorization in (OWNER, OTHER),
        camera_check=lambda authorization, camera_id: authorization == OWNER,
    )
    Thread(target=server.serve_forever, daemon=True).start()
    yield manager, f"http://127.0.0.1:{server.server_address[1]}"
    stop.set()
    server.shutdown()
    server.server_close()


def post(base, path, body, content_type="image/jpeg", authorization=OWNER, origin=ORIGIN):
    request = Request(base + path, data=body, method="POST", headers={
        "Content-Type": content_type, "Authorization": authorization, "Origin": origin})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status
    except HTTPError as error:
        return error.code


def connect(base):
    return post(base, "/cameras", json.dumps({"browser": True, "camera_id": "cam-1"}).encode(), "application/json")


def test_the_owner_connects_it_and_sends_pictures(service):
    manager, base = service
    assert connect(base) == 202
    assert manager.get("cam-1").source == PUSHED_SOURCE
    assert post(base, "/cameras/cam-1/frame", jpeg()) == 204
    assert fetch_next(manager.get("cam-1").stream).shape == (48, 64, 3)


def test_another_account_cannot_send_pictures_to_it(service):
    _, base = service
    connect(base)
    assert post(base, "/cameras/cam-1/frame", jpeg(), authorization=OTHER) == 403


def test_another_page_cannot_send_pictures_to_it(service):
    _, base = service
    connect(base)
    assert post(base, "/cameras/cam-1/frame", jpeg(), origin="https://evil.example") == 403


def test_a_camera_that_is_not_connected_tells_the_phone(service):
    _, base = service
    assert post(base, "/cameras/cam-1/frame", jpeg()) == 404


def test_broken_and_oversized_pictures_are_refused(service):
    _, base = service
    connect(base)
    assert post(base, "/cameras/cam-1/frame", b"not a picture") == 422
    assert post(base, "/cameras/cam-1/frame", b"x" * (MAX_FRAME_BODY + 1)) == 422
