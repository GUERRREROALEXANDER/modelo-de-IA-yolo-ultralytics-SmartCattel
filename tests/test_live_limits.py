"""What keeps the service up, and its picture private, once it can be reached from the internet."""
import json
import socket
import time
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from smartcattle_ai import live as live_module
from smartcattle_ai.limits import RateLimit, SessionCache, StreamSlots, client_address
from smartcattle_ai.live import LatestFrame, LiveState, capture_worker, detection_worker, make_server
from smartcattle_ai.stream import CameraStream
from smartcattle_ai.zones import ZoneSet

ZONE = ZoneSet.from_rectangle((0.1, 0.1, 0.9, 0.9))
ORIGIN = "http://localhost:5173"


def status_of(url, headers=None, method="GET"):
    try:
        with urlopen(Request(url, headers=headers or {}, method=method), timeout=5) as response:
            return response.status
    except HTTPError as error:
        return error.code


@pytest.fixture
def serve():
    servers = []

    def start(state=None, session_check=None, manager=None):
        state = state or LiveState("camera-01", ZONE)
        server = make_server(state, Event(), "127.0.0.1", 0, [ORIGIN], manager=manager, session_check=session_check)
        Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", state

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_behind_a_proxy_each_visitor_is_told_apart_and_the_first_forwarded_address_is_only_a_fallback():
    assert client_address({"CF-Connecting-IP": "1.1.1.1", "X-Forwarded-For": "9.9.9.9"}, "10.0.0.1", True) == "1.1.1.1"
    assert client_address({"X-Forwarded-For": "2.2.2.2, 10.0.0.2"}, "10.0.0.1", True) == "2.2.2.2"
    assert client_address({}, "10.0.0.1", True) == "10.0.0.1"


def test_without_a_proxy_the_headers_are_the_clients_own_words_and_are_ignored():
    assert client_address({"CF-Connecting-IP": "1.1.1.1"}, "10.0.0.1", False) == "10.0.0.1"


def test_a_rate_limit_counts_per_address_and_starts_again_in_the_next_window():
    now = [0.0]
    limit = RateLimit(2, window_s=60, clock=lambda: now[0])
    assert [limit.hit("a"), limit.hit("a")] == [0, 0]
    assert limit.hit("a") > 0
    assert limit.hit("b") == 0
    now[0] = 61.0
    assert limit.hit("a") == 0


def test_stream_slots_bound_each_address_and_the_total_and_come_back():
    slots = StreamSlots(total=3, per_address=2)
    assert slots.take("a") and slots.take("a")
    assert not slots.take("a")
    assert slots.take("b")
    assert not slots.take("c")
    slots.give_back("a")
    assert slots.take("c")


def test_the_session_answer_is_remembered_so_a_burst_is_not_a_burst_on_the_backend():
    calls = []
    now = [0.0]
    cache = SessionCache(lambda value: calls.append(value) or value == "good", valid_s=100, invalid_s=10,
                         clock=lambda: now[0])
    assert [cache("good"), cache("good"), cache("bad"), cache("bad")] == [True, True, False, False]
    assert calls == ["good", "bad"]
    now[0] = 11.0
    cache("bad")
    assert calls == ["good", "bad", "bad"]


def test_the_picture_needs_a_session_on_the_internet(serve):
    url, state = serve(session_check=lambda authorization: authorization == "Bearer good")
    state.publish(b"jpeg", [], 2, 2, None)
    assert status_of(f"{url}/snapshot.jpg") == 403
    assert status_of(f"{url}/snapshot.jpg?token=bad") == 403
    assert status_of(f"{url}/video.mjpg") == 403
    assert status_of(f"{url}/snapshot.jpg?token=good") == 200


def test_on_a_private_network_the_picture_stays_open_as_before(serve):
    url, state = serve()
    state.publish(b"jpeg", [], 2, 2, None)
    assert status_of(f"{url}/snapshot.jpg") == 200


def test_one_address_cannot_exhaust_a_public_service_but_another_still_gets_in(serve, monkeypatch):
    monkeypatch.setattr(live_module, "MAX_REQUESTS_PER_MINUTE", 3)
    url, _ = serve(session_check=lambda authorization: False)
    attacker, visitor = {"CF-Connecting-IP": "6.6.6.6"}, {"CF-Connecting-IP": "7.7.7.7"}
    assert [status_of(f"{url}/status", attacker) for _ in range(4)] == [200, 200, 200, 429]
    assert status_of(f"{url}/status", visitor) == 200


def test_health_keeps_answering_over_the_limit(serve, monkeypatch):
    monkeypatch.setattr(live_module, "MAX_REQUESTS_PER_MINUTE", 1)
    url, _ = serve(session_check=lambda authorization: False)
    assert [status_of(f"{url}/health") for _ in range(4)] == [200] * 4


def test_trying_passwords_on_the_camera_endpoints_stops_reaching_the_backend(serve, monkeypatch):
    monkeypatch.setattr(live_module, "MAX_DASHBOARD_CALLS_PER_MINUTE", 2)
    asked = []
    url, _ = serve(session_check=lambda authorization: asked.append(authorization) or False)
    answers = [status_of(f"{url}/cameras/scan", {"Origin": ORIGIN, "Authorization": f"Bearer {n}"}) for n in range(4)]
    assert answers == [403, 403, 429, 429]
    assert len(asked) == 2


def test_video_streams_are_bounded(serve, monkeypatch):
    monkeypatch.setattr(live_module, "MAX_STREAMS", 1)
    url, state = serve()
    state.publish(b"jpeg", [], 2, 2, None)
    first = urlopen(f"{url}/video.mjpg", timeout=5)
    try:
        assert status_of(f"{url}/video.mjpg") == 503
    finally:
        first.close()


def test_a_client_that_never_finishes_its_request_gives_its_thread_back(serve, monkeypatch):
    monkeypatch.setattr(live_module, "SOCKET_TIMEOUT_S", 1)
    url, _ = serve()
    port = int(url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout=5) as idle:
        idle.sendall(b"GET /status HT")
        assert idle.recv(1) == b""  # The server closed it instead of waiting forever.


def test_past_the_connection_limit_a_connection_is_closed_and_the_service_recovers(serve, monkeypatch):
    monkeypatch.setattr(live_module, "MAX_CONNECTIONS", 1)
    url, _ = serve()
    port = int(url.rsplit(":", 1)[1])
    holder = socket.create_connection(("127.0.0.1", port), timeout=5)
    time.sleep(0.2)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as refused:
        assert refused.recv(1) == b""
    holder.close()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if status_of(f"{url}/health") == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    pytest.fail("the service did not recover")


def test_health_reports_a_detector_that_hung_so_the_host_restarts_the_service(serve):
    url, state = serve()
    assert status_of(f"{url}/health") == 200
    state.working_since = time.monotonic() - LiveState.STALLED_AFTER_S - 1
    assert status_of(f"{url}/health") == 503
    with pytest.raises(HTTPError) as answer:
        urlopen(f"{url}/health", timeout=5)
    assert json.loads(answer.value.read()) == {"status": "stalled"}


class BrokenOnce:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("one bad frame")
        return []


def test_a_frame_that_breaks_the_detector_does_not_end_detection():
    state, frames, stop = LiveState("camera-01", ZONE), LatestFrame(), Event()
    worker = Thread(target=detection_worker, args=(BrokenOnce(), frames, state, stop), daemon=True)
    worker.start()
    picture = np.zeros((20, 20, 3), dtype=np.uint8)
    deadline = time.monotonic() + 8
    try:
        while state.latest_jpeg() is None and time.monotonic() < deadline:
            frames.put(picture)
            time.sleep(0.05)
        assert state.latest_jpeg() is not None
        assert worker.is_alive()
        assert state.working_since is None or not state.stalled()
    finally:
        stop.set()
        worker.join(timeout=5)


def test_capture_starts_again_after_something_the_stream_did_not_foresee():
    class Stream:
        def __init__(self):
            self.runs = 0

        def frames(self):
            self.runs += 1
            if self.runs == 1:
                raise RuntimeError("unforeseen")
            yield "frame"

        def release(self):
            pass

    frames, stop, stream = LatestFrame(), Event(), Stream()
    capture_worker(stream, frames, stop)
    assert stream.runs == 2
    assert frames.take(timeout=1) == "frame"


def test_a_capture_that_raises_counts_as_a_failed_read():
    class Capture:
        def read(self):
            raise ValueError("cv2.error")

        def release(self):
            pass

    stream = CameraStream("rtsp://camera", capture_factory=lambda *args: Capture())
    stream._capture = Capture()
    assert stream.read() is None
    assert stream.state == "error"
