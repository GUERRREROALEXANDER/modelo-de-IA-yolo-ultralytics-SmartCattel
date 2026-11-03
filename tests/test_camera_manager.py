"""POST and DELETE /cameras add and remove cameras next to the service's own, without replacing it."""

import json
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from smartcattle_ai.camera_manager import (
    CameraLimitError, CameraManager, ManagedCamera, SharedDetector, camera_id_for, check_reachable_from_cloud,
    check_source, is_public_url,
)
from smartcattle_ai.backend_client import BackendError
from smartcattle_ai.live import LiveState, lan_origins, make_server, new_camera_source, source_worker
from smartcattle_ai.stream import address_only

ZONE = (0.1, 0.1, 0.9, 0.9)
ORIGIN = "http://localhost:5173"
PASSWORD = "s3cret!"
ENV = {"CAMERA_HOST": "192.168.1.30", "CAMERA_USERNAME": "owner", "CAMERA_PASSWORD": PASSWORD,
       "CAMERA_RTSP_PORT": "554", "CAMERA_RTSP_PATH": "/live"}


@pytest.fixture(autouse=True)
def every_address_answers(monkeypatch):
    """These tests use made-up addresses: nothing is really there to answer."""
    monkeypatch.setattr("smartcattle_ai.live.no_answer", lambda source: None)



class FakeStream:
    def __init__(self):
        self.switched, self.stopped = [], False

    def switch(self, source):
        self.switched.append(source)

    def stop(self):
        self.stopped = True


def manager_for(path=None, max_cameras=8):
    def start(camera_id, source):
        return ManagedCamera(camera_id, source, LiveState(camera_id, ZONE), FakeStream(), Event())
    return CameraManager(start, lambda camera_id: f"http://farm:8090/cameras/{camera_id}", path=path,
                         max_cameras=max_cameras)


@pytest.fixture
def service(tmp_path):
    own, stop, manager = LiveState("camera-01", ZONE), Event(), manager_for(tmp_path / "cameras.json")
    server = make_server(own, stop, "127.0.0.1", 0, [ORIGIN], manager=manager)
    Thread(target=server.serve_forever, daemon=True).start()
    yield own, manager, f"http://127.0.0.1:{server.server_address[1]}"
    stop.set()
    server.shutdown()
    server.server_close()


def call(base, path, method="GET", body=None, origin=ORIGIN):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base + path, data=data, method=method, headers={"Content-Type": "application/json", "Origin": origin})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_adding_a_camera_keeps_the_service_camera_and_serves_the_new_one_on_its_own_path(service):
    own, manager, base = service
    status, body = call(base, "/cameras", "POST", {"url": "192.168.1.40:8080"})
    assert status == 202
    assert body["stream_url"] == f"http://farm:8090/cameras/{body['camera_id']}"
    assert call(base, "/status")[1]["camera_id"] == own.camera_id
    assert call(base, f"/cameras/{body['camera_id']}/status")[1]["camera_id"] == body["camera_id"]


def test_the_password_is_saved_on_this_pc_but_never_returned(service, tmp_path):
    _, _, base = service
    _, body = call(base, "/cameras", "POST", {"host": "192.168.1.50", "password": PASSWORD})
    assert PASSWORD not in json.dumps(body) and "s3cret" not in json.dumps(body)
    assert "s3cret" in (tmp_path / "cameras.json").read_text(encoding="utf-8")


def test_retrying_a_camera_with_a_new_password_replaces_the_entry(service):
    _, manager, base = service
    _, first = call(base, "/cameras", "POST", {"host": "192.168.1.50", "password": "wrong"})
    _, second = call(base, "/cameras", "POST", {"host": "192.168.1.50", "password": "right"})
    assert first["camera_id"] == second["camera_id"]
    entry = manager.get(first["camera_id"])
    assert "right" in entry.stream.switched[0] and entry.state.snapshot()["status"] == "connecting"


def test_removing_stops_the_camera_and_its_video(service):
    _, manager, base = service
    _, body = call(base, "/cameras", "POST", {"url": "192.168.1.40:8080"})
    entry = manager.get(body["camera_id"])
    assert call(base, f"/cameras/{body['camera_id']}", "DELETE")[0] == 200
    assert entry.stop.is_set() and entry.stream.stopped
    assert call(base, f"/cameras/{body['camera_id']}/status")[0] == 404
    assert call(base, f"/cameras/{body['camera_id']}", "DELETE")[0] == 404


def test_other_origins_can_neither_add_nor_remove(service):
    _, manager, base = service
    assert call(base, "/cameras", "POST", {"url": "192.168.1.40:8080"}, origin="https://evil.example")[0] == 403
    _, body = call(base, "/cameras", "POST", {"url": "192.168.1.40:8080"})
    assert call(base, f"/cameras/{body['camera_id']}", "DELETE", origin="https://evil.example")[0] == 403
    assert manager.get(body["camera_id"]) is not None


def test_scan_and_unknown_cameras_are_not_mistaken_for_added_ones(service):
    _, _, base = service
    assert call(base, "/cameras/nope/status")[0] == 404
    assert call(base, "/cameras/scan", origin="https://evil.example")[0] == 403


def test_a_typed_address_never_receives_another_cameras_saved_password():
    with pytest.raises(ValueError, match="password"):
        new_camera_source({"host": "10.0.0.99"}, ENV)
    assert new_camera_source({"host": "192.168.1.30"}, ENV) == "rtsp://owner:s3cret%21@192.168.1.30:554/live"
    # The saved port and path still help a camera that brings its own password.
    assert new_camera_source({"host": "10.0.0.99", "password": "x"}, ENV) == "rtsp://admin:x@10.0.0.99:554/live"


@pytest.mark.parametrize("body", [
    {"url": "file:///C:/Windows/win.ini"}, {"url": "concat:a|b"}, {"url": "ftp://cam/x"},
    {"host": "user:pw@evil.example/x", "password": "x"}, {"host": "a b", "password": "x"}, {},
])
def test_rejects_sources_that_are_not_cameras(body):
    with pytest.raises(ValueError):
        new_camera_source(body, {})


def test_accepts_every_kind_of_camera_source():
    for source in ("rtsp://cam/live", "RTSP://cam/live", "http://cam/video", "https://cam/video", "0", "1"):
        assert check_source(source) == source


def test_ids_are_stable_valid_and_free_of_credentials():
    camera_id = camera_id_for("rtsp://admin:s3cret@192.168.1.50:554/live")
    assert camera_id == camera_id_for("rtsp://admin:other@192.168.1.50:554/live")
    assert "s3cret" not in camera_id and "admin" not in camera_id
    assert camera_id != camera_id_for("rtsp://admin:s3cret@192.168.1.50:554/other")
    assert camera_id_for("1") == "cam-webcam-1"
    long_host = camera_id_for("rtsp://" + "a" * 200 + ".example.com/live")
    assert len(long_host) <= 64


def test_limits_how_many_cameras_one_pc_runs():
    manager = manager_for(max_cameras=1)
    manager.add("rtsp://a/live")
    with pytest.raises(CameraLimitError):
        manager.add("rtsp://b/live")
    manager.add("rtsp://a/live")  # Replacing one is still allowed at the limit.


def test_cameras_return_after_a_restart_and_removed_ones_do_not(tmp_path):
    path = tmp_path / "cameras.json"
    before = manager_for(path)
    keep, drop = before.add("rtsp://a/live"), before.add("rtsp://b/live")
    before.remove(drop.camera_id)
    assert before.stop_all() == [keep.camera_id]
    after = manager_for(path)
    after.restore()
    assert after.get(keep.camera_id).source == "rtsp://a/live" and after.get(drop.camera_id) is None


@pytest.mark.parametrize("content", ["not json", '{"a": 1}', '[{"source": "file:///x"}]', '[{"other": 1}]'])
def test_a_broken_camera_file_does_not_stop_the_service(tmp_path, content):
    path = tmp_path / "cameras.json"
    path.write_text(content, encoding="utf-8")
    manager = manager_for(path)
    manager.restore()
    assert manager.stop_all() == []


def test_the_shared_detector_lets_one_frame_through_at_a_time():
    running, overlapped = [], []

    class Model:
        def detect(self, frame):
            overlapped.append(bool(running))
            running.append(frame)
            Event().wait(0.01)
            running.pop()
            return []

    shared = SharedDetector(Model())
    threads = [Thread(target=shared.detect, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert overlapped == [False] * 5


class FakeClient:
    def __init__(self, *failures):
        self.failures, self.reports = list(failures), []

    def report_camera_source(self, camera_id, source, active=True):
        if self.failures:
            raise self.failures.pop(0)
        self.reports.append((camera_id, source, active))


def test_only_the_address_leaves_this_pc():
    assert address_only("rtsp://admin:s3cret%21@192.168.1.50:554/live?channel=1") == "rtsp://192.168.1.50:554/live?channel=1"
    assert address_only("1") == "1"


def test_records_the_source_retrying_while_the_backend_is_down():
    client = FakeClient(BackendError("down"))
    source_worker(client, "cam-1", "rtsp://cam/live", Event(), retry_s=0)
    assert client.reports == [("cam-1", "rtsp://cam/live", True)]


def test_a_backend_without_the_endpoint_is_not_retried():
    client = FakeClient(BackendError("missing", 404), BackendError("missing", 404))
    source_worker(client, "cam-1", "rtsp://cam/live", Event(), retry_s=0)
    assert len(client.failures) == 1 and client.reports == []


def test_other_devices_are_let_in_by_this_pcs_address():
    assert lan_origins("192.168.1.10") == ["http://192.168.1.10:5173", "http://192.168.1.10:4173"]
    assert lan_origins(None) == []


def resolves_to(*addresses):
    return lambda host, port: [(None, None, None, None, (address, port)) for address in addresses]


@pytest.mark.parametrize("addresses", [("192.168.1.50",), ("127.0.0.1",), ("169.254.169.254",), ("10.0.0.1",), ("8.8.8.8", "10.0.0.1"), ("::1",)])
def test_the_cloud_refuses_addresses_inside_any_network(addresses):
    with pytest.raises(ValueError, match="cloud"):
        check_reachable_from_cloud("rtsp://camera.example/live", resolve=resolves_to(*addresses))


def test_the_cloud_accepts_a_public_address():
    source = "rtsp://user:pw@camera.example:554/live"
    assert check_reachable_from_cloud(source, resolve=resolves_to("8.8.8.8")) == source


def test_the_cloud_refuses_webcams_and_names_that_do_not_resolve():
    def fails(host, port):
        raise OSError("no such host")

    with pytest.raises(ValueError, match="cloud"):
        check_reachable_from_cloud("1")
    with pytest.raises(ValueError, match="cloud"):
        check_reachable_from_cloud("rtsp://nowhere.example/live", resolve=fails)


def test_the_cloud_setting_applies_to_typed_addresses():
    with pytest.raises(ValueError, match="cloud"):
        new_camera_source({"host": "192.168.1.50", "password": "x"}, {"SMARTCATTLE_CLOUD": "1"})
    assert new_camera_source({"host": "192.168.1.50", "password": "x"}, {}).startswith("rtsp://admin:x@192.168.1.50")


@pytest.mark.parametrize("url,public", [
    ("https://abc.trycloudflare.com", True), ("https://live.onrender.com", True), ("http://8.8.8.8:8090", True),
    ("http://localhost:8091", False), ("http://127.0.0.1:8090", False), ("http://192.168.1.10:8090", False), (None, False),
])
def test_tells_which_urls_the_internet_can_reach(url, public):
    assert is_public_url(url) is public


@pytest.fixture
def exposed_service():
    tokens = []

    def session_check(authorization):
        tokens.append(authorization)
        return authorization == "Bearer good"

    stop, manager = Event(), manager_for()
    server = make_server(LiveState("camera-01", ZONE), stop, "127.0.0.1", 0, [ORIGIN], manager=manager,
                         session_check=session_check,
                         camera_check=lambda authorization, camera_id: authorization == "Bearer good")
    Thread(target=server.serve_forever, daemon=True).start()
    yield manager, f"http://127.0.0.1:{server.server_address[1]}"
    stop.set()
    server.shutdown()
    server.server_close()


def call_with(base, path, method, authorization, body=None):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json", "Origin": ORIGIN}
    if authorization:
        headers["Authorization"] = authorization
    try:
        with urlopen(Request(base + path, data=data, method=method, headers=headers), timeout=5) as response:
            return response.status
    except HTTPError as error:
        return error.code


@pytest.mark.parametrize("authorization", [None, "Bearer bad"])
def test_an_exposed_service_needs_a_session_even_with_the_right_origin(exposed_service, authorization):
    manager, base = exposed_service
    body = {"url": "192.168.1.40:8080"}
    assert call_with(base, "/cameras", "POST", authorization, body) == 403
    assert call_with(base, "/cameras/scan", "GET", authorization) == 403
    added = manager.add("rtsp://a/live")
    assert call_with(base, f"/cameras/{added.camera_id}", "DELETE", authorization) == 403
    assert manager.get(added.camera_id) is not None


def test_an_exposed_service_accepts_a_signed_in_person(exposed_service):
    manager, base = exposed_service
    assert call_with(base, "/cameras", "POST", "Bearer good",
                     {"url": "192.168.1.40:8080", "camera_id": "cam-mine"}) == 202


def test_an_exposed_service_only_runs_cameras_saved_in_the_account(exposed_service):
    _, base = exposed_service
    assert call_with(base, "/cameras", "POST", "Bearer good", {"url": "192.168.1.40:8080"}) == 403


def test_watching_an_exposed_service_needs_no_session(exposed_service):
    _, base = exposed_service
    assert call_with(base, "/status", "GET", None) == 200


def test_removing_does_not_wait_for_the_backend():
    release, called = Event(), Event()

    def slow_backend(camera_id, source):
        release.wait(5)
        called.set()

    manager = manager_for()
    manager._on_remove = slow_backend
    camera = manager.add("rtsp://a/live")
    assert manager.remove(camera.camera_id) is True
    assert camera.stop.is_set() and not called.is_set()
    release.set()
    assert called.wait(5)


def test_a_camera_runs_under_the_id_it_was_saved_with_and_keeps_it_after_a_restart(tmp_path):
    path = tmp_path / "cameras.json"
    before = manager_for(path)
    assert before.add("rtsp://a/live", "cam-of-this-farm").camera_id == "cam-of-this-farm"
    before.stop_all()
    after = manager_for(path)
    after.restore()
    assert after.get("cam-of-this-farm").source == "rtsp://a/live"
