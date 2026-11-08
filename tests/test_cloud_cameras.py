"""Imou cameras connected in the dashboard, opened from the cloud by their serial number."""

from threading import Event

import pytest

from live import cloud_cameras_enabled
from smartcattle_ai.backend_client import BackendError
from smartcattle_ai.camera_manager import (CameraManager, ManagedCamera, camera_id_for, check_reachable_from_cloud,
                                           check_source)
from smartcattle_ai.cloud_cameras import cloud_camera_worker, sync_cloud_cameras
from smartcattle_ai.imou import ImouUrlProvider, imou_provider_for, imou_serial

SERIAL = "04797CEPSF59DA6"


class FakeStream:
    def __init__(self):
        self.switched = []
        self.stopped = False

    def switch(self, source):
        self.switched.append(source)

    def stop(self):
        self.stopped = True


class FakeState:
    def set_camera_status(self, status):
        self.status = status

    def stalled(self):
        return False


def make_manager(opened, opener=None):
    def start(camera_id, source):
        opened.append((camera_id, source))
        return ManagedCamera(camera_id, source, FakeState(), FakeStream(), Event())
    return CameraManager(start, lambda camera_id: f"https://live.example/cameras/{camera_id}", opener=opener)


class FakeClient:
    def __init__(self, items):
        self.items = items

    def cloud_cameras(self):
        if isinstance(self.items, Exception):
            raise self.items
        return self.items


def test_an_imou_source_is_recognised_and_reachable_from_the_cloud():
    assert imou_serial(f"imou:{SERIAL}") == SERIAL
    assert imou_serial("imou:../etc") is None
    assert imou_serial("rtsp://192.168.1.5/live") is None
    assert check_source(f"imou:{SERIAL}") == f"imou:{SERIAL}"
    assert check_reachable_from_cloud(f"imou:{SERIAL}") == f"imou:{SERIAL}"
    assert camera_id_for(f"imou:{SERIAL}") == f"cam-imou-{SERIAL.lower()}"


def test_sync_opens_the_listed_cameras_and_closes_the_removed_ones():
    opened = []
    manager = make_manager(opened)
    client = FakeClient([{"camera_id": "cam-1", "source": f"imou:{SERIAL}"},
                         {"camera_id": "cam-2", "source": "rtsp://bad"}, "junk"])
    assert sync_cloud_cameras(client, manager) == ["opened cam-1"]
    assert opened == [("cam-1", f"imou:{SERIAL}")]
    assert sync_cloud_cameras(client, manager) == []  # Already open: not opened twice.

    manager.add("1", "cam-phone")  # Not a cloud camera: the sync leaves it alone.
    client.items = []
    assert sync_cloud_cameras(client, manager) == ["closed cam-1"]
    assert [camera.camera_id for camera in manager.all()] == ["cam-phone"]


def test_a_changed_source_is_switched_through_the_opener():
    opened = []
    manager = make_manager(opened, opener=lambda source: f"opened:{source}")
    sync_cloud_cameras(FakeClient([{"camera_id": "cam-1", "source": f"imou:{SERIAL}"}]), manager)
    sync_cloud_cameras(FakeClient([{"camera_id": "cam-1", "source": "imou:ABCDEF123456"}]), manager)
    assert manager.get("cam-1").stream.switched == ["opened:imou:ABCDEF123456"]


def test_the_worker_survives_a_backend_outage(capsys):
    stop = Event()
    client = FakeClient(BackendError("Backend HTTP 503"))
    original = client.cloud_cameras

    def once():
        stop.set()
        return original()

    client.cloud_cameras = once
    cloud_camera_worker(client, make_manager([]), stop, interval_s=0)
    assert "could not read the cloud cameras" in capsys.readouterr().err


def test_a_provider_needs_the_imou_account_and_shares_its_client():
    with pytest.raises(ValueError, match="IMOU_APP_ID"):
        imou_provider_for(SERIAL, env={})
    env = {"IMOU_APP_ID": "lc-app", "IMOU_APP_SECRET": "secret"}
    first, second = imou_provider_for(SERIAL, env), imou_provider_for("ABCDEF123456", env)
    assert isinstance(first, ImouUrlProvider)
    assert first.client is second.client
    assert str(first) == f"Imou cloud device {SERIAL} (SD)"


@pytest.mark.parametrize(("env", "expected"), [
    ({}, False),
    ({"IMOU_APP_ID": "a", "IMOU_APP_SECRET": "b"}, False),  # A farm PC does not open them unless told to.
    ({"IMOU_APP_ID": "a", "IMOU_APP_SECRET": "b", "SMARTCATTLE_CLOUD": "1"}, True),
    ({"IMOU_APP_ID": "a", "IMOU_APP_SECRET": "b", "SMARTCATTLE_IMOU_SYNC": "1"}, True),
    ({"IMOU_APP_ID": "a", "SMARTCATTLE_CLOUD": "1"}, False),
])
def test_only_the_chosen_server_with_an_imou_account_opens_them(env, expected):
    assert cloud_cameras_enabled(env) is expected
