"""Keeps open, from the cloud, the Imou cameras people connected in the dashboard.

Someone types a camera's address on the Cameras page and connects the Imou camera it recommends; the backend saves
it as imou:<serial>. This worker asks the backend for that list every few seconds, opens each camera through the
Imou cloud and closes those removed from it. Nothing on the farm's network is needed, so it runs on a server.
"""

import sys
from threading import Event, Thread

from .backend_client import BackendError
from .camera_manager import CameraLimitError
from .imou import imou_serial

#: Seconds between two looks at the list: a newly connected camera opens within this time.
SYNC_EVERY_S = 15.0


def sync_cloud_cameras(client, manager) -> list[str]:
    """Opens the listed cameras and closes the cloud cameras no longer listed. Returns what changed, for the log."""
    wanted = {}
    for item in client.cloud_cameras():
        camera_id, source = (item.get("camera_id"), item.get("source")) if isinstance(item, dict) else (None, None)
        if isinstance(camera_id, str) and imou_serial(source) is not None:
            wanted[camera_id] = source
    changes = []
    # Only cloud cameras are this worker's: a phone or a webcam added on this server is left alone.
    for camera in manager.all():
        if imou_serial(camera.source) is not None and camera.camera_id not in wanted:
            manager.remove(camera.camera_id)
            changes.append(f"closed {camera.camera_id}")
    for camera_id, source in wanted.items():
        current = manager.get(camera_id)
        if current is not None and current.source == source:
            continue
        try:
            manager.add(source, camera_id)
            changes.append(f"opened {camera_id}")
        except (CameraLimitError, ValueError) as exc:
            print(f"Warning: cloud camera {camera_id} not opened: {exc}", file=sys.stderr)
    return changes


def cloud_camera_worker(client, manager, stop: Event, interval_s: float = SYNC_EVERY_S) -> None:
    warned = False
    while not stop.is_set():
        try:
            for change in sync_cloud_cameras(client, manager):
                print(f"Cloud camera {change}", flush=True)
            warned = False
        except BackendError as exc:
            # The open cameras keep running; the list is asked for again on the next round.
            if not warned:
                print(f"Warning: could not read the cloud cameras from the backend: {exc}", file=sys.stderr)
                warned = True
        stop.wait(interval_s)


def start_cloud_cameras(client, manager, stop: Event, interval_s: float = SYNC_EVERY_S) -> Thread:
    thread = Thread(target=cloud_camera_worker, args=(client, manager, stop, interval_s), daemon=True)
    thread.start()
    return thread
