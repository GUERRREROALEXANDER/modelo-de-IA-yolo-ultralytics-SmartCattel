"""Cameras added from the dashboard while the service runs: one pipeline each, kept across restarts."""

import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
from urllib.parse import urlsplit

from .pushed import PUSHED_SOURCE
from .stream import host_and_port, is_device_index

# Each camera costs CPU for detection, so an open-ended list would slow every camera down.
MAX_CAMERAS = 8
ALLOWED_SCHEMES = frozenset({"rtsp", "rtsps", "http", "https"})


class CameraLimitError(Exception):
    pass


class SharedDetector:
    """One model for every camera: frames take turns, because predict() is not safe to run concurrently."""

    def __init__(self, detector):
        self._detector = detector
        self._lock = Lock()

    def detect(self, frame):
        with self._lock:
            return self._detector.detect(frame)


@dataclass
class ManagedCamera:
    camera_id: str
    source: str
    state: Any
    stream: Any
    stop: Event


def check_source(source: str) -> str:
    """A webcam index or a network stream. Anything else (file://, concat:...) would let FFmpeg read this PC."""
    if is_device_index(source) or source == PUSHED_SOURCE:
        return source
    if urlsplit(source).scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("The camera address must be rtsp://, http://, an IP address or a webcam number")
    host_and_port(source)  # Raises ValueError for a missing host or a port that is not a number.
    return source


def check_reachable_from_cloud(source: str, resolve=socket.getaddrinfo) -> str:
    """A cloud server cannot reach a farm's network, and must not be pointed at its own: only internet addresses."""
    message = ("This server runs in the cloud and cannot reach that address. Use the camera's public address, "
               "or run live.py on a PC in the camera's network")
    if is_device_index(source):
        raise ValueError(message)
    host, port = host_and_port(source)
    try:
        addresses = {info[4][0] for info in resolve(host, port)}
    except OSError as exc:
        raise ValueError(message) from exc
    if not addresses or not all(ipaddress.ip_address(address.split("%")[0]).is_global for address in addresses):
        raise ValueError(message)
    return source


def is_public_url(url: str | None) -> bool:
    """Whether a URL can be reached from the internet, as far as its host name tells."""
    host = urlsplit(url).hostname if url else None
    if not host or host == "localhost":
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return True  # A name: a tunnel or a hosting provider's domain.


def camera_id_for(source: str) -> str:
    """Stable for one camera and free of credentials, so retrying with a new password reuses the same entry."""
    if is_device_index(source):
        return f"cam-webcam-{int(source)}"
    host, port = host_and_port(source)
    parts = urlsplit(source)
    # The path separates the channels of a recorder, which share one host and port.
    digest = hashlib.sha1(f"{parts.path}?{parts.query}".encode()).hexdigest()[:4]
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", host).strip("-")[:40]
    return f"cam-{slug}-{port}-{digest}"


class CameraManager:
    """Starts, replaces and stops the extra cameras.

    start(camera_id, source) builds and starts a pipeline and returns a ManagedCamera. The list is saved
    to `path`; the file holds camera passwords, like .env, and stays on this PC.
    """

    def __init__(self, start, stream_url, on_remove=None, path=None, max_cameras: int = MAX_CAMERAS):
        self._start = start
        self._stream_url = stream_url
        self._on_remove = on_remove
        self._path = Path(path) if path else None
        self._max = max_cameras
        self._cameras: dict[str, ManagedCamera] = {}
        self._lock = Lock()

    def stream_url(self, camera_id: str) -> str:
        return self._stream_url(camera_id)

    def get(self, camera_id: str) -> ManagedCamera | None:
        with self._lock:
            return self._cameras.get(camera_id)

    def all(self) -> list[ManagedCamera]:
        with self._lock:
            return list(self._cameras.values())

    def any_stalled(self) -> bool:
        with self._lock:
            return any(camera.state.stalled() for camera in self._cameras.values())

    def add(self, source: str, camera_id: str | None = None) -> ManagedCamera:
        """camera_id is the id the dashboard saved the camera under, so its reports reach that person's account."""
        camera_id = camera_id or camera_id_for(check_source(source))
        with self._lock:
            camera = self._cameras.get(camera_id)
            if camera is not None:
                camera.source = source
                camera.stream.switch(source)
                camera.state.set_camera_status("connecting")
            else:
                if len(self._cameras) >= self._max:
                    raise CameraLimitError(f"This PC already runs {self._max} added cameras; remove one first")
                camera = self._cameras[camera_id] = self._start(camera_id, source)
            self._save()
            return camera

    def remove(self, camera_id: str) -> bool:
        with self._lock:
            camera = self._cameras.pop(camera_id, None)
            if camera is None:
                return False
            self._save()
        self._halt(camera)
        if self._on_remove is not None:
            # It talks to the backend, which may be slow or down: the person is waiting for this answer.
            Thread(target=self._on_remove, args=(camera_id, camera.source), daemon=True).start()
        return True

    def restore(self) -> None:
        """Reconnects the saved cameras. A broken entry is skipped: it must not stop the service from starting."""
        if self._path is None or not self._path.is_file():
            return
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
            entries = [(item.get("camera_id") or camera_id_for(check_source(item["source"])),
                        check_source(item["source"])) for item in saved]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"Warning: could not read the saved cameras: {type(exc).__name__}", file=sys.stderr)
            return
        with self._lock:
            for camera_id, source in entries[:self._max]:
                self._cameras[camera_id] = self._start(camera_id, source)

    def stop_all(self) -> list[str]:
        """Stops every pipeline and keeps the saved list, so they return on the next start."""
        with self._lock:
            cameras = list(self._cameras.values())
            self._cameras.clear()
        for camera in cameras:
            self._halt(camera)
        return [camera.camera_id for camera in cameras]

    @staticmethod
    def _halt(camera: ManagedCamera) -> None:
        camera.stop.set()
        camera.stream.stop()

    def _save(self) -> None:
        if self._path is None:
            return
        body = json.dumps([{"camera_id": camera.camera_id, "source": camera.source}
                           for camera in self._cameras.values()])
        temporary = self._path.with_name(self._path.name + ".tmp")
        try:
            temporary.write_text(body, encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(self._path)
        except OSError as exc:
            print(f"Warning: could not save the camera list: {type(exc).__name__}", file=sys.stderr)
