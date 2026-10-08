"""One reconnecting OpenCV capture for an RTSP camera, with credentials kept out of logs."""

import os
import re
import time
from collections.abc import Callable
from urllib.parse import quote, urlsplit

import cv2
import numpy as np

_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")


def redact_url(text: str) -> str:
    """Hide user:password in any URL inside text."""
    return _CREDENTIALS.sub("***:***@", str(text))


def camera_url_from_env(env=os.environ) -> str:
    """CAMERA_RTSP_URL, or a URL built from CAMERA_HOST/USERNAME/PASSWORD/RTSP_PORT/RTSP_PATH."""
    url = env.get("CAMERA_RTSP_URL", "").strip()
    if url:
        return url
    missing = [name for name in ("CAMERA_HOST", "CAMERA_USERNAME", "CAMERA_PASSWORD") if not env.get(name, "").strip()]
    if missing:
        raise ValueError("Camera not configured: set CAMERA_RTSP_URL, or " + ", ".join(missing))
    port = env.get("CAMERA_RTSP_PORT", "").strip() or "554"
    path = env.get("CAMERA_RTSP_PATH", "").strip() or "/cam/realmonitor?channel=1&subtype=1"
    user = quote(env["CAMERA_USERNAME"].strip(), safe="")
    password = quote(env["CAMERA_PASSWORD"], safe="")
    return f"rtsp://{user}:{password}@{env['CAMERA_HOST'].strip()}:{port}/{path.lstrip('/')}"


def host_and_port(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    if not parts.hostname:
        raise ValueError(f"Camera URL has no host: {redact_url(url)}")
    return parts.hostname, parts.port or (554 if parts.scheme == "rtsp" else 80)


def is_valid_frame(frame) -> bool:
    return isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3 and frame.size > 0


def _open_capture(url: str, open_timeout_s: float, read_timeout_s: float):
    # RTSP over TCP: UDP loses packets on Wi-Fi and produces gray, smeared frames.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000),
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(read_timeout_s * 1000)]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


class CameraStream:
    """Owns at most one capture; frames() reconnects with exponential backoff."""

    def __init__(self, url: str | Callable[[], str], open_timeout_s: float = 10, read_timeout_s: float = 10,
                 max_backoff_s: float = 30, capture_factory=None, sleep=time.sleep):
        self.url = url
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_backoff_s = max_backoff_s
        self._factory = capture_factory or _open_capture
        self._sleep = sleep
        self._capture = None
        self._stopped = False
        self._callbacks = []
        self.state = "connecting"
        self.last_error = None
        self.reconnects = 0
        self.frame_size = None
        self.reported_fps = None

    def on_state_change(self, callback) -> None:
        self._callbacks.append(callback)

    def _set_state(self, state: str, error: str | None = None) -> None:
        self.last_error = redact_url(error)[:200] if error else None
        if state != self.state:
            self.state = state
            for callback in self._callbacks:
                callback(state, self.last_error)

    def open(self) -> bool:
        self.release()
        try:
            # A callable url (e.g. the Imou cloud) is resolved again on every reconnect: its URLs can change.
            url = self.url() if callable(self.url) else self.url
            capture = self._factory(url, self.open_timeout_s, self.read_timeout_s)
        except Exception as exc:  # OpenCV raises cv2.error for some backends
            self._set_state("error", f"Could not open stream: {exc}")
            return False
        if not capture.isOpened():
            capture.release()
            self._set_state("error", "Could not open stream (unreachable, wrong credentials or path)")
            return False
        self._capture = capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.reported_fps = fps if fps and 0 < fps <= 240 else None
        return True

    def read(self):
        """A valid BGR frame, or None after marking the stream as failed."""
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or not is_valid_frame(frame):
            self._set_state("error", "Stream read failed or returned an invalid frame")
            return None
        self.frame_size = (frame.shape[1], frame.shape[0])
        self._set_state("online")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def stop(self) -> None:
        self._stopped = True

    def frames(self):
        backoff = 1.0
        while not self._stopped:
            if self._capture is None and not self.open():
                self._sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                self.reconnects += 1
                continue
            frame = self.read()
            if frame is None:
                self.release()
                self._sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                self.reconnects += 1
                continue
            backoff = 1.0
            yield frame
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
