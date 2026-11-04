"""A camera that sends its own pictures: a phone's browser, which no server can open from the outside.

It needs no address, no password and no shared network. It only has to reach this service, and it stands in for a
CameraStream in the same pipeline.
"""

import time
from threading import Condition

import cv2
import numpy as np

from .stream import is_valid_frame

#: What a dashboard asks for, and the source saved with the camera, in place of an address.
PUSHED_SOURCE = "browser"
#: A phone sends a few pictures a second. This long without one means it closed the page or lost its signal.
QUIET_AFTER_S = 10.0


class PushedStream:
    """Holds the newest picture a phone sent; older ones are dropped, as the detector can only use the latest."""

    def __init__(self, quiet_after_s: float = QUIET_AFTER_S, clock=time.monotonic):
        self._quiet_after_s = quiet_after_s
        self._clock = clock
        self._condition = Condition()
        self._frame = None
        self._stopped = False
        self._callbacks = []
        self._last_sent = clock()
        self.state = "connecting"
        self.last_error = None
        self.frame_size = None
        self.reported_fps = None
        self.reconnects = 0

    def on_state_change(self, callback) -> None:
        self._callbacks.append(callback)

    def _set_state(self, state: str, error: str | None = None) -> None:
        self.last_error = error
        if state != self.state:
            self.state = state
            for callback in self._callbacks:
                callback(state, error)

    def push(self, jpeg: bytes) -> None:
        """Raises ValueError when the body is not a picture."""
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        if not is_valid_frame(frame):
            raise ValueError("That is not a picture")
        with self._condition:
            self._frame = frame
            self._last_sent = self._clock()
            self.frame_size = (frame.shape[1], frame.shape[0])
            self._condition.notify_all()

    def frames(self):
        while not self._stopped:
            with self._condition:
                if self._frame is None:
                    self._condition.wait(0.5)
                frame, self._frame = self._frame, None
                quiet = self._clock() - self._last_sent > self._quiet_after_s
            if frame is not None:
                self._set_state("online")
                yield frame
            elif quiet:
                self._set_state("error", "El celular dejó de enviar video. Abre la página en él y vuelve a conectar.")

    def switch(self, source) -> None:
        """Connecting again: the same phone goes on sending, so there is nothing to reopen."""
        with self._condition:
            self._last_sent = self._clock()
        self._set_state("connecting")

    def release(self) -> None:
        pass

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
