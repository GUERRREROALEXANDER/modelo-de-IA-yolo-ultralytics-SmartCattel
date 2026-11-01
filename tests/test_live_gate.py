"""The gate counter inside the live service: counted, shown in /status and drawn."""

from threading import Event, Thread

import numpy as np

from smartcattle_ai.line_counter import LineCounter
from smartcattle_ai.live import LatestFrame, LiveState, detection_worker
from smartcattle_ai.media import draw_detections
from smartcattle_ai.tracking import IoUTracker

ZONE = (0.0, 0.0, 1.0, 1.0)
# A vertical gate down the middle. Going left to right is "out" for a line drawn top to bottom.
GATE = (0.5, 0.0, 0.5, 1.0)


class WalkingCow:
    """A cow walking left to right across the gate of a 100 px frame."""

    def __init__(self):
        self.positions = iter([20, 30, 40, 50, 60, 70])

    def detect(self, frame):
        x = next(self.positions, 70)
        # 30 px wide moving 10 px per frame: consecutive boxes overlap, so the tracker keeps its number.
        return [{"class": "cow", "confidence": 0.9, "bbox": [x, 40, x + 30, 60]}]


def test_the_worker_counts_a_crossing_and_status_reports_it(capsys):
    frames, state, stop = LatestFrame(), LiveState("camera-01", ZONE), Event()
    state.gate = LineCounter(GATE)
    worker = Thread(target=detection_worker, args=(WalkingCow(), frames, state, stop),
                    kwargs={"tracker": IoUTracker()})
    worker.start()
    try:
        seq = 0
        for _ in range(6):
            frames.put(np.zeros((100, 100, 3), dtype=np.uint8))
            seq, _ = state.wait_frame(seq, timeout=5)
    finally:
        stop.set()
        worker.join()
    assert state.snapshot()["gate"] == {"line": list(GATE), "entered": 0, "exited": 1}
    assert capsys.readouterr().out.count("Gate: cow #1 went out") == 1


def test_without_a_gate_status_says_none():
    assert LiveState("camera-01", ZONE).snapshot()["gate"] is None


def test_the_gate_and_its_totals_are_drawn():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert not np.array_equal(draw_detections(frame, []), draw_detections(frame, [], gate=LineCounter(GATE)))
