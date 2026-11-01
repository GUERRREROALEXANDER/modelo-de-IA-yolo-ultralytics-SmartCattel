"""A cow that stops moving, as the live service sees it: marked, drawn and warned about once."""

from threading import Event, Thread

import numpy as np

from smartcattle_ai.live import LatestFrame, LiveState, detection_worker, mark_still
from smartcattle_ai.media import draw_detections
from smartcattle_ai.stillness import StillnessMonitor
from smartcattle_ai.tracking import IoUTracker

ZONE = (0.0, 0.0, 1.0, 1.0)


def resting_cow():
    return {"class": "cow", "confidence": 0.9, "bbox": [10, 10, 30, 30], "inside_zone": True, "track_id": 4}


def test_a_cow_past_the_limit_is_marked_and_warned_about_once(capsys):
    monitor = StillnessMonitor(still_after_s=60)
    assert "still" not in mark_still(monitor, [resting_cow()], 64, 64, 0, "camera-01")[0]
    marked = mark_still(monitor, [resting_cow()], 64, 64, 60, "camera-01")
    assert marked[0]["still"] is True and marked[0]["still_for_s"] == 60
    mark_still(monitor, [resting_cow()], 64, 64, 61, "camera-01")
    out = capsys.readouterr().out
    assert out.count("cow #4 has not moved for 1 min") == 1


def test_a_still_cow_is_drawn_differently():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    plain = draw_detections(frame, [resting_cow()])
    still = draw_detections(frame, [{**resting_cow(), "still": True, "still_for_s": 1800}])
    # Magenta box instead of green: the two frames must differ.
    assert not np.array_equal(plain, still)


def test_the_worker_passes_detections_through_the_monitor():
    class Detector:
        def detect(self, frame):
            return [{"class": "cow", "confidence": 0.9, "bbox": [10, 10, 30, 30]}]

    frames, state, stop = LatestFrame(), LiveState("camera-01", ZONE), Event()
    frames.put(np.zeros((64, 64, 3), dtype=np.uint8))
    worker = Thread(target=detection_worker, args=(Detector(), frames, state, stop),
                    kwargs={"tracker": IoUTracker(), "stillness": StillnessMonitor()})
    worker.start()
    try:
        assert state.wait_frame(0, timeout=5) is not None
    finally:
        stop.set()
        worker.join()
    cow = state.snapshot()["detections"][0]
    assert cow["track_id"] == 1 and cow["still_for_s"] == 0
