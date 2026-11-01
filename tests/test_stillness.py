import pytest

from smartcattle_ai.stillness import StillnessMonitor


def cow(x, y=100, track_id=1, cls="cow"):
    return {"class": cls, "confidence": 0.9, "bbox": [x, y, x + 40, y + 40], "track_id": track_id}


def test_a_still_cow_alerts_exactly_once():
    monitor = StillnessMonitor(still_after_s=60)
    monitor.update([cow(100)], 1000, 1000, now=0)
    before = monitor.update([cow(100)], 1000, 1000, now=59)
    assert before[0]["still_for_s"] == 59.0
    assert monitor.new_alerts(before) == []
    reached = monitor.update([cow(100)], 1000, 1000, now=60)
    assert reached[0]["still_for_s"] == 60.0
    assert monitor.new_alerts(reached) == [1]
    assert monitor.new_alerts(reached) == []
    later = monitor.update([cow(100)], 1000, 1000, now=120)
    assert monitor.new_alerts(later) == []


def test_small_jitter_under_the_tolerance_does_not_reset():
    monitor = StillnessMonitor(still_after_s=60)
    monitor.update([cow(100)], 1000, 1000, now=0)
    # 10 px of drift is 0.01 of the frame, below the default 0.02 tolerance.
    out = monitor.update([cow(110)], 1000, 1000, now=60)
    assert out[0]["still_for_s"] == 60.0
    assert monitor.new_alerts(out) == [1]


def test_moving_resets_and_allows_a_second_alert():
    monitor = StillnessMonitor(still_after_s=60, forget_after_s=1000)
    monitor.update([cow(100)], 1000, 1000, now=0)
    assert monitor.new_alerts(monitor.update([cow(100)], 1000, 1000, now=60)) == [1]
    moved = monitor.update([cow(300)], 1000, 1000, now=70)
    assert moved[0]["still_for_s"] == 0.0
    assert monitor.new_alerts(moved) == []
    again = monitor.update([cow(300)], 1000, 1000, now=130)
    assert again[0]["still_for_s"] == 60.0
    assert monitor.new_alerts(again) == [1]


def test_persons_and_untracked_cows_are_not_monitored():
    monitor = StillnessMonitor()
    detections = [cow(100, cls="person"), cow(100, track_id=None)]
    out = monitor.update(detections, 1000, 1000, now=0)
    assert out == detections
    assert out is not detections
    assert out[0] is not detections[0]
    assert monitor.tracked == 0
    assert monitor.new_alerts(out) == []


def test_the_input_is_not_modified():
    detections = [cow(100)]
    StillnessMonitor().update(detections, 1000, 1000, now=0)
    assert detections[0] == cow(100)
    assert "still_for_s" not in detections[0]


def test_a_forgotten_track_is_removed():
    monitor = StillnessMonitor(forget_after_s=10)
    monitor.update([cow(100)], 1000, 1000, now=0)
    assert monitor.tracked == 1
    monitor.update([], 1000, 1000, now=10)
    assert monitor.tracked == 1
    monitor.update([], 1000, 1000, now=10.5)
    assert monitor.tracked == 0
    # Coming back after being forgotten starts a fresh clock.
    assert monitor.update([cow(100)], 1000, 1000, now=20)[0]["still_for_s"] == 0.0


@pytest.mark.parametrize("options", [
    {"still_after_s": 0},
    {"still_after_s": -1},
    {"tolerance": 0},
    {"tolerance": 1},
    {"tolerance": 1.5},
    {"forget_after_s": 0},
    {"forget_after_s": -5},
])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        StillnessMonitor(**options)
