"""People at night on the live camera: flagged, drawn in red and warned about once."""

from datetime import datetime, timezone

import numpy as np

from smartcattle_ai.live import mark_intruders
from smartcattle_ai.media import draw_detections
from smartcattle_ai.night_watch import NightWatch, parse_window

# 02:00 UTC is 21:00 in Bogota, inside 20:00-05:30; 17:00 UTC is noon there.
NIGHT = datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
NOON = datetime(2026, 11, 1, 17, 0, tzinfo=timezone.utc)


def person(track_id=1):
    return {"class": "person", "confidence": 0.7, "bbox": [10, 10, 30, 50], "track_id": track_id}


def watch():
    return NightWatch(parse_window("20:00-05:30"), "America/Bogota")


def test_by_day_nobody_is_flagged(capsys):
    detections = [person()]
    assert mark_intruders(watch(), detections, NOON, "camera-01") == detections
    assert capsys.readouterr().out == ""


def test_at_night_people_are_flagged_and_warned_about_once(capsys):
    night = watch()
    cow = {"class": "cow", "confidence": 0.9, "bbox": [40, 10, 60, 30], "track_id": 2}
    first = mark_intruders(night, [person(), cow, {**person(), "track_id": None}], NIGHT, "camera-01")
    assert [item.get("night") for item in first] == [True, None, None]
    # Still flagged on the next frame, but the warning is not repeated.
    assert mark_intruders(night, [person()], NIGHT, "camera-01")[0]["night"] is True
    assert capsys.readouterr().out.count("person #1 in view at night") == 1


def test_a_person_at_night_is_drawn_differently():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert not np.array_equal(draw_detections(frame, [person()]), draw_detections(frame, [{**person(), "night": True}]))
