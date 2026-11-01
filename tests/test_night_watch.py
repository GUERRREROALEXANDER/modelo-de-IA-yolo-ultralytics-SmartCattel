from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from smartcattle_ai.night_watch import NightWatch, parse_window

BOGOTA = ZoneInfo("America/Bogota")


def at(hour, minute=0):
    return datetime(2026, 1, 15, hour, minute, tzinfo=BOGOTA)


def person(track_id=1, cls="person"):
    return {"class": cls, "confidence": 0.9, "bbox": [0, 0, 10, 10], "track_id": track_id}


@pytest.mark.parametrize("value, expected", [
    ("20:00-05:30", (time(20, 0), time(5, 30))),
    (" 01:00-04:00 ", (time(1, 0), time(4, 0))),
    ("20:00 - 05:30", (time(20, 0), time(5, 30))),
    ("00:00-23:59", (time(0, 0), time(23, 59))),
])
def test_parse_window_accepts_valid_windows(value, expected):
    assert parse_window(value) == expected


@pytest.mark.parametrize("value", [
    "",
    "20:00",
    "20:00-",
    "-05:30",
    "20:00-05:30-06:00",
    "24:00-05:00",
    "20:00-24:00",
    "20:60-05:00",
    "20:00-05:60",
    "8:00-17:00",
    "aa:bb-cc:dd",
])
def test_parse_window_rejects_bad_windows(value):
    with pytest.raises(ValueError):
        parse_window(value)


def test_window_crossing_midnight():
    watch = NightWatch(parse_window("20:00-05:30"))
    assert watch.is_night(at(23, 0))
    assert watch.is_night(at(2, 0))
    assert watch.is_night(at(0, 0))
    assert watch.is_night(at(5, 29))
    assert not watch.is_night(at(12, 0))


def test_same_day_window():
    watch = NightWatch(parse_window("01:00-04:00"))
    assert watch.is_night(at(2, 0))
    assert not watch.is_night(at(23, 0))
    assert not watch.is_night(at(4, 0))


def test_start_is_inclusive_and_end_is_exclusive():
    crossing = NightWatch(parse_window("20:00-05:30"))
    assert crossing.is_night(at(20, 0))
    assert not crossing.is_night(at(5, 30))
    same_day = NightWatch(parse_window("01:00-04:00"))
    assert same_day.is_night(at(1, 0))
    assert not same_day.is_night(at(4, 0))


def test_utc_is_converted_to_the_configured_zone():
    watch = NightWatch(parse_window("20:00-05:30"))
    # 02:00 UTC is 21:00 the previous day in Bogota.
    assert watch.is_night(datetime(2026, 1, 16, 2, 0, tzinfo=timezone.utc))
    # 17:00 UTC is noon in Bogota.
    assert not watch.is_night(datetime(2026, 1, 16, 17, 0, tzinfo=timezone.utc))


def test_naive_datetime_is_rejected():
    watch = NightWatch(parse_window("20:00-05:30"))
    with pytest.raises(ValueError):
        watch.is_night(datetime(2026, 1, 15, 23, 0))
    with pytest.raises(ValueError):
        watch.new_intruders([person()], datetime(2026, 1, 15, 23, 0))


def test_unknown_timezone_is_rejected():
    with pytest.raises(ValueError, match="Mars/Olympus"):
        NightWatch(parse_window("20:00-05:30"), "Mars/Olympus")


def test_person_at_night_is_announced_once():
    watch = NightWatch(parse_window("20:00-05:30"))
    detection = person(7)
    assert watch.new_intruders([detection], at(23)) == [detection]
    assert watch.new_intruders([detection], at(23, 30)) == []


def test_person_by_day_is_not_announced_and_later_is():
    watch = NightWatch(parse_window("20:00-05:30"))
    detection = person(7)
    assert watch.new_intruders([detection], at(12)) == []
    assert watch.new_intruders([detection], at(21)) == [detection]


def test_cows_and_untracked_persons_are_ignored():
    watch = NightWatch(parse_window("20:00-05:30"))
    detections = [person(1, cls="cow"), person(None)]
    assert watch.new_intruders(detections, at(23)) == []
    assert len(watch._announced) == 0


def test_announced_set_is_bounded():
    watch = NightWatch(parse_window("20:00-05:30"))
    for track_id in range(1005):
        watch.new_intruders([person(track_id)], at(23))
    assert len(watch._announced) == 1000
    assert 0 not in watch._announced
    assert 1004 in watch._announced
    # An evicted id looks new again and is announced once more.
    assert watch.new_intruders([person(0)], at(23)) == [person(0)]


def test_the_input_is_not_modified():
    detections = [person(7)]
    NightWatch(parse_window("20:00-05:30")).new_intruders(detections, at(23))
    assert detections == [person(7)]
