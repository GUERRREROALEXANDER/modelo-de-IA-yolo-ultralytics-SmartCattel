import pytest

from smartcattle_ai.line_counter import LineCounter, parse_line


# A vertical gate in the middle of the frame, spanning y from 0.4 to 0.6.
GATE = (0.5, 0.6, 0.5, 0.4)


def cow(x, y=500, track_id=1, cls="cow"):
    return {"class": cls, "confidence": 0.9, "bbox": [x - 20, y - 40, x + 20, y], "track_id": track_id}


def test_a_cow_walking_across_the_gate_counts_in_once():
    counter = LineCounter(GATE)
    assert counter.update([cow(400)], 1000, 1000, now=0) == []
    crossings = counter.update([cow(600)], 1000, 1000, now=1)
    assert crossings == [{"track_id": 1, "direction": "in"}]
    assert counter.entered == 1
    assert counter.exited == 0
    assert counter.inside_count == 1
    # Staying on the far side is not another crossing.
    assert counter.update([cow(650)], 1000, 1000, now=2) == []


def test_walking_back_counts_out():
    counter = LineCounter(GATE)
    counter.update([cow(600)], 1000, 1000, now=0)
    crossings = counter.update([cow(400)], 1000, 1000, now=1)
    assert crossings == [{"track_id": 1, "direction": "out"}]
    assert counter.entered == 0
    assert counter.exited == 1
    assert counter.inside_count == -1


def test_staying_on_one_side_or_standing_on_the_line_counts_nothing():
    counter = LineCounter(GATE)
    for x in (400, 410, 420):
        assert counter.update([cow(x)], 1000, 1000, now=x) == []
    # Zero side means on the line: ignored, the previous side is kept.
    assert counter.update([cow(500)], 1000, 1000, now=500) == []
    assert counter.update([cow(500)], 1000, 1000, now=501) == []
    assert counter.update([cow(400)], 1000, 1000, now=502) == []
    assert counter.entered == 0
    assert counter.exited == 0


def test_passing_beyond_the_end_of_the_segment_does_not_count():
    counter = LineCounter(GATE)
    counter.update([cow(400, y=800)], 1000, 1000, now=0)
    assert counter.update([cow(600, y=800)], 1000, 1000, now=1) == []
    assert counter.entered == 0
    assert counter.exited == 0


def test_persons_and_cows_without_a_track_id_are_ignored():
    counter = LineCounter(GATE)
    counter.update([cow(400, cls="person"), cow(400, track_id=None)], 1000, 1000, now=0)
    crossings = counter.update([cow(600, cls="person"), cow(600, track_id=None)], 1000, 1000, now=1)
    assert crossings == []
    assert counter.entered == 0
    assert counter.exited == 0


def test_two_cows_crossing_at_once_count_twice():
    counter = LineCounter(GATE)
    counter.update([cow(400, y=450, track_id=7), cow(400, y=550, track_id=9)], 1000, 1000, now=0)
    crossings = counter.update([cow(600, y=450, track_id=7), cow(600, y=550, track_id=9)], 1000, 1000, now=1)
    assert crossings == [
        {"track_id": 7, "direction": "in"},
        {"track_id": 9, "direction": "in"},
    ]
    assert counter.entered == 2


def test_a_forgotten_track_starts_fresh():
    counter = LineCounter(GATE, forget_after_s=10)
    counter.update([cow(400)], 1000, 1000, now=0)
    # The gap is longer than the memory, so the old side is gone and this is no crossing.
    assert counter.update([cow(600)], 1000, 1000, now=20) == []
    assert counter.inside_count == 0
    # The same cow now counts normally from the side it was just seen on.
    assert counter.update([cow(400)], 1000, 1000, now=21) == [{"track_id": 1, "direction": "out"}]


def test_describe_reports_the_line_and_the_totals():
    counter = LineCounter(GATE)
    counter.update([cow(400)], 1000, 1000, now=0)
    counter.update([cow(600)], 1000, 1000, now=1)
    assert counter.describe() == {"line": [0.5, 0.6, 0.5, 0.4], "entered": 1, "exited": 0}


def test_the_input_is_not_modified():
    detections = [cow(400), cow(400, cls="person", track_id=None)]
    before = [dict(detection) for detection in detections]
    LineCounter(GATE).update(detections, 1000, 1000, now=0)
    assert detections == before
    assert detections[0]["bbox"] == [380, 460, 420, 500]


@pytest.mark.parametrize("value,expected", [
    ("0.5,0.6,0.5,0.4", (0.5, 0.6, 0.5, 0.4)),
    (" 0 , 0 , 1 , 1 ", (0.0, 0.0, 1.0, 1.0)),
    ("0,0,0.25,1", (0.0, 0.0, 0.25, 1.0)),
])
def test_parse_line_accepts_normalized_lines(value, expected):
    assert parse_line(value) == expected


@pytest.mark.parametrize("value", [
    "",
    "0.1,0.2,0.3",
    "0.1,0.2,0.3,0.4,0.5",
    "0.1,0.2,0.3,wide",
    "1.5,0.2,0.3,0.4",
    "-0.1,0.2,0.3,0.4",
    "0.1,nan,0.3,0.4",
    "0.1,inf,0.3,0.4",
    "0.5,0.5,0.5,0.5",
])
def test_parse_line_rejects_bad_lines(value):
    with pytest.raises(ValueError):
        parse_line(value)


@pytest.mark.parametrize("options", [
    {"forget_after_s": 0},
    {"forget_after_s": -1},
])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        LineCounter(GATE, **options)


def test_a_gate_with_equal_points_is_invalid():
    with pytest.raises(ValueError):
        LineCounter((0.5, 0.5, 0.5, 0.5))
