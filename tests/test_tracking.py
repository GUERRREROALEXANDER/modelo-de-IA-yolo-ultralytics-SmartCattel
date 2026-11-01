import pytest

from smartcattle_ai.tracking import IoUTracker, iou


def cow(x, y=0, size=10, cls="cow"):
    return {"class": cls, "confidence": 0.9, "bbox": [x, y, x + size, y + size]}


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_an_animal_keeps_its_number_while_it_walks():
    tracker = IoUTracker()
    first = tracker.update([cow(0), cow(100)])
    moved = tracker.update([cow(102), cow(2)])
    assert [item["track_id"] for item in first] == [1, 2]
    # Matched by position, not by order in the list.
    assert [item["track_id"] for item in moved] == [2, 1]


def test_a_new_animal_gets_a_new_number():
    tracker = IoUTracker()
    tracker.update([cow(0)])
    assert [item["track_id"] for item in tracker.update([cow(0), cow(200)])] == [1, 2]


def test_a_person_never_takes_a_cows_number():
    tracker = IoUTracker()
    tracker.update([cow(0)])
    assert tracker.update([cow(0, cls="person")])[0]["track_id"] == 2


def test_a_hidden_animal_survives_a_few_frames_then_is_forgotten():
    tracker = IoUTracker(max_missed=2)
    tracker.update([cow(0)])
    tracker.update([])
    tracker.update([])
    assert tracker.update([cow(0)])[0]["track_id"] == 1
    for _ in range(3):
        tracker.update([])
    assert tracker.active == 0
    assert tracker.update([cow(0)])[0]["track_id"] == 2


def test_the_input_is_not_modified():
    detections = [cow(0)]
    IoUTracker().update(detections)
    assert "track_id" not in detections[0]


@pytest.mark.parametrize("options", [{"iou_threshold": 0}, {"iou_threshold": 1.5}, {"max_missed": -1}])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        IoUTracker(**options)
