from copy import deepcopy

from smartcattle_ai.rules import evaluate


def test_inside_outside_and_boundary_without_mutation():
    detections = [
        {"bbox": [20, 10, 40, 50], "class": "cow"},
        {"bbox": [80, 10, 100, 50], "class": "cow"},
        {"bbox": [10, 10, 30, 20], "class": "cow"},
    ]
    original = deepcopy(detections)
    evaluated, events = evaluate(detections, 100, 100, (0.2, 0.2, 0.8, 0.5))
    assert [item["inside_zone"] for item in evaluated] == [True, False, True]
    assert len(events) == 1
    assert events[0]["type"] == "cattle_outside_zone"
    assert events[0]["detection"] == evaluated[1]
    assert detections == original


def test_empty():
    assert evaluate([], 100, 100, (0, 0, 1, 1)) == ([], [])
