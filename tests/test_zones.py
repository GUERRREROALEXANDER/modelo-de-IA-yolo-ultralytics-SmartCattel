import pytest

from smartcattle_ai.rules import evaluate
from smartcattle_ai.zones import SafeArea, ZoneSet, as_zone_set

TRIANGLE = ({"x": 0.2, "y": 0.2}, {"x": 0.8, "y": 0.2}, {"x": 0.5, "y": 0.8})


def backend_zone(name="Pen", active=True, outline=None, **rect):
    return {"id": 1, "camera_id": "camera-01", "name": name, "active": active, "outline": outline,
            "x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9, "created_at": "2026-10-31T00:00:00Z", **rect}


def test_a_rectangle_contains_its_border():
    area = SafeArea(0.1, 0.1, 0.9, 0.9)
    assert area.contains(0.5, 0.5) and area.contains(0.1, 0.9)
    assert not area.contains(0.95, 0.5)


def test_a_polygon_excludes_the_corners_of_its_bounding_box():
    zones = ZoneSet.from_backend([backend_zone(outline=list(TRIANGLE), x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.8)])
    assert zones.contains(0.5, 0.4)
    # Inside the bounding box, outside the triangle: an animal there has escaped.
    assert not zones.contains(0.25, 0.75)
    # On an edge counts as inside, like the rectangle's border.
    assert zones.contains(0.5, 0.2)


def test_an_animal_in_any_zone_is_safe():
    zones = ZoneSet.from_backend([
        backend_zone("West", x_min=0.0, x_max=0.3), backend_zone("East", x_min=0.7, x_max=1.0),
    ])
    assert zones.contains(0.1, 0.5) and zones.contains(0.9, 0.5)
    assert not zones.contains(0.5, 0.5)
    assert [item["name"] for item in zones.describe()] == ["West", "East"]


def test_inactive_zones_are_ignored_and_none_means_no_zone():
    assert ZoneSet.from_backend([backend_zone(active=False)]) is None
    assert ZoneSet.from_backend([]) is None
    assert ZoneSet.from_backend([backend_zone()]).source == "backend"


def test_equal_zone_sets_compare_equal():
    assert ZoneSet.from_backend([backend_zone()]) == ZoneSet.from_backend([backend_zone()])
    assert ZoneSet.from_backend([backend_zone(name="")]) == ZoneSet.from_rectangle((0.1, 0.1, 0.9, 0.9))
    # A rename in the dashboard is a change, so the service announces it.
    assert ZoneSet.from_backend([backend_zone("Pen")]) != ZoneSet.from_backend([backend_zone("Corral")])
    assert ZoneSet.from_rectangle((0, 0, 1, 1)) != ZoneSet.from_rectangle((0.1, 0.1, 0.9, 0.9))


def test_the_classic_tuple_still_works_everywhere():
    assert as_zone_set((0.1, 0.1, 0.9, 0.9)).areas == (SafeArea(0.1, 0.1, 0.9, 0.9),)
    evaluated, _ = evaluate([{"class": "cow", "bbox": [0, 0, 10, 100]}], 100, 100, (0.2, 0.2, 0.8, 0.8))
    assert evaluated[0]["inside_zone"] is False


def test_a_zone_set_needs_an_area():
    with pytest.raises(ValueError):
        ZoneSet([])
