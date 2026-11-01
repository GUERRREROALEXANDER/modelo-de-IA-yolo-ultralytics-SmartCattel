import numpy as np
import pytest

from smartcattle_ai.heatmap import HeatMap


def cow(x1, y1, x2, y2, cls="cow"):
    return {"class": cls, "confidence": 0.9, "bbox": [x1, y1, x2, y2]}


def test_a_cow_lands_in_the_cell_holding_its_bottom_center():
    heat = HeatMap(columns=4, rows=2)
    # Bottom center is (50, 50), i.e. (0.5, 0.5): column 2 of 4, row 1 of 2.
    heat.add([cow(40, 20, 60, 50)], 100, 100)
    grid = heat.grid()
    assert grid[1][2] == 1.0
    assert grid.sum() == 1.0
    assert heat.samples == 1


def test_a_point_exactly_at_1_0_goes_in_the_last_cell():
    heat = HeatMap(columns=4, rows=2)
    heat.add([cow(100, 0, 100, 100)], 100, 100)
    grid = heat.grid()
    assert grid[1][3] == 1.0


def test_boxes_outside_the_frame_are_clipped_to_the_edges():
    negative = HeatMap(columns=4, rows=2)
    negative.add([cow(-100, -100, -40, 50)], 100, 100)
    assert negative.grid()[1][0] == 1.0
    beyond = HeatMap(columns=4, rows=2)
    beyond.add([cow(140, 250, 160, 250)], 100, 100)
    assert beyond.grid()[1][3] == 1.0


def test_other_classes_are_ignored():
    heat = HeatMap(columns=4, rows=2)
    heat.add([cow(40, 20, 60, 50, cls="person"), cow(40, 20, 60, 50, cls="dog")], 100, 100)
    assert heat.samples == 0
    assert not heat.grid().any()


def test_grid_is_normalized_and_returned_as_a_copy():
    heat = HeatMap(columns=4, rows=2)
    heat.add([cow(40, 20, 60, 50)] * 3, 100, 100)
    grid = heat.grid()
    assert grid[1][2] == 1.0
    assert grid.max() == 1.0
    assert grid.sum() == 1.0
    grid[1][2] = 0.0
    assert heat.grid()[1][2] == 1.0
    assert heat.grid() is not heat.grid()


def test_an_empty_map_is_all_zeros_and_overlay_returns_an_equal_image():
    heat = HeatMap()
    frame = np.full((40, 60, 3), 128, dtype=np.uint8)
    assert not heat.grid().any()
    out = heat.overlay(frame)
    assert out is not frame
    assert np.array_equal(out, frame)


def test_overlay_keeps_shape_and_dtype_and_does_not_modify_the_input():
    heat = HeatMap(columns=8, rows=4)
    # Bottom center (40, 40) of an 80x80 frame: column 4 of 8, row 2 of 4.
    heat.add([cow(30, 30, 50, 50)], 80, 80)
    frame = np.full((80, 80, 3), 128, dtype=np.uint8)
    before = frame.copy()
    out = heat.overlay(frame)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert np.array_equal(frame, before)
    assert not np.array_equal(out[40, 40], frame[40, 40])
    assert np.array_equal(out[0, 0], frame[0, 0])


def test_reset_clears_the_counts_and_the_samples():
    heat = HeatMap(columns=4, rows=2)
    heat.add([cow(40, 20, 60, 50)], 100, 100)
    heat.reset()
    assert heat.samples == 0
    assert not heat.grid().any()


def test_describe_reports_geometry_and_a_rounded_grid():
    heat = HeatMap(columns=4, rows=2)
    # Three cows in cell (1, 0) and one in (1, 2), so the cold cell is exactly 1/3.
    heat.add([cow(10, 10, 10, 50)] * 3 + [cow(50, 10, 50, 50)], 100, 100)
    described = heat.describe()
    assert described["columns"] == 4
    assert described["rows"] == 2
    assert described["samples"] == 4
    assert len(described["grid"]) == 2
    assert all(len(row) == 4 for row in described["grid"])
    assert described["grid"][1][0] == 1.0
    assert described["grid"][1][2] == 0.333
    assert all(isinstance(value, float) for row in described["grid"] for value in row)


@pytest.mark.parametrize("options", [
    {"columns": 1},
    {"columns": 257},
    {"columns": 0},
    {"columns": -5},
    {"columns": 2.5},
    {"columns": "32"},
    {"columns": True},
    {"rows": 1},
    {"rows": 257},
    {"rows": 0},
    {"rows": -5},
    {"rows": 2.5},
    {"rows": "18"},
    {"rows": True},
])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        HeatMap(**options)


@pytest.mark.parametrize("alpha", [-0.1, -1, 1.1, 2, "x", None])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        HeatMap().overlay(np.zeros((10, 10, 3), dtype=np.uint8), alpha)
