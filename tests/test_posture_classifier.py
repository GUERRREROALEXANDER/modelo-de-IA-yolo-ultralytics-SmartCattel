"""The trained classifier that ships with the repository, and the pictures it is given."""

import numpy as np
import pytest

from smartcattle_ai.posture_classifier import (
    DEFAULT_WEIGHTS,
    INPUT_SIZE,
    PostureClassifier,
    as_infrared_night,
    load_default_classifier,
    square_crop,
)


def test_a_crop_keeps_the_proportions_of_the_body():
    frame = np.full((480, 640, 3), 255, np.uint8)
    crop = square_crop(frame, [100, 200, 300, 250])
    assert crop.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    # Padding above and below, rather than a stretched body.
    assert crop[0, INPUT_SIZE // 2].sum() == 0 and crop[INPUT_SIZE // 2, INPUT_SIZE // 2].sum() > 0


@pytest.mark.parametrize("bbox", [[-20, -20, 50, 50], [630, 470, 700, 700], [10, 10, 10, 10]])
def test_a_box_at_or_past_the_edge_still_gives_a_crop(bbox):
    assert square_crop(np.zeros((480, 640, 3), np.uint8), bbox).shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_the_night_version_is_grey_and_dimmer_and_always_the_same():
    image = np.random.default_rng(1).integers(100, 255, (64, 64, 3), np.uint8)
    night = as_infrared_night(image, seed=3)
    assert (night[:, :, 0] == night[:, :, 1]).all() and night.mean() < image.mean()
    assert (night == as_infrared_night(image, seed=3)).all()


def test_no_boxes_loads_nothing():
    assert PostureClassifier("does-not-exist.pt").predict(np.zeros((8, 8, 3), np.uint8), []) == []


def test_a_copy_without_the_weights_runs_without_the_classifier(monkeypatch, tmp_path):
    monkeypatch.setattr("smartcattle_ai.posture_classifier.DEFAULT_WEIGHTS", tmp_path / "missing.pt")
    assert load_default_classifier() is None


def test_the_installed_model_answers_one_of_the_two_postures_for_each_box():
    assert DEFAULT_WEIGHTS.is_file(), "models/cattle_posture_cls.pt must ship with the repository"
    frame = np.random.default_rng(2).integers(0, 255, (240, 320, 3), np.uint8)
    answers = load_default_classifier().predict(frame, [[10, 10, 100, 80], [100, 50, 300, 200]])
    assert len(answers) == 2
    assert all(name in ("lying", "standing") and 0 <= confidence <= 1 for name, confidence in answers)
