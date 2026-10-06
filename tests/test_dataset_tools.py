"""Offline checks for dataset integrity and cattle metrics."""

from pathlib import Path

import numpy as np
import pytest

from scripts.build_dataset import yolo_box
from scripts.validate_dataset import validate
from smartcattle_ai.evaluation import summarize
from smartcattle_ai.media import save_image


def fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "cattle"
    for split, names in {"train": ("a", "b"), "val": ("c",), "test": ("d",)}.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for index, name in enumerate(names):
            image = np.full((32, 32, 3), index * 40 + len(split), dtype=np.uint8)
            save_image(root / "images" / split / f"{name}.jpg", image)
            (root / "labels" / split / f"{name}.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    yaml = root / "data.yaml"
    yaml.write_text(f'path: "{root.as_posix()}"\ntrain: images/train\nval: images/val\n'
                    'test: images/test\nnames: {0: cow}\n', encoding="utf-8")
    return yaml


def test_validation_detects_bad_box_and_cross_split_duplicate(tmp_path):
    yaml = fake_dataset(tmp_path)
    root = yaml.parent
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0 0.25\n", encoding="utf-8")
    (root / "images" / "test" / "d.jpg").write_bytes(
        (root / "images" / "train" / "a.jpg").read_bytes())
    result = validate(yaml, tmp_path / "report.json")
    assert result["total_images"] == 4
    assert any("Zero or negative area" in error for error in result["errors"])
    assert any("duplicate" in error.lower() for error in result["errors"])
    assert any(item["kind"] == "exact" and item["cross_split"] for item in result["duplicates"])
    assert not result["valid"]


def test_missing_or_malformed_labels_and_corrupt_image(tmp_path):
    yaml = fake_dataset(tmp_path)
    root = yaml.parent
    (root / "labels" / "train" / "a.txt").unlink()
    (root / "labels" / "val" / "c.txt").write_text("3 2 0.5 0.2 0.2\ninvalid\n",
                                                      encoding="utf-8")
    (root / "images" / "test" / "d.jpg").write_bytes(b"broken")
    (root / "labels" / "test" / "orphan.txt").write_text("", encoding="utf-8")
    errors = "\n".join(validate(yaml)["errors"])
    for phrase in ("Image without label", "Label file without image", "Class ID other than 0",
                   "Coordinates outside", "Malformed label line", "Corrupt or unreadable"):
        assert phrase in errors


def test_open_images_box_conversion():
    box = yolo_box({"XMin": "-0.2", "XMax": "0.6", "YMin": "0.2", "YMax": "1.2"})
    assert box == pytest.approx((0.3, 0.6, 0.6, 0.8))
    assert yolo_box({"XMin": "0.1", "XMax": "0.1001",
                     "YMin": "0.2", "YMax": "0.5"}) is None


def test_perfect_and_missing_predictions():
    truth = [[10, 10, 30, 30]]
    perfect = summarize([{"ground_truth": truth, "predictions": truth,
                          "confidences": [0.9]}])
    missing = summarize([{"ground_truth": truth, "predictions": [],
                          "confidences": []}])
    assert perfect["mAP50"] == pytest.approx(1.0, abs=0.01)
    assert perfect["operating"]["TP"] == 1
    assert missing["mAP50"] == 0
    assert missing["operating"]["FN"] == 1
