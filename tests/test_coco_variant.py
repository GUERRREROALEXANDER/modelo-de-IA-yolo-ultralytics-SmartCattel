"""Offline checks for the COCO-indexed dataset and semantic evaluation mapping."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import build_coco_variant as builder
from scripts.validate_dataset import parse_labels
from smartcattle_ai import evaluation


def annotation(image_id="sample", **changes):
    row = {"ImageID": image_id, "LabelName": "/m/01g317", "IsGroupOf": "0",
           "IsDepiction": "0", "XMin": "0.1", "XMax": "0.5", "YMin": "0.2", "YMax": "0.8"}
    return {**row, **changes}


def test_cow_conversion_preserves_coordinates():
    assert builder.convert_cow_labels("0 0.12345678 0.5 0.1 0.2\n\n") == [
        "19 0.12345678 0.5 0.1 0.2"]
    assert builder.convert_cow_labels("") == []
    with pytest.raises(ValueError, match="class-0"):
        builder.convert_cow_labels("19 0.5 0.5 0.1 0.1")
    assert len(builder.COCO_NAMES) == 80
    assert builder.COCO_NAMES[0] == "person"
    assert builder.COCO_NAMES[19] == "cow"


@pytest.mark.parametrize("label", sorted(builder.PERSON_LABELS))
def test_person_categories_and_clipping(label):
    assert builder.person_box(annotation(LabelName=label, XMin="-0.2", YMax="1.2")) == pytest.approx(
        (0.25, 0.6, 0.5, 0.8))


@pytest.mark.parametrize("changes", [
    {"IsGroupOf": "1"}, {"IsDepiction": "1"}, {"LabelName": "/m/0cnyhnx"},
    {"XMin": "0.499"}, {"YMin": "0.799"}, {"XMin": "0.6"}, {"XMin": "nan"},
    {"XMax": "inf"},
])
def test_person_box_exclusions(changes):
    assert builder.person_box(annotation(**changes)) is None


def test_dedup_keeps_distinct_people():
    first = (0.3, 0.5, 0.4, 0.6)
    duplicate = (0.301, 0.501, 0.4, 0.6)
    other = (0.8, 0.5, 0.2, 0.5)
    assert builder.box_iou(first, duplicate) > 0.9
    assert builder.deduplicate_person_boxes([first, first, duplicate, other]) == [first, other]


def test_build_reuses_images_and_metadata_and_is_repeatable(tmp_path, monkeypatch):
    source, out, raw = (tmp_path / name for name in ("cattle", "cattle_coco", "raw"))
    raw.mkdir()
    for split in builder.SPLITS:
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        (source / "images" / split / f"{split}.jpg").write_bytes(f"image-{split}".encode())
        (source / "labels" / split / f"{split}.txt").write_text(
            "" if split == "val" else "0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    for source_split, rows in (
        ("test", [annotation("train"), annotation("train", LabelName="/m/04yx4"),
                  annotation("train", IsGroupOf="1"), annotation("unselected")]),
        ("validation", [annotation("test", LabelName="/m/03bt1vf")]),
    ):
        with (raw / f"{source_split}-annotations-bbox.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(annotation()))
            writer.writeheader()
            writer.writerows(rows)

    def unavailable_link(*args):
        raise OSError("Hard links unavailable")

    monkeypatch.setattr(builder.os, "link", unavailable_link)
    stats_path = tmp_path / "stats.json"
    first = builder.build(source, out, raw, stats_path)
    (out / "images" / "train" / "stale.jpg").write_bytes(b"stale")
    (out / "labels" / "train" / "stale.txt").write_text("", encoding="utf-8")
    assert builder.build(source, out, raw, stats_path) == first
    for split in builder.SPLITS:
        assert [path.name for path in (out / "images" / split).iterdir()] == [f"{split}.jpg"]
        assert (out / "images" / split / f"{split}.jpg").read_bytes() == (
            source / "images" / split / f"{split}.jpg").read_bytes()
        expected = 0 if split == "val" else 1
        assert first["splits"][split] == {"images": 1, "cow_boxes": expected, "person_boxes": expected}
        label = out / "labels" / split / f"{split}.txt"
        assert not parse_labels(label, (0, 19))[1]
        assert len(label.read_text().splitlines()) == expected * 2
    assert not (out / "labels" / "train" / "stale.txt").exists()
    assert (source / "labels" / "train" / "train.txt").read_text().startswith("0 ")
    config = yaml.safe_load((out / "data.yaml").read_text())
    assert config["path"] == out.resolve().as_posix()
    assert config["names"] == builder.COCO_NAMES
    assert json.loads(stats_path.read_text()) == first


def test_validator_defaults_and_allowed_classes(tmp_path):
    label = tmp_path / "labels.txt"
    label.write_text("19 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert "Class ID other than 0" in parse_labels(label)[1][0]
    assert parse_labels(label, (0, 19))[1] == []
    label.write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert parse_labels(label, (0, 19))[1]


@pytest.mark.parametrize("names, expected", [
    ({0: "cow"}, 0), ({"0": "cow"}, 0), (builder.COCO_NAMES, 19),
])
def test_class_resolution(names, expected):
    assert evaluation.resolve_class_ids(names, ("cow",)) == {"cow": expected}


def test_missing_class_fails():
    with pytest.raises(ValueError, match="person"):
        evaluation.resolve_class_ids({0: "cow"}, ("person",))


def test_ground_truth_filters_by_name(tmp_path):
    label = tmp_path / "labels.txt"
    label.write_text("19 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    truth = evaluation.ground_truth_by_name(label, builder.COCO_NAMES, ("cow", "person"), 100, 100)
    assert truth["cow"][0] == pytest.approx([30, 30, 70, 70])
    assert truth["person"][0] == pytest.approx([10, 10, 30, 30])


@pytest.mark.parametrize("names, gt_class, requested, model_class", [
    ({0: "cow"}, 0, "cow", 19), (builder.COCO_NAMES, 19, "cow", 0),
    (builder.COCO_NAMES, 0, "person", 0),
])
def test_evaluate_remaps_independent_class_ids(tmp_path, monkeypatch, names, gt_class, requested, model_class):
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "images" / "test" / "sample.jpg").write_bytes(b"mock image")
    (tmp_path / "labels" / "test" / "sample.txt").write_text(
        f"{gt_class} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": tmp_path.as_posix(), "names": names,
                                  **{split: f"images/{split}" for split in builder.SPLITS}}), encoding="utf-8")
    box = SimpleNamespace(cls=torch.tensor(model_class), conf=torch.tensor(0.9),
                          xyxy=torch.tensor([[30., 30., 70., 70.]]))

    def predict(frame, **kwargs):
        assert kwargs["classes"] == [model_class]
        return [SimpleNamespace(boxes=[box], speed={"inference": 1.0})]

    monkeypatch.setattr(evaluation, "load_model", lambda weights: SimpleNamespace(
        names={model_class: requested}, predict=predict))
    monkeypatch.setattr(evaluation, "read_image", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    result = evaluation.evaluate("mock.pt", data, class_names=(requested,))
    assert result["dataset_class_ids"] == {requested: gt_class}
    assert result["model_class_ids"] == [model_class]
    assert result["operating"]["TP"] == 1
    assert result["operating"]["FP"] == result["operating"]["FN"] == 0


def test_evaluation_never_matches_different_classes(tmp_path, monkeypatch):
    (tmp_path / "images" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "images" / "test" / "sample.jpg").write_bytes(b"mock image")
    (tmp_path / "labels" / "test" / "sample.txt").write_text(
        "19 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": tmp_path.as_posix(), "names": builder.COCO_NAMES,
                                  **{split: f"images/{split}" for split in builder.SPLITS}}), encoding="utf-8")
    # A person detection on exactly the cow's box must be FP, leaving the cow FN.
    box = SimpleNamespace(cls=torch.tensor(0), conf=torch.tensor(0.9),
                          xyxy=torch.tensor([[30., 30., 70., 70.]]))
    model = SimpleNamespace(names={0: "person", 19: "cow"}, predict=lambda *args, **kwargs: [
        SimpleNamespace(boxes=[box], speed={"inference": 1.0})])
    monkeypatch.setattr(evaluation, "load_model", lambda weights: model)
    monkeypatch.setattr(evaluation, "read_image", lambda path: np.zeros((100, 100, 3), dtype=np.uint8))
    result = evaluation.evaluate("mock.pt", data, class_names=("cow", "person"))
    assert result["operating"]["TP"] == 0
    assert result["operating"]["FP"] == result["operating"]["FN"] == 1
    assert result["per_class"]["person"]["operating"]["false_positives_on_negative_images"] == 1
