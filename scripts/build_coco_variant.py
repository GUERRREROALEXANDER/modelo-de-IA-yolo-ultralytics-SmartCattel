"""Build an 80-class COCO-indexed variant using only local images and metadata."""

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_LABELS = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
assert len(COCO_NAMES) == 80
assert COCO_NAMES[0] == "person" and COCO_NAMES[19] == "cow"


def convert_cow_labels(text: str) -> list[str]:
    """Remap cleaned cow labels without rounding their coordinates."""
    converted = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Expected a cleaned class-0 cow label: {line!r}")
        converted.append("19 " + " ".join(fields[1:]))
    return converted


def person_box(row: dict) -> tuple[float, ...] | None:
    if row["LabelName"] not in PERSON_LABELS:
        return None
    if float(row["IsGroupOf"]) == 1 or float(row["IsDepiction"]) == 1:
        return None
    corners = [float(row[key]) for key in ("XMin", "XMax", "YMin", "YMax")]
    if not all(math.isfinite(value) for value in corners):
        return None
    left, right, top, bottom = [min(1.0, max(0.0, value)) for value in corners]
    width, height = right - left, bottom - top
    if width < 0.002 or height < 0.002:
        return None
    return ((left + right) / 2, (top + bottom) / 2, width, height)


def box_iou(first, second) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    width = max(0.0, min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2))
    height = max(0.0, min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2))
    intersection = width * height
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_person_boxes(boxes) -> list[tuple[float, ...]]:
    """Keep the first box when IoU with a previously retained box exceeds 0.9."""
    kept = []
    for box in boxes:
        if not any(box_iou(box, existing) > 0.9 for existing in kept):
            kept.append(box)
    return kept


def build(source: Path, out: Path, raw: Path, stats_path: Path) -> dict:
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    images = {}
    cow_labels = {}
    for split in SPLITS:
        directory = source / "images" / split
        images[split] = sorted(path for path in directory.iterdir()
                               if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for image in images[split]:
            label = source / "labels" / split / (image.stem + ".txt")
            cow_labels[split, image.stem] = convert_cow_labels(label.read_text(encoding="utf-8"))
    image_ids = {image.stem for items in images.values() for image in items}
    people = defaultdict(list)
    for source_split in ("validation", "test"):
        with (raw / f"{source_split}-annotations-bbox.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["ImageID"] in image_ids:
                    box = person_box(row)
                    if box is not None:
                        people[row["ImageID"]].append(box)
    people = {key: deduplicate_person_boxes(boxes) for key, boxes in people.items()}
    stats = {"splits": {}}
    for split in SPLITS:
        image_dir, label_dir = out / "images" / split, out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale variant files so reruns preserve the cleaned dataset's splits.
        expected_images = {image.name for image in images[split]}
        expected_labels = {image.stem + ".txt" for image in images[split]}
        for directory, expected, suffixes in (
                (image_dir, expected_images, IMAGE_SUFFIXES), (label_dir, expected_labels, {".txt"})):
            for existing in directory.iterdir():
                if existing.is_file() and existing.suffix.lower() in suffixes and existing.name not in expected:
                    existing.unlink()
        counts = {"images": len(images[split]), "cow_boxes": 0, "person_boxes": 0}
        for image in images[split]:
            target = image_dir / image.name
            if not target.exists() or not os.path.samefile(image, target):
                target.unlink(missing_ok=True)
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)
            cows = cow_labels[split, image.stem]
            persons = people.get(image.stem, [])
            lines = cows + ["0 " + " ".join(f"{value:.8f}" for value in box) for box in persons]
            (label_dir / (image.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["cow_boxes"] += len(cows)
            counts["person_boxes"] += len(persons)
        stats["splits"][split] = counts
        print(f'{split}: images={counts["images"]}, cow boxes={counts["cow_boxes"]}, '
              f'person boxes={counts["person_boxes"]}')
    config = {"path": out.as_posix(), **{split: f"images/{split}" for split in SPLITS},
              "names": COCO_NAMES}
    (out / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets" / "cattle")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "cattle_coco")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "openimages")
    parser.add_argument("--stats", type=Path, default=ROOT / "data" / "dataset_coco_stats.json")
    args = parser.parse_args(argv)
    build(args.source, args.out, args.raw, args.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
