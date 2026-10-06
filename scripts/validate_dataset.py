"""Validate cattle YOLO labels, image integrity, and split leakage."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai.media import read_image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dataset_paths(data_yaml: Path) -> dict[str, Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(config.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = data_yaml.parent / base
    return {split: (base / config[split]).resolve() for split in SPLITS}


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise ValueError(f"Image path has no images directory: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_labels(path: Path) -> tuple[list[tuple[float, ...]], list[str]]:
    boxes, errors = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        location = f"{path}:{line_number}"
        if len(fields) != 5:
            errors.append(f"Malformed label line: {location}")
            continue
        if fields[0] != "0":
            errors.append(f"Class ID other than 0: {location}")
        try:
            values = tuple(float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"Malformed coordinates: {location}")
            continue
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            errors.append(f"Coordinates outside [0,1]: {location}")
        if values[2] <= 0 or values[3] <= 0:
            errors.append(f"Zero or negative area box: {location}")
        if values[0] - values[2] / 2 < -1e-6 or values[0] + values[2] / 2 > 1 + 1e-6 or \
                values[1] - values[3] / 2 < -1e-6 or values[1] + values[3] / 2 > 1 + 1e-6:
            errors.append(f"Box extends outside image: {location}")
        boxes.append(values)
    return boxes, errors


def dhash(frame) -> int:
    gray = cv2.cvtColor(cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    value = 0
    for bit in (gray[:, :-1] > gray[:, 1:]).flat:
        value = (value << 1) | int(bit)
    return value


def validate(data_yaml: Path, report_path: Path | None = None) -> dict:
    paths = dataset_paths(data_yaml)
    errors, warnings, records = [], [], []
    stats = {}
    classes = set()
    for split, directory in paths.items():
        if not directory.is_dir():
            errors.append(f"Missing image directory: {directory}")
        images = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if directory.exists() else []
        labels_dir = label_path(directory / "placeholder.jpg").parent
        if not labels_dir.is_dir():
            errors.append(f"Missing label directory: {labels_dir}")
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for path in images:
            if path.stem not in label_stems:
                errors.append(f"Image without label file: {path}")
        for path in labels:
            if path.stem not in image_stems:
                errors.append(f"Label file without image: {path}")
        counts = []
        positives = negatives = 0
        for image in images:
            label = label_path(image)
            boxes = []
            if label.exists():
                boxes, label_errors = parse_labels(label)
                errors.extend(label_errors)
                for line in label.read_text(encoding="utf-8").splitlines():
                    if line.split():
                        classes.add(line.split()[0])
            counts.append(len(boxes))
            positives += bool(boxes)
            negatives += not boxes
            try:
                frame = read_image(image)
                records.append((split, image, hashlib.md5(image.read_bytes()).hexdigest(), dhash(frame)))
            except (OSError, ValueError, cv2.error):
                errors.append(f"Corrupt or unreadable image: {image}")
        stats[split] = {"images": len(images), "positives": positives, "negatives": negatives,
                        "annotations": sum(counts), "boxes_per_image": {
                            "min": min(counts, default=0),
                            "mean": sum(counts) / len(counts) if counts else 0,
                            "max": max(counts, default=0)}}
    duplicates = []
    for index, (split_a, path_a, md5_a, hash_a) in enumerate(records):
        for split_b, path_b, md5_b, hash_b in records[index + 1:]:
            kind = "exact" if md5_a == md5_b else "near" if (hash_a ^ hash_b).bit_count() <= 4 else None
            if kind:
                entry = {"kind": kind, "first": str(path_a), "second": str(path_b),
                         "cross_split": split_a != split_b}
                duplicates.append(entry)
                message = f"{kind.capitalize()} duplicate: {path_a} <> {path_b}"
                (errors if split_a != split_b else warnings).append(message)
    summary = {"total_images": sum(item["images"] for item in stats.values()),
               "total_annotations": sum(item["annotations"] for item in stats.values()),
               "classes_present": sorted(classes), "splits": stats,
               "duplicates": duplicates, "errors": errors, "warnings": warnings,
               "valid": not errors}
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def print_report(report: dict) -> None:
    print("| Split | Images | Positive | Negative | Annotations | Boxes min/mean/max |")
    print("|---|---:|---:|---:|---:|---:|")
    for split, item in report["splits"].items():
        counts = item["boxes_per_image"]
        print(f'| {split} | {item["images"]} | {item["positives"]} | {item["negatives"]} | '
              f'{item["annotations"]} | {counts["min"]}/{counts["mean"]:.2f}/{counts["max"]} |')
    print(f'Total images: {report["total_images"]}; total annotations: {report["total_annotations"]}; '
          f'classes present: {", ".join(report["classes_present"])}')
    print(f'Errors: {len(report["errors"])}; warnings: {len(report["warnings"])}')
    for message in report["errors"] + report["warnings"]:
        print(message)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "cattle" / "data.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "dataset_validation.json")
    args = parser.parse_args(argv)
    report = validate(args.data, args.out)
    print_report(report)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
