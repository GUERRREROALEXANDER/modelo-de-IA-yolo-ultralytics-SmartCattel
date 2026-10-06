"""Build a licensed, single-class cattle dataset from Open Images V7."""

import argparse
import csv
import json
import math
import random
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai.media import read_image, save_image

CATTLE = {"/m/01xq0k1", "/m/0cnyhnx"}
BACKGROUND = {"/m/03k3r", "/m/07bgp", "/m/03fwl", "/m/0bt9lr", "/m/09kx5"}
RAW = ROOT / "data" / "raw" / "openimages"
ATTRIBUTION_COLUMNS = ("image_id", "split", "role", "license", "author", "title",
                       "original_url", "landing_url")


def download(url: str, target: Path) -> None:
    """Download with the system store, certifi when available, then curl.exe."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    contexts: list[ssl.SSLContext | None] = [None]
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    last_error = None
    for context in contexts:
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30, context=context) as response:
                    with temporary.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                temporary.replace(target)
                return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
    for _ in range(3):
        try:
            subprocess.run(["curl.exe", "--fail", "--location", "--silent", "--show-error",
                            "--max-time", "30", "--output", str(temporary), url],
                           check=True, timeout=35, capture_output=True)
            temporary.replace(target)
            return
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed: {url}: {last_error}")


def metadata_path(split: str, kind: str) -> Path:
    if kind == "annotations":
        name = f"{split}-annotations-bbox.csv"
        url = f"https://storage.googleapis.com/openimages/v5/{name}"
    else:
        name = f"{split}-images-with-rotation.csv"
        url = f"https://storage.googleapis.com/openimages/2018_04/{split}/{name}"
    path = RAW / name
    if not path.exists():
        download(url, path)
    return path


def yolo_box(row: dict) -> tuple[float, float, float, float] | None:
    """Clip an Open Images box and return normalized xywh, or None if too small."""
    try:
        left, right, top, bottom = (float(row[key]) for key in ("XMin", "XMax", "YMin", "YMax"))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid Open Images box: {row}") from exc
    if not all(math.isfinite(value) for value in (left, right, top, bottom)):
        return None
    left, right, top, bottom = (max(0.0, min(1.0, value)) for value in
                                (left, right, top, bottom))
    width, height = right - left, bottom - top
    if width < 0.002 or height < 0.002:
        return None
    return ((left + right) / 2, (top + bottom) / 2, width, height)


def load_review() -> set[str]:
    """Image IDs excluded after visual review (see data/label_review.csv)."""
    path = ROOT / "data" / "label_review.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["image_id"] for row in csv.DictReader(stream) if row["decision"] == "exclude"}


def collect(split: str, limit: int | None) -> tuple[list[dict], dict]:
    cattle = defaultdict(list)
    background = set()
    with metadata_path(split, "annotations").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["LabelName"] in CATTLE:
                cattle[row["ImageID"]].append(row)
            elif row["LabelName"] in BACKGROUND:
                background.add(row["ImageID"])
    exclusions = {"group": 0, "depiction": 0, "invalid_boxes": 0}
    positives = []
    for image_id, rows in sorted(cattle.items()):
        group = any(row["IsGroupOf"] == "1" for row in rows)
        depiction = any(row["IsDepiction"] == "1" for row in rows)
        exclusions["group"] += group
        exclusions["depiction"] += depiction
        if group or depiction:
            continue
        boxes = [box for row in rows if (box := yolo_box(row)) is not None]
        if not boxes:
            exclusions["invalid_boxes"] += 1
            continue
        positives.append({"image_id": image_id, "source_split": split,
                          "role": "positive", "boxes": boxes})
    if limit is not None:
        positives = positives[:limit]
    candidates = sorted(background - cattle.keys())
    negatives = random.Random(42).sample(candidates, min(len(candidates), round(len(positives) * 0.12)))
    selected = positives + [{"image_id": image_id, "source_split": split,
                             "role": "negative", "boxes": []} for image_id in negatives]
    by_id = {item["image_id"]: item for item in selected}
    with metadata_path(split, "images").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            item = by_id.get(row["ImageID"])
            if item is not None:
                item["metadata"] = row
    missing = [item["image_id"] for item in selected if "metadata" not in item]
    if missing:
        raise ValueError(f"Missing image metadata for {split}: {missing[:5]}")
    # Boxes refer to the rotated image while S3 serves it unrotated, so skip rotated images.
    rotated = [item for item in selected if item["metadata"]["Rotation"] not in ("", "0.0")]
    exclusions["rotated"] = len(rotated)
    selected = [item for item in selected if item not in rotated]
    # Manual review: Open Images "Cattle" also tags bison, buffalo, yaks, goats and sheep.
    reviewed = load_review()
    exclusions["manual_review"] = sum(item["image_id"] in reviewed for item in selected)
    selected = [item for item in selected if item["image_id"] not in reviewed]
    return selected, {"cattle_images": len(cattle), "exclusions": exclusions,
                      "selected_positive": sum(item["role"] == "positive" for item in selected),
                      "selected_negative": sum(item["role"] == "negative" for item in selected)}


def prepare_image(item: dict, out: Path) -> tuple[dict, str | None]:
    target = out / "images" / item["split"] / f'{item["image_id"]}.jpg'
    try:
        if target.exists():
            try:
                read_image(target)
                return item, None
            except ValueError:
                target.unlink()
        url = (f'https://s3.amazonaws.com/open-images-dataset/{item["source_split"]}/'
               f'{item["image_id"]}.jpg')
        raw = RAW / item["source_split"] / f'{item["image_id"]}.jpg'
        if not raw.exists():
            download(url, raw)
        try:
            frame = read_image(raw)
        except ValueError:
            raw.unlink(missing_ok=True)
            download(url, raw)
            frame = read_image(raw)
        longest = max(frame.shape[:2])
        if longest > 1024:
            scale = 1024 / longest
            size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        save_image(target, frame)
        return item, None
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        target.unlink(missing_ok=True)
        return item, str(exc)


def build(out: Path, limit: int | None = None, workers: int = 8) -> dict:
    source_test, validation_stats = collect("validation", limit)
    source_training, test_stats = collect("test", limit)
    overlap = {item["image_id"] for item in source_test} & {
        item["image_id"] for item in source_training}
    if overlap:
        raise ValueError(f"Open Images source splits overlap: {sorted(overlap)[:5]}")
    rng = random.Random(42)
    rng.shuffle(source_training)
    boundary = round(len(source_training) * 0.85)
    for index, item in enumerate(source_training):
        item["split"] = "train" if index < boundary else "val"
    for item in source_test:
        item["split"] = "test"
    items = source_training + source_test
    selected = {(item["split"], item["image_id"]) for item in items}
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for folder, extension in (("images", ".jpg"), ("labels", ".txt")):
            for existing in (out / folder / split).glob(f"*{extension}"):
                if (split, existing.stem) not in selected:
                    existing.unlink()
    failures = []
    completed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_image, item, out) for item in items]
        for future in as_completed(futures):
            item, error = future.result()
            if error:
                failures.append({"image_id": item["image_id"], "error": error})
                (out / "labels" / item["split"] / f'{item["image_id"]}.txt').unlink(missing_ok=True)
            else:
                completed.append(item)
    completed.sort(key=lambda item: (item["split"], item["image_id"]))
    attribution = ROOT / "data" / "ATTRIBUTION.csv"
    attribution.parent.mkdir(parents=True, exist_ok=True)
    with attribution.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ATTRIBUTION_COLUMNS)
        writer.writeheader()
        for item in completed:
            label = out / "labels" / item["split"] / f'{item["image_id"]}.txt'
            label.write_text("".join("0 " + " ".join(f"{value:.8f}" for value in box) + "\n"
                                     for box in item["boxes"]), encoding="utf-8")
            metadata = item["metadata"]
            writer.writerow({"image_id": item["image_id"], "split": item["split"],
                             "role": item["role"], "license": metadata["License"],
                             "author": metadata["Author"], "title": metadata["Title"],
                             "original_url": metadata["OriginalURL"],
                             "landing_url": metadata["OriginalLandingURL"]})
    out = out.resolve()
    (out / "data.yaml").write_text(
        f'path: "{out.as_posix()}"\ntrain: images/train\nval: images/val\n'
        'test: images/test\nnames: {0: cow}\n', encoding="utf-8")
    stats = {"source_validation": validation_stats, "source_test": test_stats,
             "splits": {split: {"images": sum(item["split"] == split for item in completed),
                                "positives": sum(item["split"] == split and item["role"] == "positive"
                                                 for item in completed),
                                "negatives": sum(item["split"] == split and item["role"] == "negative"
                                                 for item in completed)}
                        for split in ("train", "val", "test")},
             "failed_or_corrupt": failures}
    (ROOT / "data" / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Maximum positive images per source split")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "cattle")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1 or args.workers < 1:
        parser.error("--limit and --workers must be positive")
    print(json.dumps(build(args.out, args.limit, args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
