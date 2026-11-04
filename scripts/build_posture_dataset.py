"""Builds the classification dataset for standing and lying cattle from data/posture_labels.csv.

Each crop is padded to a square rather than stretched: a body along the ground is the strongest thing the classifier
can see, and stretching throws its proportions away. The training split also gets a night version of every crop,
because the cameras switch to grey infrared after dark and the labelled pictures are almost all taken by day.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartcattle_ai.posture_classifier import as_infrared_night, square_crop  # noqa: E402
from smartcattle_ai.tracking import iou  # noqa: E402

LABELS = ROOT / "data" / "posture_labels.csv"
IMAGES = ROOT / "datasets" / "cattle_coco" / "images"
#: Free-licence photographs of resting cattle (data/posture_commons_credits.csv), which the first set has too few of.
COMMONS_SPLIT = "commons"
COMMONS_IMAGES = ROOT / "datasets" / "cattle_posture_commons"
#: Open Images often labels one animal twice, as Cattle and as Bull. Counting it twice would flatter the test numbers.
SAME_ANIMAL_IOU = 0.8


def load_labels(path: Path = LABELS) -> list[dict]:
    """The labelled boxes, one per animal: boxes of one picture that overlap this much are the same animal."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kept: list[dict] = []
    for row in rows:
        row["bbox"] = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        twin = next((other for other in kept
                     if other["image"] == row["image"] and iou(other["bbox"], row["bbox"]) >= SAME_ANIMAL_IOU), None)
        if twin is None:
            kept.append(row)
    return kept


def image_path(row: dict, images: Path = IMAGES) -> Path:
    if row["split"] == COMMONS_SPLIT:
        return COMMONS_IMAGES / row["image"]
    return images / row["split"] / row["image"]


def build(out: Path, labels: list[dict], images: Path = IMAGES) -> dict:
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, int] = {}
    for row in labels:
        if row["posture"] not in ("standing", "lying"):
            continue
        image = cv2.imread(str(image_path(row, images)))
        if image is None:
            raise FileNotFoundError(row["image"])
        # Keras-style folders: Ultralytics names the classes after them, and "val" is what it validates on.
        split = "train" if row["split"] == COMMONS_SPLIT else row["split"]
        folder = out / split / row["posture"]
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{Path(row['image']).stem}_{row['box_index']}"
        crop = square_crop(image, row["bbox"])
        cv2.imwrite(str(folder / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if split == "train":
            cv2.imwrite(str(folder / f"{stem}_night.jpg"), as_infrared_night(crop, seed=hash_of(stem)),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        key = f"{split}/{row['posture']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def hash_of(text: str) -> int:
    """A seed that does not change between runs, unlike hash()."""
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "cattle_posture_cls")
    args = parser.parse_args(argv)
    print(build(args.out, load_labels()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
