"""Trains the standing/lying classifier and measures it honestly, on pictures it never saw.

The labelled set is small, so one held-out split would rest on a handful of lying animals. Instead: five folds with
every picture wholly on one side, because two cows of one photograph share its light, grass and camera. Each fold uses
its last epoch rather than its best, since picking the best on the held-out part would flatter the result.

Every fold is scored by day and on a simulated infrared night, against the box-shape rule it has to beat. The result is
split by source too: the free-licence photographs were searched for as resting cattle, so a model could learn to tell
the sources apart instead of the postures, and only the score on the other source would expose that.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_posture_dataset import COMMONS_SPLIT, hash_of, image_path, load_labels  # noqa: E402
from smartcattle_ai.model_loader import load_model  # noqa: E402
from smartcattle_ai.posture import LYING_ASPECT  # noqa: E402
from smartcattle_ai.posture_classifier import (  # noqa: E402
    DEFAULT_WEIGHTS,
    INPUT_SIZE,
    MIN_CONFIDENCE,
    PostureClassifier,
    as_infrared_night,
    square_crop,
)

FOLDS = 5
EPOCHS = 15
REPORT = ROOT / "reports" / "posture_evaluation.json"
WORK = ROOT / "runs" / "posture_cls"


def shape_says(bbox) -> str:
    x1, y1, x2, y2 = bbox
    return "lying" if (x2 - x1) / max(y2 - y1, 1) >= LYING_ASPECT else "standing"


def write_split(folder: Path, rows: list[dict], with_night: bool) -> None:
    for row in rows:
        target = folder / row["posture"]
        target.mkdir(parents=True, exist_ok=True)
        stem = f"{Path(row['image']).stem}_{row['box_index']}"
        crop = square_crop(cv2.imread(str(image_path(row))), row["bbox"])
        cv2.imwrite(str(target / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if with_night:
            night = as_infrared_night(crop, seed=hash_of(stem))
            cv2.imwrite(str(target / f"{stem}_night.jpg"), night, [cv2.IMWRITE_JPEG_QUALITY, 92])


def train(name: str, train_rows: list[dict], check_rows: list[dict]) -> Path:
    data = WORK / name / "data"
    if data.exists():
        shutil.rmtree(data)
    write_split(data / "train", train_rows, with_night=True)
    write_split(data / "val", check_rows, with_night=False)
    model = load_model(str(ROOT / "yolo11n-cls.pt"))
    # Cattle are not upside down, and the crops are padded: flips and small changes of light are what is left.
    model.train(data=str(data), epochs=EPOCHS, imgsz=INPUT_SIZE, batch=32, device="cpu", seed=7, plots=False,
                project=str(WORK / name), name="model", exist_ok=True, verbose=False, fliplr=0.5, flipud=0.0,
                degrees=5.0, scale=0.2, erasing=0.1, workers=0)
    return Path(model.trainer.last)


def scores(pairs: list[tuple[str, str]]) -> dict:
    """Recall and precision of lying, the rare class that matters, plus the plain share of right answers."""
    hits = sum(truth == said == "lying" for truth, said in pairs)
    lying = sum(truth == "lying" for truth, _ in pairs)
    called = sum(said == "lying" for _, said in pairs)
    return {
        "animals": len(pairs), "lying": lying, "recall_lying": round(hits / lying, 3) if lying else None,
        "precision_lying": round(hits / called, 3) if called else None,
        "accuracy": round(sum(truth == said for truth, said in pairs) / len(pairs), 3) if pairs else None,
    }


def shipped_rule(answer: tuple[str, float], bbox) -> str:
    """What the service does with one look: the classifier when it is sure, the shape when not."""
    name, confidence = answer
    return name if confidence >= MIN_CONFIDENCE else shape_says(bbox)


def cross_validate(rows: list[dict]) -> dict:
    folds: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        folds[hash_of(row["image"]) % FOLDS].append(row)
    results: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for fold, held_out in sorted(folds.items()):
        others = [row for other, group in folds.items() if other != fold for row in group]
        classifier = PostureClassifier(train(f"fold{fold}", others, held_out), "cpu")
        for row in held_out:
            image = cv2.imread(str(image_path(row)))
            source = "commons" if row["split"] == COMMONS_SPLIT else "open_images"
            truth = row["posture"]
            day = classifier.predict(image, [row["bbox"]])[0]
            night_image = as_infrared_night(image, seed=hash_of(row["image"]))
            night = classifier.predict(night_image, [row["bbox"]])[0]
            shape = shape_says(row["bbox"])
            for scope in ("all", source):
                results[f"{scope}/day/shape_only"].append((truth, shape))
                results[f"{scope}/day/classifier"].append((truth, day[0]))
                results[f"{scope}/day/shipped"].append((truth, shipped_rule(day, row["bbox"])))
                results[f"{scope}/night/classifier"].append((truth, night[0]))
                results[f"{scope}/night/shipped"].append((truth, shipped_rule(night, row["bbox"])))
        print(f"fold {fold} done", flush=True)
    return {key: scores(pairs) for key, pairs in sorted(results.items())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-final", action="store_true", help="Measure only; leave the installed model alone")
    args = parser.parse_args(argv)
    rows = [row for row in load_labels() if row["posture"] in ("standing", "lying")]
    print(len(rows), "animals", flush=True)
    measured = cross_validate(rows)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        "method": f"{FOLDS}-fold cross-validation, whole pictures on one side, last of {EPOCHS} epochs",
        "labels": "Labelled by eye from the pictures by an AI assistant, not by a veterinarian",
        "night": "Simulated: pictures made grey, dim and noisy. No real night footage was measured.",
        "min_confidence": MIN_CONFIDENCE,
        "results": measured,
    }, indent=2) + "\n", encoding="utf-8")
    if not args.skip_final:
        last = train("final", rows, [row for row in rows if row["split"] == "test"])
        DEFAULT_WEIGHTS.parent.mkdir(exist_ok=True)
        shutil.copyfile(last, DEFAULT_WEIGHTS)
    print(json.dumps(measured, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
