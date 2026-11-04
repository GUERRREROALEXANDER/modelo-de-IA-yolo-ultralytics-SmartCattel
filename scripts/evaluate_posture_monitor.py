"""Measures the whole rule the service uses, classifier included, on animals that models of each fold never saw.

A labelled photograph is shown as a still scene for some seconds, so the stillness and the hidden-legs checks act as they
would on a resting herd. Reuses the folds left by train_posture.py and adds the result to its report.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from build_posture_dataset import COMMONS_SPLIT, hash_of, image_path, load_labels  # noqa: E402
from smartcattle_ai.posture import WALKING, PostureMonitor  # noqa: E402
from smartcattle_ai.posture_classifier import PostureClassifier, as_infrared_night  # noqa: E402
from train_posture import FOLDS, REPORT, scores  # noqa: E402

SCENE_S = 20
STEP_S = 0.5


def fold_weights(fold: int) -> Path:
    found = sorted(ROOT.glob(f"runs/**/fold{fold}/**/weights/last.pt"), key=lambda path: path.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"fold {fold}: run train_posture.py first")
    return found[-1]


def watch(classifier, image, rows: list[dict]) -> list[str]:
    """What the monitor settles on for each animal after watching the still scene."""
    monitor = PostureMonitor(classifier)
    height, width = image.shape[:2]
    scene = [{"class": "cow", "confidence": 0.9, "track_id": index, "bbox": row["bbox"]} for index, row in enumerate(rows)]
    answers = scene
    for tick in range(int(SCENE_S / STEP_S)):
        answers = monitor.update(scene, width, height, tick * STEP_S, image)
    return [item.get("posture") for item in answers]


def main() -> int:
    rows = [row for row in load_labels() if row["posture"] in ("standing", "lying")]
    by_fold: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold[hash_of(row["image"]) % FOLDS][row["image"]].append(row)
    results: dict[str, list[tuple[str, str]]] = defaultdict(list)
    unanswered: dict[str, int] = defaultdict(int)
    for fold in range(FOLDS):
        classifier = PostureClassifier(fold_weights(fold), "cpu")
        for name, group in by_fold[fold].items():
            image = cv2.imread(str(image_path(group[0])))
            source = "commons" if group[0]["split"] == COMMONS_SPLIT else "open_images"
            night = as_infrared_night(image, seed=hash_of(name))
            for light, picture in (("day", image), ("night", night)):
                for row, said in zip(group, watch(classifier, picture, group)):
                    # No answer is not a wrong one: it counts as unknown in the report, and is measured apart.
                    said = "standing" if said == WALKING else said
                    for scope in ("all", source):
                        if said is None:
                            unanswered[f"{scope}/{light}/whole_rule"] += 1
                            continue
                        results[f"{scope}/{light}/whole_rule"].append((row["posture"], said))
        print(f"fold {fold} done", flush=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["results"].update({key: {**scores(pairs), "no_answer": unanswered[key]} for key, pairs in sorted(results.items())})
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    for key in sorted(results):
        print(key, report["results"][key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
