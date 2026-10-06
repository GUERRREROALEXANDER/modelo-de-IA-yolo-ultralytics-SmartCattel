"""Report cattle counts and optional box errors for licensed scenario images."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smartcattle_ai import Detector, Settings
from smartcattle_ai.evaluation import match_boxes, xywh_to_xyxy
from smartcattle_ai.media import draw_detections, read_image, save_image
from scripts.validate_dataset import parse_labels


def report(manifest: Path, weights: str, output: Path) -> None:
    # Scenario labels only contain cattle, so persons are excluded from the comparison.
    detector = Detector(Settings(weights=weights, class_names=("cow",)))
    image_output = ROOT / "outputs" / "scenarios" / Path(weights).stem
    image_output.mkdir(parents=True, exist_ok=True)
    rows = []
    credits = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for entry in csv.DictReader(stream):
            image = manifest.parent / entry["file"]
            frame = read_image(image)
            detections = detector.detect(frame)
            expected = int(entry["expected"])
            labels = manifest.parent / "labels" / f"{image.stem}.txt"
            false_positives = missed = "n/a"
            if labels.exists():
                boxes, errors = parse_labels(labels)
                if errors:
                    raise ValueError(errors[0])
                height, width = frame.shape[:2]
                truth = np.asarray([xywh_to_xyxy(box, width, height) for box in boxes],
                                   dtype=np.float32).reshape(-1, 4)
                predicted = np.asarray([item["bbox"] for item in detections],
                                       dtype=np.float32).reshape(-1, 4)
                matched = int(match_boxes(truth, predicted)[:, 0].sum())
                false_positives, missed = len(detections) - matched, len(truth) - matched
            save_image(image_output / image.name, draw_detections(frame, detections))
            rows.append((entry["file"], entry["scenario"], expected, len(detections),
                         ", ".join(f'{item["confidence"]:.3f}' for item in detections),
                         false_positives, missed))
            credits.append((entry["file"], entry["license"], entry["author"], entry["source_url"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Scenario report: {Path(weights).name}", "",
             "| File | Scenario | Expected | Detected | Confidences | False positives | Missed |",
             "|---|---|---:|---:|---|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    lines.extend(("", "## Image credits", "",
                  "| File | License | Author | Source URL |",
                  "|---|---|---|---|"))
    for credit in credits:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in credit) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    print(f"Annotated images: {image_output}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "samples" / "manifest.csv")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report(args.manifest, args.weights,
           args.out or ROOT / "reports" / f"scenarios_{Path(args.weights).stem}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
