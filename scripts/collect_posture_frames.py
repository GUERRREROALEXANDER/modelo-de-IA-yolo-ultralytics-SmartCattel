"""Saves pictures of the cows a camera sees, sorted by the posture estimated for each one, ready to be labelled.

The estimate only sorts them, to save work: each picture must still be checked by a person before training, or the
model would learn the estimate's mistakes.

python scripts/collect_posture_frames.py --source 1 --out datasets/cattle_posture/raw --every 5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartcattle_ai import Detector, Settings  # noqa: E402
from smartcattle_ai.posture import PostureMonitor  # noqa: E402
from smartcattle_ai.stream import normalize_source  # noqa: E402
from smartcattle_ai.tracking import IoUTracker  # noqa: E402


def folder_for(detections: list[dict]) -> str | None:
    """Where a picture goes: "lying" if any cow lies, "standing" if any stands, None without cows."""
    postures = {item.get("posture") for item in detections if item["class"] == "cow"}
    if "lying" in postures:
        return "lying"
    return "standing" if postures else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Webcam number, phone app address or stream link")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "cattle_posture" / "raw")
    parser.add_argument("--every", type=float, default=5.0, help="Seconds between saved pictures")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)
    source = normalize_source(args.source)
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        print("Error: could not open the camera", file=sys.stderr)
        return 1
    detector, tracker, monitor = Detector(Settings.from_env()), IoUTracker(), PostureMonitor()
    saved, last = 0, 0.0
    try:
        while saved < args.limit:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.time()
            height, width = frame.shape[:2]
            found = monitor.update(tracker.update(detector.detect(frame)), width, height, now)
            folder = folder_for(found)
            if folder is None or now - last < args.every:
                continue
            target = args.out / folder
            target.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target / f"{int(now * 1000)}.jpg"), frame)
            saved, last = saved + 1, now
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
    print(f"Saved {saved} pictures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
