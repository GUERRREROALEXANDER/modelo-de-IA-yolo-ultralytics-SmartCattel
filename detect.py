"""Command line detection for images, videos, and image folders."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from smartcattle_ai import Detector, Settings
from smartcattle_ai.media import is_video, process_image, process_video


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _summary(result: dict) -> None:
    print(f"File: {result['source']}")
    if "frames" in result:
        print(f"Frames: {result['frames_processed']} (read: {result['frames_read']})")
        print(f"Max cattle: {result['max_cattle']}; average cattle: {result['avg_cattle']:.2f}")
        print(f"Annotated video: {result['output'] or 'not saved'}")
    else:
        print(f"Cattle: {result['total_cattle']}")
        for detection in result["detections"]:
            print(f"{detection['class']}  conf={detection['confidence']:.2f} "
                  f"bbox={detection['bbox']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Detect cattle in images or videos.")
    parser.add_argument("--source", required=True, help="Image, video, or image folder")
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.source)
    if not source.exists():
        print(f"Source does not exist: {source}", file=sys.stderr)
        return 1
    try:
        settings = Settings.from_env()
        changes = {field: value for field, value in {
            "weights": args.weights, "confidence": args.conf, "iou": args.iou,
            "imgsz": args.imgsz, "device": args.device,
        }.items() if value is not None}
        detector = Detector(replace(settings, **changes))
        output_dir = None if args.no_save else args.output_dir
        if source.is_dir():
            images = sorted(path for path in source.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if not images:
                raise ValueError(f"No images found in folder: {source}")
            results = [process_image(detector, path, output_dir) for path in images]
        elif is_video(source):
            results = process_video(detector, source, output_dir,
                                    args.vid_stride, args.max_frames)
        else:
            results = process_image(detector, source, output_dir)
        if args.json:
            print(json.dumps(results))
        elif isinstance(results, list):
            for result in results:
                _summary(result)
        else:
            _summary(results)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
