"""Command line detection for images, videos, and image folders."""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from smartcattle_ai import Detector, Settings
from smartcattle_ai.media import is_video, process_image, process_video
from smartcattle_ai.backend_client import BackendClient, BackendError, EventThrottle
from smartcattle_ai.pipeline import EventPublisher, parse_zone


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _summary(result: dict) -> None:
    print(f"File: {result['source']}")
    if "frames" in result:
        print(f"Frames: {result['frames_processed']} (read: {result['frames_read']})")
        print(f"Cattle: {result['total_cattle']}; Persons: {result['total_persons']}")
        print(f"Max cattle: {result['max_cattle']}; max persons: {result['max_persons']}; "
              f"average cattle: {result['avg_cattle']:.2f}")
        print(f"Annotated video: {result['output'] or 'not saved'}")
    else:
        print(f"Cattle: {result['total_cattle']}; Persons: {result['total_persons']}")
        for detection in result["detections"]:
            print(f"{detection['class']}  conf={detection['confidence']:.2f} "
                  f"bbox={detection['bbox']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Detect cattle in images or videos.")
    parser.add_argument("--source", required=True, help="Image, video, or image folder")
    parser.add_argument("--weights")
    parser.add_argument("--classes", help="Comma-separated classes: cow,person")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--backend-url", default=os.getenv("SMARTCATTLE_BACKEND_URL"))
    parser.add_argument("--camera-id", default=os.getenv("SMARTCATTLE_CAMERA_ID") or "camera-01")
    parser.add_argument("--zone", default=os.getenv("SMARTCATTLE_ZONE") or "0.1,0.1,0.9,0.9")
    parser.add_argument("--event-cooldown", type=float, default=10.0)
    args = parser.parse_args(argv)
    source = Path(args.source)
    if not source.exists():
        print(f"Source does not exist: {source}", file=sys.stderr)
        return 1
    try:
        publisher = None
        sent_count = 0
        zone = None
        if args.backend_url:
            zone = parse_zone(args.zone)
            if not 1 <= len(args.camera_id) <= 64:
                raise ValueError("camera-id must contain 1 to 64 characters")
            client = BackendClient(args.backend_url, os.getenv("SMARTCATTLE_API_KEY"))
            if not client.health():
                raise BackendError("Backend health check failed")
            publisher = EventPublisher(client, args.camera_id, zone,
                                       EventThrottle(args.event_cooldown if is_video(source) else 0))
        settings = Settings.from_env()
        changes = {field: value for field, value in {
            "weights": args.weights, "confidence": args.conf, "iou": args.iou,
            "imgsz": args.imgsz, "device": args.device,
        }.items() if value is not None}
        if args.classes is not None:
            changes["class_names"] = tuple(name.strip() for name in args.classes.split(","))
        settings = replace(settings, **changes)
        detector = Detector(settings)
        if not args.json:
            print(f"Model: {settings.weights}")
        output_dir = None if args.no_save else args.output_dir
        if source.is_dir():
            images = sorted(path for path in source.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            if not images:
                raise ValueError(f"No images found in folder: {source}")
            results = []
            for path in images:
                result = process_image(detector, path, output_dir, zone=zone)
                results.append(result)
                if publisher:
                    sent_count += len(publisher.handle(result["detections"], result["width"],
                                                      result["height"], time.time()))
        elif is_video(source):
            def on_frame(item):
                nonlocal sent_count
                sent_count += len(publisher.handle(item["detections"], item["width"],
                                                   item["height"], item["time_s"]))

            results = process_video(detector, source, output_dir,
                                    args.vid_stride, args.max_frames,
                                    on_frame=on_frame if publisher else None, zone=zone)
        else:
            results = process_image(detector, source, output_dir, zone=zone)
            if publisher:
                sent_count += len(publisher.handle(results["detections"], results["width"],
                                                   results["height"], time.time()))
        if args.json:
            print(json.dumps(results))
        elif isinstance(results, list):
            for result in results:
                _summary(result)
        else:
            _summary(results)
        if publisher:
            print(f"Backend events sent: {sent_count}")
        return 0
    except (ValueError, RuntimeError, OSError, BackendError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
