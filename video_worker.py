"""Process the videos uploaded in the SmartCattle dashboard with YOLO, one at a time."""

import argparse
import os
import sys
from threading import Event

from smartcattle_ai import Detector, Settings
from smartcattle_ai.backend_client import BackendClient
from smartcattle_ai.env import load_env_file
from smartcattle_ai.pipeline import parse_zone
from smartcattle_ai.video_jobs import run_worker
from smartcattle_ai.zones import ZoneSet


def main(argv=None, detector_factory=Detector) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default=os.getenv("SMARTCATTLE_BACKEND_URL"))
    parser.add_argument("--zone", default=os.getenv("SMARTCATTLE_ZONE") or "0.1,0.1,0.9,0.9",
                        help="Safe zone for videos whose lot has no camera or no zone drawn")
    parser.add_argument("--vid-stride", type=int, default=2, help="Analyze one frame of every N")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between checks of an empty queue")
    parser.add_argument("--once", action="store_true", help="Process at most one video and exit")
    args = parser.parse_args(argv)
    try:
        if not args.backend_url:
            raise ValueError("Set --backend-url or SMARTCATTLE_BACKEND_URL")
        if args.vid_stride < 1:
            raise ValueError("--vid-stride must be at least 1")
        if args.poll_interval <= 0:
            raise ValueError("--poll-interval must be positive")
        zone = ZoneSet.from_rectangle(parse_zone(args.zone))
        detector = detector_factory(Settings.from_env())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    client = BackendClient(args.backend_url, os.getenv("SMARTCATTLE_API_KEY") or None, timeout=30.0)
    print(f"Waiting for uploaded videos at {args.backend_url} (Ctrl+C to stop)", flush=True)
    stop = Event()
    try:
        run_worker(client, detector, zone, stop, poll_s=args.poll_interval, once=args.once,
                   vid_stride=args.vid_stride)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
