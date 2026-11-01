"""Run YOLO on the live camera and serve the annotated video to the SmartCattle frontend."""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from threading import Event, Thread

from smartcattle_ai import Detector, Settings
from smartcattle_ai.backend_client import BackendClient, BackendError, EventThrottle
from smartcattle_ai.env import load_env_file
from smartcattle_ai.imou import imou_configured, imou_url_provider
from smartcattle_ai.live import (
    DEFAULT_ORIGINS, LatestFrame, LiveState, capture_worker, detection_worker, make_server, occupancy_worker,
    status_worker, zone_sync_worker,
)
from smartcattle_ai.line_counter import LineCounter, parse_line
from smartcattle_ai.pipeline import EventPublisher, parse_zone
from smartcattle_ai.stream import CameraStream, camera_url_from_env, normalize_source, redact_url
from smartcattle_ai.stillness import StillnessMonitor
from smartcattle_ai.tracking import IoUTracker
from smartcattle_ai.tunnel import QuickTunnel, find_cloudflared
from smartcattle_ai.zones import ZoneSet

CAMERA_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def main(argv=None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Live camera detection served as MJPEG.")
    parser.add_argument("--url", help="Stream URL (rtsp://, http://), a phone camera app's address "
                                      "(192.168.1.20:8080) or a webcam index (0, 1). Default: the Imou cloud "
                                      "if IMOU_APP_ID is set, else CAMERA_URL, CAMERA_RTSP_URL or CAMERA_* variables")
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID") or os.getenv("SMARTCATTLE_CAMERA_ID") or "camera-01")
    parser.add_argument("--zone", default=os.getenv("SMARTCATTLE_ZONE") or "0.1,0.1,0.9,0.9")
    parser.add_argument("--backend-url", default=os.getenv("SMARTCATTLE_BACKEND_URL"))
    parser.add_argument("--event-cooldown", type=float, default=10.0)
    parser.add_argument("--status-interval", type=float, default=15.0)
    parser.add_argument("--occupancy-interval", type=float, default=1.0,
                        help="Seconds between head counts sent to the backend")
    parser.add_argument("--zone-sync-interval", type=float, default=30.0,
                        help="Seconds between reloads of the safe zones drawn in the dashboard")
    parser.add_argument("--still-after-minutes", type=float, default=30.0,
                        help="Warn about a cow that has not moved for this long (0 turns it off)")
    parser.add_argument("--gate-line", default=os.getenv("SMARTCATTLE_GATE_LINE") or None,
                        help="Count cattle crossing the line x1,y1,x2,y2 (fractions of the frame), e.g. a gate")
    parser.add_argument("--no-tracking", action="store_true",
                        help="Do not number animals across frames (one alert per cooldown instead of per escape)")
    parser.add_argument("--host", default=os.getenv("SMARTCATTLE_LIVE_HOST") or "127.0.0.1")
    # PORT is what hosts such as Render assign to the service.
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("SMARTCATTLE_LIVE_PORT") or os.getenv("PORT") or 8090))
    parser.add_argument("--origins", default=os.getenv("SMARTCATTLE_LIVE_ORIGINS") or ",".join(DEFAULT_ORIGINS),
                        help="Comma-separated frontend origins allowed to read /status")
    parser.add_argument("--public-url",
                        default=os.getenv("SMARTCATTLE_LIVE_PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL") or None,
                        help="URL where browsers reach this server; reported to the backend")
    parser.add_argument("--tunnel", action="store_true", default=os.getenv("SMARTCATTLE_LIVE_TUNNEL") == "1",
                        help="Publish the video through a Cloudflare quick tunnel (anyone with the URL can watch)")
    args = parser.parse_args(argv)

    try:
        url = normalize_source(args.url) if args.url else (
            imou_url_provider() if imou_configured() else camera_url_from_env())
        zone = ZoneSet.from_rectangle(parse_zone(args.zone))
        if not CAMERA_ID.fullmatch(args.camera_id):
            # The backend's rule for camera ids; anything else would be a 422 on every report.
            raise ValueError("camera-id must be 1 to 64 letters, digits, '-' or '_'")
        gate_line = parse_line(args.gate_line) if args.gate_line else None
        if gate_line is not None and args.no_tracking:
            raise ValueError("--gate-line needs tracking; remove --no-tracking")
        if args.still_after_minutes < 0:
            raise ValueError("--still-after-minutes must be 0 or more")
        for name in ("occupancy_interval", "zone_sync_interval", "status_interval"):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        detector = Detector(Settings.from_env())
        detector.class_ids  # Load the model now, not on the first frame.
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    state = LiveState(args.camera_id, zone, args.public_url.rstrip("/") if args.public_url else None)
    if gate_line is not None:
        state.gate = LineCounter(gate_line)
    stop = Event()
    frames = LatestFrame()
    # Short backoff so a camera that comes back is picked up within seconds.
    stream = CameraStream(url, max_backoff_s=5)

    def on_camera_state(status, error):
        state.set_camera_status(status, error)
        print(f"Camera {status}" + (f": {error}" if error else ""), flush=True)

    stream.on_state_change(on_camera_state)
    publisher = client = None
    workers = [Thread(target=capture_worker, args=(stream, frames, stop), daemon=True)]
    if args.backend_url:
        client = BackendClient(args.backend_url, os.getenv("SMARTCATTLE_API_KEY") or None)
        try:
            client.health()
        except BackendError as exc:
            print(f"Warning: backend not reachable yet ({exc}); retrying in the background", file=sys.stderr)
        publisher = EventPublisher(client, args.camera_id, zone, EventThrottle(args.event_cooldown))
        workers.append(Thread(target=status_worker, args=(client, state, stop, args.status_interval), daemon=True))
        workers.append(Thread(target=occupancy_worker, args=(client, state, stop, args.occupancy_interval),
                              daemon=True))
        workers.append(Thread(target=zone_sync_worker,
                              args=(client, state, zone, stop, args.zone_sync_interval, publisher), daemon=True))
    tracker = None if args.no_tracking else IoUTracker()
    # Stillness needs the animal's number, so it is off without tracking.
    stillness = (StillnessMonitor(still_after_s=args.still_after_minutes * 60)
                 if tracker is not None and args.still_after_minutes > 0 else None)
    workers.append(Thread(target=detection_worker, args=(detector, frames, state, stop, publisher),
                          kwargs={"tracker": tracker, "stillness": stillness}, daemon=True))

    origins = [origin.strip() for origin in args.origins.split(",") if origin.strip()]
    try:
        server = make_server(state, stop, args.host, args.port, origins)
    except OSError as exc:
        print(f"Error: cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    tunnel = None
    if args.tunnel:
        executable = find_cloudflared()
        if executable is None:
            print("Error: --tunnel needs cloudflared (PATH, SMARTCATTLE_CLOUDFLARED or "
                  "%LOCALAPPDATA%\\cloudflared\\cloudflared.exe)", file=sys.stderr)
            server.server_close()
            return 1
        tunnel = QuickTunnel(server.server_address[1], executable)
        try:
            state.public_url = tunnel.start()
        except (RuntimeError, OSError) as exc:
            print(f"Error: tunnel failed: {exc}", file=sys.stderr)
            server.server_close()
            return 1
    for worker in workers:
        worker.start()
    base = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"Camera {args.camera_id}: {redact_url(str(url))}")
    print(f"Backend: {args.backend_url or 'disabled'}")
    print(f"Video: {base}/video.mjpg  Status: {base}/status  (Ctrl+C to stop)")
    if state.public_url:
        print(f"Public video: {state.public_url}/video.mjpg")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        stream.stop()
        server.server_close()
        if tunnel is not None:
            tunnel.stop()
        if client is not None:
            # Lets the frontend drop the camera now instead of after the backend's offline timeout.
            try:
                client.report_camera_status(args.camera_id, {
                    "status": "offline", "observed_at": datetime.now(timezone.utc).isoformat()})
            except BackendError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
