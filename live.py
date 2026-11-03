"""Run YOLO on the live camera and serve the annotated video to the SmartCattle frontend."""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from threading import Event, Thread

from smartcattle_ai import Detector, Settings
from smartcattle_ai.backend_client import (
    BackendClient,
    BackendError,
    EventThrottle,
    camera_is_owned,
    session_is_valid,
    session_on_this_farm,
)
from smartcattle_ai.farm import FarmLink, is_farm_key
from smartcattle_ai.pushed import PUSHED_SOURCE, PushedStream
from smartcattle_ai.camera_manager import MAX_CAMERAS, CameraManager, ManagedCamera, SharedDetector, is_public_url
from smartcattle_ai.env import load_env_file
from smartcattle_ai.imou import imou_configured, imou_url_provider
from smartcattle_ai.discovery import local_ipv4
from smartcattle_ai.live import (
    DEFAULT_ORIGINS, LatestFrame, LiveState, VisitorLog, capture_worker, detection_worker, lan_origins, make_server,
    occupancy_worker, source_worker, status_worker, zone_sync_worker,
)
from smartcattle_ai.heatmap import HeatMap
from smartcattle_ai.line_counter import LineCounter, parse_line
from smartcattle_ai.night_watch import NightWatch, parse_window
from smartcattle_ai.pipeline import EventPublisher, parse_zone
from smartcattle_ai.posture import PostureMonitor
from smartcattle_ai.stream import CameraStream, address_only, camera_url_from_env, normalize_source, redact_url
from smartcattle_ai.stillness import StillnessMonitor
from smartcattle_ai.tracking import IoUTracker
from smartcattle_ai.tunnel import QuickTunnel, find_cloudflared
from smartcattle_ai.zones import ZoneSet

CAMERA_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def max_added_cameras(env) -> int:
    """How many cameras the dashboard may add. A server's memory and CPU say how many it can really analyse."""
    value = env.get("SMARTCATTLE_MAX_CAMERAS", "")
    return int(value) if value.isdigit() else MAX_CAMERAS


def main(argv=None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description="Live camera detection served as MJPEG.")
    parser.add_argument("--url", help="Stream URL (rtsp://, http://), a phone camera app's address "
                                      "(192.168.1.20:8080) or a webcam index (0, 1), to open at start. "
                                      "Default: no camera; each person connects theirs from the dashboard")
    parser.add_argument("--auto-camera", action="store_true", default=os.getenv("SMARTCATTLE_AUTOSTART_CAMERA") == "1",
                        help="Open the camera set in .env at start (Imou cloud if IMOU_APP_ID is set, else CAMERA_URL, "
                             "CAMERA_RTSP_URL or CAMERA_* variables) and reconnect the saved dashboard cameras")
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
    parser.add_argument("--night-window", default=os.getenv("SMARTCATTLE_NIGHT_WINDOW") or None,
                        help="Warn about people seen in this local window, e.g. 20:00-05:30")
    parser.add_argument("--timezone", default=os.getenv("SMARTCATTLE_TIMEZONE") or "America/Bogota",
                        help="Time zone of --night-window")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Do not keep the heat map served at /heatmap.jpg")
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
        # Nothing opens by itself: a .env copied from another PC must not point this one at somebody else's camera.
        url = normalize_source(args.url) if args.url else (
            (imou_url_provider() if imou_configured() else camera_url_from_env()) if args.auto_camera else None)
        zone = ZoneSet.from_rectangle(parse_zone(args.zone))
        if not CAMERA_ID.fullmatch(args.camera_id):
            # The backend's rule for camera ids; anything else would be a 422 on every report.
            raise ValueError("camera-id must be 1 to 64 letters, digits, '-' or '_'")
        gate_line = parse_line(args.gate_line) if args.gate_line else None
        if gate_line is not None and args.no_tracking:
            raise ValueError("--gate-line needs tracking; remove --no-tracking")
        night_window = parse_window(args.night_window) if args.night_window else None
        if night_window is not None:
            NightWatch(night_window, args.timezone)  # Reject a bad time zone now, not on the first camera.
        if night_window is not None and args.no_tracking:
            raise ValueError("--night-window needs tracking; remove --no-tracking")
        if args.still_after_minutes < 0:
            raise ValueError("--still-after-minutes must be 0 or more")
        for name in ("occupancy_interval", "zone_sync_interval", "status_interval"):
            if getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        model = Detector(Settings.from_env())
        if url is not None:
            model.class_ids  # Load the model now, not on the first frame. With no camera it waits for the first one.
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    # One model in memory, however many cameras the dashboard adds.
    detector = SharedDetector(model)

    client = None
    if args.backend_url:
        client = BackendClient(args.backend_url, os.getenv("SMARTCATTLE_API_KEY") or None)
        try:
            client.health()
        except BackendError as exc:
            print(f"Warning: backend not reachable yet ({exc}); retrying in the background", file=sys.stderr)

    def build_camera(camera_id, source, camera_stop, public_url=None, gate=None):
        """One camera's state and threads, not started yet."""
        camera_state = LiveState(camera_id, zone, public_url)
        camera_state.gate = gate
        camera_state.posture = PostureMonitor()
        if not args.no_heatmap:
            camera_state.heat = HeatMap()
        frames = LatestFrame()
        # Short backoff so a camera that comes back is picked up within seconds. Not for the Imou cloud (a callable
        # source): every retry spends calls of its monthly allowance, and a day offline must not use it all up.
        stream = PushedStream() if source == PUSHED_SOURCE \
            else CameraStream(source, max_backoff_s=5 if isinstance(source, str) else 300)

        def on_camera_state(status, error):
            camera_state.set_camera_status(status, error)
            print(f"Camera {camera_id} {status}" + (f": {error}" if error else ""), flush=True)

        stream.on_state_change(on_camera_state)
        publisher = None
        workers = [Thread(target=capture_worker, args=(stream, frames, camera_stop), daemon=True)]
        if client is not None:
            publisher = EventPublisher(client, camera_id, zone, EventThrottle(args.event_cooldown))
            if isinstance(source, str):
                workers.append(Thread(target=source_worker, args=(client, camera_id, address_only(source), camera_stop),
                                      daemon=True))
            workers.append(Thread(target=status_worker, args=(client, camera_state, camera_stop, args.status_interval),
                                  daemon=True))
            workers.append(Thread(target=occupancy_worker,
                                  args=(client, camera_state, camera_stop, args.occupancy_interval), daemon=True))
            workers.append(Thread(target=zone_sync_worker,
                                  args=(client, camera_state, zone, camera_stop, args.zone_sync_interval, publisher),
                                  daemon=True))
        tracker = None if args.no_tracking else IoUTracker()
        # Stillness needs the animal's number, so it is off without tracking.
        stillness = (StillnessMonitor(still_after_s=args.still_after_minutes * 60)
                     if tracker is not None and args.still_after_minutes > 0 else None)
        night = NightWatch(night_window, args.timezone) if night_window is not None else None
        workers.append(Thread(target=detection_worker, args=(detector, frames, camera_state, camera_stop, publisher),
                              kwargs={"tracker": tracker, "stillness": stillness,
                                      "visitors": VisitorLog() if tracker is not None else None,
                                      "night": night}, daemon=True))
        return camera_state, stream, workers

    def on_camera_removed(camera_id, source):
        if client is None:
            return
        report_offline(camera_id)
        try:
            client.report_camera_source(camera_id, address_only(source), active=False)
        except BackendError as exc:
            print(f"Warning: could not record the removal of camera {camera_id}: {exc}", file=sys.stderr)

    def report_offline(camera_id):
        # Lets the frontend drop the camera now instead of after the backend's offline timeout.
        if client is None:
            return
        try:
            client.report_camera_status(camera_id, {
                "status": "offline", "observed_at": datetime.now(timezone.utc).isoformat()})
        except BackendError:
            pass

    stop = Event()
    public_url = args.public_url.rstrip("/") if args.public_url else None
    if url is not None:
        state, stream, workers = build_camera(args.camera_id, url, stop, public_url,
                                              LineCounter(gate_line) if gate_line is not None else None)
    else:
        # Waiting for someone to connect a camera from the dashboard; /status and /health answer meanwhile.
        state, stream, workers = LiveState(args.camera_id, zone, public_url), None, []
        state.set_camera_status("idle")

    def added_stream_url(camera_id):
        return f"{state.public_url}/cameras/{camera_id}"

    def start_added_camera(camera_id, source):
        camera_stop = Event()
        camera_state, camera_stream, camera_workers = build_camera(camera_id, source, camera_stop,
                                                                   added_stream_url(camera_id))
        for worker in camera_workers:
            worker.start()
        print(f"Camera {camera_id} added: {redact_url(source)}", flush=True)
        return ManagedCamera(camera_id, source, camera_state, camera_stream, camera_stop)

    # The list holds camera passwords and only serves to reconnect by itself, so it is kept only when asked for that.
    manager = CameraManager(start_added_camera, added_stream_url, on_remove=on_camera_removed,
                            path=(os.getenv("SMARTCATTLE_CAMERAS_FILE") or ".cameras.json") if args.auto_camera else None,
                            max_cameras=max_added_cameras(os.environ))
    origins = [origin.strip() for origin in args.origins.split(",") if origin.strip()]
    # Listening on every interface is how other devices get here, so their browsers must be let in too,
    # and they must be told this PC's address: "localhost" would mean their own device.
    lan_address = local_ipv4() if args.host == "0.0.0.0" else None
    origins += lan_origins(lan_address)
    session_check = camera_check = None
    if args.tunnel or is_public_url(args.public_url):
        # Reachable from the internet: changing cameras takes a dashboard session. No backend to ask means no one can.
        farm_key = os.getenv("SMARTCATTLE_API_KEY")
        if args.backend_url and is_farm_key(farm_key):
            # Being signed in is not enough here: this PC's network is only for the people of its account.
            session_check = lambda authorization: session_on_this_farm(args.backend_url, authorization, farm_key)
        elif args.backend_url:
            session_check = lambda authorization: session_is_valid(args.backend_url, authorization)
        else:
            session_check = lambda authorization: False
        if args.backend_url:
            camera_check = lambda authorization, camera_id: camera_is_owned(args.backend_url, authorization, camera_id)
    try:
        server = make_server(state, stop, args.host, args.port, origins, camera=stream, manager=manager,
                             session_check=session_check, camera_check=camera_check)
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
    base = f"http://{lan_address or ('localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host)}:{args.port}"
    print(f"Camera {args.camera_id}: {redact_url(str(url))}" if url is not None
          else "No camera yet: connect one from the Cameras page of the dashboard")
    print(f"Backend: {args.backend_url or 'disabled'}")
    print(f"Status: {base}/status  (Ctrl+C to stop)" + (f"  Video: {base}/video.mjpg" if url is not None else ""))
    if state.heat is not None:
        print(f"Heat map: {base}/heatmap.jpg")
    if state.public_url:
        print(f"Public video: {state.public_url}/video.mjpg")
    # Without a public address, the dashboard on this PC still needs to find this service and the cameras it adds.
    state.public_url = state.public_url or base
    for worker in workers:
        worker.start()
    manager.restore()
    if client is not None and is_farm_key(os.getenv("SMARTCATTLE_API_KEY")):
        def moved_to(new_url):
            # Every camera's link hangs from this address, and the panel reads them from their next report.
            state.public_url = new_url
            for camera in manager.all():
                camera.state.public_url = f"{new_url}/cameras/{camera.camera_id}"

        FarmLink(client.report_farm_pc, lambda: state.public_url, tunnel, moved_to).start(stop)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if stream is not None:
            stream.stop()
        added = manager.stop_all()
        server.server_close()
        if tunnel is not None:
            tunnel.stop()
        for camera_id in ((args.camera_id, *added) if url is not None else added):
            report_offline(camera_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
