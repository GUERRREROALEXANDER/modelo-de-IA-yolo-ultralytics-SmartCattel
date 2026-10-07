"""Check that the camera stream really reaches this PC and OpenCV, before running YOLO."""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2

from smartcattle_ai.env import load_env_file
from smartcattle_ai.stream import CameraStream, camera_url_from_env, host_and_port, redact_url


def _step(ok: bool, text: str, quiet: bool) -> None:
    if not quiet:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")


def tcp_check(host: str, port: int, timeout: float = 3.0) -> str | None:
    try:
        with socket.create_connection((host, port), timeout):
            return None
    except OSError as exc:
        return str(exc)


def main(argv=None, stream_factory=CameraStream) -> int:
    parser = argparse.ArgumentParser(description="Probe the camera stream with OpenCV.")
    parser.add_argument("--url", help="Stream URL (default: CAMERA_RTSP_URL or CAMERA_* variables)")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to read frames")
    parser.add_argument("--save-frame", default="outputs/probe_frame.jpg", help="JPEG proof frame ('' to skip)")
    parser.add_argument("--reconnect-test", action="store_true", help="Close and reopen the stream once")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    load_env_file()
    quiet = args.json
    summary = {"url": None, "tcp": False, "opened": False, "frames": 0, "invalid_frames": 0,
               "resolution": None, "measured_fps": None, "reported_fps": None, "reconnect": None, "error": None}

    def finish(code: int) -> int:
        if args.json:
            print(json.dumps(summary))
        return code

    try:
        url = args.url or camera_url_from_env()
        host, port = host_and_port(url)
    except ValueError as exc:
        summary["error"] = str(exc)
        _step(False, str(exc), quiet)
        return finish(1)
    summary["url"] = redact_url(url)
    _step(True, f"Stream URL: {summary['url']}", quiet)

    error = tcp_check(host, port)
    summary["tcp"] = error is None
    if error:
        summary["error"] = f"TCP {host}:{port} unreachable: {error}"
        _step(False, summary["error"], quiet)
        if not quiet:
            print("  Camera off, wrong IP, different network/AP isolation, or RTSP disabled in Imou Life.")
        return finish(2)
    _step(True, f"TCP {host}:{port} reachable", quiet)

    stream = stream_factory(url)
    if not stream.open():
        summary["error"] = stream.last_error
        _step(False, f"OpenCV could not open the stream: {stream.last_error}", quiet)
        if not quiet:
            print("  Usually a wrong password (safety code / RTSP password) or a wrong path.")
        return finish(3)
    summary["opened"] = True
    summary["reported_fps"] = stream.reported_fps
    _step(True, "OpenCV opened the stream", quiet)

    saved = None
    start = time.monotonic()
    first = last = None
    while time.monotonic() - start < args.duration:
        frame = stream.read()
        if frame is None:
            summary["invalid_frames"] += 1
            if summary["invalid_frames"] >= 5:
                break
            continue
        last = time.monotonic()
        first = first or last
        summary["frames"] += 1
        summary["resolution"] = list(stream.frame_size)
        if saved is None and args.save_frame:
            Path(args.save_frame).parent.mkdir(parents=True, exist_ok=True)
            saved = args.save_frame if cv2.imwrite(args.save_frame, frame) else None
    stream.release()
    if summary["frames"] > 1 and last > first:
        summary["measured_fps"] = round((summary["frames"] - 1) / (last - first), 2)
    ok = summary["frames"] > 0
    _step(ok, f"Frames: {summary['frames']} valid, {summary['invalid_frames']} invalid; "
              f"resolution {summary['resolution']}; measured FPS {summary['measured_fps']}; "
              f"stream FPS {summary['reported_fps']}", quiet)
    if saved and not quiet:
        print(f"  Proof frame saved to {saved}")
    if not ok:
        summary["error"] = stream.last_error
        return finish(4)

    if args.reconnect_test:
        again = stream_factory(url)
        summary["reconnect"] = again.open() and again.read() is not None
        again.release()
        _step(summary["reconnect"], "Reconnect: stream reopened and delivered a frame", quiet)
    return finish(0 if summary["reconnect"] is not False else 5)


if __name__ == "__main__":
    sys.exit(main())
