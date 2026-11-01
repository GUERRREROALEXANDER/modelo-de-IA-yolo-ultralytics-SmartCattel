"""Worker for videos uploaded in the dashboard: claim, download, analyze, report.

The backend queues each upload; this claims the oldest, downloads it from
storage, runs the same detector as the live camera over it and delivers the
summary. Progress, with the boxes of the current frame, goes out every few
seconds: it renews the job's lease and lets the dashboard draw the boxes over
the playback.
"""

import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2

from .backend_client import BackendError, backend_detections
from .rules import evaluate
from .tracking import IoUTracker
from .zones import ZoneSet

#: The backend's ceiling on alerts per analysis.
MAX_ALERTS = 500
#: The backend's ceiling on uploads; anything larger is not what it handed out.
MAX_DOWNLOAD_BYTES = 1024 ** 3 + 1024 ** 2


class JobLost(Exception):
    """The backend gave the job to someone else (lease expired) or it no longer exists."""


class VideoAnalyzer:
    """Accumulates one video's statistics, frame by frame. No I/O, so it is tested on plain lists."""

    def __init__(self):
        self.frames = 0
        self.max_cattle = 0
        self.max_people = 0
        self._confidence_sum = 0.0
        self._detections = 0
        self._cattle_seen = 0
        self._cattle_outside = 0
        self._outside_tracks: set[int] = set()
        self._people_tracks: set[int] = set()
        self.alerts: list[dict] = []

    def add(self, detections: list[dict], time_sec: float) -> None:
        """detections: tracked (track_id) and evaluated against the zone (inside_zone for cattle)."""
        self.frames += 1
        cattle = [item for item in detections if item["class"] == "cow"]
        people = [item for item in detections if item["class"] == "person"]
        self.max_cattle = max(self.max_cattle, len(cattle))
        self.max_people = max(self.max_people, len(people))
        for item in detections:
            self._confidence_sum += item["confidence"]
            self._detections += 1
        for item in cattle:
            self._cattle_seen += 1
            track_id = item.get("track_id")
            if item["inside_zone"]:
                self._outside_tracks.discard(track_id)
                continue
            self._cattle_outside += 1
            # One alert per escape: when the animal leaves, not on every frame it stays out.
            if track_id not in self._outside_tracks:
                self._outside_tracks.add(track_id)
                self._alert(track_id, "cattle", time_sec)
        for item in people:
            track_id = item.get("track_id")
            if track_id not in self._people_tracks:
                self._people_tracks.add(track_id)
                self._alert(track_id, "person", time_sec)

    def _alert(self, track_id, cls: str, time_sec: float) -> None:
        if len(self.alerts) < MAX_ALERTS:
            self.alerts.append({"track_id": track_id or 0, "cls": cls, "time_sec": round(max(time_sec, 0.0), 2)})

    def summary(self, video_name: str, duration_sec: float) -> dict:
        """The body of POST /api/ai/video-jobs/{id}/complete."""
        name = " ".join(video_name.split())[:200] or "video"
        return {
            "video_name": name,
            "duration_sec": round(max(duration_sec, 0.0), 2),
            "frames_analyzed": self.frames,
            "max_cattle": self.max_cattle,
            "max_people": self.max_people,
            "alerts": self.alerts,
            "avg_confidence": round(min(self._confidence_sum / self._detections, 1.0), 4) if self._detections else 0.0,
            "outside_ratio": round(self._cattle_outside / self._cattle_seen, 4) if self._cattle_seen else 0.0,
        }


def download(url: str, target: Path, timeout: float = 30.0) -> None:
    """Streams the video to disk. No API key: this URL is the storage's, signed, not the backend's."""
    with urlopen(Request(url), timeout=timeout) as response, open(target, "wb") as out:
        written = 0
        while chunk := response.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_DOWNLOAD_BYTES:
                raise ValueError("The video is larger than the upload limit")
            out.write(chunk)


def job_zones(client, job: dict, fallback: ZoneSet) -> ZoneSet:
    """The safe zones of the lot's camera; the configured zone when the lot has none."""
    camera_id = job.get("camera_id")
    if not camera_id:
        return fallback
    return ZoneSet.from_backend(client.camera_zones(camera_id)) or fallback


def process_job(client, detector, job: dict, fallback_zone: ZoneSet, *, vid_stride: int = 2,
                progress_every_s: float = 2.0, work_dir: Path | None = None,
                fetch=download, clock=time.monotonic) -> dict | None:
    """Analyzes one claimed job and delivers the result. Returns the summary, or None if it failed."""
    job_id = str(job["job_id"])
    directory = Path(tempfile.mkdtemp(prefix="smartcattle-job-", dir=work_dir))
    try:
        video = directory / ("video" + Path(job.get("file_name") or "").suffix[:10])
        try:
            fetch(job["download_url"], video)
        except (URLError, OSError, ValueError) as exc:
            _fail(client, job_id, f"Could not download the video: {type(exc).__name__}")
            return None
        zones = job_zones(client, job, fallback_zone)
        summary = _analyze(client, detector, job_id, video, job.get("file_name") or "video", zones,
                           vid_stride, progress_every_s, clock)
        if summary is None:
            return None
        client.complete_video_job(job_id, summary)
        return summary
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _analyze(client, detector, job_id, video: Path, name: str, zones: ZoneSet,
             vid_stride: int, progress_every_s: float, clock) -> dict | None:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            _fail(client, job_id, "The video could not be decoded (unsupported or damaged file)")
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 0 < fps <= 1000 else 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        expected = math.ceil(frame_count / vid_stride) if frame_count > 0 else None
        tracker = IoUTracker(max_missed=max(1, round(fps / vid_stride)))
        analyzer = VideoAnalyzer()
        frames_read = 0
        started = last_progress = clock()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            if (frames_read - 1) % vid_stride:
                continue
            height, width = frame.shape[:2]
            detections = tracker.update(detector.detect(frame))
            cattle, _ = evaluate([item for item in detections if item["class"] == "cow"], width, height, zones)
            detections = cattle + [{**item, "inside_zone": None} for item in detections if item["class"] != "cow"]
            time_sec = (frames_read - 1) / fps
            analyzer.add(detections, time_sec)
            now = clock()
            if now - last_progress >= progress_every_s:
                last_progress = now
                _progress(client, job_id, analyzer.frames, expected, now - started, time_sec,
                          backend_detections(detections, width, height))
        if analyzer.frames == 0:
            _fail(client, job_id, "The video has no readable frames")
            return None
        return analyzer.summary(name, frames_read / fps)
    finally:
        capture.release()


def _progress(client, job_id, processed, expected, elapsed, time_sec, boxes) -> None:
    progress = {"frames_processed": processed, "video_time_sec": round(time_sec, 3), "detections": boxes}
    # The container's frame count is an estimate; the backend rejects processed > total.
    if expected is not None and expected >= processed:
        progress["frames_total"] = expected
    if elapsed > 0:
        progress["fps"] = round(min(processed / elapsed, 1000), 2)
    try:
        client.report_job_progress(job_id, progress)
    except BackendError as exc:
        if exc.status in (404, 409):
            raise JobLost(str(exc)) from exc
        # A missed heartbeat is not fatal: the lease is ten minutes and the next one renews it.
        print(f"Warning: could not report progress: {exc}", file=sys.stderr)


def _fail(client, job_id: str, error: str) -> None:
    print(f"Job {job_id} failed: {error}", file=sys.stderr)
    try:
        client.fail_video_job(job_id, error)
    except BackendError as exc:
        print(f"Warning: could not mark job {job_id} as failed: {exc}", file=sys.stderr)


def run_worker(client, detector, fallback_zone: ZoneSet, stop, *, poll_s: float = 5.0, once: bool = False,
               **options) -> int:
    """Claims and processes jobs until stop is set. With once, returns after the first job or an empty queue."""
    done = 0
    warned = False
    while not stop.is_set():
        try:
            job = client.claim_video_job()
            warned = False
        except BackendError as exc:
            if not warned:
                print(f"Warning: could not reach the backend: {exc}", file=sys.stderr)
            warned = True
            job = None
        if job is None:
            if once:
                return done
            stop.wait(poll_s)
            continue
        print(f"Job {job['job_id']}: {job.get('file_name')} (attempt {job.get('attempt')})", flush=True)
        try:
            summary = process_job(client, detector, job, fallback_zone, **options)
        except JobLost as exc:
            print(f"Job {job['job_id']} was taken over: {exc}", file=sys.stderr)
            summary = None
        except BackendError as exc:
            print(f"Warning: job {job['job_id']} could not be delivered: {exc}", file=sys.stderr)
            summary = None
        if summary is not None:
            done += 1
            print(f"Job {job['job_id']} done: {summary['frames_analyzed']} frames, max cattle "
                  f"{summary['max_cattle']}, {len(summary['alerts'])} alerts", flush=True)
        if once:
            return done
    return done
