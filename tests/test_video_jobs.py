"""The uploaded-video worker, with a synthetic video, a fake detector and a fake backend."""

from threading import Event

import cv2
import numpy as np
import pytest

import video_worker
from smartcattle_ai.backend_client import BackendError
from smartcattle_ai.video_jobs import JobLost, VideoAnalyzer, process_job, run_worker
from smartcattle_ai.zones import ZoneSet

ZONE = ZoneSet.from_rectangle((0.0, 0.0, 0.5, 1.0))


def cow(x, inside, track_id, confidence=0.8):
    return {"class": "cow", "confidence": confidence, "bbox": [x, 0, x + 1, 1],
            "inside_zone": inside, "track_id": track_id}


class FakeBackend:
    def __init__(self, zones=None, progress_error=None):
        self.zones = zones or []
        self.progress_error = progress_error
        self.progress, self.completed, self.failed, self.jobs = [], [], [], []

    def camera_zones(self, camera_id):
        return self.zones

    def claim_video_job(self):
        return self.jobs.pop(0) if self.jobs else None

    def report_job_progress(self, job_id, progress):
        if self.progress_error:
            raise self.progress_error
        self.progress.append(progress)

    def complete_video_job(self, job_id, summary):
        self.completed.append((job_id, summary))

    def fail_video_job(self, job_id, error):
        self.failed.append((job_id, error))


class WalkingCow:
    """One cow walking left to right: inside the left-half zone, then out."""

    def __init__(self):
        self.frame = 0

    def detect(self, frame):
        x = 4 + self.frame * 4
        self.frame += 1
        return [{"class": "cow", "confidence": 0.9, "bbox": [x, 20, x + 10, 40]}]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "pasture.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def job(video, **changes):
    return {"job_id": "11111111-1111-4111-8111-111111111111", "download_url": "https://storage/signed",
            "file_name": "pasture.mp4", "content_type": "video/mp4", "size_bytes": video.stat().st_size,
            "lot_id": 1, "camera_id": "camera-01", "attempt": 1, **changes}


def copy_from(source):
    def fetch(url, target):
        target.write_bytes(source.read_bytes())
    return fetch


def test_the_analyzer_alerts_once_per_escape_and_per_person():
    analyzer = VideoAnalyzer()
    analyzer.add([cow(0, True, 1)], 0.0)
    analyzer.add([cow(9, False, 1), cow(0, True, 2)], 1.0)
    analyzer.add([cow(9, False, 1), {"class": "person", "confidence": 0.5, "bbox": [0, 0, 1, 1],
                                     "inside_zone": None, "track_id": 3}], 2.0)
    analyzer.add([cow(0, True, 1)], 3.0)
    analyzer.add([cow(9, False, 1)], 4.0)
    summary = analyzer.summary("  pasture\ntest.mp4 ", 5.0)
    assert summary["alerts"] == [
        {"track_id": 1, "cls": "cattle", "time_sec": 1.0},
        {"track_id": 3, "cls": "person", "time_sec": 2.0},
        {"track_id": 1, "cls": "cattle", "time_sec": 4.0},
    ]
    assert summary["video_name"] == "pasture test.mp4"
    assert (summary["frames_analyzed"], summary["max_cattle"], summary["max_people"]) == (5, 2, 1)
    assert summary["outside_ratio"] == pytest.approx(3 / 6, abs=1e-4)
    assert summary["avg_confidence"] == pytest.approx((0.8 * 6 + 0.5) / 7, abs=1e-4)


def test_an_empty_video_summary_has_no_division_by_zero():
    summary = VideoAnalyzer().summary("", 0)
    assert summary["video_name"] == "video"
    assert summary["avg_confidence"] == summary["outside_ratio"] == 0.0


def test_a_job_is_analyzed_with_its_cameras_zone_and_delivered(video, tmp_path):
    backend = FakeBackend(zones=[{"name": "Left", "active": True, "outline": None,
                                  "x_min": 0.0, "y_min": 0.0, "x_max": 0.5, "y_max": 1.0}])
    ticks = iter(range(1000))
    summary = process_job(backend, WalkingCow(), job(video), ZoneSet.from_rectangle((0, 0, 1, 1)),
                          vid_stride=1, progress_every_s=5, work_dir=tmp_path, fetch=copy_from(video),
                          clock=lambda: next(ticks))
    assert backend.failed == []
    assert backend.completed == [("11111111-1111-4111-8111-111111111111", summary)]
    assert summary["frames_analyzed"] == 20
    assert summary["duration_sec"] == pytest.approx(2.0)
    # The cow crosses the camera's left-half zone once: one escape, at the frame it left.
    assert [alert["cls"] for alert in summary["alerts"]] == ["cattle"]
    assert summary["alerts"][0]["track_id"] == 1
    assert 0 < summary["outside_ratio"] < 1
    # Progress was sent, with normalized boxes, and never claimed more frames than the total.
    assert backend.progress
    for progress in backend.progress:
        assert progress["frames_processed"] <= progress.get("frames_total", progress["frames_processed"])
        assert all(0 <= value <= 1 for box in progress["detections"] for value in box["bbox"])
    # The downloaded copy is gone.
    assert list(tmp_path.glob("smartcattle-job-*")) == []


def test_without_a_camera_the_configured_zone_applies(video, tmp_path):
    backend = FakeBackend(zones=[{"name": "Unused", "active": True, "outline": None,
                                  "x_min": 0.0, "y_min": 0.0, "x_max": 0.1, "y_max": 0.1}])
    summary = process_job(backend, WalkingCow(), job(video, camera_id=None), ZoneSet.from_rectangle((0, 0, 1, 1)),
                          vid_stride=2, work_dir=tmp_path, fetch=copy_from(video))
    assert summary["alerts"] == []
    assert summary["frames_analyzed"] == 10


def test_an_undecodable_file_fails_the_job(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    backend = FakeBackend()
    assert process_job(backend, WalkingCow(), job(broken), ZONE, work_dir=tmp_path, fetch=copy_from(broken)) is None
    assert backend.completed == []
    assert backend.failed[0][1].startswith("The video could not be decoded")


def test_a_failed_download_fails_the_job(video, tmp_path):
    def unreachable(url, target):
        raise OSError("connection reset")

    backend = FakeBackend()
    assert process_job(backend, WalkingCow(), job(video), ZONE, work_dir=tmp_path, fetch=unreachable) is None
    assert backend.failed == [("11111111-1111-4111-8111-111111111111", "Could not download the video: OSError")]


def test_a_job_taken_over_by_another_worker_is_abandoned(video, tmp_path):
    backend = FakeBackend(progress_error=BackendError("Backend HTTP 409: The job is queued", 409))
    with pytest.raises(JobLost):
        process_job(backend, WalkingCow(), job(video), ZONE, progress_every_s=0, work_dir=tmp_path,
                    fetch=copy_from(video))
    assert backend.completed == [] and backend.failed == []


def test_a_missed_heartbeat_is_not_fatal(video, tmp_path, capsys):
    backend = FakeBackend(progress_error=BackendError("Backend HTTP 503: busy", 503))
    assert process_job(backend, WalkingCow(), job(video), ZONE, progress_every_s=0, work_dir=tmp_path,
                       fetch=copy_from(video)) is not None
    assert "could not report progress" in capsys.readouterr().err


def test_the_worker_processes_the_queue(video, tmp_path):
    backend = FakeBackend()
    backend.jobs = [job(video)]
    assert run_worker(backend, WalkingCow(), ZONE, Event(), once=True, work_dir=tmp_path,
                      fetch=copy_from(video)) == 1
    assert len(backend.completed) == 1
    # Empty queue.
    assert run_worker(backend, WalkingCow(), ZONE, Event(), once=True) == 0


def test_the_worker_survives_a_backend_that_is_down(capsys):
    class Down(FakeBackend):
        def claim_video_job(self):
            raise BackendError("Could not connect to backend")

    assert run_worker(Down(), WalkingCow(), ZONE, Event(), once=True) == 0
    assert "could not reach the backend" in capsys.readouterr().err


def test_the_command_needs_a_backend(monkeypatch, capsys):
    monkeypatch.delenv("SMARTCATTLE_BACKEND_URL", raising=False)
    monkeypatch.setattr(video_worker, "load_env_file", lambda: None)
    assert video_worker.main([], detector_factory=lambda settings: WalkingCow()) == 1
    assert "SMARTCATTLE_BACKEND_URL" in capsys.readouterr().err
