"""What reaches a person: each anomaly once, nothing for a normal rest, and nothing lost to a failed send."""

import numpy as np
import pytest

from smartcattle_ai.anomalies import DOWN_AFTER_MINUTES, FORGET_AFTER_S, RETRY_AFTER_S, AnomalyPublisher
from smartcattle_ai.backend_client import BackendError, detection_log_report
from smartcattle_ai.posture import PostureMonitor
from smartcattle_ai.stillness import StillnessMonitor
from smartcattle_ai.tracking import IoUTracker, track_frame

DOWN_S = DOWN_AFTER_MINUTES * 60


class Client:
    def __init__(self, failures=0):
        self.sent = []
        self.failures = failures

    def send_event(self, event):
        if self.failures:
            self.failures -= 1
            raise BackendError("down")
        self.sent.append(event)


def cow(track=1, posture="lying", still=DOWN_S, since=100.0):
    return {"class": "cow", "confidence": 0.5, "track_id": track, "bbox": [0, 0, 10, 10], "posture": posture,
            "still_for_s": still, "still_since": since}


def person(track=2, night=True):
    item = {"class": "person", "confidence": 0.4, "track_id": track, "bbox": [0, 0, 10, 10]}
    return item | ({"night": True} if night else {})


def types(client):
    return [event["event_type"] for event in client.sent]


def test_a_cow_down_for_hours_is_reported_with_what_the_backend_needs():
    client = Client()
    AnomalyPublisher(client, "cam-1").handle([cow()], 0.0)
    event = client.sent[0]
    assert event["event_type"] == "cattle_down" and event["camera_id"] == "cam-1"
    assert event["detected_object"] == "cow" and event["confidence"] == 0.5 and event["timestamp"].endswith("+00:00")


@pytest.mark.parametrize("item", [
    cow(still=DOWN_S - 1),  # A long rest is normal: cattle lie down around half the day.
    cow(posture="standing"),  # Standing asleep, or a rock the tracker follows.
    cow(posture=None),
    {"class": "cow", "confidence": 0.5, "track_id": 1, "bbox": [0, 0, 1, 1]},
    person(night=False),  # People by day are the farm working.
    person(track=None),
])
def test_what_is_normal_stays_quiet(item):
    client = Client()
    AnomalyPublisher(client, "cam-1").handle([item], 0.0)
    assert client.sent == []


def test_a_person_at_night_and_an_outsider_are_reported():
    client = Client()
    outsider = {"class": "dog", "confidence": 0.3, "track_id": 5, "bbox": [0, 0, 1, 1]}
    AnomalyPublisher(client, "cam-1").handle([person(), outsider], 0.0)
    assert types(client) == ["possible_intrusion", "external_animal_detected"]


def test_each_anomaly_goes_once_however_many_frames_show_it():
    client = Client()
    publisher = AnomalyPublisher(client, "cam-1")
    for second in range(50):
        publisher.handle([cow(), person()], float(second))
    assert sorted(types(client)) == ["cattle_down", "possible_intrusion"]


def test_the_tracker_renumbering_a_resting_cow_is_not_a_second_alarm():
    client = Client()
    publisher = AnomalyPublisher(client, "cam-1")
    publisher.handle([cow(track=1)], 0.0)
    publisher.handle([cow(track=9)], 1.0)
    assert len(client.sent) == 1


def test_a_failed_send_waits_and_then_gets_through():
    client = Client(failures=1)
    publisher = AnomalyPublisher(client, "cam-1")
    publisher.handle([cow()], 0.0)
    publisher.handle([cow()], RETRY_AFTER_S - 1)
    assert client.sent == []
    publisher.handle([cow()], RETRY_AFTER_S)
    assert types(client) == ["cattle_down"]


def test_someone_coming_back_much_later_is_news():
    client = Client()
    publisher = AnomalyPublisher(client, "cam-1")
    publisher.handle([person()], 0.0)
    publisher.handle([], FORGET_AFTER_S + 1)
    publisher.handle([person()], FORGET_AFTER_S + 2)
    assert len(client.sent) == 2


def test_a_cow_lost_by_the_tracker_keeps_its_clock_when_it_returns_in_the_same_spot():
    monitor = StillnessMonitor(still_after_s=60)
    box = {"class": "cow", "confidence": 0.9, "bbox": [100, 100, 200, 200]}
    monitor.update([{**box, "track_id": 1}], 640, 480, 0.0)
    monitor.update([{**box, "track_id": 1}], 640, 480, 300.0)
    monitor.update([], 640, 480, 500.0)
    item = monitor.update([{**box, "track_id": 2}], 640, 480, 1000.0)[0]
    assert item["still_for_s"] == 1000.0 and item["still_since"] == 0.0


def test_a_different_cow_elsewhere_starts_its_own_clock():
    monitor = StillnessMonitor(still_after_s=60)
    monitor.update([{"class": "cow", "confidence": 0.9, "bbox": [0, 0, 50, 50], "track_id": 1}], 640, 480, 0.0)
    item = monitor.update([{"class": "cow", "confidence": 0.9, "bbox": [400, 300, 500, 400], "track_id": 2}],
                          640, 480, 100.0)[0]
    assert item["still_for_s"] == 0.0


def test_two_cows_resting_apart_never_share_a_clock():
    monitor = StillnessMonitor(still_after_s=60)
    near = {"class": "cow", "confidence": 0.9, "bbox": [100, 100, 200, 200]}
    monitor.update([{**near, "track_id": 1}], 640, 480, 0.0)
    both = monitor.update([{**near, "track_id": 1}, {**near, "track_id": 2}], 640, 480, 100.0)
    assert [item["still_for_s"] for item in both] == [100.0, 0.0]


class Weak:
    def __init__(self, strong, weak):
        self.answer = (strong, weak)

    def detect_with_weak(self, frame):
        return self.answer


def box(confidence, x=100):
    return {"class": "cow", "confidence": confidence, "bbox": [x, 100, x + 100, 200]}


def test_a_followed_animal_survives_frames_where_it_reads_weak():
    tracker = IoUTracker()
    first = track_frame(Weak([box(0.9)], []), tracker, None)
    second = track_frame(Weak([], [box(0.2)]), tracker, None)
    assert second[0]["track_id"] == first[0]["track_id"]


def test_a_weak_detection_alone_is_ignored_because_it_may_be_a_rock():
    assert track_frame(Weak([], [box(0.2)]), IoUTracker(), None) == []


def test_a_strong_detection_takes_the_track_over_a_weak_one():
    tracker = IoUTracker()
    track_frame(Weak([box(0.9)], []), tracker, None)
    found = track_frame(Weak([box(0.8)], [box(0.2)]), tracker, None)
    assert [item["confidence"] for item in found] == [0.8]


class Fixed:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def predict(self, frame, boxes):
        self.calls += 1
        return [self.answers.pop(0) for _ in boxes]


FRAME = np.zeros((480, 640, 3), np.uint8)
# Wide: the shape alone says lying.
WIDE = {"class": "cow", "confidence": 0.9, "track_id": 1, "bbox": [100, 100, 300, 180]}
# Tall: the shape alone says standing.
TALL = {"class": "cow", "confidence": 0.9, "track_id": 1, "bbox": [100, 100, 220, 220]}


def settled(monitor, item, seconds=30, frame=FRAME):
    result = None
    for second in range(seconds):
        result = monitor.update([dict(item)], 640, 480, float(second), frame)[0]
    return result


def test_the_classifier_finds_the_cow_lying_facing_the_camera_that_the_shape_misses():
    assert settled(PostureMonitor(), TALL)["posture"] == "standing"
    assert settled(PostureMonitor(Fixed([("lying", 0.95)] * 30)), TALL)["posture"] == "lying"


def test_an_unsure_classifier_leaves_the_decision_to_the_shape():
    assert settled(PostureMonitor(Fixed([("standing", 0.5)] * 30)), WIDE)["posture"] == "lying"


def test_one_odd_picture_does_not_change_the_posture():
    answers = [("lying", 0.9)] * 3 + [("standing", 0.9)] + [("lying", 0.9)] * 20
    assert settled(PostureMonitor(Fixed(answers)), TALL)["posture"] == "lying"


def test_the_classifier_runs_every_couple_of_seconds_rather_than_every_frame():
    classifier = Fixed([("lying", 0.9)] * 30)
    settled(PostureMonitor(classifier), TALL, seconds=10)
    assert classifier.calls == 5


def test_a_broken_classifier_falls_back_on_the_shape_and_keeps_watching():
    class Broken:
        def predict(self, frame, boxes):
            raise RuntimeError("corrupt weights")

    assert settled(PostureMonitor(Broken()), WIDE)["posture"] == "lying"


def test_a_walking_cow_the_classifier_calls_lying_stays_on_its_feet():
    """Seen on real herd video: walking, with the legs hidden or blurred, it looks like a cow lying down."""
    monitor = PostureMonitor(Fixed([("lying", 0.95)] * 60))
    item = None
    for second in range(40):
        moved = {**TALL, "bbox": [100 + second * 10, 100, 220 + second * 10, 220]}
        item = monitor.update([moved], 640, 480, second / 2, FRAME)[0]
    assert item["posture"] != "lying"


def test_a_cow_is_believed_lying_only_after_holding_its_place():
    monitor = PostureMonitor(Fixed([("lying", 0.95)] * 60))
    assert settled(monitor, TALL, seconds=4)["posture"] == "standing"
    late = None
    for second in range(4, 12):
        late = monitor.update([dict(TALL)], 640, 480, float(second), FRAME)[0]
    assert late["posture"] == "lying"


def test_half_a_cow_at_the_edge_is_judged_by_its_look():
    at_edge = {"class": "cow", "confidence": 0.9, "track_id": 1, "bbox": [0, 100, 120, 220]}
    assert settled(PostureMonitor(Fixed([("lying", 0.95)] * 30)), at_edge)["posture"] == "lying"


def test_half_a_cow_at_the_edge_gets_no_posture_without_a_classifier():
    at_edge = {"class": "cow", "confidence": 0.9, "track_id": 1, "bbox": [0, 100, 120, 220]}
    assert "posture" not in settled(PostureMonitor(), at_edge)


def test_the_daily_report_counts_what_the_frame_holds_by_posture():
    report = detection_log_report([
        cow(posture="lying"), cow(posture="standing"), cow(posture="walking"), cow(posture=None), person(),
        {"class": "dog", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
    ], "2026-11-03T03:00:00+00:00")
    assert report == {
        "cattle_lying": 1, "cattle_standing": 1, "cattle_walking": 1, "cattle_unknown": 1, "people": 1,
        "avg_confidence": pytest.approx(0.48), "observed_at": "2026-11-03T03:00:00+00:00",
    }


def test_an_empty_scene_is_still_a_report_without_a_confidence():
    report = detection_log_report([], "2026-11-03T03:00:00+00:00")
    assert report["cattle_lying"] == 0 and report["people"] == 0 and "avg_confidence" not in report
