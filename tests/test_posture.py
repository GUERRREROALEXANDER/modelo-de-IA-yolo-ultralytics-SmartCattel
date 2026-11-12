from smartcattle_ai.posture import LYING, STANDING, WALKING, PostureMonitor

SIZE = 1000


def cow(bbox, track_id=1):
    return {"class": "cow", "confidence": 0.9, "bbox": bbox, "track_id": track_id}


def run(monitor, boxes, start=0.0, step=0.5):
    """Feeds one box per step; returns the last estimate."""
    posture = None
    for index, bbox in enumerate(boxes):
        posture = monitor.update([cow(bbox)], SIZE, SIZE, start + index * step)[0].get("posture")
    return posture


def test_a_cow_that_stays_put_is_standing():
    assert run(PostureMonitor(), [[300, 300, 500, 500]] * 10) == STANDING


def test_a_cow_whose_box_crosses_the_picture_is_walking():
    boxes = [[300 + 40 * i, 300, 500 + 40 * i, 500] for i in range(10)]
    assert run(PostureMonitor(), boxes) == WALKING


def test_a_jittering_box_is_not_walking():
    boxes = [[300 + (i % 2) * 3, 300, 500 + (i % 2) * 3, 500] for i in range(20)]
    assert run(PostureMonitor(), boxes) == STANDING


def test_a_calf_only_ever_seen_lying_is_lying():
    # Nothing to compare it with: a calf that never stood up in view must still read as lying.
    assert run(PostureMonitor(), [[300, 400, 600, 500]] * 10) == LYING


def test_a_cow_that_lies_down_where_it_stood():
    monitor = PostureMonitor()
    run(monitor, [[300, 200, 560, 500]] * 10)
    assert run(monitor, [[300, 320, 560, 500]] * 10, start=5.0) == LYING


def test_a_cow_far_away_is_not_lying_because_it_looks_small():
    monitor = PostureMonitor()
    run(monitor, [[300, 200, 560, 500]] * 4)
    assert run(monitor, [[400, 100, 500, 200]] * 10, start=60.0) == STANDING


def test_a_long_rest_stays_lying_and_getting_up_ends_it():
    monitor = PostureMonitor()
    run(monitor, [[300, 200, 560, 500]] * 10)
    run(monitor, [[300, 320, 560, 500]] * 10, start=5.0)
    # Seen all along, every half minute, for ten minutes: far longer than the samples remember.
    assert run(monitor, [[300, 320, 560, 500]] * 20, start=10.0, step=30.0) == LYING
    assert run(monitor, [[300, 200, 560, 500]] * 10, start=610.0) == STANDING


def test_one_bad_frame_does_not_change_the_posture():
    monitor = PostureMonitor()
    run(monitor, [[300, 200, 560, 500]] * 10)
    assert run(monitor, [[300, 320, 560, 500]], start=5.0) == STANDING
    assert run(monitor, [[300, 200, 560, 500]] * 3, start=5.5) == STANDING


def test_a_cut_off_cow_says_nothing_new():
    monitor = PostureMonitor()
    assert "posture" not in monitor.update([cow([0, 300, 300, 400])], SIZE, SIZE, 0.0)[0]
    run(monitor, [[300, 200, 560, 500]] * 3)
    assert monitor.update([cow([0, 300, 300, 400])], SIZE, SIZE, 2.0)[0]["posture"] == STANDING


def test_without_tracking_only_a_flat_box_says_anything():
    monitor = PostureMonitor()
    flat = {"class": "cow", "confidence": 0.9, "bbox": [300, 400, 600, 500]}
    tall = {"class": "cow", "confidence": 0.9, "bbox": [300, 200, 500, 500]}
    assert monitor.update([flat], SIZE, SIZE, 0.0)[0]["posture"] == LYING
    assert "posture" not in monitor.update([tall], SIZE, SIZE, 0.0)[0]


def test_people_and_a_trained_models_answer_are_left_alone():
    monitor = PostureMonitor()
    person = {"class": "person", "confidence": 0.9, "bbox": [300, 400, 600, 500], "track_id": 2}
    trained = {**cow([300, 200, 500, 500]), "posture": LYING}
    assert monitor.update([person, trained], SIZE, SIZE, 0.0) == [person, trained]


def test_animals_gone_from_view_are_forgotten():
    monitor = PostureMonitor()
    run(monitor, [[300, 200, 560, 500]] * 3)
    monitor.update([], SIZE, SIZE, 10_000.0)
    assert monitor._tracks == {}


def test_the_posture_travels_to_the_backend_and_to_the_picture():
    import numpy as np

    from smartcattle_ai.backend_client import backend_detections
    from smartcattle_ai.media import draw_detections

    lying = {**cow([300, 400, 600, 500]), "inside_zone": True, "posture": LYING}
    plain = {**cow([300, 200, 500, 500]), "inside_zone": True}
    reported = backend_detections([lying, plain], SIZE, SIZE)
    assert reported[0]["posture"] == LYING and "posture" not in reported[1]
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    assert (draw_detections(frame, [lying]) != draw_detections(frame, [{**lying, "posture": STANDING}])).any()


def test_the_live_service_adds_the_posture_to_what_it_publishes():
    import numpy as np

    from smartcattle_ai.live import LiveState, _process_frame
    from smartcattle_ai.zones import ZoneSet

    class Flat:
        def detect(self, frame):
            return [{"class": "cow", "confidence": 0.9, "bbox": [100, 300, 400, 400]}]

    state = LiveState("camera-01", ZoneSet.from_rectangle((0.0, 0.0, 1.0, 1.0)))
    state.posture = PostureMonitor()
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    _process_frame(frame, Flat(), state, None, 70, None, None, None, None, None, None, None)
    assert state.snapshot()["detections"][0]["posture"] == LYING
