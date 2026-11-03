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
    assert run(monitor, [[300, 320, 560, 500]] * 10, start=600.0) == LYING
    assert run(monitor, [[300, 200, 560, 500]] * 10, start=700.0) == STANDING


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
