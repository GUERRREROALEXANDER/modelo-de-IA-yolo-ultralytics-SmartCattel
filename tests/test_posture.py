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
