"""Whether a cow is walking, standing or lying, estimated from how its box behaves.

The detector only says "cow". This reads the box over time, so it is an estimate and needs no other model; a model
trained on cow_standing and cow_lying replaces it (see Detector).
"""

from collections import deque
from dataclasses import dataclass, field

WALKING = "walking"
STANDING = "standing"
LYING = "lying"

#: The middle of the box moving this many body lengths per second is an animal walking, not a box jittering.
WALKING_BODY_LENGTHS_PER_S = 0.15
#: A box this much wider than tall is a body along the ground: a cow on its feet, seen from the side, stays below it.
LYING_ASPECT = 2.0
#: The box falling to this share of its recent height, with the feet in the same place, is the animal lying down.
HEIGHT_DROP = 0.72
#: Feet this close, as a share of the box height, mean the animal has not changed its distance to the camera.
SAME_PLACE = 0.15
#: How long a sample stays in memory: enough to compare with how the animal looked a moment ago.
WINDOW_S = 60.0
#: Speed is measured over at least this long: from one frame to the next, jitter looks like movement.
SPEED_OVER_S = 1.0


@dataclass
class _Track:
    samples: deque = field(default_factory=deque)
    state: str = STANDING


def _center(bbox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


class PostureMonitor:
    def __init__(self):
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[dict], width: int, height: int, now: float) -> list[dict]:
        """The same detections, with "posture" on the cows it can say something about."""
        result = []
        for item in detections:
            if item["class"] != "cow" or item.get("track_id") is None:
                result.append(item)
                continue
            result.append({**item, "posture": self._estimate(item, now)})
        return result

    def _estimate(self, item: dict, now: float) -> str:
        track = self._tracks.setdefault(item["track_id"], _Track())
        bbox = item["bbox"]
        track.samples.append((now, bbox))
        while track.samples and now - track.samples[0][0] > WINDOW_S:
            track.samples.popleft()
        if self._is_walking(track, bbox, now):
            track.state = WALKING
        else:
            track.state = LYING if self._is_flat(bbox) or self._has_dropped(track, bbox) else STANDING
        return track.state

    @staticmethod
    def _has_dropped(track: _Track, bbox) -> bool:
        """Lower than it just was, in the same spot. A cow walking away also shrinks, but its feet move up the picture."""
        height = bbox[3] - bbox[1]
        tallest = max(
            (old[3] - old[1] for _, old in track.samples if abs(old[3] - bbox[3]) <= SAME_PLACE * height), default=height,
        )
        return height <= HEIGHT_DROP * tallest

    @staticmethod
    def _is_flat(bbox) -> bool:
        return (bbox[2] - bbox[0]) >= LYING_ASPECT * max(bbox[3] - bbox[1], 1.0)

    @staticmethod
    def _is_walking(track: _Track, bbox, now: float) -> bool:
        older = next((sample for sample in reversed(track.samples) if now - sample[0] >= SPEED_OVER_S), None)
        if older is None:
            return False
        (x, y), (old_x, old_y) = _center(bbox), _center(older[1])
        body = max(bbox[2] - bbox[0], 1.0)
        moved = ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5
        return moved / body / (now - older[0]) >= WALKING_BODY_LENGTHS_PER_S
