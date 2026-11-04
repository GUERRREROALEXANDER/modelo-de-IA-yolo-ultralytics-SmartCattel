"""Whether a cow is walking, standing or lying, estimated from how its box behaves.

The detector only says "cow". The box over time tells walking apart, and the look of the animal, through a small
classifier when it is installed, tells standing from lying. Without it, or when it is unsure, the box's shape decides.
A detector trained on cow_standing and cow_lying replaces all of this (see Detector).
"""

import sys
from collections import Counter, deque
from dataclasses import dataclass, field

from .posture_classifier import MIN_CONFIDENCE

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
#: Back to this share of the height it stood at, the animal has got up.
HEIGHT_RECOVERED = 0.85
#: A new posture must hold this long: one bad frame must not make an animal drop and rise on screen.
SETTLE_S = 2.0
#: A box this close to the edge of the picture is a cut-off animal, so its shape says nothing.
EDGE_MARGIN = 0.01
#: A track unseen for this long is gone.
FORGET_AFTER_S = 300.0
#: How long a sample stays in memory: enough to compare with how the animal looked a moment ago.
WINDOW_S = 60.0
#: Speed is measured over at least this long: from one frame to the next, jitter looks like movement.
SPEED_OVER_S = 1.0
#: A crop this often is plenty: cattle change posture over seconds, and the detector needs the CPU more.
CLASSIFY_EVERY_S = 2.0
#: The classifier's latest answers about one animal: one odd picture, a passing shadow, must not outvote the others.
VOTES = 5
#: The classifier alone calls a walking cow with hidden or blurred legs "lying" (measured on real herd video). A cow that
#: lies down stays put, so it must have held this spot this long, within this share of its body, before that is believed.
LYING_NEEDS_STILL_S = 5.0
LYING_STILL_BODY_SHARE = 0.25


@dataclass
class _Track:
    samples: deque = field(default_factory=deque)
    state: str = STANDING
    # How tall it stood when it lay down: past WINDOW_S the samples no longer remember.
    standing_height: float | None = None
    candidate: str | None = None
    candidate_since: float = 0.0
    votes: deque = field(default_factory=lambda: deque(maxlen=VOTES))
    classified_at: float = float("-inf")


def _center(bbox) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


class PostureMonitor:
    def __init__(self, classifier=None):
        self._tracks: dict[int, _Track] = {}
        self._classifier = classifier

    def update(self, detections: list[dict], width: int, height: int, now: float, frame=None) -> list[dict]:
        """The same detections, with "posture" on the cows it can say something about."""
        self._forget_gone(now)
        self._classify(detections, frame, now)
        result = []
        for item in detections:
            if item["class"] != "cow" or "posture" in item:
                result.append(item)  # A model trained on postures has already said.
                continue
            if item.get("track_id") is None:
                # Tracking is off, so there is no history: only the shape can say anything.
                result.append({**item, "posture": LYING} if self._is_flat(item["bbox"]) else item)
                continue
            if self._is_cut_off(item["bbox"], width, height):
                # Its shape says nothing, but its look does: half a cow lying down still looks like one.
                known = self._tracks.get(item["track_id"])
                if known is None or not known.votes:
                    result.append({**item, "posture": known.state} if known is not None and known.samples else item)
                    continue
            result.append({**item, "posture": self._estimate(item, now)})
        return result

    def _classify(self, detections: list[dict], frame, now: float) -> None:
        """Adds a vote for each tracked cow that is due for one, all in one batch."""
        if self._classifier is None or frame is None:
            return
        due = []
        for item in detections:
            track_id = item.get("track_id")
            if item["class"] != "cow" or "posture" in item or track_id is None:
                continue
            track = self._tracks.setdefault(track_id, _Track())
            if now - track.classified_at >= CLASSIFY_EVERY_S:
                track.classified_at = now
                due.append((track, item["bbox"]))
        if not due:
            return
        try:
            answers = self._classifier.predict(frame, [bbox for _, bbox in due])
        except Exception as exc:  # A broken classifier must leave the box's estimate, not skip the frame.
            print(f"Warning: posture classifier off after {type(exc).__name__}", file=sys.stderr, flush=True)
            self._classifier = None
            return
        for (track, _), (name, confidence) in zip(due, answers):
            if confidence >= MIN_CONFIDENCE:
                track.votes.append(LYING if name == LYING else STANDING)

    def _forget_gone(self, now: float) -> None:
        """A service left running for months must not keep every animal it ever numbered."""
        gone = [key for key, track in self._tracks.items()
                if track.samples and now - track.samples[-1][0] > FORGET_AFTER_S]
        for key in gone:
            del self._tracks[key]

    @staticmethod
    def _is_cut_off(bbox, width: int, height: int) -> bool:
        margin_x, margin_y = EDGE_MARGIN * width, EDGE_MARGIN * height
        return (bbox[0] <= margin_x or bbox[1] <= margin_y
                or bbox[2] >= width - margin_x or bbox[3] >= height - margin_y)

    def _estimate(self, item: dict, now: float) -> str:
        track = self._tracks.setdefault(item["track_id"], _Track())
        bbox = item["bbox"]
        track.samples.append((now, bbox))
        while track.samples and now - track.samples[0][0] > WINDOW_S:
            track.samples.popleft()
        if self._is_walking(track, bbox, now):
            seen = WALKING
        elif track.votes:
            seen = Counter(track.votes).most_common(1)[0][0]
            if seen == LYING and not self._has_stayed(track, bbox, now):
                seen = STANDING
        else:
            seen = LYING if self._is_flat(bbox) or self._is_down(track, bbox) else STANDING
        if len(track.samples) == 1:
            track.state = seen
        return self._settled(track, seen, now)

    @staticmethod
    def _settled(track: _Track, seen: str, now: float) -> str:
        if seen == track.state:
            track.candidate = None
        elif seen != track.candidate:
            track.candidate, track.candidate_since = seen, now
        elif now - track.candidate_since >= SETTLE_S:
            track.state, track.candidate = seen, None
        return track.state

    def _is_down(self, track: _Track, bbox) -> bool:
        height = bbox[3] - bbox[1]
        if track.standing_height is not None and height < HEIGHT_RECOVERED * track.standing_height:
            return True
        track.standing_height = None
        tallest = self._tallest_here(track, bbox)
        if height <= HEIGHT_DROP * tallest:
            track.standing_height = tallest
            return True
        return False

    @staticmethod
    def _tallest_here(track: _Track, bbox) -> float:
        """Its recent height in this spot. A cow walking away also shrinks, but its feet move up the picture."""
        height = bbox[3] - bbox[1]
        return max(
            (old[3] - old[1] for _, old in track.samples if abs(old[3] - bbox[3]) <= SAME_PLACE * height), default=height,
        )

    @staticmethod
    def _has_stayed(track: _Track, bbox, now: float) -> bool:
        if now - track.samples[0][0] < LYING_NEEDS_STILL_S:
            return False
        x, y = _center(bbox)
        reach = LYING_STILL_BODY_SHARE * max(bbox[2] - bbox[0], 1.0)
        for moment, old in track.samples:
            old_x, old_y = _center(old)
            if now - moment <= LYING_NEEDS_STILL_S and ((x - old_x) ** 2 + (y - old_y) ** 2) ** 0.5 > reach:
                return False
        return True

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
