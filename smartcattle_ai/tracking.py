"""Lightweight IoU tracker: the same animal keeps its number from frame to frame.

Without it every frame is a stranger: a cow standing outside the zone for ten
seconds is a hundred "escapes". With it, the escape is one event per animal.

Greedy IoU matching is enough here: cattle move slowly against the frame rate,
and it needs no extra dependency (ByteTrack in Ultralytics pulls in `lap`).
"""


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def track_frame(detector, tracker, frame) -> list[dict]:
    """A frame's tracked detections, keeping followed animals through the frames where they read weak."""
    split = getattr(detector, "detect_with_weak", None)
    if split is None:
        return tracker.update(detector.detect(frame))
    strong, weak = split(frame)
    return tracker.update(strong, weak)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be greater than 0 and at most 1")
        if max_missed < 0:
            raise ValueError("max_missed must be nonnegative")
        self.iou_threshold = iou_threshold
        # Frames a track survives unseen: an animal hidden behind another one for a moment keeps its number.
        self.max_missed = max_missed
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def _match(self, detections: list[dict], taken: set[int]) -> dict[int, int]:
        """Index of each detection that continues a track, best overlaps first. Adds what it uses to `taken`."""
        candidates = sorted(
            ((iou(track["bbox"], detection["bbox"]), track_id, index)
             for track_id, track in self._tracks.items() if track_id not in taken
             for index, detection in enumerate(detections)
             if track["class"] == detection["class"]),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        for score, track_id, index in candidates:
            if score < self.iou_threshold:
                break
            if track_id in taken or index in assigned:
                continue
            assigned[index] = track_id
            taken.add(track_id)
        return assigned

    def update(self, detections: list[dict], weak: list[dict] = ()) -> list[dict]:
        """The same detections, each with a track_id. Boxes are matched only within their class.

        A weak detection continues a track that nothing else claimed and otherwise is dropped: alone it may be a rock,
        but on top of an animal seen a moment ago it is that animal, in the dark or half hidden.
        """
        used_tracks: set[int] = set()
        assigned = self._match(detections, used_tracks)
        kept_weak = self._match(list(weak), used_tracks) if weak else {}

        result = []
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = {"bbox": detection["bbox"], "class": detection["class"], "missed": 0}
            result.append({**detection, "track_id": track_id})
        for index, track_id in kept_weak.items():
            detection = weak[index]
            self._tracks[track_id] = {"bbox": detection["bbox"], "class": detection["class"], "missed": 0}
            result.append({**detection, "track_id": track_id})

        seen = {item["track_id"] for item in result}
        for track_id in [track_id for track_id in self._tracks if track_id not in seen]:
            self._tracks[track_id]["missed"] += 1
            if self._tracks[track_id]["missed"] > self.max_missed:
                del self._tracks[track_id]
        return result

    @property
    def active(self) -> int:
        return len(self._tracks)
