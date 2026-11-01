"""Stillness monitor: a cow that has not moved for half an hour may be down or calving.

The tracker gives every animal a number; this module gives that number a memory.
The clock starts at the first sighting and only restarts when the center drifts
further than a small tolerance, so a bounding box breathing with the animal does
not keep resetting it. No new dependency: a dict per track and Euclidean distance
in normalized units are enough.
"""


def _distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class StillnessMonitor:
    def __init__(self, still_after_s: float = 1800.0, tolerance: float = 0.02, forget_after_s: float = 120.0):
        if still_after_s <= 0:
            raise ValueError("still_after_s must be greater than 0")
        if not 0 < tolerance < 1:
            raise ValueError("tolerance must be greater than 0 and less than 1")
        if forget_after_s <= 0:
            raise ValueError("forget_after_s must be greater than 0")
        self.still_after_s = still_after_s
        # Fraction of the frame the center may drift before it counts as movement.
        self.tolerance = tolerance
        # Seconds a track survives unseen: a cow that leaves the frame comes back clean.
        self.forget_after_s = forget_after_s
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[dict], width: int, height: int, now: float) -> list[dict]:
        """The same detections, each monitored cow tagged with how long it has held still."""
        result = []
        for detection in detections:
            item = dict(detection)
            track_id = detection.get("track_id")
            if detection.get("class") != "cow" or track_id is None:
                result.append(item)
                continue
            x1, y1, x2, y2 = detection["bbox"]
            center = ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height)
            track = self._tracks.get(track_id)
            if track is None or _distance(center, track["center"]) > self.tolerance:
                track = {"center": center, "since": now, "seen": now, "alerted": False}
                self._tracks[track_id] = track
            track["seen"] = now
            item["still_for_s"] = round(now - track["since"], 1)
            result.append(item)
        for track_id in [track_id for track_id, track in self._tracks.items()
                         if now - track["seen"] > self.forget_after_s]:
            del self._tracks[track_id]
        return result

    def new_alerts(self, detections: list[dict]) -> list[int]:
        """The cows that just crossed the line; each anchor alerts only once."""
        alerts = []
        for detection in detections:
            track_id = detection.get("track_id")
            track = self._tracks.get(track_id)
            if track is None or track["alerted"]:
                continue
            if detection.get("still_for_s", 0) >= self.still_after_s:
                track["alerted"] = True
                alerts.append(track_id)
        return alerts

    @property
    def tracked(self) -> int:
        return len(self._tracks)
