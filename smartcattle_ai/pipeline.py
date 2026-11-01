"""Connect zone rule events to backend publishing."""

import sys
from datetime import datetime, timezone

from .backend_client import BackendError, to_backend_event
from .rules import evaluate


def parse_zone(value: str) -> tuple[float, float, float, float]:
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("zone must contain four numeric values") from exc
    if (len(parts) != 4 or any(not 0 <= part <= 1 for part in parts)
            or parts[0] >= parts[2] or parts[1] >= parts[3]):
        raise ValueError("zone must be left,top,right,bottom in [0,1] with left<right and top<bottom")
    return parts


class EventPublisher:
    """Sends an escape to the backend.

    Tracked animals (with a track_id) are reported once per escape: when they
    leave the zone, not on every frame they stay outside. They can escape
    again after being seen back inside. Untracked detections fall back to the
    per-camera cooldown alone. The cooldown applies to both, so a whole herd
    walking out at once becomes a trickle of alerts, not a flood.
    """

    #: Seconds an escaped animal is remembered without being seen, before its number is forgotten.
    FORGET_AFTER_S = 120.0

    def __init__(self, client, camera_id: str, zone, throttle):
        self.client = client
        self.camera_id = camera_id
        self.zone = zone
        self.throttle = throttle
        self._escaped: dict[int, float] = {}

    def handle(self, detections, width: int, height: int, now: float) -> list[dict]:
        cattle = [item for item in detections if item["class"] == "cow"]
        if any(item.get("inside_zone") is None for item in cattle):
            cattle, _ = evaluate(cattle, width, height, self.zone)
        self._forget(now)
        sent = []
        for item in cattle:
            track_id = item.get("track_id")
            if item["inside_zone"]:
                self._escaped.pop(track_id, None)
                continue
            if track_id is not None and track_id in self._escaped:
                self._escaped[track_id] = now
                continue
            if not self.throttle.allow(self.camera_id, now):
                continue
            payload = to_backend_event({"type": "cattle_outside_zone",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "detection": item}, self.camera_id)
            try:
                self.client.send_event(payload)
            except BackendError as exc:
                print(f"Warning: could not send backend event: {exc}", file=sys.stderr)
                continue
            sent.append(payload)
            if track_id is not None:
                self._escaped[track_id] = now
        return sent

    def _forget(self, now: float) -> None:
        for track_id in [key for key, seen in self._escaped.items() if now - seen > self.FORGET_AFTER_S]:
            del self._escaped[track_id]
