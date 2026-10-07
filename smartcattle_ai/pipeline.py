"""Connect zone rule events to backend publishing."""

import sys

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
    def __init__(self, client, camera_id: str, zone: tuple, throttle):
        self.client = client
        self.camera_id = camera_id
        self.zone = zone
        self.throttle = throttle

    def handle(self, detections, width: int, height: int, now: float) -> list[dict]:
        cattle = [item for item in detections if item["class"] == "cow"]
        _, events = evaluate(cattle, width, height, self.zone)
        sent = []
        for event in events:
            if not self.throttle.allow(self.camera_id, now):
                continue
            payload = to_backend_event(event, self.camera_id)
            try:
                self.client.send_event(payload)
                sent.append(payload)
            except BackendError as exc:
                print(f"Warning: could not send backend event: {exc}", file=sys.stderr)
        return sent
