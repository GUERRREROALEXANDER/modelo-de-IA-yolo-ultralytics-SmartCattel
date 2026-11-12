"""What must reach a person: sends the backend the anomalies the detection thread can only notice.

The escape has its own publisher (pipeline.py). These three are about the herd's safety rather than its position.
A cow lying down is not one of them by itself: cattle rest around half the day and most of the night, so it takes far
longer than a normal rest, without moving, before it is worth waking someone.
"""

import sys
from datetime import datetime, timezone

from .backend_client import BackendError

CATTLE_DOWN = "cattle_down"
POSSIBLE_INTRUSION = "possible_intrusion"
EXTERNAL_ANIMAL = "external_animal_detected"

#: A starting point, not a veterinary rule: longer than the rests a healthy cow normally takes in one go.
DOWN_AFTER_MINUTES = 240.0
#: Animals that can scatter or attack a herd. They only appear if SMARTCATTLE_CLASSES includes them.
OUTSIDERS = frozenset({"dog", "bear"})
#: A failed send is tried again this much later, not on every frame, while the backend is down.
RETRY_AFTER_S = 60.0
#: An animal unseen for this long may be reported again if it returns.
FORGET_AFTER_S = 1800.0


class AnomalyPublisher:
    """Reports each anomaly once, and only counts it as reported once the backend has it."""

    def __init__(self, client, camera_id: str, down_after_s: float = DOWN_AFTER_MINUTES * 60):
        self.client = client
        self.camera_id = camera_id
        self.down_after_s = down_after_s
        self._reported: dict[tuple, float] = {}
        self._retry_at: dict[tuple, float] = {}
        self._warned = False

    def handle(self, detections: list[dict], now: float) -> list[dict]:
        sent = []
        for item in detections:
            found = self._anomaly(item)
            if found is None:
                continue
            kind, key = found
            if key in self._reported:
                self._reported[key] = now
            elif now >= self._retry_at.get(key, 0.0) and self._send(kind, item):
                self._reported[key] = now
                self._retry_at.pop(key, None)
                sent.append({"event_type": kind, "track_id": item.get("track_id")})
            elif key not in self._retry_at or now >= self._retry_at[key]:
                self._retry_at[key] = now + RETRY_AFTER_S
        self._forget(now)
        return sent

    def _anomaly(self, item: dict) -> tuple[str, tuple] | None:
        kind = item["class"]
        if kind == "cow":
            still = item.get("still_for_s", 0)
            if self.down_after_s > 0 and item.get("posture") == "lying" and still >= self.down_after_s:
                # By when its stillness began rather than its number, which can change while it rests.
                return CATTLE_DOWN, (CATTLE_DOWN, item.get("still_since"))
            return None
        track_id = item.get("track_id")
        if track_id is None:
            return None
        if kind == "person" and item.get("night"):
            return POSSIBLE_INTRUSION, (POSSIBLE_INTRUSION, track_id)
        if kind in OUTSIDERS:
            return EXTERNAL_ANIMAL, (EXTERNAL_ANIMAL, track_id)
        return None

    def _send(self, kind: str, item: dict) -> bool:
        try:
            self.client.send_event({
                "event_type": kind, "camera_id": self.camera_id, "detected_object": item["class"],
                "confidence": item["confidence"], "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except BackendError as exc:
            if not self._warned:
                print(f"Warning: could not send {kind} from camera {self.camera_id}: {exc}", file=sys.stderr)
            self._warned = True
            return False
        self._warned = False
        return True

    def _forget(self, now: float) -> None:
        for store in (self._reported,):
            for key in [key for key, seen in store.items() if now - seen > FORGET_AFTER_S]:
                del store[key]
        for key in [key for key, at in self._retry_at.items() if now - at > FORGET_AFTER_S]:
            del self._retry_at[key]
