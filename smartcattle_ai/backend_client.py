"""Minimal HTTP client for the SmartCattle backend's AI endpoints."""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class BackendError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def session_is_valid(base_url: str, authorization: str, timeout: float = 5.0) -> bool:
    """Asks the backend whether this Authorization header belongs to a signed-in person."""
    if not authorization.startswith("Bearer "):
        return False
    request = Request(f"{base_url.rstrip('/')}/api/auth/me", headers={"Authorization": authorization})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (HTTPError, URLError, OSError):
        return False


class BackendClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None, method: str | None = None) -> tuple[int, dict]:
        # The key goes on every request: the AI endpoints that only read (zones) need it too.
        headers = {} if self.api_key is None else {"X-API-Key": self.api_key}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                # 204 No Content: "nothing to claim", or an accepted head count.
                return response.status, json.loads(body) if body else {}
        except HTTPError as exc:
            try:
                detail = json.load(exc).get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise BackendError(f"Backend HTTP {exc.code}: {detail or exc.reason}", exc.code) from exc
        except (URLError, OSError) as exc:
            raise BackendError(f"Could not connect to backend: {exc.reason if isinstance(exc, URLError) else exc}") from exc
        except ValueError as exc:
            raise BackendError(f"Invalid backend response: {exc}") from exc

    def health(self) -> bool:
        _, response = self._request("/health")
        return response.get("status") == "ok"

    def send_event(self, event: dict) -> dict:
        status, response = self._request("/api/ai/events", event)
        if status != 201:
            raise BackendError(f"Backend returned HTTP {status}; expected 201", status)
        return response

    def report_camera_status(self, camera_id: str, report: dict) -> dict:
        _, response = self._request(f"/api/ai/cameras/{quote(camera_id, safe='')}/status", report, "PUT")
        return response

    def report_camera_source(self, camera_id: str, source: str, active: bool = True) -> None:
        """Where the camera is reached, for the database. Callers pass it without credentials."""
        self._request(f"/api/ai/cameras/{quote(camera_id, safe='')}/source", {"source": source, "active": active}, "PUT")

    def camera_zones(self, camera_id: str) -> list[dict]:
        """The camera's active safe zones, as drawn in the dashboard."""
        _, response = self._request(f"/api/ai/cameras/{quote(camera_id, safe='')}/zones")
        return response.get("items", [])

    def report_occupancy(self, camera_id: str, report: dict) -> None:
        self._request(f"/api/ai/cameras/{quote(camera_id, safe='')}/occupancy", report, "PUT")

    def claim_video_job(self) -> dict | None:
        """The oldest uploaded video waiting for analysis, or None when the queue is empty."""
        status, response = self._request("/api/ai/video-jobs/claim", {}, "POST")
        return None if status == 204 else response

    def report_job_progress(self, job_id: str, progress: dict) -> None:
        self._request(f"/api/ai/video-jobs/{quote(job_id, safe='')}/progress", progress, "PUT")

    def complete_video_job(self, job_id: str, summary: dict) -> dict:
        _, response = self._request(f"/api/ai/video-jobs/{quote(job_id, safe='')}/complete", summary, "POST")
        return response

    def fail_video_job(self, job_id: str, error: str) -> dict:
        _, response = self._request(f"/api/ai/video-jobs/{quote(job_id, safe='')}/fail",
                                    {"error": safe_error(error)}, "POST")
        return response


def safe_error(text: str) -> str:
    """Error text the backend accepts and shows to people: no URLs (they can carry credentials), 1..300 chars."""
    text = " ".join(str(text).split())
    if "://" in text:
        text = "Video processing failed"
    return (text or "Video processing failed")[:300]


def normalized_box(bbox, width: int, height: int) -> list[float] | None:
    """Pixel [x1, y1, x2, y2] to fractions of the frame, clipped; None when the box has no area left."""
    x1, y1, x2, y2 = (min(max(value / size, 0.0), 1.0)
                      for value, size in zip(bbox, (width, height, width, height)))
    if x1 >= x2 or y1 >= y2:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def backend_detections(detections: list[dict], width: int, height: int, limit: int = 200) -> list[dict]:
    """Boxes in the backend's Detection shape. Cattle only: inside_zone has no meaning for a person."""
    result = []
    for item in detections:
        if item["class"] != "cow" or item.get("inside_zone") is None:
            continue
        box = normalized_box(item["bbox"], width, height)
        if box is None:
            continue
        result.append({"track_id": item.get("track_id"), "label": "cow", "confidence": item["confidence"],
                       "bbox": box, "inside_zone": bool(item["inside_zone"])})
        if len(result) == limit:
            break
    return result


def occupancy_report(detections: list[dict], width: int, height: int, observed_at: str) -> dict:
    """The body of PUT /api/ai/cameras/{id}/occupancy for one evaluated frame."""
    cattle = [item for item in detections if item["class"] == "cow"]
    inside = sum(item.get("inside_zone") is True for item in cattle)
    return {"cattle_inside": inside, "cattle_outside": len(cattle) - inside, "observed_at": observed_at,
            "detections": backend_detections(detections, width, height)}


def to_backend_event(rule_event: dict, camera_id: str) -> dict:
    if rule_event["type"] != "cattle_outside_zone":
        raise ValueError(f"Unsupported rule event: {rule_event['type']}")
    detection = rule_event["detection"]
    return {
        "event_type": "cattle_out_of_zone",
        "camera_id": camera_id,
        "detected_object": detection["class"],
        "confidence": detection["confidence"],
        "timestamp": rule_event["timestamp"],
    }


class EventThrottle:
    def __init__(self, cooldown_seconds: float = 10.0):
        if cooldown_seconds < 0:
            raise ValueError("event cooldown must be nonnegative")
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[str, float] = {}

    def allow(self, key: str, now: float) -> bool:
        last = self._last.get(key)
        if last is not None and now - last < self.cooldown_seconds:
            return False
        self._last[key] = now
        return True
