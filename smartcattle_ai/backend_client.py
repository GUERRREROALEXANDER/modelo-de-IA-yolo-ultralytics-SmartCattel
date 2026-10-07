"""Minimal HTTP client for SmartCattle event ingestion."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BackendError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class BackendClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> tuple[int, dict]:
        headers = {}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
            if self.api_key is not None:
                headers["X-API-Key"] = self.api_key
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.status, json.load(response)
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
