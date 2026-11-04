"""Limits for a service reachable from the internet: one address must not be able to take it down."""

import time
from threading import Lock

MAX_TRACKED_ADDRESSES = 20_000


def client_address(headers, peer: str, behind_proxy: bool) -> str:
    """Who is asking. Behind a proxy the peer is the proxy, so every visitor would share one limit.

    Cloudflare, in front of Render and of the quick tunnel, overwrites CF-Connecting-IP. X-Forwarded-For can start
    with whatever the client sent, so it is only the fallback. Without a proxy these headers are the client's own
    words and are ignored.
    """
    if behind_proxy:
        address = (headers.get("CF-Connecting-IP") or headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        if address:
            return address[:64]
    return peer


class RateLimit:
    """At most `limit` requests per address in each window. hit() answers the seconds to wait, or 0 if allowed."""

    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = Lock()

    def hit(self, address: str) -> int:
        now = self._clock()
        with self._lock:
            if len(self._hits) >= MAX_TRACKED_ADDRESSES:
                self._hits = {key: value for key, value in self._hits.items() if value[0] > now}
                if len(self._hits) >= MAX_TRACKED_ADDRESSES:
                    self._hits.clear()  # Forgetting counts is better than running out of memory.
            ends, count = self._hits.get(address, (now + self.window_s, 0))
            if ends <= now:
                ends, count = now + self.window_s, 0
            if count >= self.limit:
                return max(1, int(ends - now) + 1)
            self._hits[address] = (ends, count + 1)
            return 0


class StreamSlots:
    """Open video streams, in total and per address. Each one is a thread and a constant flow of paid bandwidth."""

    def __init__(self, total: int, per_address: int):
        self.total = total
        self.per_address = per_address
        self._open: dict[str, int] = {}
        self._lock = Lock()

    def take(self, address: str) -> bool:
        with self._lock:
            if sum(self._open.values()) >= self.total or self._open.get(address, 0) >= self.per_address:
                return False
            self._open[address] = self._open.get(address, 0) + 1
            return True

    def give_back(self, address: str) -> None:
        with self._lock:
            left = self._open.get(address, 0) - 1
            if left > 0:
                self._open[address] = left
            else:
                self._open.pop(address, None)


class SessionCache:
    """Remembers the backend's answer about a token, so a burst of requests is not a burst of calls to the backend."""

    def __init__(self, check, valid_s: float = 120.0, invalid_s: float = 10.0, max_size: int = 2000,
                 clock=time.monotonic):
        self._check = check
        self._valid_s = valid_s
        self._invalid_s = invalid_s
        self._max_size = max_size
        self._clock = clock
        self._answers: dict[str, tuple[float, bool]] = {}
        self._lock = Lock()

    def __call__(self, authorization: str) -> bool:
        now = self._clock()
        with self._lock:
            known = self._answers.get(authorization)
        if known is not None and known[0] > now:
            return known[1]
        valid = self._check(authorization)
        with self._lock:
            if len(self._answers) >= self._max_size:
                self._answers.clear()
            self._answers[authorization] = (now + (self._valid_s if valid else self._invalid_s), valid)
        return valid
