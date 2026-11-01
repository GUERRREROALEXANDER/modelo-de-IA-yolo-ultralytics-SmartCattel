"""Gate line counter: a cow crossing a drawn line is one event, not one per frame.

The tracker gives every animal a stable number; this module remembers which side
of a line that number was on. A crossing is recorded only when the side flips
and the animal's path actually cuts the segment, so a cow that walks around the
end of the gate is not counted. No new dependency: a cross product and the
proper segment intersection test are enough.

The position used is the bottom center of the box, the same ground point the
safe-zone rule uses, normalized to the frame so the line survives any resolution.
"""

import math


def parse_line(value: str) -> tuple[float, float, float, float]:
    """Parse "x1,y1,x2,y2" normalized coordinates, rejecting sloppy or degenerate lines."""
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError("line must have four numbers: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (float(part) for part in parts)
    except ValueError:
        raise ValueError("line must contain only numbers") from None
    for number in (x1, y1, x2, y2):
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError("line coordinates must be between 0 and 1")
    if (x1, y1) == (x2, y2):
        raise ValueError("line points must be different")
    return (x1, y1, x2, y2)


def _side(a, b, p) -> float:
    """(B-A) x (P-A): positive on one side, negative on the other, zero on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _cuts(a, b, c, d) -> bool:
    """True only when segment CD crosses AB through their interiors, not at an end."""
    d1 = _side(a, b, c)
    d2 = _side(a, b, d)
    d3 = _side(c, d, a)
    d4 = _side(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0


class LineCounter:
    def __init__(self, line: tuple[float, float, float, float], forget_after_s: float = 120.0):
        if forget_after_s <= 0:
            raise ValueError("forget_after_s must be greater than 0")
        x1, y1, x2, y2 = line
        if (x1, y1) == (x2, y2):
            raise ValueError("line points must be different")
        self.line = (float(x1), float(y1), float(x2), float(y2))
        # Seconds a track survives unseen: a cow that leaves comes back with a clean side.
        self.forget_after_s = forget_after_s
        self.entered = 0
        self.exited = 0
        self._tracks: dict[int, dict] = {}

    def update(self, detections: list[dict], width: int, height: int, now: float) -> list[dict]:
        """The crossings produced by this update; detections themselves are left untouched."""
        # Drop stale tracks before their old side can fake a crossing on reappearance.
        for track_id in [track_id for track_id, track in self._tracks.items()
                         if now - track["seen"] > self.forget_after_s]:
            del self._tracks[track_id]

        a = self.line[:2]
        b = self.line[2:]
        crossings = []
        for detection in detections:
            track_id = detection.get("track_id")
            if detection.get("class") != "cow" or track_id is None:
                continue
            x1, y1, x2, y2 = detection["bbox"]
            position = ((x1 + x2) / 2 / width, y2 / height)
            side = _side(a, b, position)
            track = self._tracks.get(track_id)
            if track is None:
                track = {"side": None, "anchor": position, "seen": now}
                self._tracks[track_id] = track
            if side != 0:
                sign = 1 if side > 0 else -1
                previous = track["side"]
                if previous is not None and sign != previous:
                    if _cuts(a, b, track["anchor"], position):
                        direction = "in" if sign > 0 else "out"
                        if direction == "in":
                            self.entered += 1
                        else:
                            self.exited += 1
                        crossings.append({"track_id": track_id, "direction": direction})
                track["side"] = sign
                track["anchor"] = position
            track["seen"] = now
        return crossings

    @property
    def inside_count(self) -> int:
        """entered - exited, relative to when counting started, so it may be negative."""
        return self.entered - self.exited

    def describe(self) -> dict:
        return {"line": list(self.line), "entered": self.entered, "exited": self.exited}
