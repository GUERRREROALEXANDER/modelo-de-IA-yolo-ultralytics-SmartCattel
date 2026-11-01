"""Safe zones as the backend stores them: rectangles, optional polygons, several per camera."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SafeArea:
    """One zone, normalized to the frame (0..1), so it survives a change of resolution."""

    left: float
    top: float
    right: float
    bottom: float
    outline: tuple[tuple[float, float], ...] | None = None
    name: str = ""

    def contains(self, x: float, y: float) -> bool:
        # The rectangle is the polygon's bounding box: a cheap rejection first.
        if not (self.left <= x <= self.right and self.top <= y <= self.bottom):
            return False
        return self.outline is None or _inside_polygon(x, y, self.outline)


def _inside_polygon(x: float, y: float, points) -> bool:
    """Ray casting, counting a point on an edge as inside like the rectangle does."""
    inside = False
    count = len(points)
    for index in range(count):
        (x1, y1), (x2, y2) = points[index], points[(index + 1) % count]
        if _on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _on_segment(x, y, x1, y1, x2, y2, tolerance=1e-9) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return (abs(cross) <= tolerance and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance)


class ZoneSet:
    """Every active zone of a camera. An animal is safe if it stands in any of them."""

    def __init__(self, areas, source: str = "config"):
        self.areas = tuple(areas)
        if not self.areas:
            raise ValueError("a camera needs at least one safe zone")
        self.source = source

    @classmethod
    def from_rectangle(cls, zone, source: str = "config") -> "ZoneSet":
        left, top, right, bottom = zone
        return cls([SafeArea(left, top, right, bottom)], source)

    @classmethod
    def from_backend(cls, items: list[dict]) -> "ZoneSet | None":
        """The body of GET /api/ai/cameras/{id}/zones; None when the camera has no active zone."""
        areas = []
        for item in items:
            if item.get("active") is False:
                continue
            outline = item.get("outline")
            areas.append(SafeArea(
                float(item["x_min"]), float(item["y_min"]), float(item["x_max"]), float(item["y_max"]),
                tuple((float(point["x"]), float(point["y"])) for point in outline) if outline else None,
                str(item.get("name", "")),
            ))
        return cls(areas, "backend") if areas else None

    def contains(self, x: float, y: float) -> bool:
        return any(area.contains(x, y) for area in self.areas)

    def describe(self) -> list[dict]:
        """For the /status endpoint: what the frontend needs to draw the zones."""
        return [{"name": area.name, "rect": [area.left, area.top, area.right, area.bottom],
                 "outline": [list(point) for point in area.outline] if area.outline else None}
                for area in self.areas]

    def __eq__(self, other) -> bool:
        return isinstance(other, ZoneSet) and self.areas == other.areas

    def __hash__(self) -> int:
        return hash(self.areas)


def as_zone_set(zone) -> ZoneSet:
    """Accepts the classic (left, top, right, bottom) tuple wherever a ZoneSet is expected."""
    return zone if isinstance(zone, ZoneSet) else ZoneSet.from_rectangle(zone)
