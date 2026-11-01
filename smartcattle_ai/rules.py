"""Evaluate the safe zones using each box's bottom center: where the animal stands, not its back."""

from datetime import datetime, timezone

from .zones import as_zone_set


def evaluate(detections, width, height, zone) -> tuple[list[dict], list[dict]]:
    """zone is a ZoneSet, or the classic (left, top, right, bottom) rectangle."""
    zones = as_zone_set(zone)
    evaluated = []
    events = []
    for detection in detections:
        x1, _, x2, y2 = detection["bbox"]
        x, y = (x1 + x2) / (2 * width), y2 / height
        item = {**detection, "inside_zone": zones.contains(x, y)}
        evaluated.append(item)
        if not item["inside_zone"]:
            events.append({
                "type": "cattle_outside_zone",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detection": item.copy(),
            })
    return evaluated, events
