"""Evaluate a normalized safe zone using each box's bottom center."""

from datetime import datetime, timezone


def evaluate(detections, width, height, zone) -> tuple[list[dict], list[dict]]:
    left, top, right, bottom = zone
    evaluated = []
    events = []
    for detection in detections:
        x1, _, x2, y2 = detection["bbox"]
        x, y = (x1 + x2) / (2 * width), y2 / height
        item = {**detection, "inside_zone": left <= x <= right and top <= y <= bottom}
        evaluated.append(item)
        if not item["inside_zone"]:
            events.append({
                "type": "cattle_outside_zone",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "detection": item.copy(),
            })
    return evaluated, events
