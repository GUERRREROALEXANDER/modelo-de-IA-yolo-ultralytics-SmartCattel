"""Night watch: an alert only matters after dark, and only once per person.

The detector runs around the clock, but a farmhand crossing the yard at noon is
not news. This module keeps the schedule (a local window that may cross midnight)
and the memory (which track ids have already been reported), so the pipeline can
ask two plain questions: are we inside the window, and who is new? Timezone
conversion is delegated to zoneinfo and the input detections are never touched.
Windows has no time zone database of its own, so requirements.txt adds tzdata.
"""

import re
from collections import OrderedDict
from datetime import datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_WINDOW_PATTERN = re.compile(r"^(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})$")

# Ids kept forever would grow without bound on a long-running camera.
_MAX_ANNOUNCED = 1000


def parse_window(value: str) -> tuple[time, time]:
    """Turn "HH:MM-HH:MM" into a pair of times, rejecting anything ambiguous."""
    match = _WINDOW_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"window must look like HH:MM-HH:MM, got {value!r}")
    start_hour, start_minute, end_hour, end_minute = (int(part) for part in match.groups())
    if start_hour > 23 or end_hour > 23:
        raise ValueError(f"hour must be between 0 and 23, got {value!r}")
    if start_minute > 59 or end_minute > 59:
        raise ValueError(f"minute must be between 0 and 59, got {value!r}")
    start = time(start_hour, start_minute)
    end = time(end_hour, end_minute)
    if start == end:
        raise ValueError(f"window start and end must differ, got {value!r}")
    return start, end


class NightWatch:
    def __init__(self, window: tuple[time, time], timezone: str = "America/Bogota"):
        if not isinstance(window, tuple) or len(window) != 2:
            raise ValueError("window must be a (start, end) pair of times")
        start, end = window
        if not isinstance(start, time) or not isinstance(end, time):
            raise ValueError("window must contain datetime.time values")
        if start == end:
            raise ValueError("window start and end must differ")
        try:
            self.timezone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            raise ValueError(f"unknown timezone: {timezone!r}") from None
        self.window = window
        # Insertion order is the order in which people were announced, so the
        # oldest id is the first one evicted once the cap is reached.
        self._announced: OrderedDict[int, None] = OrderedDict()

    def is_night(self, now: datetime) -> bool:
        """Whether the local time of an aware datetime falls inside the window."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        local = now.astimezone(self.timezone).time()
        start, end = self.window
        if start < end:
            return start <= local < end
        # A window crossing midnight wraps around: it is two intervals in one.
        return local >= start or local < end

    def new_intruders(self, detections: list[dict], now: datetime) -> list[dict]:
        """The tracked people seen for the first time tonight; nobody else."""
        if not self.is_night(now):
            return []
        intruders = []
        for detection in detections:
            if detection.get("class") != "person":
                continue
            track_id = detection.get("track_id")
            if track_id is None or track_id in self._announced:
                continue
            self._remember(track_id)
            intruders.append(detection)
        return intruders

    def _remember(self, track_id: int) -> None:
        self._announced[track_id] = None
        while len(self._announced) > _MAX_ANNOUNCED:
            self._announced.popitem(last=False)
