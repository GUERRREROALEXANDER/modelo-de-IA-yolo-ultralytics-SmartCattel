"""What keeps a farm's PC reachable from anywhere: linking it to an account, and telling the backend where it is.

Only the PC at the farm can open the farm's cameras, and nothing on the internet can start a connection to it. So it
opens a tunnel outwards, and reports the tunnel's address, which changes whenever the tunnel restarts.
"""

import sys
from pathlib import Path
from threading import Event, Thread

FARM_KEY_PREFIX = "farm-"
REPORT_EVERY_S = 30.0


def is_farm_key(key: str | None) -> bool:
    return bool(key) and key.startswith(FARM_KEY_PREFIX)


def update_env_file(path, values: dict[str, str]) -> None:
    """Sets these variables in a .env, leaving every other line as it was."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    pending = dict(values)
    result = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            result.append(f"{key}={pending.pop(key)}")
        else:
            result.append(line)
    result.extend(f"{key}={value}" for key, value in pending.items())
    path.write_text("\n".join(result) + "\n", encoding="utf-8")


class FarmLink:
    """Every so often: brings the tunnel back if it died, and tells the backend the current address."""

    def __init__(self, report, current_url, tunnel=None, on_new_url=None, every_s: float = REPORT_EVERY_S):
        self._report = report
        self._current_url = current_url
        self._tunnel = tunnel
        self._on_new_url = on_new_url
        self._every_s = every_s

    def tick(self) -> None:
        if self._tunnel is not None and not self._tunnel.alive():
            try:
                url = self._tunnel.start()
            except (RuntimeError, OSError) as error:
                # No internet, most likely. The next round tries again, and the panel shows this PC as off meanwhile.
                print(f"Warning: tunnel is down: {error}", file=sys.stderr)
                return
            print(f"Tunnel back at {url}", flush=True)
            if self._on_new_url is not None:
                self._on_new_url(url)
        try:
            self._report(self._current_url())
        except OSError as error:  # BackendError is one too.
            print(f"Warning: could not reach the backend: {error}", file=sys.stderr)

    def run(self, stop: Event) -> None:
        while not stop.is_set():
            self.tick()
            stop.wait(self._every_s)

    def start(self, stop: Event) -> Thread:
        worker = Thread(target=self.run, args=(stop,), daemon=True, name="farm-link")
        worker.start()
        return worker
