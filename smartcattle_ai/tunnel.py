"""Cloudflare quick tunnel: publishes the local video server on a random https://*.trycloudflare.com URL."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from threading import Thread

_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def find_cloudflared() -> str | None:
    """SMARTCATTLE_CLOUDFLARED, cloudflared on PATH, or %LOCALAPPDATA%\\cloudflared\\cloudflared.exe."""
    candidates = [os.getenv("SMARTCATTLE_CLOUDFLARED"), shutil.which("cloudflared")]
    if os.getenv("LOCALAPPDATA"):
        candidates.append(str(Path(os.environ["LOCALAPPDATA"]) / "cloudflared" / "cloudflared.exe"))
    return next((path for path in candidates if path and Path(path).is_file()), None)


def parse_tunnel_url(line: str) -> str | None:
    match = _URL.search(line)
    return match.group(0) if match else None


class QuickTunnel:
    """Runs cloudflared as a child process; the URL changes on every start."""

    def __init__(self, port: int, executable: str):
        self.port = port
        self.executable = executable
        self.url = None
        self._process = None

    def start(self) -> str:
        self._process = subprocess.Popen(
            [self.executable, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
        stderr = self._process.stderr
        assert stderr is not None
        # cloudflared logs the assigned URL on stderr; keep draining it afterwards so the pipe never fills.
        for line in stderr:
            self.url = parse_tunnel_url(line)
            if self.url:
                break
        if not self.url:
            self.stop()
            raise RuntimeError("cloudflared exited without a tunnel URL")
        Thread(target=stderr.read, daemon=True).start()
        return self.url

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
