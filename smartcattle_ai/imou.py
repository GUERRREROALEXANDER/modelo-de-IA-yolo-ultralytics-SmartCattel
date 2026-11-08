"""Imou Open Platform client: gets the camera's live HLS URL from the Imou cloud, so no LAN access is needed."""

import hashlib
import json
import os
import re
import secrets
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_API_URL = "https://openapi.easy4ip.com/openapi"
_STREAM_IDS = {"HD": 0, "SD": 1}
#: A camera opened through the Imou cloud by its serial number, as the backend saves it: imou:04797CEPSF59DA6.
IMOU_PREFIX = "imou:"
_SERIAL = re.compile(r"^[A-Z0-9]{8,32}$")


class ImouError(Exception):
    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class ImouClient:
    """Signed calls to the Imou HTTP API (open.imoulife.com/book/http/develop.html)."""

    def __init__(self, app_id: str, app_secret: str, base_url: str = DEFAULT_API_URL, timeout: float = 10.0,
                 clock=time.time):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._clock = clock
        self._token = None
        self._token_expires_at = 0.0

    def _call(self, method: str, params: dict) -> dict:
        now = round(self._clock())
        nonce = secrets.token_hex(16)
        sign = hashlib.md5(f"time:{now},nonce:{nonce},appSecret:{self.app_secret}".encode("utf-8")).hexdigest()
        body = {
            "system": {"ver": "1.0", "sign": sign, "appId": self.app_id, "time": now, "nonce": nonce},
            "params": params,
            "id": secrets.token_hex(8),
        }
        request = Request(f"{self.base_url}/{method}", data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response).get("result") or {}
        except HTTPError as exc:
            raise ImouError(f"Imou {method}: HTTP {exc.code}") from exc
        except (URLError, OSError) as exc:
            raise ImouError(f"Imou {method}: could not connect ({exc.reason if isinstance(exc, URLError) else exc})") from exc
        except (ValueError, AttributeError) as exc:
            raise ImouError(f"Imou {method}: invalid response") from exc
        code = str(result.get("code", ""))
        if code != "0":
            raise ImouError(f"Imou {method}: {code} {result.get('msg', '')}".strip(), code)
        return result.get("data") or {}

    def _token_call(self, method: str, params: dict) -> dict:
        for attempt in range(2):
            if self._token is None or self._clock() >= self._token_expires_at:
                data = self._call("accessToken", {})
                if "accessToken" not in data:
                    raise ImouError("Imou accessToken: no token in response")
                self._token = data["accessToken"]
                # Renew a minute early; expireTime is the token lifetime in seconds.
                self._token_expires_at = self._clock() + max(float(data.get("expireTime") or 0) - 60, 60)
            try:
                return self._call(method, {**params, "token": self._token})
            except ImouError as exc:
                if exc.code != "TK1002" or attempt:
                    raise
                self._token = None  # Expired or revoked token: fetch a new one and retry once.
        raise AssertionError("unreachable")

    def _find_hls(self, device_id: str, stream_id: int) -> str | None:
        try:
            data = self._token_call("getLiveStreamInfo", {"deviceId": device_id, "channelId": "0"})
        except ImouError as exc:
            if exc.code == "LV1002":  # No live address created yet.
                return None
            raise
        for stream in data.get("streams") or []:
            hls = str(stream.get("hls", ""))
            if stream.get("streamId") == stream_id and hls.startswith("https://") and str(stream.get("status")) == "1":
                return hls
        return None

    def live_hls_url(self, device_id: str, profile: str = "SD") -> str:
        """HLS URL of the device live stream, creating the live address the first time."""
        stream_id = _STREAM_IDS.get(profile.upper())
        if stream_id is None:
            raise ValueError("profile must be HD or SD")
        hls = self._find_hls(device_id, stream_id)
        if hls:
            return hls
        try:
            self._token_call("bindDeviceLive", {"deviceId": device_id, "channelId": "0", "streamId": stream_id})
        except ImouError as exc:
            if exc.code != "LV1001":  # LV1001: the live address already exists.
                raise
        hls = self._find_hls(device_id, stream_id)
        if not hls:
            raise ImouError(f"Imou: no active {profile.upper()} HLS stream for device {device_id} (is it online?)")
        return hls


def imou_configured(env=os.environ) -> bool:
    return bool(env.get("IMOU_APP_ID", "").strip())


class ImouUrlProvider:
    """Callable returning a fresh HLS URL; str() is a log-safe description."""

    def __init__(self, client: ImouClient, device_id: str, profile: str):
        self.client = client
        self.device_id = device_id
        self.profile = profile

    def __call__(self) -> str:
        return self.client.live_hls_url(self.device_id, self.profile)

    def __str__(self) -> str:
        return f"Imou cloud device {self.device_id} ({self.profile})"


def imou_serial(source) -> str | None:
    """The serial number of an imou:<serial> source; None for any other source."""
    if not isinstance(source, str) or not source.startswith(IMOU_PREFIX):
        return None
    serial = source[len(IMOU_PREFIX):]
    return serial if _SERIAL.fullmatch(serial) else None


_shared: dict = {"client": None, "key": None}
_shared_lock = Lock()


class _LockedProvider(ImouUrlProvider):
    """Cameras share one client and its token; they ask for their URL one at a time."""

    def __call__(self) -> str:
        with _shared_lock:
            return super().__call__()


def imou_provider_for(serial: str, env=os.environ) -> ImouUrlProvider:
    """The live URL provider of one camera of the developer account, sharing a client with the others."""
    app_id, app_secret = env.get("IMOU_APP_ID", "").strip(), env.get("IMOU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise ValueError("This server has no Imou account: set IMOU_APP_ID and IMOU_APP_SECRET")
    profile = env.get("IMOU_PROFILE", "").strip().upper() or "SD"
    if profile not in _STREAM_IDS:
        raise ValueError("IMOU_PROFILE must be HD or SD")
    base_url = env.get("IMOU_API_URL", "").strip() or DEFAULT_API_URL
    with _shared_lock:
        key = (app_id, app_secret, base_url)
        if _shared["key"] != key:
            _shared["client"], _shared["key"] = ImouClient(app_id, app_secret, base_url), key
        client = _shared["client"]
    return _LockedProvider(client, serial, profile)


def imou_url_provider(env=os.environ) -> ImouUrlProvider:
    """Provider built from IMOU_APP_ID/APP_SECRET/DEVICE_ID, plus optional IMOU_API_URL and IMOU_PROFILE."""
    missing = [name for name in ("IMOU_APP_ID", "IMOU_APP_SECRET", "IMOU_DEVICE_ID") if not env.get(name, "").strip()]
    if missing:
        raise ValueError("Imou not configured: set " + ", ".join(missing))
    profile = env.get("IMOU_PROFILE", "").strip().upper() or "SD"
    if profile not in _STREAM_IDS:
        raise ValueError("IMOU_PROFILE must be HD or SD")
    client = ImouClient(env["IMOU_APP_ID"].strip(), env["IMOU_APP_SECRET"].strip(),
                        env.get("IMOU_API_URL", "").strip() or DEFAULT_API_URL)
    return ImouUrlProvider(client, env["IMOU_DEVICE_ID"].strip(), profile)
