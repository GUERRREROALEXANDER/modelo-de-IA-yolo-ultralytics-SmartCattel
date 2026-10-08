"""Imou cloud client tests against a local fake of the Imou HTTP API."""

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from smartcattle_ai.imou import ImouClient, ImouError, imou_configured, imou_url_provider
from smartcattle_ai.stream import CameraStream

HLS = "https://cmgw.example/live/abc.m3u8"


@pytest.fixture
def imou():
    state = {"calls": [], "bound": False, "expire_token": False, "tokens": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data=None):
            body = json.dumps({"id": "1", "result": {"code": code, "msg": "m", "data": data or {}}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = self.path.rsplit("/", 1)[-1]
            system, params = body["system"], body["params"]
            state["calls"].append((method, params))
            expected = hashlib.md5(
                f"time:{system['time']},nonce:{system['nonce']},appSecret:secret".encode("utf-8")).hexdigest()
            if system["appId"] != "app" or system["sign"] != expected:
                self.reply("SN1001")
            elif method == "accessToken":
                state["tokens"] += 1
                self.reply("0", {"accessToken": f"tok{state['tokens']}", "expireTime": 3600})
            elif state["expire_token"]:
                state["expire_token"] = False
                self.reply("TK1002")
            elif method == "getLiveStreamInfo":
                if not state["bound"]:
                    self.reply("LV1002")
                else:
                    self.reply("0", {"streams": [
                        {"streamId": 0, "status": "1", "hls": "https://cmgw.example/hd.m3u8", "liveToken": "a"},
                        {"streamId": 1, "status": "1", "hls": HLS, "liveToken": "b"}]})
            elif method == "bindDeviceLive":
                state["bound"] = True
                self.reply("0", {"liveToken": "b", "streams": [{"hls": HLS}]})
            else:
                self.reply("OP1011")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}/openapi"
    yield state
    server.shutdown()
    server.server_close()


def test_creates_live_address_once_then_reuses_it(imou):
    client = ImouClient("app", "secret", imou["url"])
    assert client.live_hls_url("DEV1") == HLS
    assert [method for method, _ in imou["calls"]] == [
        "accessToken", "getLiveStreamInfo", "bindDeviceLive", "getLiveStreamInfo"]
    bind_params = imou["calls"][2][1]
    assert bind_params == {"deviceId": "DEV1", "channelId": "0", "streamId": 1, "token": "tok1"}

    imou["calls"].clear()
    assert client.live_hls_url("DEV1", "hd") == "https://cmgw.example/hd.m3u8"
    assert [method for method, _ in imou["calls"]] == ["getLiveStreamInfo"]


def test_expired_token_is_renewed_once(imou):
    imou["bound"] = True
    client = ImouClient("app", "secret", imou["url"])
    client.live_hls_url("DEV1")
    imou["expire_token"] = True
    assert client.live_hls_url("DEV1") == HLS
    assert imou["tokens"] == 2


def test_bad_secret_and_unreachable_api_raise(imou):
    with pytest.raises(ImouError) as error:
        ImouClient("app", "wrong", imou["url"]).live_hls_url("DEV1")
    assert error.value.code == "SN1001"
    with pytest.raises(ImouError, match="could not connect"):
        ImouClient("app", "secret", "http://127.0.0.1:9/openapi", timeout=1).live_hls_url("DEV1")


def test_provider_from_env_and_camera_stream_resolves_it(imou):
    assert not imou_configured({})
    with pytest.raises(ValueError, match="IMOU_APP_SECRET, IMOU_DEVICE_ID"):
        imou_url_provider({"IMOU_APP_ID": "app"})
    provider = imou_url_provider({"IMOU_APP_ID": "app", "IMOU_APP_SECRET": "secret", "IMOU_DEVICE_ID": "DEV1",
                                  "IMOU_API_URL": imou["url"]})
    assert str(provider) == "Imou cloud device DEV1 (SD)"
    assert "secret" not in str(provider)

    opened = []

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            return 0

        def release(self):
            pass

    stream = CameraStream(provider, capture_factory=lambda url, *timeouts: opened.append(url) or Capture())
    assert stream.open()
    assert opened == [HLS]


def test_camera_stream_reports_provider_errors():
    def failing():
        raise ImouError("Imou getLiveStreamInfo: LV1003 device offline", "LV1003")

    stream = CameraStream(failing)
    assert not stream.open()
    assert stream.state == "error"
    assert "device offline" in stream.last_error
