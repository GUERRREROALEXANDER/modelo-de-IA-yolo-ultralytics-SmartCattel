"""/heatmap.jpg on the live server: served over the latest frame, 404 when off, 503 before a frame."""

import json
from threading import Event, Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from smartcattle_ai.heatmap import HeatMap
from smartcattle_ai.live import LiveState, make_server

ZONE = (0.0, 0.0, 1.0, 1.0)


@pytest.fixture
def serve():
    servers = []

    def start(state):
        server = make_server(state, Event(), "127.0.0.1", 0)
        Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def status_of(url):
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()


def test_the_heat_map_is_painted_over_the_latest_frame(serve):
    state = LiveState("camera-01", ZONE)
    state.heat = HeatMap()
    base = serve(state)
    assert status_of(base + "/heatmap.jpg")[0] == 503

    state.heat.add([{"class": "cow", "bbox": [10, 10, 30, 60]}], 64, 64)
    _, jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    state.publish(jpeg.tobytes(), [], 64, 64, None)

    code, body = status_of(base + "/heatmap.jpg")
    assert code == 200 and body.startswith(b"\xff\xd8")
    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (64, 64, 3) and image.max() > 0
    assert json.loads(status_of(base + "/status")[1])["heatmap_samples"] == 1


def test_the_heat_map_can_be_off(serve):
    base = serve(LiveState("camera-01", ZONE))
    code, body = status_of(base + "/heatmap.jpg")
    assert code == 404 and json.loads(body) == {"detail": "Heat map is off"}
    assert json.loads(status_of(base + "/status")[1])["heatmap_samples"] is None
