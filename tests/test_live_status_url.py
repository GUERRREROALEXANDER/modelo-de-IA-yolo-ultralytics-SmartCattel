import json
from threading import Event, Thread
from urllib.request import urlopen

from smartcattle_ai.live import LiveState, make_server
from smartcattle_ai.zones import ZoneSet


def test_status_says_where_other_devices_reach_the_service():
    state = LiveState("camera-01", ZoneSet.from_rectangle((0.1, 0.1, 0.9, 0.9)), "http://192.168.1.5:8090")
    stop = Event()
    server = make_server(state, stop, "127.0.0.1", 0)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/status", timeout=5) as response:
            assert json.load(response)["service_url"] == "http://192.168.1.5:8090"
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
