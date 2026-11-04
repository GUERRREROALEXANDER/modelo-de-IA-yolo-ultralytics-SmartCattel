import io
import json
from threading import Event
from urllib.error import HTTPError, URLError

from pair_farm import main as pair_main
from pair_farm import pair
from smartcattle_ai.backend_client import BackendError
from smartcattle_ai.env import load_env_file
from smartcattle_ai.farm import FarmLink, is_farm_key, update_env_file


class Tunnel:
    def __init__(self, urls, alive=True):
        self.urls = list(urls)
        self._alive = alive

    def alive(self):
        return self._alive

    def start(self):
        if not self.urls:
            raise RuntimeError("no internet")
        self._alive = True
        return self.urls.pop(0)


def test_only_keys_from_linking_count_as_a_farms():
    assert is_farm_key("farm-3-abc")
    assert not is_farm_key("la-clave-compartida") and not is_farm_key(None) and not is_farm_key("")


def test_env_file_keeps_what_was_there(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# nota\nCAMERA_HOST=1.2.3.4\nSMARTCATTLE_API_KEY=vieja\n", encoding="utf-8")
    update_env_file(path, {"SMARTCATTLE_API_KEY": "farm-1-x", "SMARTCATTLE_LIVE_TUNNEL": "1"})
    assert path.read_text(encoding="utf-8") == (
        "# nota\nCAMERA_HOST=1.2.3.4\nSMARTCATTLE_API_KEY=farm-1-x\nSMARTCATTLE_LIVE_TUNNEL=1\n"
    )


def test_linking_writes_everything_the_service_needs(tmp_path):
    env = tmp_path / ".env"
    seen = {}

    def opener(request, timeout):
        seen["url"], seen["body"] = request.full_url, json.loads(request.data)
        return io.BytesIO(json.dumps({"key": "farm-1-x", "origins": ["https://panel.example"]}).encode())

    pair("ABCD2345", "https://api.example/", env, opener)
    assert seen == {"url": "https://api.example/api/farm-pc/paired", "body": {"code": "ABCD2345"}}
    text = env.read_text(encoding="utf-8")
    for line in ("SMARTCATTLE_BACKEND_URL=https://api.example", "SMARTCATTLE_API_KEY=farm-1-x",
                 "SMARTCATTLE_LIVE_ORIGINS=https://panel.example", "SMARTCATTLE_LIVE_TUNNEL=1"):
        assert line in text


def test_a_wrong_code_and_no_internet_say_so_in_plain_words(tmp_path, monkeypatch, capsys):
    env = str(tmp_path / ".env")
    monkeypatch.chdir(tmp_path)

    def refused(*args, **kwargs):
        raise HTTPError("u", 403, "no", None, None)

    monkeypatch.setattr("pair_farm.pair", refused)
    assert pair_main(["MALO", "--backend", "https://api.example", "--env-file", env]) == 1
    assert "codigo" in capsys.readouterr().err

    def offline(*args, **kwargs):
        raise URLError("sin red")

    monkeypatch.setattr("pair_farm.pair", offline)
    assert pair_main(["MALO", "--backend", "https://api.example", "--env-file", env]) == 1
    assert "internet" in capsys.readouterr().err


def test_the_address_is_reported_every_round():
    reported = []
    FarmLink(reported.append, lambda: "https://a.example").tick()
    assert reported == ["https://a.example"]


def test_a_dead_tunnel_comes_back_and_its_new_address_is_used():
    current, reported, moved = ["https://vieja"], [], []

    def on_new_url(url):
        current[0] = url
        moved.append(url)

    FarmLink(reported.append, lambda: current[0], Tunnel(["https://nueva"], alive=False), on_new_url).tick()
    assert moved == ["https://nueva"] and reported == ["https://nueva"]


def test_a_tunnel_that_cannot_start_does_not_report_a_dead_address(capsys):
    reported = []
    FarmLink(reported.append, lambda: "https://vieja", Tunnel([], alive=False)).tick()
    assert reported == []
    assert "tunnel is down" in capsys.readouterr().err


def test_no_internet_does_not_stop_the_loop(capsys):
    def report(url):
        raise OSError("sin red")

    FarmLink(report, lambda: "https://a").tick()
    assert "could not reach" in capsys.readouterr().err


def test_a_backend_that_fails_does_not_stop_the_loop(capsys):
    def report(url):
        raise BackendError("502")

    FarmLink(report, lambda: "https://a").tick()
    assert "could not reach" in capsys.readouterr().err


def test_the_service_reads_back_what_linking_wrote(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    update_env_file(env, {"SMARTCATTLE_BACKEND_URL": "https://api.example", "SMARTCATTLE_API_KEY": "farm-1-x"})
    monkeypatch.delenv("SMARTCATTLE_BACKEND_URL", raising=False)
    monkeypatch.delenv("SMARTCATTLE_API_KEY", raising=False)
    load_env_file(env)
    import os
    assert os.environ["SMARTCATTLE_BACKEND_URL"] == "https://api.example"
    assert os.environ["SMARTCATTLE_API_KEY"] == "farm-1-x"


def test_it_stops_when_asked():
    stop = Event()
    stop.set()
    reported = []
    FarmLink(reported.append, lambda: "https://a").run(stop)
    assert reported == []
