"""Find cameras on the local network, so the dashboard can offer them instead of asking for an IP."""

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor

# Port -> what usually listens there. 554 is RTSP (Imou/Dahua, Hikvision, most IP cameras); the others are
# the phone apps the README suggests (IP Webcam on 8080, DroidCam on 4747).
CAMERA_PORTS = {554: "rtsp", 8080: "ip-webcam", 4747: "droidcam"}


def local_ipv4() -> str | None:
    """This PC's address on the LAN. Connecting a UDP socket sends nothing; it only picks the route."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def _open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def scan_cameras(own_address: str | None = None, timeout_s: float = 0.35, workers: int = 96,
                 probe=_open) -> dict:
    """Every host of this PC's /24 with a camera port open. Only private networks are scanned."""
    own = own_address or local_ipv4()
    if not own:
        return {"network": None, "cameras": []}
    network = ipaddress.ip_network(f"{own}/24", strict=False)
    if not network.is_private:
        return {"network": str(network), "cameras": []}
    targets = [(str(host), port) for host in network.hosts() if str(host) != own for port in CAMERA_PORTS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda target: probe(target[0], target[1], timeout_s), targets))
    found: dict[str, dict] = {}
    for (host, port), is_open in zip(targets, results):
        if is_open:
            entry = found.setdefault(host, {"host": host, "ports": [], "kinds": []})
            entry["ports"].append(port)
            entry["kinds"].append(CAMERA_PORTS[port])
    cameras = sorted(found.values(), key=lambda item: ipaddress.ip_address(item["host"]))
    return {"network": str(network), "cameras": cameras}
