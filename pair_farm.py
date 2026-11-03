"""Links this PC to an account, with the code shown on the Cameras page. Run once, by `Iniciar SmartCattle.bat`."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from smartcattle_ai.env import load_env_file
from smartcattle_ai.farm import update_env_file

DEFAULT_BACKEND = "https://smartcattle-backend.onrender.com"


def pair(code: str, backend: str, env_file: str = ".env", opener=urlopen) -> None:
    request = Request(f"{backend.rstrip('/')}/api/farm-pc/paired", data=json.dumps({"code": code}).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with opener(request, timeout=20) as response:
        answer = json.load(response)
    update_env_file(env_file, {
        "SMARTCATTLE_BACKEND_URL": backend.rstrip("/"),
        "SMARTCATTLE_API_KEY": answer["key"],
        "SMARTCATTLE_LIVE_ORIGINS": ",".join(answer["origins"]),
        "SMARTCATTLE_LIVE_TUNNEL": "1",
    })


def main(argv=None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", help="The code shown on the Cameras page")
    parser.add_argument("--backend", default=os.getenv("SMARTCATTLE_BACKEND_URL") or DEFAULT_BACKEND)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    try:
        pair(args.code, args.backend, args.env_file)
    except HTTPError as error:
        print("Ese codigo no sirve o ya vencio. Genera otro en la pagina de Camaras." if error.code == 403
              else f"El servidor respondio con un error ({error.code}). Intenta de nuevo en un momento.", file=sys.stderr)
        return 1
    except (URLError, OSError):
        print("No se pudo llegar al servidor. Revisa que este PC tenga internet.", file=sys.stderr)
        return 1
    print("Listo: este PC quedo enlazado con tu cuenta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
