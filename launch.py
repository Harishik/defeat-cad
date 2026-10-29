"""Cross-platform launcher: starts Ollama if installed, builds the web app if needed, serves on :8000.

    python launch.py [--port 8000] [--no-browser]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if not ollama_up():
        exe = shutil.which("ollama")
        if exe:
            print("Starting Ollama for the assistant…")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        else:
            print("Ollama not found — the assistant will use its offline keyword mode. Install from https://ollama.com "
                  "and run:  ollama pull qwen3.5:4b")

    dist = os.path.join(HERE, "web", "dist", "index.html")
    if not os.path.exists(dist):
        npm = shutil.which("npm")
        if not npm:
            sys.exit("web/dist is missing and npm is not installed. Install Node.js 18+ and re-run.")
        subprocess.check_call([npm, "install"], cwd=os.path.join(HERE, "web"))
        subprocess.check_call([npm, "run", "build"], cwd=os.path.join(HERE, "web"))

    url = f"http://{a.host}:{a.port}"
    if not a.no_browser:
        threading.Thread(target=lambda: (time.sleep(2.5), webbrowser.open(url)), daemon=True).start()
    print(f"DeFeat running at {url}  (Ctrl+C to stop)")
    os.chdir(HERE)
    import uvicorn
    uvicorn.run("defeat.server:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
