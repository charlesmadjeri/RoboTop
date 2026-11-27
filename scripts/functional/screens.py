import os
import subprocess
from pathlib import Path


def grab(display, path):
    env = {key: value for key, value in os.environ.items() if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    env["DISPLAY"] = display
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["import", "-window", "root", str(path)], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"screenshot failed, install imagemagick: {done.stderr.strip()}")
