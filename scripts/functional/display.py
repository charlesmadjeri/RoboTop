import subprocess


def start(size="1280x720x24"):
    try:
        proc = subprocess.Popen(
            ["Xvfb", "-displayfd", "1", "-screen", "0", size, "-nolisten", "tcp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("Xvfb not found, install xvfb")
    number = proc.stdout.readline().strip()
    if not number:
        raise RuntimeError("Xvfb did not start")
    return proc, f":{number}"
