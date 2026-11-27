import os
import subprocess

AXIS_KEYS = {
    "azerty": {"z": (0, 1.0), "s": (0, -1.0), "q": (1, 1.0), "d": (1, -1.0), "p": (2, 1.0), "m": (2, -1.0)},
    "qwerty": {"w": (0, 1.0), "s": (0, -1.0), "a": (1, 1.0), "d": (1, -1.0), "p": (2, 1.0), ";": (2, -1.0)},
    "qwertz": {"w": (0, 1.0), "s": (0, -1.0), "a": (1, 1.0), "d": (1, -1.0), "p": (2, 1.0), "ö": (2, -1.0)},
}
LAYOUT_FAMILY = {"fr": "azerty", "be": "azerty", "de": "qwertz", "ch": "qwertz", "at": "qwertz"}
DEFAULT_LAYOUT = "azerty"
SHIFTED_SYMBOLS = {":": ";"}
LAYOUT_QUERIES = (
    (["setxkbmap", "-query"], "layout:"),
    (["localectl", "status"], "x11 layout:"),
)


def key_name(char):
    if not char:
        return None
    return SHIFTED_SYMBOLS.get(char, char).lower()


def is_shifted(char):
    return bool(char) and (char in SHIFTED_SYMBOLS or char.isupper())


def system_layout_name():
    for command, marker in LAYOUT_QUERIES:
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in output.splitlines():
            if line.strip().lower().startswith(marker):
                return line.split(":", 1)[1].split(",")[0].strip().lower()
    return os.environ.get("XKB_DEFAULT_LAYOUT", "").split(",")[0].strip().lower()


def resolve_layout(requested, layout_name=system_layout_name):
    if requested in AXIS_KEYS:
        return requested, "parameter"
    name = layout_name()
    if not name:
        return DEFAULT_LAYOUT, "default, the system layout could not be read"
    return LAYOUT_FAMILY.get(name, "qwerty"), f"detected from the system layout '{name}'"


def describe_keys(layout):
    keys = {axis: [] for axis in range(3)}
    for name, (axis, sign) in AXIS_KEYS[layout].items():
        keys[axis].append((sign, name.upper()))
    return ", ".join(
        f"{'xyz'[axis]}: {'/'.join(n for s, n in sorted(pairs, reverse=True))}" for axis, pairs in keys.items()
    )
