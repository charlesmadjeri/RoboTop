import os
import re
import select
import sys
import termios
import threading
import time
import tty

try:
    from keyboard_layout import is_shifted, key_name
except ImportError:
    from teleoperator.keyboard_layout import is_shifted, key_name

HOLD_S = 0.12
ESCAPE_SEQUENCE = re.compile(r"\x1b(\[[0-9;]*[A-Za-z~]|O[A-Za-z])")


class TerminalKeys:

    def __init__(self, keymap, hold_s=HOLD_S, clock=time.monotonic, stream=sys.stdin):
        self._keymap = keymap
        self._hold_s = hold_s
        self._clock = clock
        self._stream = stream
        self._lock = threading.Lock()
        self._last_seen = {}
        self._running = False
        self._thread = None
        self._saved_attributes = None

    def feed(self, char):
        name = key_name(char)
        if name in self._keymap:
            with self._lock:
                self._last_seen[name] = (self._clock(), is_shifted(char))

    def feed_text(self, text):
        for char in ESCAPE_SEQUENCE.sub("", text):
            self.feed(char)

    def snapshot(self):
        now = self._clock()
        with self._lock:
            held = {n: shifted for n, (t, shifted) in self._last_seen.items() if now - t < self._hold_s}
        return set(held), any(held.values())

    def start(self):
        if not self._stream.isatty():
            raise RuntimeError("terminal input needs an interactive terminal on stdin")
        fd = self._stream.fileno()
        self._saved_attributes = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, args=(fd,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._saved_attributes:
            termios.tcsetattr(self._stream.fileno(), termios.TCSADRAIN, self._saved_attributes)
            self._saved_attributes = None

    def _read_loop(self, fd):
        while self._running:
            if select.select([fd], [], [], 0.1)[0]:
                self.feed_text(os.read(fd, 64).decode(errors="ignore"))
