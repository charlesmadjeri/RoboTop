import threading

try:
    from keyboard_layout import key_name
except ImportError:
    from teleoperator.keyboard_layout import key_name


class PynputKeys:

    def __init__(self, keymap):
        self._keymap = keymap
        self._lock = threading.Lock()
        self._pressed = set()
        self._boost = False
        self._listener = None

    def start(self):
        from pynput import keyboard

        self._shift_keys = (keyboard.Key.shift_l, keyboard.Key.shift)
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def snapshot(self):
        with self._lock:
            return set(self._pressed), self._boost

    def _on_press(self, key):
        with self._lock:
            if key in self._shift_keys:
                self._boost = True
                return
            name = key_name(getattr(key, "char", None))
            if name in self._keymap:
                self._pressed.add(name)

    def _on_release(self, key):
        with self._lock:
            if key in self._shift_keys:
                self._boost = False
                return
            self._pressed.discard(key_name(getattr(key, "char", None)))
