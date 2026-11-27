import os
import signal
import subprocess
from pathlib import Path


class Managed:
    def __init__(self, name, command, log, env=None, cwd=None):
        self.name = name
        self.log_path = Path(log)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            command, stdout=self._log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True
        )

    def alive(self):
        return self.proc.poll() is None

    def check(self):
        if not self.alive():
            raise RuntimeError(f"{self.name} exited with code {self.proc.returncode}\n{self.tail()}")

    def tail(self, lines=15):
        text = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def stop(self, timeout=20):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        self._log.close()
