from __future__ import annotations

from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "log"


class Log:
    def __init__(self, name: str):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.path = str(LOG_DIR / f"{name}.log")
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        for line in msg.rstrip("\n").split("\n"):
            self._file.write(f"[{ts}] {line}\n")
        self._file.flush()

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


_loggers: dict[str, Log] = {}

_stdout_log: Log | None = None


def get_log(name: str) -> Log:
    if name not in _loggers:
        _loggers[name] = Log(name)
    return _loggers[name]


def redirect_stdout_to_log(name: str) -> None:
    import sys

    global _stdout_log
    _stdout_log = get_log(name)
    sys.stdout = _stdout_log  # type: ignore[assignment]
    sys.stderr = _stdout_log


class TeeLog:
    def __init__(self, log: Log, console):
        self.log = log
        self.console = console

    def write(self, msg: str) -> None:
        self.log.write(msg)
        self.console.write(msg)

    def flush(self) -> None:
        self.log.flush()
        self.console.flush()


def tee_stdout(name: str) -> None:
    import sys
    global _stdout_log
    _stdout_log = get_log(name)
    sys.stdout = TeeLog(_stdout_log, sys.stdout)  # type: ignore[assignment]
    sys.stderr = TeeLog(_stdout_log, sys.stderr)
