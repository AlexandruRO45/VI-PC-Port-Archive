"""Opt-in live progress for the long shader steps, for Leonida.exe (launcher-wpf/) to show while a build runs.

Nothing happens unless LEONIDA_PROGRESS_DIR is set: Build.bat and make_game.py run exactly as before. When it is set
(the launcher sets it to a folder of its own, never inside the work folder), a tool writes <dir>/<name>.json with its
counts at most twice a second and once more when it finishes. Each write goes to a temporary file that replaces the
report, so a reader sees a whole report or the previous one; a report the reader holds open is skipped, not waited
for. Only counts and the shader name being worked on are written, never shader contents.
"""
import json
import os
import time

ENV = "LEONIDA_PROGRESS_DIR"
INTERVAL = 0.5   # seconds between writes


class Progress:
    def __init__(self, name: str):
        folder = os.environ.get(ENV)
        self.path = os.path.join(folder, name + ".json") if folder else None
        self.started = time.time()
        self._last = 0.0

    def __bool__(self) -> bool:
        return self.path is not None

    def update(self, force: bool = False, **counts) -> None:
        """Write the report when INTERVAL has passed since the last one, or now with force=True."""
        if not self.path:
            return
        now = time.time()
        if not force and now - self._last < INTERVAL:
            return
        self._last = now
        report = dict(counts, started=self.started, updated=now, pid=os.getpid())
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(report, f)
            os.replace(tmp, self.path)
        except OSError:   # the folder went away, or the reader holds the report: the next write tries again
            try:
                os.remove(tmp)
            except OSError:
                pass
