"""Compare two report paths the way every tool needs before it writes one.

A tool that writes `--out`, `--csv` or `--json` next to inputs it reads must refuse a report that would replace an
input or another report. `same_report_path(first, second)` is True when writing `first` would replace `second`:
the same file, a symlink or hard-link alias of it, or a spelling that differs only in letter case on a volume that
ignores case (Windows, and macOS's default APFS volume). Neither path has to exist; a report that is not written yet
is compared by its resolved name.

Standard library only. Tools import it with a sibling import after putting their own folder on `sys.path` (a
`python -I` run does not do that by itself):

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from report_paths import same_report_path  # noqa: E402
"""
from __future__ import annotations

import os
import sys
import unicodedata

PC_CASE_SENSITIVE = 11  # <sys/unistd.h> on macOS; os.pathconf_names doesn't list it


def case_insensitive_volume(path):
    """True when the volume that will hold `path` ignores letter case, as macOS's default APFS volume does.
    os.path.normcase folds case only on Windows, so on macOS `R.json` and `r.json` would pass as two reports."""
    if sys.platform != "darwin":
        return False
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return os.pathconf(path, PC_CASE_SENSITIVE) == 0
    except (OSError, ValueError):
        return True  # the macOS default; folding can only refuse a pair, never overwrite one


def path_key(path):
    """A comparison key for `path`: the resolved name, with case and accent form folded where the volume does."""
    path = os.path.normcase(os.path.realpath(path))
    if case_insensitive_volume(path):
        # APFS and HFS+ also treat composed and decomposed accents as one name.
        path = unicodedata.normalize("NFD", path.casefold())
    return path


def same_report_path(first, second):
    """True when writing `first` would replace `second` (same file, alias of it, or a case/accent-only difference)."""
    if path_key(first) == path_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False
