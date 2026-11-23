"""Do two report paths name the same file? One answer for the tools that refuse an output that would overwrite
an input or another output.

    from report_paths import same_report_path        # after sys.path.insert(0, <this folder>), as -I needs

`same_report_path(first, second)` is true when the two paths resolve to one name (`realpath`, then `normcase`, and on a
case-insensitive macOS volume also case-folded and with composed and decomposed accents treated alike) or when both
exist and `os.path.samefile` says they are one file (a hard link, another spelling the name rules missed). Neither
path has to exist: two report paths that don't exist yet still collide when they differ only in case on a volume that
ignores case. Standard library only; no game data.
"""
import os
import posixpath
import sys
import tempfile
import unicodedata

_PC_CASE_SENSITIVE = 11  # <sys/unistd.h> on macOS; os.pathconf_names doesn't list it


def case_insensitive_volume(path):
    """True when the volume that will hold `path` ignores letter case, as macOS's default APFS volume does.
    os.path.normcase folds case only on Windows, so on macOS `R.json` and `r.json` would pass as two reports."""
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if sys.platform == "darwin":
        try:
            return os.pathconf(path, _PC_CASE_SENSITIVE) == 0
        except (OSError, ValueError):
            return True  # the macOS default; folding can only refuse a pair, never overwrite one
    try:
        with tempfile.TemporaryDirectory(prefix=".case-probe-", dir=path) as probe:
            with open(os.path.join(probe, "case_test_a"), "xb"):
                return os.path.exists(os.path.join(probe, "CASE_TEST_A"))
    except OSError:
        return os.name == "nt"


def _resolve_nonexistent(path):
    path = os.path.abspath(path)
    missing = []
    while not os.path.lexists(path):
        parent, name = os.path.split(path)
        if not name or parent == path:
            break
        missing.append(name)
        path = parent
    return os.path.join(os.path.realpath(path), *reversed(missing))


def path_key(path):
    """A name two spellings of one path share: realpath, normcase, and on a case-insensitive volume case-folded NFD."""
    path = _resolve_nonexistent(path)
    # Use the path rules of the platform we are resolving for. On macOS os.path is
    # posixpath and keeps case; on Windows ntpath lowercases. Resolving a simulated
    # macOS path through ntpath would hide the case-sensitive branch under test.
    path = posixpath.normcase(path) if sys.platform == "darwin" else os.path.normcase(path)
    if os.name == "nt":
        extended_prefix = "\\\\?\\"
        unc_prefix = extended_prefix + "unc\\"
        if path.startswith(unc_prefix):
            path = "\\\\" + path[len(unc_prefix):]
        elif path.startswith(extended_prefix) and len(path) >= 6 and path[5] == ":":
            path = path[len(extended_prefix):]
    if case_insensitive_volume(path):
        # APFS and HFS+ also treat composed and decomposed accents as one name.
        path = unicodedata.normalize("NFD", path.casefold())
    return path


def same_report_path(first, second):
    if path_key(first) == path_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False
