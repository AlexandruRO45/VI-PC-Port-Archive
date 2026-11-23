"""A tool's output path: check it before any work, refuse one that would replace an input or another output, and
write a JSON report. One copy for every tool instead of one per tool.

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I keeps the script folder off sys.path
    from report_output import check_output_path, output_collision, write_json_report

    for flag, path in (("--json", args.json), ("--csv", args.csv)):
        error = path and check_output_path(path, flag)
        if error:
            parser.error(error)                      # or: print(f"ERROR: {error}", file=sys.stderr); return 2
    error = output_collision((("--json", args.json), ("--csv", args.csv)), inputs=[args.input])
    if error:
        parser.error(error)
    ...                                              # the work
    error = write_json_report(args.json, report)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

Each function returns None when all is well, or one message without an "ERROR:" prefix, so the tool decides how to
print it and which exit code to use. `check_output_path` is the check the probes carried as `validate_json_path`
(`f16_trans_sweep.py` and four more), `output_collision` compares through `report_paths.same_report_path` (case,
symlinks and hard links), and `write_json_report` is the probes' writer with a final newline.
`tests/test_tool_output_guards.py` lists the tools that still carry their own copies or no guard at all.

Standard library only; no game data.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402


def check_output_path(path, flag="--out"):
    """Why `path` cannot be an output file (empty, a NUL byte, a folder, a file where a parent folder should be), or None.

    Meant to run before any work, so a bad path costs nothing. Parent folders that don't exist yet are fine:
    `write_json_report` creates them."""
    if not path or not path.strip():
        return f"{flag} path must not be empty"
    if "\x00" in path:
        return f"{flag} path contains a NUL byte"
    if path.endswith(os.sep) or (os.altsep and path.endswith(os.altsep)) or os.path.isdir(path):
        return f"{flag} path is a directory: {path}"
    cur = os.path.dirname(os.path.abspath(path))
    while cur:
        if os.path.isdir(cur):
            break
        if os.path.lexists(cur):
            return f"{flag} parent is not a directory: {cur}"
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def output_collision(outputs, inputs=()):
    """Why the outputs cannot be written: one would replace an input, or two name the same file. None if neither.

    `outputs` is (flag, path) pairs; a pair whose path is empty or None (an option left unset) is skipped. `inputs` is
    the input files and folders the tool reads. The comparison is `same_report_path`: another spelling of the same
    name (letter case on a volume that ignores it, `..`, a symlink) or, for paths that exist, a hard link."""
    outs = [(flag, path) for flag, path in outputs if path]
    srcs = [src for src in inputs if src]
    for i, (flag, path) in enumerate(outs):
        for src in srcs:
            if same_report_path(path, src):
                return f"{flag} {path} would overwrite the input {src}"
        for other_flag, other in outs[:i]:
            if same_report_path(path, other):
                return f"{flag} {path} and {other_flag} {other} name the same file"
    return None


def write_json_report(path, data, flag="--json", indent=1):
    """Write `data` as UTF-8 JSON with LF line endings and a final newline, creating missing parent folders.

    Returns None, or one message when a folder cannot be created or the file cannot be written (an OSError), so a
    failed write is one line, not a traceback. Data that is not JSON-serialisable is a bug in the caller and raises."""
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        return f"cannot create directory for {flag} {path}: {e.strerror or e}"
    text = json.dumps(data, indent=indent) + "\n"
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except OSError as e:
        return f"cannot write {flag} {path}: {e.strerror or e}"
    return None
