#!/usr/bin/env python3
"""Find what a hand-resolved rebase or merge leaves behind: conflict markers and unclosed CMake blocks.

A rebase that keeps both sides of a conflict is not wrong by itself, but the two shapes that break the build or the
docs are easy to miss and cheap to catch:

* a conflict marker block that was committed as is (``<<<<<<<``, ``=======``, ``>>>>>>>``);
* a CMake block keyword without its ender. Resolving ``CMakeLists.txt`` by concatenating the two sides while the
  conflict sat inside an ``if()`` block leaves the enclosing ``if()`` unclosed, and the file only fails later at
  configure time, on a machine with CMake, from a message that points at the wrong line.

    python -I tools/rebase_check.py                     # the whole checkout
    python -I tools/rebase_check.py --base origin/main  # only the files changed since that revision
    python -I tools/rebase_check.py --json
    python -I tools/rebase_check.py --root <dir> --no-git

Exit code: 0 clean, 1 findings, 2 the checkout could not be read. Standard library only; no game files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

CONFLICT_START = re.compile(r"^<{7}(?:\s|$)")
CONFLICT_MIDDLE = re.compile(r"^={7}$")
CONFLICT_END = re.compile(r"^>{7}(?:\s|$)")

# CMake block keywords and the ender each one needs. `block`/`endblock` came in CMake 3.25 but are harmless to list.
OPENERS = {
    "if": "endif",
    "foreach": "endforeach",
    "function": "endfunction",
    "macro": "endmacro",
    "while": "endwhile",
    "block": "endblock",
}
ENDERS = {end: start for start, end in OPENERS.items()}
KEYWORD = re.compile(r"^\s*(%s)\s*\(" % "|".join(sorted(list(OPENERS) + list(ENDERS))))

CMAKELISTS = "CMakeLists.txt"
SKIP_DIRS = (".git", "__pycache__", "build", "node_modules", ".venv")
BINARY_SNIFF = 8192


def strip_comment(line: str) -> str:
    """Drop a CMake comment from one line.

    `#` starts a comment unless it is inside a quoted argument, which is why the quote count in front of it matters.
    Not handled: bracket arguments ([[ ... ]]), which no CMakeLists.txt in this repository uses for keywords.
    """
    stripped = line.lstrip()
    if stripped.startswith("#"):
        return ""
    quotes = 0
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            quotes += 1
        elif c == "#" and quotes % 2 == 0 and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
        i += 1
    return line


def is_text(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" not in f.read(BINARY_SNIFF)
    except OSError:
        return False


def read_lines(path: str) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read().replace("\r\n", "\n").split("\n")


def find_conflicts(path: str) -> list:
    """Conflict marker blocks in one file, as (line number of the `<<<<<<<`, line number of the `>>>>>>>`)."""
    found = []
    start = None
    middle = False
    for number, line in enumerate(read_lines(path), 1):
        if CONFLICT_START.match(line):
            if start is not None:
                found.append((start, None))
            start, middle = number, False
        elif start is not None and CONFLICT_MIDDLE.match(line):
            middle = True
        elif start is not None and CONFLICT_END.match(line):
            found.append((start, number))
            start, middle = None, False
    if start is not None:
        found.append((start, None))
    return found


def find_cmake_blocks(path: str) -> list:
    """Unclosed openers and orphan enders in one CMake file, as (line number, keyword, problem)."""
    stack = []
    found = []
    for number, raw in enumerate(read_lines(path), 1):
        line = strip_comment(raw)
        m = KEYWORD.match(line)
        if not m:
            continue
        keyword = m.group(1)
        if keyword in OPENERS:
            stack.append((number, keyword))
        elif stack and stack[-1][1] == ENDERS[keyword]:
            stack.pop()
        else:
            wanted = ", ".join("%s(" % k for k in OPENERS if ENDERS[keyword] == k) or "nothing"
            found.append((number, keyword, "no open %s to close" % wanted))
    for number, keyword in stack:
        found.append((number, keyword, "unclosed, needs %s(" % OPENERS[keyword]))
    return found


def git(root: str, *args: str):
    try:
        return subprocess.run(("git",) + args, cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def tracked_files(root: str):
    out = git(root, "ls-files", "-z")
    if out is None:
        return None
    return [p for p in out.split("\0") if p]


def changed_files(root: str, base: str):
    out = git(root, "diff", "--name-only", "--diff-filter=ACMR", "-z", base, "--")
    if out is None:
        return None
    return [p for p in out.split("\0") if p]


def walk_files(root: str) -> list:
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            found.append(os.path.relpath(os.path.join(folder, name), root).replace(os.sep, "/"))
    return found


def is_cmake(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name == CMAKELISTS or name.endswith(".cmake")


def check(root: str, files: list) -> list:
    findings = []
    for rel in files:
        path = os.path.join(root, rel)
        if not os.path.isfile(path) or not is_text(path):
            continue
        for start, end in find_conflicts(path):
            where = "lines %d-%d" % (start, end) if end else "line %d on" % start
            findings.append({"kind": "conflict-marker", "path": rel, "line": start,
                             "detail": "conflict marker %s, no closing >>>>>>>" % where if not end
                             else "conflict marker block %s" % where})
        if is_cmake(rel):
            for number, keyword, problem in find_cmake_blocks(path):
                findings.append({"kind": "cmake-block", "path": rel, "line": number,
                                 "detail": "'%s(' at line %d: %s" % (keyword, number, problem)})
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="rebase_check.py", allow_abbrev=False,
        description="Report leftover conflict markers and unclosed CMake blocks.",
        epilog="Exit code: 0 clean, 1 findings, 2 the checkout could not be read.")
    parser.add_argument("--root", default=".", metavar="DIR", help="checkout to read (default: the current folder)")
    parser.add_argument("--base", metavar="REV", help="only files changed against this revision (default: every tracked file)")
    parser.add_argument("--no-git", action="store_true", help="walk the tree instead of asking git for the file list")
    parser.add_argument("--json", action="store_true", help="write one JSON object instead of text")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print("rebase_check: not a folder: %s" % root, file=sys.stderr)
        return 2

    use_git = not args.no_git
    if use_git:
        files = changed_files(root, args.base) if args.base else tracked_files(root)
        if files is None:
            if args.base:
                print("rebase_check: git could not resolve %r in %s" % (args.base, root), file=sys.stderr)
                return 2
            files = walk_files(root)
    else:
        if args.base:
            print("rebase_check: --base needs git; drop --no-git", file=sys.stderr)
            return 2
        files = walk_files(root)

    findings = check(root, files)

    if args.json:
        print(json.dumps({"root": root, "base": args.base, "checked": len(files), "findings": findings,
                          "clean": not findings}, indent=2))
    else:
        for f in findings:
            print("%s:%d: %s" % (f["path"], f["line"], f["detail"]))
        if findings:
            kinds = {}
            for f in findings:
                kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
            print("rebase_check: %d finding(s) in %d file(s): %s" % (
                len(findings), len({f["path"] for f in findings}),
                ", ".join("%s %d" % (k, n) for k, n in sorted(kinds.items()))))
        else:
            print("rebase_check: clean, %d file(s) checked" % len(files))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
