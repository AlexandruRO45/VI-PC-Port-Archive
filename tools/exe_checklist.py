"""The things to confirm against the game executable: every ASSUMPTION, UNVERIFIED and VERIFY note, in one list.

    python -I tools/exe_checklist.py [--gdk-calls REPORT.json]... [--out checklist.md] [--root DIR]

The runtime and the docs mark what was written without the executable: `// ASSUMPTION ...` or `// ... UNVERIFIED`
comments in runtime/src and runtime/shims (C++), and `**VERIFY:**`, `**ASSUMPTION:**` or `**UNVERIFIED:**` notes in
docs/*.md. This collects them with their file and line, the whole comment block or list item each starts, so the
first look at the executable (scripts/first_look.ps1) has one list to work through.

--gdk-calls takes gdk_runtime_calls.py's JSON for the executable (its --out file or its --json output, one image or
several). A shim note then says whether the executable calls a runtime class that file serves: "called" items are
the ones a boot can reach, so they come first; "not called" ones can wait. The classes of a file are its
`constexpr GUID k...Class` ids, as gdk_runtime_calls reads them.
If any input was skipped, unreadable or cut off, an unobserved class is marked "unknown".

Output: Markdown, to stdout or --out (LF). Exit status 0; 1 when a --gdk-calls file can't be read. Reads files only.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdk_runtime_calls import served_classes  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
SOURCE_DIRS = ("runtime/src", "runtime/shims")
SOURCE_EXTS = (".cpp", ".h", ".hpp", ".inl")
CODE_MARK = re.compile(r"\b(ASSUMPTION|UNVERIFIED)\b")
DOC_MARK = re.compile(r"\*\*(VERIFY|ASSUMPTION|UNVERIFIED)\b[^*]*:?\*\*")


def comment_text(line: str) -> str | None:
    """The text of a whole-line `//` comment, or None (code, or code with a trailing comment)."""
    stripped = line.lstrip()
    return stripped[2:].strip() if stripped.startswith("//") else None


def without_marker(text: str) -> str:
    """The note without its leading marker word ("ASSUMPTION: x" -> "x"), which the list already shows."""
    return re.sub(r"^\**\s*(ASSUMPTION|UNVERIFIED|VERIFY)\b[^:]*:\**\s*", "", text, count=1)


def code_notes(path: str, rel: str) -> list[dict]:
    """Each whole-line comment run holding a marker, as one note from the run's first line to its last. Trailing
    comments (`x = 18;  // UNVERIFIED, see the header`) point at such a run and are not notes of their own."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    notes, i = [], 0
    while i < len(lines):
        if comment_text(lines[i]) is None:
            i += 1
            continue
        start = i
        parts = []
        while i < len(lines) and comment_text(lines[i]) is not None:
            parts.append(comment_text(lines[i]))
            i += 1
        # A run can hold several notes, and a note starts at its marker (a file's header describes the file first):
        # each one runs from its marker to the next marker or the end of the run.
        marks = [k for k, p in enumerate(parts) if CODE_MARK.search(p)]
        for n, at in enumerate(marks):
            stop = marks[n + 1] if n + 1 < len(marks) else len(parts)
            first = parts[at][CODE_MARK.search(parts[at]).start():]
            text = " ".join(p for p in [first] + parts[at + 1:stop] if p)
            notes.append({"file": rel, "line": start + at + 1, "kind": CODE_MARK.search(first).group(1),
                          "text": without_marker(text)})
    return notes


def doc_notes(path: str, rel: str) -> list[dict]:
    """Each paragraph or list item holding a bold marker, from the line it starts on to the next blank line or item."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    notes = []
    for i, line in enumerate(lines):
        m = DOC_MARK.search(line)
        if not m:
            continue
        start = i
        while start > 0 and lines[start].strip() and not lines[start].lstrip().startswith(("- ", "* ", "|")) \
                and lines[start - 1].strip() and not lines[start - 1].lstrip().startswith("#"):
            start -= 1
        end = i + 1
        while end < len(lines) and lines[end].strip() and not lines[end].lstrip().startswith(("- ", "* ", "#", "|")):
            end += 1
        text = " ".join(re.sub(r"^[-*] ", "", l.strip()) for l in lines[start:end])
        notes.append({"file": rel, "line": i + 1, "kind": m.group(1), "text": without_marker(text)})
    return notes


def collect(root: str) -> list[dict]:
    notes = []
    for d in SOURCE_DIRS:
        base = os.path.join(root, d)
        for dirpath, _dirs, files in sorted(os.walk(base)):
            for name in sorted(files):
                if name.endswith(SOURCE_EXTS):
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, root).replace(os.sep, "/")
                    for n in code_notes(path, rel):
                        with open(path, encoding="utf-8", errors="replace") as f:
                            n["classes"] = sorted(served_classes(f.read()))
                        notes.append(n)
    docs = os.path.join(root, "docs")
    if os.path.isdir(docs):
        for name in sorted(os.listdir(docs)):
            if name.endswith(".md"):
                notes += doc_notes(os.path.join(docs, name), f"docs/{name}")
    return notes


def read_calls(paths: list[str]) -> tuple[set[str], bool]:
    """Collect observed class ids and whether every reported input was scanned in full."""
    ids = set()
    complete = True
    for p in paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"{p}: expected a report object")
        unreadable = doc.get("unreadable_images", 0)
        if type(unreadable) is not int or unreadable < 0:
            raise ValueError(f"{p}: expected a nonnegative unreadable_images count")
        if unreadable:
            complete = False
        images = doc.get("images", [doc])
        if not isinstance(images, list):
            raise ValueError(f"{p}: expected an images array")
        if not images:
            raise ValueError(f"{p}: expected at least one image")
        for index, image in enumerate(images):
            location = f"{p}: image {index + 1}"
            if not isinstance(image, dict) or not isinstance(image.get("classes"), list):
                raise ValueError(f"{location}: expected an object with a classes array")
            if image.get("cut_off_sections") or image.get("skipped"):
                complete = False
            for c in image["classes"]:
                if not isinstance(c, dict) or not isinstance(c.get("class_id"), str) or not c["class_id"]:
                    raise ValueError(f"{location}: expected each class to have a nonempty class_id string")
                ids.add(c["class_id"].lower())
    return ids, complete


def called_classes(paths: list[str]) -> set[str]:
    """The class ids observed in any image, including partially scanned images."""
    return read_calls(paths)[0]


def render(notes: list[dict], called: set[str] | None, *, complete: bool = True) -> str:
    def status(n):
        if called is None or not n.get("classes"):
            return None
        return any(c in called for c in n["classes"])

    out = ["# Checklist: what to confirm against the executable", ""]
    out.append(f"{len(notes)} notes from the runtime sources and the docs (tools/exe_checklist.py).")
    if called is not None and not complete:
        out.append("The GDK scan was incomplete. Observed calls come first. Other class-linked notes are marked unknown.")
    elif called is not None:
        out.append("Runtime notes are marked by whether the executable calls a class their file serves "
                   "(gdk_runtime_calls); the called ones come first.")
    out.append("")
    tags = {True: " **called**", False: " (not called)" if complete else " (unknown)", None: ""}
    groups = [("Runtime", [n for n in notes if not n["file"].startswith("docs/")]),
              ("Docs", [n for n in notes if n["file"].startswith("docs/")])]
    for title, group in groups:
        if not group:
            continue
        out.append(f"## {title}")
        out.append("")
        order = {True: 0, None: 1, False: 2}
        for n in sorted(group, key=lambda n: (order[status(n)], n["file"], n["line"])):
            s = status(n)
            tag = tags[s]
            out.append(f"- [ ] `{n['file']}:{n['line']}` {n['kind']}{tag}: {n['text']}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--gdk-calls", action="append", default=[], help="gdk_runtime_calls.py JSON for the executable")
    ap.add_argument("--out", help="write the checklist here instead of stdout")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the repository (default: %(default)s)")
    a = ap.parse_args(argv)
    called = None
    complete = True
    if a.gdk_calls:
        try:
            called, complete = read_calls(a.gdk_calls)
        except (OSError, ValueError, RecursionError) as e:
            print(f"cannot read the gdk_runtime_calls report: {e}", file=sys.stderr)
            return 1
    text = render(collect(a.root), called, complete=complete)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote {a.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
