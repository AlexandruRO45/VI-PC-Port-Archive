"""Repository paths the docs name in backticks exist.

    python -I tests/test_docs_repo_paths.py

`tests/test_docs_links.py` checks Markdown links; this checks the other way the docs point at the tree, a backticked
`tools/...`, `runtime/...`, `tests/...`, `scripts/...`, `docs/...`, `viewer/...` or `game/...` path (TEST-871: until
the last three were added, `docs/audio_config.md` named a `docs/music.md` that was never in the repository).
`third_party/` is left out: its files are a submodule's, which `git ls-files` doesn't list.
A rename, or a page written against an unmerged pull request, leaves such a path pointing at nothing
(docs/saves.md described #279's `xgameruntime_services.cpp` as the runtime's save store while #279 was still
open). Reads tracked files only (`git ls-files`, with a directory walk when git is not available); no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = re.compile(r"`((?:tools|runtime|tests|scripts|docs|viewer|game)/[A-Za-z0-9_./\-]+)`")
# Pages that quote another project's tree (SCARAB, americas-recomp): their paths are not this repository's.
OTHER_PROJECTS = {"docs/fssm_crosscheck.md", "docs/d3d12_xs_surface.md"}
# Build output, not tracked.
BUILD_OUTPUT = ("runtime/build",)
# Skip remote inline links and autolinks, not other code spans on the same line.
URI_PREFIX = r"[A-Za-z][A-Za-z0-9+.-]*://"
REMOTE_MARKUP = re.compile(
    rf"\[[^\]\n]*\]\([ \t]*(?:<{URI_PREFIX}[^>\n]+>|{URI_PREFIX}[^)\s]+)"
    r"""(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*\)"""
    rf"|<{URI_PREFIX}[^>\n]+>"
)


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
            found += [os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/") for f in files]
        return found


def missing_paths(pages, files):
    """[(page:line, path)] for backticked repository paths in `pages` ({name: text}) that `files` lacks."""
    known = set(files)
    for f in files:
        parts = f.split("/")
        known.update("/".join(parts[:i]) for i in range(1, len(parts)))
    out = []
    for page, text in sorted(pages.items()):
        in_fence = False
        for n, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if "://" in line:
                line = REMOTE_MARKUP.sub(" ", line)
            for m in PATH.finditer(line):
                path = m.group(1).rstrip("/.")
                if "*" in path or path.startswith(BUILD_OUTPUT) or path in known:
                    continue
                out.append((f"{page}:{n}", path))
    return out


class DocsRepoPaths(unittest.TestCase):
    def test_checker_reports_only_the_missing_path(self):
        pages = {"docs/a.md": "Run `tools/real.py`, then `tools/gone.py`.\n```\n`tools/in_fence.py`\n```\n"
                              "`runtime/build/x.exe` and `tests/` and [`tools/x.py`](https://example.org/tools/x.py)"}
        self.assertEqual(missing_paths(pages, ["tools/real.py", "tests/t.py"]), [("docs/a.md:1", "tools/gone.py")])

    def test_docs_viewer_and_game_paths_are_checked(self):
        # TEST-871: a missing page, viewer file or package manifest is reported like a missing tool.
        pages = {"docs/b.md": "See `docs/real.md`, `docs/gone.md`, `viewer/index.html`, `viewer/gone.js`, "
                              "`game/xbox-1.0.2.1.json` and `game/gone.json`."}
        files = ["docs/real.md", "viewer/index.html", "game/xbox-1.0.2.1.json"]
        self.assertEqual(missing_paths(pages, files),
                         [("docs/b.md:1", "docs/gone.md"), ("docs/b.md:1", "viewer/gone.js"),
                          ("docs/b.md:1", "game/gone.json")])

    def test_remote_markup_does_not_hide_local_paths(self):
        pages = {"docs/a.md": "\n".join((
            "Run `tools/gone.py`; see [upstream](https://example.invalid/ref).",
            "See [`tools/foreign.py`](https://example.invalid/src); then run `runtime/gone.cpp`.",
            "See https://example.invalid/ref and run `viewer/gone.js`.",
            "See [`docs/gone.md`](./gone.md) and [upstream](https://example.invalid/ref).",
        ))}
        self.assertEqual(missing_paths(pages, []), [
            ("docs/a.md:1", "tools/gone.py"), ("docs/a.md:2", "runtime/gone.cpp"),
            ("docs/a.md:3", "viewer/gone.js"), ("docs/a.md:4", "docs/gone.md"),
        ])

    def test_remote_labels_and_autolinks_stay_exempt(self):
        pages = {"docs/a.md": "\n".join((
            "See [`tools/foreign.py`](https://example.invalid/src).",
            'See [the `tools/foreign.py` script](<https://example.invalid/src> "Upstream").',
            "See <https://example.invalid/`tools/foreign.py`>.",
        ))}
        self.assertEqual(missing_paths(pages, []), [])

    def test_docs_name_existing_paths(self):
        files = tracked_files()
        pages = {}
        for p in files:
            if p.startswith("docs/") and p.endswith(".md") and p not in OTHER_PROJECTS:
                with open(os.path.join(ROOT, p), encoding="utf-8") as f:
                    pages[p] = f.read()
        self.assertTrue(pages)
        missing = missing_paths(pages, files)
        self.assertEqual(missing, [], "backticked paths that name no tracked file: " +
                         "; ".join(f"{where} {path}" for where, path in missing))


if __name__ == "__main__":
    unittest.main()
