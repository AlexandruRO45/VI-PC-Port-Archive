"""No list item in the Markdown docs is repeated a few lines later.

    python -I tests/test_docs_list_items.py

When two branches rewrite the same bullet and a merge keeps both sides, the page shows the bullet twice: the current
text and the one it replaced, often next to each other (docs/d3d12_vulkan.md had its RTV/DSV bullet twice and its
"Hash-table cost" / "Wave64" risks twice each; docs/character_models.md kept the first line of a rewritten bullet).
A reader cannot tell which copy is current. This test flags two list items within WINDOW lines of each other whose
openings match. Reads tracked Markdown only; no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(\S.*)$")
BACKTICK = chr(96)
FENCE_OPENING = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
WINDOW = 8      # lines between the two items
OPENING = 60    # characters of the item's text that must match (after the list marker)


def repeated_items(text):
    """[(line a, line b)] for list items within WINDOW lines whose first OPENING characters are equal."""
    items, fence = [], None
    for n, line in enumerate(text.split("\n"), 1):
        line = line.rstrip("\r")
        if fence is not None:
            marker, length = fence
            closing = rf"^ {{0,3}}{re.escape(marker)}{{{length},}}[ \t]*$"
            if re.fullmatch(closing, line):
                fence = None
            continue
        opening = FENCE_OPENING.match(line)
        if opening:
            marker, info = opening.groups()
            if marker[0] != BACKTICK or BACKTICK not in info:
                fence = (marker[0], len(marker))
                continue
        m = ITEM.match(line)
        if m and len(m.group(2)) >= OPENING:
            items.append((n, len(m.group(1)), m.group(2)[:OPENING]))
    found = []
    for i, (n, indent, head) in enumerate(items):
        for n2, indent2, head2 in items[i + 1:]:
            if n2 - n > WINDOW:
                break
            if indent2 == indent and head2 == head:
                found.append((n, n2))
    return found


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = out.split()
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedListItems(unittest.TestCase):
    def test_checker_flags_a_repeated_item_only(self):
        current = "- **Wave64.** A native wave64 shader needs a 64-wide subgroup in its stage: AMD can require it"
        old = "- **Wave64.** A native wave64 shader needs a 64-wide subgroup in its stage: AMD (old wording)"
        text = "\n".join([current, "  continued", "- **Other.** " + "x" * 70, old, "",
                          "- **19 two-slice 2D arrays** (`depth = 2`, `depth - 1 = 1`) appeared as `depth = 1`",
                          "- **99 three-slice 2D arrays** (`depth = 3`, `depth - 1 = 2`) appeared as `depth = 2`",
                          "```", current, current, "```"])
        self.assertEqual(repeated_items(text), [(1, 4)])

    def test_shorter_backtick_line_does_not_close_longer_fence(self):
        tick = BACKTICK
        repeated = "- **Synthetic repeated item.** " + "x" * OPENING
        text = "\n".join([
            repeated,
            tick * 4 + "markdown",
            tick * 3,
            repeated,
            tick * 3,
            tick * 4,
        ])
        self.assertEqual(repeated_items(text), [])

    def test_tilde_fence_content_is_not_scanned(self):
        repeated = "- **Synthetic repeated item.** " + "x" * OPENING
        text = "\n".join([
            repeated,
            "~~~~markdown",
            repeated,
            "~~~~",
        ])
        self.assertEqual(repeated_items(text), [])

    def test_docs_have_no_repeated_list_items(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                found += [f"{path}:{a} and :{b}" for a, b in repeated_items(f.read())]
        self.assertEqual(found, [], "a list item repeated a few lines later (keep the current copy): " +
                         "; ".join(found))


if __name__ == "__main__":
    unittest.main()
