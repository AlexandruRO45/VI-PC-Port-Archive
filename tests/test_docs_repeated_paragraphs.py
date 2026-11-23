"""No paragraph of a Markdown doc appears twice in the same file.

    python -I tests/test_docs_repeated_paragraphs.py

When two branches add the same section and a merge keeps both copies, the page carries the whole block twice, often
dozens of lines apart: docs/design.md has two "PAL stage descriptors" sections, docs/models.md two copies of the
effect-record section. tests/test_docs_repeated_lines.py only compares lines within a few lines of each other, so it
misses these. This test splits each tracked Markdown file into paragraphs (blank-line separated, outside code fences,
whitespace folded) and flags any paragraph of MIN_CHARS or more that occurs twice in one file. Table rows and headings
are left out: tests/test_docs_table_rows.py checks rows, and a heading may repeat under different parents ("### Layout").
Reads tracked Markdown only; no game files.

KNOWN lists the repeats that were on main when this test was added, each with the open pull request that removes it,
so the test fails only on a new one. A pull request that removes a repeat may delete its KNOWN entry; an entry whose
repeat is gone is not an error, so those pull requests merge in any order.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CHARS = 80   # shorter paragraphs ("See below.") may repeat on purpose
OPENING = 50     # characters of a paragraph that identify it in KNOWN
HEADING = re.compile(r"^#{1,6}\s")

# (path, the paragraph's first OPENING characters): repeats on main when the test was added, and who removes them.
KNOWN = {
    # docs/animation.md, the MoVE state-machine section twice: #5755
    ("docs/animation.md", "`tools/anim_move.py` decodes all 324 `MoVE` (versi"),
    ("docs/animation.md", "All 324 resources (325 zlib streams, 331,771 store"),
    ("docs/animation.md", "After the fixed 32-byte header (`\"MoVE\"`, `u32 ver"),
    ("docs/animation.md", "The root state container starts immediately after "),
    ("docs/animation.md", "Every node in the graph starts with `{u16 type; u1"),
    ("docs/animation.md", "- **Outgoing state transitions (1,097 total: 77 on"),
    # docs/design.md, the PAL stage descriptors section twice: #6552
    ("docs/design.md", "The FSSM header starts a table of 0x88-byte hardwa"),
    # docs/gbuffer.md, sections 2 and 3 twice: #5350, #5169
    ("docs/gbuffer.md", "Depending on whether container names are matched c"),
    ("docs/gbuffer.md", "The two six-colour-target rows total **1,702** pix"),
    ("docs/gbuffer.md", "Comparing #1891 against the 5-target (`mrt0..mrt4`"),
    # docs/models.md, the effect-record section twice: #5598, #4855
    ("docs/models.md", "So a material leads to its effect. Through the car"),
    ("docs/models.md", "- None of the 23,619 graphics programs (2,236 VS, "),
    ("docs/models.md", "They are reachable through the **effect records** "),
    ("docs/models.md", "1. **The effect record.** It starts with a one-byt"),
    ("docs/models.md", "The u32 is a size computed from the record's own c"),
    ("docs/models.md", "This holds for all 932 records of this build (910 "),
    ("docs/models.md", "`shaders.rpf` holds **two shader databases** (SGD5"),
    ("docs/models.md", "The two databases add up to the whole carve: 11,51"),
    ("docs/models.md", "On the rim and the tire, list 5's three entries ar"),
    ("docs/models.md", "The group walk and its counts above were reproduce"),
    ("docs/models.md", "- **The u32 after an effect's name.** It is reprod"),
    ("docs/models.md", "Until both are settled, a real model is drawn with"),
    # docs/shader_validation.md, the SHADER-302 packed 16-bit and ps_oracle sections: #6969, #5358
    ("docs/shader_validation.md", "*Measured on an RDNA2 iGPU (gfx1036), Windows 11, "),
    ("docs/shader_validation.md", "380 raw compute kernels covering all 11 packed 16-"),
    ("docs/shader_validation.md", "- **Constant operands under `op_sel` / `op_sel_hi`"),
    ("docs/shader_validation.md", "`tools/hw_oracle` only launches compute kernels, s"),
    ("docs/shader_validation.md", "1. `vk_harness` builds a pipeline from a template "),
    ("docs/shader_validation.md", "The template sets what the code finds when it star"),
    # docs/testing.md, the 128-byte push-constant section twice: #5345
    ("docs/testing.md", "The translated shaders take 144 bytes of push cons"),
    ("docs/testing.md", "To run them anyway, set `LEONIDA_PC_TAIL_UBO=1` **"),
}


def paragraphs(text):
    """[(first line number, paragraph)] outside code fences, each paragraph's lines joined with single spaces."""
    out, lines, start, in_fence = [], [], 0, False
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            if lines:
                out.append((start, " ".join(lines)))
            lines = []
            continue
        s = " ".join(line.split())
        if in_fence or not s:
            if lines:
                out.append((start, " ".join(lines)))
            lines = []
            continue
        if not lines:
            start = n
        lines.append(s)
    if lines:
        out.append((start, " ".join(lines)))
    return out


def repeated_paragraphs(text):
    """[(opening, [line numbers])] for prose paragraphs of MIN_CHARS or more that occur more than once."""
    seen = {}
    for n, p in paragraphs(text):
        if len(p) < MIN_CHARS or p.startswith("|") or HEADING.match(p):
            continue
        seen.setdefault(p, []).append(n)
    return [(p[:OPENING], ns) for p, ns in seen.items() if len(ns) > 1]


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = out.split()
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r.split(os.sep) for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedParagraphs(unittest.TestCase):
    def test_checker_flags_whole_repeated_paragraphs_only(self):
        a = "The disassembly count is unchanged: the same 42 programs have no invalid word, and the rest decode."
        b = "A second paragraph that is long enough to be judged, but only appears a single time in the text."
        wrap = a.index(" ", 50)
        text = "\n".join([
            a, "",                                                   # 1
            b, "",                                                   # 3
            "x " * 50, "",                                           # filler between the two copies
            "  " + a.replace(" ", "  "), "",                         # 7: the same paragraph, other whitespace
            "short", "", "short", "",                                # shorter than MIN_CHARS
            "| " + a + " |", "", "| " + a + " |", "",                # table rows are tests/test_docs_table_rows.py's
            "## " + a, "", "## " + a, "",                            # a heading may repeat
            "```", a, "```",                                         # inside a code fence
            a[:wrap], a[wrap + 1:], "",                              # 24: the same words wrapped: still a repeat
        ])
        self.assertEqual(repeated_paragraphs(text), [(a[:OPENING], [1, 7, 24])])

    def test_a_paragraph_extended_by_a_line_is_not_a_repeat(self):
        a = "The disassembly count is unchanged: the same 42 programs have no invalid word, and the rest decode."
        self.assertEqual(repeated_paragraphs(a + "\n\n" + a + "\nOne more line.\n"), [])

    def test_known_entries_are_well_formed(self):
        for path, opening in KNOWN:
            self.assertTrue(path.endswith(".md"), path)
            self.assertEqual(len(opening), OPENING, (path, opening))

    def test_docs_repeat_no_paragraph(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                text = f.read().replace("\r\n", "\n")
            found += [f"{path}:{','.join(map(str, ns))} ({opening}...)" for opening, ns in repeated_paragraphs(text)
                      if (path, opening) not in KNOWN]
        self.assertEqual(found, [], "a paragraph that is in the file twice (a merge kept both copies; delete one): "
                         + "; ".join(found))


if __name__ == "__main__":
    unittest.main()
