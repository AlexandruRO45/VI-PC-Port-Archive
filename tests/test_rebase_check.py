"""`tools/rebase_check.py` finds what a hand-resolved rebase leaves behind.

Two shapes are cheap to commit and expensive to notice. A conflict marker block committed as is lands in a page or a
source file that still parses, so nothing fails until a reader sees it. A CMake block keyword without its ender
comes out of resolving `CMakeLists.txt` by keeping both sides while the conflict sat inside an `if()`: the merged
file builds nowhere, and CMake's message names the wrong line. The second one happened on 2026-11-23 while rebasing
#4348 onto `main` (the dstorage test block was appended after the 44 test-registration lines, so the `endif()` that
closes `if(BUILD_TESTING)` was consumed by the new block), which is why the tool exists.

The fixtures here are small files written to a temporary folder; no game files, nothing tracked is modified.

    python -I tests/test_rebase_check.py
"""
import contextlib
import io
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "tools", "rebase_check.py")
REAL_CMAKELISTS = os.path.join(ROOT, "runtime", "CMakeLists.txt")


def load_module():
    """Import tools/rebase_check.py, whose folder is not a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rebase_check", CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebase_check = load_module()


def write(root, rel, text, newline="\n"):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    return path


def run(root, *args):
    """Run main() over `root`, returning (exit code, text output)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = rebase_check.main(["--root", root, "--no-git"] + list(args))
    return code, out.getvalue()


class ConflictMarkers(unittest.TestCase):
    def test_a_committed_block_is_reported_with_both_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "docs/page.md", "text\n<<<<<<< HEAD\nleft\n=======\nright\n>>>>>>> abc123\n")
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("docs/page.md:2:", out)
            self.assertIn("lines 2-6", out)

    def test_an_unterminated_block_names_the_opening_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "a.txt", "<<<<<<< HEAD\nleft\n=======\nright\n")
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("line 1 on", out)

    def test_a_markdown_underline_is_not_a_conflict(self):
        # `=======` alone underlines a heading in Markdown and closes nothing: only a block that starts with
        # `<<<<<<< ` and then has both other markers is one.
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "docs/page.md", "Title\n=======\n\nBody\n")
            code, out = run(tmp)
            self.assertEqual(code, 0, out)
            self.assertIn("clean", out)

    def test_crlf_line_endings_find_the_same_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "a.txt", "<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> z\n", newline="\r\n")
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("a.txt:1:", out)


class CmakeBlocks(unittest.TestCase):
    BALANCED = """cmake_minimum_required(VERSION 3.20)
project(x)
if(WIN32)
  foreach(name a b)
    message(STATUS "${name}")
  endforeach()
endif()
"""

    def test_a_balanced_file_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "runtime/CMakeLists.txt", self.BALANCED)
            code, out = run(tmp)
            self.assertEqual(code, 0, out)

    def test_the_endif_eaten_by_a_concat_is_reported(self):
        # The shape of the 2026-11-23 rebase: the dstorage block was appended inside `if(BUILD_TESTING)`, and its own
        # `endif()` closed that block instead, leaving `if(BUILD_TESTING)` open.
        swallowed = ("if(BUILD_TESTING)\n"
                     "  if(WIN32 AND TARGET a_test)\n"
                     "    add_test(NAME a COMMAND a_test)\n"
                     "  endif()\n"
                     "  if(WIN32 AND TARGET dstorage_test)\n"
                     "    add_test(NAME dstorage COMMAND dstorage_test)\n"
                     "  endif()\n")
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "runtime/CMakeLists.txt", swallowed)
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("CMakeLists.txt:1:", out)
            self.assertIn("unclosed, needs endif(", out)

    def test_an_orphan_ender_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "runtime/CMakeLists.txt", "message(STATUS hi)\nendif()\n")
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("no open if( to close", out)

    def test_a_keyword_in_a_comment_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "runtime/CMakeLists.txt", "# if(WIN32) is explained below\nmessage(STATUS ok)  # endif()\n")
            code, out = run(tmp)
            self.assertEqual(code, 0, out)

    def test_a_keyword_inside_a_quoted_argument_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "runtime/CMakeLists.txt", 'set(reason "if( without endif(")\n')
            code, out = run(tmp)
            self.assertEqual(code, 0, out)

    def test_a_cmake_module_is_checked_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "cmake/Helper.cmake", "function(helper)\n  message(STATUS x)\n")
            code, out = run(tmp)
            self.assertEqual(code, 1)
            self.assertIn("Helper.cmake:1:", out)


class Output(unittest.TestCase):
    def test_json_lists_the_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "a.txt", "<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> z\n")
            code, out = run(tmp, "--json")
            self.assertEqual(code, 1)
            import json

            payload = json.loads(out)
            self.assertFalse(payload["clean"])
            self.assertEqual(payload["findings"][0]["kind"], "conflict-marker")
            self.assertEqual(payload["findings"][0]["path"], "a.txt")

    def test_a_binary_file_is_skipped_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob.bin")
            with open(path, "wb") as f:
                f.write(b"\x00\x01<<<<<<< HEAD\n")
            code, out = run(tmp)
            self.assertEqual(code, 0, out)

    def test_a_missing_folder_is_a_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = rebase_check.main(["--root", os.path.join(ROOT, "no-such-folder"), "--no-git"])
        self.assertEqual(code, 2)
        self.assertIn("not a folder", err.getvalue())


class TheCheckout(unittest.TestCase):
    """The repository's own files: this is what the tool is for."""

    def test_the_runtime_cmakelists_is_balanced(self):
        self.assertEqual(rebase_check.find_cmake_blocks(REAL_CMAKELISTS), [])

    def test_no_conflict_marker_is_committed(self):
        findings = rebase_check.check(ROOT, ["README.md", "AGENTS.md", "CONTRIBUTING.md",
                                             "runtime/CMakeLists.txt", "docs/design.md"])
        self.assertEqual([f for f in findings if f["kind"] == "conflict-marker"], [])


if __name__ == "__main__":
    unittest.main()
