"""tests/semantic/regen_fixtures.py finds its fixtures by their header line instead of a hand-kept list.

    python -I tests/test_regen_fixtures.py

On throwaway folders: a fixture added by name is picked up afterwards, a hand-written fixture in the same folder is
left alone, --check still reports a stale or a missing fixture, and an unknown name is a usage error. Plus the
repository: the generated set is exactly the fixtures whose first line is the script's header.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import regen_fixtures as rf  # noqa: E402


class RegenFixturesTest(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida regen ")
        self.addCleanup(td.cleanup)
        self.asm = os.path.join(td.name, "shaders")
        self.fix = os.path.join(td.name, "disasm")
        os.makedirs(self.asm)
        os.makedirs(self.fix)
        for p in (mock.patch.object(rf, "ASM_DIR", self.asm), mock.patch.object(rf, "FIXTURE_DIR", self.fix)):
            p.start()
            self.addCleanup(p.stop)
        self.write(os.path.join(self.asm, "cs_a.asm"), "; what cs_a does\nv_mov_b32 v0, 1\ns_endpgm\n")
        self.write(os.path.join(self.asm, "cs_b.asm"), "s_endpgm\n")
        self.write(os.path.join(self.fix, "hand.s"), "; written by hand, not by the script\n\ts_endpgm\n")

    @staticmethod
    def write(path, text):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rf.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_a_named_fixture_is_added_and_then_found_without_a_list(self):
        self.assertEqual(rf.generated_names(self.fix), [])
        self.assertEqual(self.run_main("cs_a")[0], 0)
        self.assertEqual(rf.generated_names(self.fix), ["cs_a"])
        with open(os.path.join(self.fix, "cs_a.s"), encoding="utf-8") as f:
            self.assertEqual(f.read(), rf.render("cs_a"))
        self.assertEqual(self.run_main("--check")[0], 0)

    def test_hand_written_fixtures_are_left_alone(self):
        self.run_main("cs_a", "cs_b")
        code, out, _ = self.run_main()  # rewrite everything it owns
        self.assertEqual(code, 0)
        self.assertNotIn("hand.s", out)
        with open(os.path.join(self.fix, "hand.s"), encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("; written by hand"))

    def test_check_reports_a_stale_fixture(self):
        self.run_main("cs_a")
        self.write(os.path.join(self.asm, "cs_a.asm"), "v_mov_b32 v0, 2\ns_endpgm\n")
        code, out, _ = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("cs_a: fixture is stale or missing", out)

    def test_check_of_a_named_fixture_that_was_never_generated(self):
        code, out, _ = self.run_main("--check", "cs_b")
        self.assertEqual(code, 1)
        self.assertIn("cs_b: fixture is stale or missing", out)

    def test_an_unknown_name_is_a_usage_error(self):
        code, _, err = self.run_main("cs_nope")
        self.assertEqual(code, 2)
        self.assertIn("cs_nope.asm", err)


class RepositoryTest(unittest.TestCase):
    def test_the_generated_set_is_the_fixtures_with_the_header(self):
        names = rf.generated_names()
        self.assertEqual(tuple(names), rf.SHADERS)
        self.assertGreater(len(names), 0)
        for fname in os.listdir(rf.FIXTURE_DIR):
            if fname.endswith(".s"):
                with open(os.path.join(rf.FIXTURE_DIR, fname), encoding="utf-8") as f:
                    self.assertEqual(rf.HEADER_MARK in f.readline(), fname[:-2] in names, fname)


if __name__ == "__main__":
    unittest.main()
