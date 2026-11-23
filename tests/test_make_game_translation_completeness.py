"""make_game.py: the shaders stage refuses a translation index that does not match its listings.

rdna2spirv.py drops a shader that is refused or whose module fails spirv-val and still exits 0, so a full run can
carve 26,503 shaders and finish "done" with an index of fewer (measured end to end at aa2988cc: 26,503 carved,
26,473 indexed, exit 0), and an equally-sized index of the wrong names hides the same loss behind a matching
count. verify_translation_completeness() compares the disassembly listing stems with the merged index keys and
fails naming both directions. The expected set is the listings, not the carve: fssm_unpack.py deliberately
tolerates carved containers that never yield a listing, and those must not block a healthy build.

python -I tests/test_make_game_translation_completeness.py
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402


def workdir(listings, index_names, carve_names=None):
    """A disassembly folder holding one .s per listing name, a merged index naming index_names, and (when given)
    a carve index listing carve_names — including containers that never yielded a listing."""
    tmp = tempfile.TemporaryDirectory()
    dxbc = os.path.join(tmp.name, "dxbc")
    spirv = os.path.join(tmp.name, "spirv")
    disasm = os.path.join(dxbc, "disasm")
    os.makedirs(disasm)
    os.makedirs(spirv)
    for name in listings:
        with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write(name + "\n")
    with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({name: {} for name in index_names}, f)
    if carve_names is not None:
        with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"name": n} for n in carve_names], f)
    return tmp, dxbc, spirv


class TranslationCompletenessTests(unittest.TestCase):
    def run_check(self, dxbc, spirv):
        """(exit code, stderr) of verify_translation_completeness, with fail() allowed to exit."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                make_game.verify_translation_completeness(dxbc, spirv)
                code = 0
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def test_index_naming_every_listing_passes_quietly(self):
        tmp, dxbc, spirv = workdir(["s0", "s1", "s2"], ["s0", "s1", "s2"],
                                   carve_names=["s0", "s1", "s2"])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_a_carved_container_without_a_listing_is_not_expected(self):
        # fssm_unpack.py tolerates parse failures and "no XBCH": a carved container that never yields a
        # listing must not be counted against the translation (tupacfan04's case).
        tmp, dxbc, spirv = workdir(["s0"], ["s0"], carve_names=["s0", "container_without_payload"])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_the_right_count_with_the_wrong_names_fails_naming_both(self):
        # shdann's case: the index holds one entry, the disassembly holds one listing, but they disagree.
        tmp, dxbc, spirv = workdir(["s0"], ["unrelated"], carve_names=["s0"])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)
            self.assertIn("s0", err)          # the missing listing is named
            self.assertIn("unrelated", err)   # so is the entry nothing backs
            self.assertIn("1 listing", err)
    def test_missing_modules_fail_with_counts_and_a_capped_name_sample(self):
        # Scaled from the aa2988cc run (26,503 carved, 26,473 indexed): 33 listings, 20 modules, 13 missing,
        # more than the eight-name sample.
        names = [f"s{i}" for i in range(33)]
        tmp, dxbc, spirv = workdir(names, names[:20], carve_names=names)
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)
            self.assertIn("13 listing(s)", err)
            for named in names[20:28]:
                self.assertIn(named + ",", err)
            self.assertIn("… (13 total)", err)
            self.assertNotIn(names[28] + ",", err)   # the sample is capped, not the whole set

    def test_an_empty_index_is_left_to_its_own_guard(self):
        # #3328 refuses a merge that publishes {} and translates a saved one again; comparing it against the
        # listings here would race that fix.
        tmp, dxbc, spirv = workdir(["s0", "s1"], [], carve_names=["s0", "s1"])
        with tmp:
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_an_index_that_is_not_an_object_fails_rather_than_passing(self):
        tmp = tempfile.TemporaryDirectory()
        with tmp:
            dxbc = os.path.join(tmp.name, "dxbc")
            spirv = os.path.join(tmp.name, "spirv")
            os.makedirs(os.path.join(dxbc, "disasm"))
            os.makedirs(spirv)
            with open(os.path.join(dxbc, "disasm", "s0.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write("s0\n")
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump([], f)          # a list where the merged index should be an object
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)

    def test_an_unparseable_index_fails_because_a_resumed_stage_reads_it_nowhere_else(self):
        tmp = tempfile.TemporaryDirectory()
        with tmp:
            dxbc = os.path.join(tmp.name, "dxbc")
            spirv = os.path.join(tmp.name, "spirv")
            os.makedirs(os.path.join(dxbc, "disasm"))
            os.makedirs(spirv)
            with open(os.path.join(dxbc, "disasm", "s0.s"), "w", encoding="utf-8", newline="\n") as f:
                f.write("s0\n")
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                f.write("not json")
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 1)
            self.assertIn("not valid JSON", err)

    def test_a_missing_translation_index_is_left_to_the_paths_that_own_it(self):
        # A translation that did not run, or a merge that failed, already fails the stage before this check
        # (rdna2spirv.py exits nonzero; an empty merge is refused at the merge).
        tmp, dxbc, spirv = workdir(["s0"], [], carve_names=["s0"])
        with tmp:
            os.remove(os.path.join(spirv, "index.json"))
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")

    def test_a_work_folder_without_disassembly_is_an_intermediate_state(self):
        tmp = tempfile.TemporaryDirectory()
        with tmp:
            dxbc = os.path.join(tmp.name, "dxbc")
            spirv = os.path.join(tmp.name, "spirv")
            os.makedirs(dxbc)
            os.makedirs(spirv)
            with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"s0": {}}, f)
            code, err = self.run_check(dxbc, spirv)
            self.assertEqual(code, 0)
            self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
