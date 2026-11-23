"""runtime/test/shaders/build.py under the PerVertexKHR fallbacks (LEONIDA_BARY_FALLBACK=1, LEONIDA_F16_INTERP_FALLBACK=1).

The translator refuses a computed I/J pair (ps_ij_computed) under --bary-fallback (no PerVertexKHR at all).
--f16-interp-fallback alone still translates f32 computed I/J via PerVertexKHR (SHADER-1274). build.py used to
treat a refusal as a failed batch: it exited before publishing anything, so check.py then reported every test
missing. check.py and run_gpu_tests.py report ps_ij_computed as SKIP under --bary-fallback; build.py now leaves
it out there and builds the rest.

Standard library only; the translator is replaced by a stand-in that writes the index and modules and, like the real
one, refuses ps_ij_computed under --bary-fallback. No llvm-mc, SPIR-V tools, GPU or game data.

    python -I tests/test_shader_build_fallbacks.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "runtime", "test", "shaders", "build.py")
FALLBACKS = ("LEONIDA_BARY_FALLBACK", "LEONIDA_F16_INTERP_FALLBACK")


def load_build():
    spec = importlib.util.spec_from_file_location("shaders_build_under_test", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildUnderFallbackTest(unittest.TestCase):
    def setUp(self):
        self.fresh()

    def fresh(self):
        """A new build.py, output folder and record of what the translator was asked for (one per subtest)."""
        self.build = load_build()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "out"
        self.asked = []

    def fake_translate(self, work, names):
        """What rdna2spirv.py leaves in <work>/spirv: a module and an index entry per shader it validated."""
        self.asked.append(list(names))
        bary = os.environ.get("LEONIDA_BARY_FALLBACK") == "1"
        spirv = Path(work) / "spirv"
        spirv.mkdir(parents=True)
        index = {}
        for n in names:
            if n == "ps_ij_computed" and bary:
                continue   # "computed I/J pair under --bary-fallback: no PerVertexKHR inputs"
            (spirv / (n + ".spv")).write_bytes(b"\x03\x02\x23\x07")
            (spirv / (n + ".spvasm")).write_text("; SPIR-V\n", encoding="utf-8")
            index[n] = {"stage": n[:2]}
        (spirv / "index.json").write_text(json.dumps(index), encoding="utf-8")
        return 0

    def run_build(self, names, env):
        stubs = {"spirv_stubs": types.SimpleNamespace(generate_for_index=lambda *a: None),
                 "bary_fallback_gs": types.SimpleNamespace(build_for_index=lambda *a: None)}
        clean = {k: v for k, v in os.environ.items() if k not in FALLBACKS}
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {**clean, **env}, clear=True), \
                mock.patch.dict(sys.modules, stubs), \
                mock.patch.object(self.build, "translate_tests", self.fake_translate), \
                mock.patch.object(sys, "argv", ["build.py", str(self.out), *names]), \
                contextlib.redirect_stdout(stdout):
            try:
                status = self.build.main()
            except SystemExit as e:
                status = e.code
        return status, stdout.getvalue()

    def test_bary_fallback_builds_the_rest_without_ps_ij_computed(self):
        status, printed = self.run_build(["cs_store", "ps_ij_computed"], {"LEONIDA_BARY_FALLBACK": "1"})
        self.assertEqual(status, 0, printed)
        self.assertEqual(self.asked, [["cs_store"]])
        self.assertTrue((self.out / "spirv" / "cs_store.spv").is_file())
        self.assertFalse((self.out / "spirv" / "ps_ij_computed.spv").exists())
        self.assertIn("ps_ij_computed", printed)
        self.assertIn("PerVertexKHR", printed)

    def test_f16_interp_fallback_builds_ps_ij_computed(self):
        # SHADER-1274: f32 computed I/J still translates under --f16-interp-fallback alone.
        status, printed = self.run_build(["cs_store", "ps_ij_computed"], {"LEONIDA_F16_INTERP_FALLBACK": "1"})
        self.assertEqual(status, 0, printed)
        self.assertEqual(self.asked, [["cs_store", "ps_ij_computed"]])
        self.assertTrue((self.out / "spirv" / "ps_ij_computed.spv").is_file())

    def test_the_whole_list_builds_under_f16_interp_fallback(self):
        # Before: "Translation did not produce validated SPIR-V for: ps_ij_computed", and an empty <out>/spirv.
        status, printed = self.run_build([], {"LEONIDA_F16_INTERP_FALLBACK": "1"})
        self.assertEqual(status, 0, printed)
        self.assertIn("ps_ij_computed", self.asked[0])   # SHADER-1274
        published = {p.stem for p in (self.out / "spirv").glob("*.spv")}
        self.assertEqual(published, set(self.asked[0]))
        self.assertIn("ps_interp_f16_high", published)   # the f16 halves still translate under this fallback
        self.assertIn("ps_ij_computed", published)

    def test_without_a_fallback_ps_ij_computed_is_still_built(self):
        status, printed = self.run_build(["cs_store", "ps_ij_computed"], {"LEONIDA_BARY_FALLBACK": "0"})
        self.assertEqual(status, 0, printed)
        self.assertEqual(self.asked, [["cs_store", "ps_ij_computed"]])
        self.assertTrue((self.out / "spirv" / "ps_ij_computed.spv").is_file())

    def test_standalone_barycentric_override_and_probe(self):
        for value, probe, want in (("1", False, True), ("0", True, False),
                                   ("", True, True), ("", False, False)):
            with self.subTest(value=value, probe=probe), \
                    mock.patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": value}, clear=False):
                proc = SimpleNamespace(stdout=json.dumps({"barycentric_fallback": probe}) + "\n")
                with mock.patch.object(self.build.subprocess, "run", return_value=proc):
                    self.assertIs(self.build.barycentric_fallback(), want)

    def test_a_missing_module_without_a_fallback_still_fails_the_batch(self):
        keep = self.fake_translate

        def refuse(work, names):
            keep(work, names)
            os.remove(os.path.join(work, "spirv", "cs_store.spv"))
            return 0
        self.fake_translate = refuse
        try:
            status, _ = self.run_build(["cs_store"], {})
        finally:
            self.fake_translate = keep
        self.assertIn("did not produce validated SPIR-V for: cs_store", str(status))


if __name__ == "__main__":
    unittest.main()
