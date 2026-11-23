"""run_gpu_tests.py: ms_tri_bary (ms_tri drawn through ps_bary) picks its fallback wiring without a GPU.

The case only runs on a device, so a refactor that drops or rewires it would otherwise go unnoticed on machines that
skip it. These checks call mesh_bary_case with the test program stubbed out: the bary fallback without
--mesh-compute must skip with a reason naming LEONIDA_MESH_COMPUTE (a geometry shader can't follow a native mesh
stage), missing fixtures must fail rather than skip, and with both fallbacks the draw must get the generated vertex
shader and ps_bary's pass-through geometry shader."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests

NATIVE_MESH = {"VKH_MESH": "1"}
MESH_COMPUTE = {"VKH_MESH_COMPUTE": "ms_tri.mesh_vs.spv", "VKH_MESH_SLOT": "3088", "VKH_MESH_PRIMS": "64"}


def never_run(*args, **kwargs):
    raise AssertionError("vk_harness must not run in this case")


class MeshBaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spirv = self.tmp.name
        for name in ("ms_tri.spv", "ps_bary.spv", "ps_bary.bary_gs.spv"):
            Path(self.spirv, name).write_bytes(b"\x03\x02\x23\x07")

    def tearDown(self):
        self.tmp.cleanup()

    def case(self, mesh_env, bary_fallback, readback=never_run):
        env = {"LEONIDA_BARY_FALLBACK": "1" if bary_fallback else "0"}
        with patch.dict(os.environ, env), patch.object(run_gpu_tests, "mesh_env", return_value=dict(mesh_env)), \
                patch.object(run_gpu_tests, "run_readback", side_effect=readback):
            return run_gpu_tests.mesh_bary_case("vk_harness", self.spirv, self.spirv, None, None)

    def test_bary_fallback_on_a_native_mesh_stage_skips_with_its_reason(self):
        ok, detail = self.case(NATIVE_MESH, bary_fallback=True)
        self.assertIsNone(ok)
        self.assertTrue(detail.startswith("SKIP"), detail)
        self.assertIn("LEONIDA_MESH_COMPUTE=1", detail)

    def test_missing_pixel_shader_is_a_failure(self):
        os.remove(os.path.join(self.spirv, "ps_bary.spv"))
        self.assertEqual(self.case(MESH_COMPUTE, bary_fallback=True), (False, "missing ps_bary.spv"))

    def test_missing_geometry_shader_is_a_failure(self):
        os.remove(os.path.join(self.spirv, "ps_bary.bary_gs.spv"))
        self.assertEqual(self.case(MESH_COMPUTE, bary_fallback=True), (False, "missing ps_bary.bary_gs.spv"))

    def test_both_fallbacks_draw_through_the_geometry_shader(self):
        seen = {}

        def readback(raw, run_cmd, cmd, env=None):
            seen.update(cmd=cmd, env=env)
            return run_gpu_tests.UNSUPPORTED, "unsupported: stub"

        ok, _ = self.case(MESH_COMPUTE, bary_fallback=True, readback=readback)
        self.assertIsNone(ok)   # the stub reports an unsupported device: a skip, judged after the wiring below
        self.assertEqual(seen["cmd"][1:3], [os.path.join(self.spirv, "ms_tri.spv"),
                                            os.path.join(self.spirv, "ps_bary.spv")])
        self.assertEqual(seen["env"]["VKH_MESH_COMPUTE"], "ms_tri.mesh_vs.spv")
        self.assertEqual(seen["env"]["VKH_GS"], os.path.join(self.spirv, "ps_bary.bary_gs.spv"))
        self.assertEqual(seen["env"]["VKH_GS_HELPER"], "1")

    def test_without_the_bary_fallback_no_geometry_shader_is_added(self):
        seen = {}

        def readback(raw, run_cmd, cmd, env=None):
            seen.update(env=env)
            return run_gpu_tests.UNSUPPORTED, "unsupported: stub"

        self.case(NATIVE_MESH, bary_fallback=False, readback=readback)
        self.assertNotIn("VKH_GS", seen["env"])
        self.assertEqual(seen["env"]["VKH_MESH"], "1")

    def test_ms_tri_brings_ps_bary(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ms_tri.asm"]), ["ms_tri.asm", "ps_bary.asm"])


if __name__ == "__main__":
    unittest.main()
