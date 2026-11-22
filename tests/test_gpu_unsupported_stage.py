"""TEST-18: a device without geometry or mesh shaders (MoltenVK, older GPUs) makes vk_harness exit 77; the
runners report that as SKIP, not as a failure."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "scripts"))
import check
import run_gpu_tests


class MeshCheckOnDeviceWithoutMeshShaders(unittest.TestCase):
    """check.py's ms_w64x_ballot with a fake vk_harness."""

    def run_mesh(self, code, stderr):
        def harness(cmd, **_kw):
            return subprocess.CompletedProcess(cmd, code, "", stderr)

        with tempfile.TemporaryDirectory() as tmp:
            for name in ("ms_w64x_ballot", "fs_passthru"):
                (Path(tmp) / f"{name}.spv").touch()
            with patch.object(check.subprocess, "run", side_effect=harness):
                return check.mesh_w64x_test(tmp, tmp, "ms_w64x_ballot")

    def test_missing_mesh_extension_skips(self):
        problem = self.run_mesh(77, "device: fake (subgroup 32..32)\nunsupported: no VK_EXT_mesh_shader\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("no VK_EXT_mesh_shader", problem)

    def test_other_harness_errors_still_fail(self):
        problem = self.run_mesh(1, "FAILED vkCreateDevice -> -8\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertTrue(problem)


class RunGpuTestsUnsupportedDetail(unittest.TestCase):
    def test_quotes_the_harness_reason(self):
        msg = "device: MoltenVK (subgroup 32..32)\nunsupported: no geometryShader"
        self.assertEqual(run_gpu_tests.unsupported_skip(msg), "SKIP (unsupported: no geometryShader)")

    def test_without_a_reason(self):
        self.assertEqual(run_gpu_tests.unsupported_skip(""), "SKIP (unsupported device)")

    def test_harness_and_runners_agree_on_the_exit_code(self):
        self.assertEqual(run_gpu_tests.UNSUPPORTED, check.UNSUPPORTED)
        harness = (ROOT / "runtime" / "test" / "vk_harness.cpp").read_text()
        for reason in ("unsupported: no VK_EXT_mesh_shader", "unsupported: no geometryShader"):
            self.assertIn(reason, harness)



class PixelTestOnDeviceWithoutAFeature(unittest.TestCase):
    """run_gpu_tests.py's pixel branch with a fake vk_harness that exits 77 (no assembler or GPU needed)."""

    def test_the_skip_is_reported_once(self):
        def fake_run_cmd(cmd, env=None, gpu=True):
            if cmd[0] == sys.executable:                    # the translator: leave the .spv it would write
                Path(cmd[cmd.index("--out") + 1], "ps_bary.spv").touch()
                return 0, ""
            return 77, "device: fake (subgroup 32..32)\nunsupported: no fragmentShaderBarycentric\n"

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter="ps_bary", llvm_mc="llvm-mc", build=tmp)
            # the fake assembler still writes its .s: the runner copies this run's inputs before translating
            fake_assemble = lambda _mc, _src, dst, _name: Path(dst).touch()
            with (patch.object(run_gpu_tests, "assemble_to_s", side_effect=fake_assemble),
                  patch.object(run_gpu_tests, "spirv_as"),
                  patch.object(run_gpu_tests, "run_cmd", side_effect=fake_run_cmd), contextlib.redirect_stdout(out)):
                self.assertEqual(run_gpu_tests.run_suite(args, tmp), 0)
        rows = [line for line in out.getvalue().splitlines() if "ps_bary" in line]
        self.assertEqual(len(rows), 1, out.getvalue())
        self.assertIn("SKIP (unsupported: no fragmentShaderBarycentric)", rows[0])
        self.assertIn("0/1 passed, 1 skipped", out.getvalue())

class GsTriWithoutAGeometryStage(unittest.TestCase):
    """run_gpu_tests.py's gs_tri: VKH_GS where there is a geometry stage, and where there is none (or with
    LEONIDA_GS_COMPUTE=1) the compute capture passes of tools/spirv_gs_to_compute.py (vk_harness VKH_GS_COMPUTE)."""

    def run_gs_tri(self, env, device):
        calls = []

        def fake_run_cmd(cmd, env=None, gpu=True):
            calls.append((cmd, env))
            if cmd[0] == sys.executable and cmd[2].endswith("rdna2spirv.py"):
                Path(cmd[cmd.index("--out") + 1], "gs_tri.spv").touch()
                return 0, ""
            if cmd[0] == sys.executable and cmd[2].endswith("spirv_gs_to_compute.py"):
                Path(cmd[cmd.index("--layout") + 1]).write_text(
                    '{"verts_in": 3, "max_verts": 3, "topology": "triangle_strip", '
                    '"inputs": ["position", "location 1"], "outputs": ["position", "location 1"]}')
                return 0, ""
            return 0, "ok"

        out = io.StringIO()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("LEONIDA_")}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter="gs_tri", llvm_mc="llvm-mc", build=tmp)
            fake_assemble = lambda _mc, _src, dst, _name: Path(dst).touch()
            with (patch.dict(os.environ, {**clean, **env}, clear=True),
                  patch.object(run_gpu_tests.fallbacks, "device", return_value=device),
                  patch.object(run_gpu_tests, "MESH_COMPUTE", False), patch.object(check, "MESH_COMPUTE", False),
                  patch.object(run_gpu_tests, "assemble_to_s", side_effect=fake_assemble),
                  patch.object(run_gpu_tests, "spirv_as"),
                  patch.object(run_gpu_tests, "check_grad", return_value=(True, "4096/4096 pixels exact")),
                  patch.object(run_gpu_tests, "run_cmd", side_effect=fake_run_cmd), contextlib.redirect_stdout(out)):
                self.assertEqual(run_gpu_tests.run_suite(args, tmp), 0, out.getvalue())
        harness = [(cmd, e) for cmd, e in calls if cmd[0].endswith("vk_harness" + (".exe" if os.name == "nt" else ""))]
        self.assertEqual(len(harness), 1, calls)
        return calls, harness[0], out.getvalue()

    def test_no_geometry_stage_runs_the_compute_capture(self):
        calls, (cmd, env), out = self.run_gs_tri({}, {"geometry_shader": False})
        lower = next(c for c, _ in calls if c[0] == sys.executable and c[2].endswith("spirv_gs_to_compute.py"))
        self.assertIn("--vs", lower)
        self.assertIn("--draw-vs", lower)
        self.assertTrue(cmd[1].endswith("gs_tri.vs_cs.spv"))
        self.assertTrue(env["VKH_GS_COMPUTE"].endswith("gs_tri.gs_cs.spv"))
        self.assertTrue(env["VKH_GS_DRAW"].endswith("gs_tri.draw_vs.spv"))
        self.assertEqual(env["VKH_GS_LAYOUT"], "3,3,2,2,2")
        self.assertNotIn("VKH_GS", env)
        self.assertIn("gs compute (no geometryShader; LEONIDA_GS_COMPUTE=0 turns it off)", out)
        self.assertIn("PASS  gs_tri", out)

    def test_a_geometry_stage_draws_with_it(self):
        calls, (cmd, env), _ = self.run_gs_tri({}, {"geometry_shader": True})
        self.assertTrue(env["VKH_GS"].endswith("gs_tri.spv"))
        self.assertTrue(cmd[1].endswith("vs_tri.spv"))
        self.assertFalse(any(c[2].endswith("spirv_gs_to_compute.py") for c, _ in calls if c[0] == sys.executable))

    def test_the_variable_overrides_the_device(self):
        _, (_, env), _ = self.run_gs_tri({"LEONIDA_GS_COMPUTE": "0"}, {"geometry_shader": False})
        self.assertIn("VKH_GS", env)
        _, (_, env), _ = self.run_gs_tri({"LEONIDA_GS_COMPUTE": "1"}, {"geometry_shader": True})
        self.assertIn("VKH_GS_COMPUTE", env)


if __name__ == "__main__":
    unittest.main()
