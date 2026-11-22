"""run_gpu_tests.py: a filtered run still builds the vertex shader a selected test draws with, and relative --build
/--work paths keep working for test programs that run in another working directory (vk_texture_test)."""
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests
from check import CONFIG_DIRS, TEST_PROGRAMS


class DrawShaders(unittest.TestCase):
    def test_texture_sample_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_texture_sample.asm"]),
                         ["ps_texture_sample.asm", "vs_ids.asm"])

    def test_cube_sample_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_cube_sample.asm"]),
                         ["ps_cube_sample.asm", "vs_ids.asm"])

    def test_flat_still_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_flat.asm"]), ["ps_flat.asm", "vs_ids.asm"])

    def test_each_filtered_clip_case_brings_ps_clip_color(self):
        for name in run_gpu_tests.CLIP_CASES:
            with self.subTest(name=name):
                selected = f"{name}.asm"
                self.assertEqual(run_gpu_tests.with_draw_shaders([selected]), [selected, "ps_clip_color.asm"])

    def test_vs_ids_is_not_added_twice(self):
        files = ["ps_flat.asm", "ps_texture_sample.asm", "vs_ids.asm"]
        self.assertEqual(run_gpu_tests.with_draw_shaders(files), files)

    def test_other_shaders_add_nothing(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_ops.asm"]), ["ps_ops.asm"])

    def test_every_shader_named_exists(self):
        shaders = Path(run_gpu_tests.SHADER_DIR)
        for name, extra in run_gpu_tests.DRAWN_WITH.items():
            # the clip cases (#193) are variants of vs_clip.asm with different signatures, not files of their own
            source = "vs_clip" if name in run_gpu_tests.CLIP_CASES else name
            self.assertTrue((shaders / f"{source}.asm").is_file(), name)
            self.assertTrue((shaders / extra).is_file(), extra)


class FilterSelection(unittest.TestCase):
    def test_reused_work_translates_only_selected_shaders_and_dependencies(self):
        passes = []

        def write_disassembly(_llvm, _asm, path, name):
            Path(path).write_text(name, encoding="utf-8")

        def translate(cmd, gpu=True):
            inputs = Path(cmd[3])
            files = sorted(inputs.glob("*.s"))
            passes.append([path.name for path in files])
            for path in files:
                self.assertEqual(path.read_text(encoding="utf-8"), path.stem)
                self.assertTrue((inputs.parent / f"{path.stem}.dxbc").is_file())
            return 1, "stop before running GPU tests"

        with tempfile.TemporaryDirectory() as tmp, \
                patch("sys.stdout", io.StringIO()), \
                patch.object(run_gpu_tests, "assemble_to_s", side_effect=write_disassembly), \
                patch.object(run_gpu_tests, "run_cmd", side_effect=translate):
            for shader_filter in ("ps_bary", "ps_flat"):
                args = SimpleNamespace(filter=shader_filter, llvm_mc="unused")
                self.assertEqual(run_gpu_tests.run_suite(args, tmp), 1)

            self.assertEqual(passes, [["ps_bary.s"], ["ps_flat.s", "vs_ids.s"]])
            disasm = Path(tmp) / "disasm"
            self.assertEqual((disasm / "ps_bary.s").read_text(encoding="utf-8"), "ps_bary")
            self.assertTrue((disasm / "ps_flat.s").is_file())
            self.assertTrue((disasm / "vs_ids.s").is_file())

    def test_unmatched_filter_fails_before_running_shader_tools(self):
        args = SimpleNamespace(filter="no_matching_shader", build="unused")
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("sys.stderr", err), \
                patch.object(run_gpu_tests, "run_cmd", return_value=(0, "")) as run, \
                patch.object(run_gpu_tests, "spirv_as") as assemble:
            work = Path(tmp) / "work"
            self.assertEqual(run_gpu_tests.run_suite(args, str(work)), 2)
            self.assertIn(args.filter, err.getvalue())
            run.assert_not_called()
            assemble.assert_not_called()
            self.assertFalse(work.exists())


class RelativePaths(unittest.TestCase):
    def test_build_and_work_become_absolute(self):
        ext = ".exe" if os.name == "nt" else ""
        seen = {}

        def suite(a, work):
            seen["build"], seen["work"] = a.build, work
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            for tool in ("vk_compute_test", "vk_store_test", "vk_harness"):
                (build / (tool + ext)).touch()
            old = os.getcwd()
            os.chdir(tmp)
            try:
                with (patch.object(sys, "argv", ["run_gpu_tests.py", "--build", "build", "--work", "work"]),
                      patch.object(run_gpu_tests, "run_suite", side_effect=suite)):
                    self.assertEqual(run_gpu_tests.main(), 0)
            finally:
                os.chdir(old)
            # Compare resolved paths: Windows can spell the temp directory with an 8.3 name (RUNNER~1).
            for got, want in ((seen["build"], build), (seen["work"], Path(tmp) / "work")):
                self.assertTrue(os.path.isabs(got), got)
                self.assertEqual(Path(got).resolve(), want.resolve())


class BuildConfigurations(unittest.TestCase):
    def test_missing_program_error_lists_per_directory_files(self):
        ext = ".exe" if os.name == "nt" else ""
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build output"
            rel = build / "RelWithDebInfo"
            rel.mkdir(parents=True)
            (rel / ("vk_compute_test" + ext)).touch()
            (rel / ("vk_store_test" + ext)).touch()
            debug = build / "Debug"
            debug.mkdir()
            (debug / ("vk_harness" + ext)).touch()
            with (patch.object(sys, "argv", ["run_gpu_tests.py", "--build", str(build)]),
                  self.assertRaises(SystemExit) as caught):
                run_gpu_tests.main()
            message = str(caught.exception)
            lines = message.splitlines()
            self.assertIn(f"  {rel}: vk_harness{ext}", lines)
            self.assertIn(f"  {debug}: vk_compute_test{ext}, vk_store_test{ext}", lines)
            root_missing = ", ".join(tool + ext for tool in TEST_PROGRAMS)
            self.assertIn(f"  {build}: {root_missing}", lines)

    def test_main_passes_visual_studio_config_directory_to_suite(self):
        ext = ".exe" if os.name == "nt" else ""
        for config_name in CONFIG_DIRS:
            with self.subTest(config=config_name), tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                config = build / config_name
                config.mkdir(parents=True)
                for tool in TEST_PROGRAMS:
                    (config / (tool + ext)).touch()
                seen = {}

                def suite(a, work):
                    seen["build"] = a.build
                    return 0

                with (patch.object(sys, "argv", [
                          "run_gpu_tests.py", "--build", str(build), "--work", str(Path(tmp) / "work")]),
                      patch.object(run_gpu_tests, "run_suite", side_effect=suite)):
                    self.assertEqual(run_gpu_tests.main(), 0)
                self.assertEqual(Path(seen["build"]).resolve(), config.resolve())


if __name__ == "__main__":
    unittest.main()
