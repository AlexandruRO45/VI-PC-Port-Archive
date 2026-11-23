"""Unit tests for gui/core.py (standard library only, no Qt or game data required)."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.core import (  # noqa: E402
    BuildConfig,
    build_make_game_args,
    detect_system_language,
    ensure_output_dir,
    format_validation_check_details,
    load_config,
    pending_install_plan,
    save_config,
    tr,
)


class GuiCoreTest(unittest.TestCase):
    def test_build_args_full_options(self):
        cfg = BuildConfig(
            game_dir="/tmp/game",
            out_dir="/tmp/out",
            work_dir="/tmp/work",
            stages=["check", "shaders"],
            full_check=True,
            wav=True,
            jobs=4,
            wave64_emulate="on",
            runtime_build="/tmp/rt",
            no_float_controls=True,
            bary_fallback=True,
            texel_offset_fallback=True,
            f16_interp_fallback=True,
            mesh_compute=True,
        )
        args = build_make_game_args(cfg, repo_root="/repo", python_exe="/usr/bin/python3")
        self.assertEqual(args[:3], ["/usr/bin/python3", "-I", os.path.join("/repo", "make_game.py")])
        self.assertIn("--game", args)
        self.assertIn("--stages", args)
        self.assertIn("check,shaders", args)
        self.assertIn("--full-check", args)
        self.assertIn("--wav", args)
        self.assertIn("--wave64-emulate", args)
        self.assertIn("on", args)
        self.assertIn("--mesh-compute", args)

    def test_config_roundtrip_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "gui_config.json")
            cfg = BuildConfig(game_dir="  /my/game  ", stages=["invalid"], jobs=0, wave64_emulate="weird", language="xx")
            save_config(cfg, cfg_path)
            loaded = load_config(cfg_path)
            self.assertEqual(loaded.game_dir, "/my/game")
            self.assertEqual(loaded.stages, ["check"])
            self.assertEqual(loaded.jobs, 1)
            self.assertEqual(loaded.wave64_emulate, "auto")
            self.assertEqual(loaded.language, "")

    def test_detect_system_language(self):
        self.assertEqual(detect_system_language({"LANG": "pt_BR.UTF-8"}), "pt-BR")
        self.assertEqual(detect_system_language({"LANG": "en_US.UTF-8"}), "en")
        self.assertEqual(detect_system_language({"LANG": "es_ES.UTF-8"}), "es")
        self.assertEqual(detect_system_language({"LANG": "fr_FR.UTF-8"}), "en")

    def test_pending_install_plan_installs_only_missing_and_deduplicates(self):
        report = {
            "checks": [
                {"id": "python", "status": "error", "required": True, "detail": "missing"},
                {"id": "dotnet", "status": "ok", "required": True, "detail": "8.0.428"},
                {"id": "spirv_as", "status": "error", "required": True, "detail": "missing"},
                {"id": "spirv_val", "status": "error", "required": True, "detail": "missing"},
                {"id": "msvc", "status": "error", "required": True, "detail": "missing"},
                {"id": "ffmpeg", "status": "warn", "required": False, "detail": "optional"},
                {"id": "llvm_mc", "status": "error", "required": True, "detail": "missing"},
            ]
        }
        plan = pending_install_plan(report, platform_name="win32")
        keys = [item.package_key for item in plan]
        self.assertEqual(keys, ["python", "vulkan_sdk", "msvc", "ffmpeg", "llvm_mc"])
        self.assertNotIn("dotnet", keys)
        by_key = {item.package_key: item for item in plan}
        self.assertFalse(by_key["python"].manual_only)
        self.assertIn("winget", by_key["python"].command)
        self.assertFalse(by_key["llvm_mc"].manual_only)
        self.assertTrue(any("install_llvm.ps1" in part for part in by_key["llvm_mc"].command))

    def test_validation_check_details_lists_missing_and_incomplete(self):
        report = {
            "kind": "game_package",
            "details": {
                "issues": [
                    {"path": "common.rpf", "kind": "MISSING", "expected_size": 1048576, "actual_size": None},
                    {"path": "audio.rpf", "kind": "SIZE", "expected_size": 2000, "actual_size": 1000},
                    {"path": "data.rpf", "kind": "CHANGED", "expected_size": 500, "actual_size": 500},
                ]
            },
        }
        chk = {"id": "integrity", "name": "File Integrity", "status": "error", "detail": "3 issue(s) found."}
        text = format_validation_check_details(report, chk, "pt-BR")
        self.assertIn("common.rpf", text)
        self.assertIn("audio.rpf", text)
        self.assertIn("50.0%", text)
        self.assertIn("data.rpf", text)
        self.assertIn("SHA-256 mismatch", text)

    def test_ensure_output_dir_creates_missing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "LeonidaOut")
            self.assertFalse(os.path.isdir(target))
            resolved, ok = ensure_output_dir(target)
            self.assertTrue(ok)
            self.assertTrue(os.path.isdir(resolved))

    def test_translations(self):
        self.assertEqual(tr("pt-BR", "nav_validate"), "Validação")
        self.assertEqual(tr("en", "nav_validate"), "Validation")
        self.assertEqual(tr("es", "nav_validate"), "Validación")


if __name__ == "__main__":
    unittest.main()
