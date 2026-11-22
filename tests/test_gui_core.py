"""Unit tests for gui/core.py (standard library only, no Qt or game data required)."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.core import BuildConfig, build_make_game_args, load_config, save_config, tr  # noqa: E402


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
            self.assertEqual(loaded.language, "pt-BR")

    def test_translations(self):
        self.assertEqual(tr("pt-BR", "nav_validate"), "Validação")
        self.assertEqual(tr("en", "nav_validate"), "Validation")


if __name__ == "__main__":
    unittest.main()
