"""`rpf9dump models --lod N`: option handling and output layout (synthetic), and a LOD 0/1 export (game files).

Set RPF9DUMP_DLL to a built rpf9dump Release DLL to run the option tests. The export test also needs the user's
own package in LEONIDA_GAME_DIR (it reads one 16 MiB window of levels.rpf and writes counts only to a temp
directory); without it, it is skipped.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rpf9dump_exit_codes import texture_dictionary_rpf  # noqa: E402

DLL = os.environ.get("RPF9DUMP_DLL")
GAME = os.environ.get("LEONIDA_GAME_DIR")
# A window of levels.rpf (package 1.0.2.1) where many drawables have a second LOD: measured with this option,
# 16 MiB holds both kinds, so every assertion below has something to check.
WINDOW_START = 68987912192
WINDOW_LENGTH = 16 << 20


def run(*args):
    return subprocess.run(["dotnet", DLL, *map(str, args)], capture_output=True, text=True)


@unittest.skipUnless(DLL, "requires a built rpf9dump (RPF9DUMP_DLL)")
class LodOptionTest(unittest.TestCase):
    def test_lod_out_of_range_or_missing_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "textures.rpf"
            texture_dictionary_rpf(src)
            for bad in (["--lod", "4"], ["--lod", "-1"], ["--lod", "x"], ["--lod"]):
                r = run("models", src, Path(tmp) / "out", *bad)
                self.assertEqual(r.returncode, 2, (bad, r.stdout, r.stderr))
                self.assertIn("--lod needs a LOD index from 0 to 3", r.stdout)

    def test_lod_is_refused_outside_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "textures.rpf"
            texture_dictionary_rpf(src)
            for value in ("0", "1"):   # even the default: --lod names a models option
                r = run("textures", src, Path(tmp) / "out", "--lod", value)
                self.assertEqual(r.returncode, 2, (value, r.stdout, r.stderr))
                self.assertIn("--lod applies to models only", r.stdout)
            self.assertFalse((Path(tmp) / "out" / "textures" / "textures").exists())

    def test_a_lod_export_has_its_own_directory_and_records_the_lod(self):
        # make_game resumes the models stage on models/manifest.json: a LOD export must not write it.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "textures.rpf"
            texture_dictionary_rpf(src)
            out = Path(tmp) / "out"
            r = run("models", src, out, "--lod", "1")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            manifest = json.loads((out / "textures" / "models_lod1" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["Lod"], 1)
            self.assertFalse((out / "textures" / "models").exists())
            r = run("models", src, out)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            manifest = json.loads((out / "textures" / "models" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["Lod"], 0)


@unittest.skipUnless(DLL and GAME, "requires a built rpf9dump (RPF9DUMP_DLL) and the package (LEONIDA_GAME_DIR)")
class LodExportTest(unittest.TestCase):
    def test_lod1_is_a_lighter_model_of_the_same_drawables(self):
        levels = Path(GAME) / "levels.rpf"
        if not levels.is_file():
            self.skipTest("no levels.rpf in LEONIDA_GAME_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            manifests = {}
            for lod in (0, 1):
                r = run("models", levels, tmp, "--start", WINDOW_START, "--length", WINDOW_LENGTH, "--lod", lod)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                sub = "models" if lod == 0 else "models_lod1"
                manifests[lod] = json.loads((Path(tmp) / "levels" / sub / "manifest.json").read_text(encoding="utf-8"))
            first = {(i["Offset"], i["Name"]): i for i in manifests[0]["Items"]}
            second = {(i["Offset"], i["Name"]): i for i in manifests[1]["Items"]}
            unavailable = manifests[1]["ModelDecodeFailures"].get("lod-unavailable", 0)
            self.assertTrue(second and unavailable, "the window should hold drawables with and without a second LOD")
            self.assertLessEqual(set(second), set(first))
            # Every drawable that decodes at LOD 0 either decodes at LOD 1 or is counted as lod-unavailable.
            self.assertEqual(len(second) + unavailable, len(first))
            self.assertTrue(all(second[k]["Tris"] < first[k]["Tris"] for k in second))
            self.assertFalse(any(Path(tmp, "levels", "models_lod1").glob("*_tex")))


if __name__ == "__main__":
    unittest.main()
