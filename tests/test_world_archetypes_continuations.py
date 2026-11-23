"""Archive/carved-folder parity through the actual CLIs, using synthetic 0x7a resources only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("archetype_fixtures", ROOT / "tests/test_world_archetypes.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


class CarvedContinuationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0
        self.definitions = F.maptypes([(F.ROCK + i, 2, *F.ROCK_BOX) for i in range(500)])
        self.assertEqual(len(F.WA.parse_archetypes(self.definitions)), 500)
        self.assertFalse(F.WI._needs_continuation(self.definitions))

    def cli(self, tool, *args):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / tool), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def carve(self, parts, gap=0):
        self.serial += 1
        folder = self.root / str(self.serial)
        folder.mkdir()
        archive = folder / "synthetic.rpf"
        if gap:
            data = bytearray(F.rpf(parts[:1]))
            data.extend(b"\0" * ((-len(data)) % 16 + gap))
            data.extend(zlib.compress(parts[1]))
        else:
            data = F.rpf(parts)
        archive.write_bytes(data)
        carved = folder / "carved"
        self.cli("rpf9_carve.py", archive, carved)
        manifest = json.loads((carved / "index.json").read_text(encoding="utf-8"))
        files = [carved / f"{entry['off']:012x}.bin" for entry in manifest["files"]]
        self.assertEqual(len(files), len(parts))
        return archive, carved, files

    def report(self, path, extents=False):
        self.serial += 1
        output = self.root / f"report-{self.serial}.json"
        self.cli("world_archetypes.py", path, "--out", output, *(["--check-extents"] if extents else []))
        doc = json.loads(output.read_text(encoding="utf-8"))
        for key in ("path", "start", "limit"):
            doc["summary"].pop(key)
        for entry in doc["archetypes"]:
            entry.pop("source")
        return doc

    def test_all_500_definitions_survive_one_and_two_continuations(self):
        for cuts in ((65536,), (32768, 65536)):
            with self.subTest(cuts=cuts):
                bounds = (0, *cuts, len(self.definitions))
                parts = [self.definitions[a:b] for a, b in zip(bounds, bounds[1:])]
                archive, carved, _ = self.carve(parts)
                before = self.report(archive)
                self.assertEqual(before["summary"]["archetypes"], 500)
                self.assertEqual(before["summary"]["names"], 500)
                self.assertEqual(len({row["name"] for row in before["archetypes"]}), 500)
                self.assertEqual(self.report(carved), before)

    def test_check_extents_uses_the_entities_and_map_root_after_page_one(self):
        entities = [F.entity(F.ROCK, (0, 0, 0)) for _ in range(499)]
        entities.append(F.entity(F.ROCK, (10, 20, 30), (0, 0, F.S, F.S), sxy=2, sz=2))
        # The rotated, scaled entity and the map root both live beyond byte 65,536. Bounds use the
        # documented conjugated rotation without scaling: its box is [8,17,30]..[11,20,33].
        map_data = F.mapfile(entities, (0, -2, 0), (11, 20, 33))
        self.assertGreater(len(map_data), 65536 + F.ENT)
        archive, carved, _ = self.carve([self.definitions, map_data[:65536], map_data[65536:]])
        before = self.report(archive, extents=True)
        self.assertEqual(before["summary"]["names"], 500)
        self.assertEqual(before["extents"]["maps_fully_defined"], 1)
        self.assertEqual(before["extents"]["rotation"]["conjugate_matches"], 1)
        self.assertEqual(before["extents"]["rotation"]["stored_matches"], 0)
        self.assertEqual(before["extents"]["scale"]["unscaled_matches"], 1)
        self.assertEqual(before["extents"]["scale"]["scaled_matches"], 0)
        self.assertEqual(self.report(carved, extents=True), before)

    def test_complete_bare_file_and_folder_keep_their_definitions(self):
        archive, carved, files = self.carve([self.definitions])
        (carved / "index.json").unlink()
        expected = self.report(archive)
        self.assertEqual(self.report(files[0]), expected)
        self.assertEqual(self.report(carved), expected)

    def test_missing_page_or_index_does_not_invent_a_continuation(self):
        for missing in ("page", "index"):
            with self.subTest(missing=missing):
                _, carved, files = self.carve([self.definitions[:65536], self.definitions[65536:]])
                (files[1] if missing == "page" else carved / "index.json").unlink()
                doc = self.report(carved)
                self.assertEqual(doc["archetypes"], [])
                self.assertEqual(doc["summary"]["names"], 0)

    def test_standalone_first_page_does_not_take_neighboring_pages(self):
        _, _, files = self.carve([self.definitions[:65536], self.definitions[65536:]])
        self.assertEqual(self.report(files[0])["archetypes"], [])

    def test_gap_and_new_resource_stop_the_join(self):
        _, carved, _ = self.carve([self.definitions[:65536], self.definitions[65536:]], gap=16)
        self.assertEqual(self.report(carved)["archetypes"], [])
        independent = F.maptypes([(F.CRATE, 1, *F.CRATE_BOX)])
        _, carved, _ = self.carve([self.definitions[:65536], independent])
        doc = self.report(carved)
        self.assertEqual([row["name"] for row in doc["archetypes"]], [F.CRATE])

    def test_world_index_keeps_its_cap_and_explicit_directory_membership(self):
        _, carved, files = self.carve([self.definitions[:65536], self.definitions[65536:]])
        for listed, cap, size in ((files, len(self.definitions), len(self.definitions)),
                                 (files[:1], len(self.definitions), 65536),
                                 (files, 65536, 65536)):
            with self.subTest(listed=len(listed), cap=cap):
                rows, _ = F.WI.scan_path(str(carved), max_inflate=cap,
                                         directory_files=[os.fspath(path) for path in listed])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["size"], size)


if __name__ == "__main__":
    unittest.main()
