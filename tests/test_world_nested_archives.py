"""Nested RPF9 resources reach the world readers through synthetic CLI inputs."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[1]


def fixture(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = fixture("test_world_archetypes")
I = fixture("test_world_interior_names")


class NestedArchivesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.serial = 0

    def cli(self, tool, *args):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / tool), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def archive(self, name, parts):
        path = self.input / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray(0x40)
        data[:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 0x40)
        spans = []
        for part in parts:
            data.extend(b"\0" * (-len(data) % 16))
            compressed = zlib.compress(part)
            spans.append((len(data), len(compressed)))
            data.extend(compressed)
        path.write_bytes(data)
        return path, spans

    def report(self, path, *args):
        self.serial += 1
        output = self.root / f"report-{self.serial}.json"
        self.cli("world_archetypes.py", path, "--out", output, *args)
        doc = json.loads(output.read_text(encoding="utf-8"))
        doc["summary"].pop("path")
        return doc

    def without_sources(self, doc):
        return {**doc, "archetypes": [{k: v for k, v in row.items() if k != "source"}
                                     for row in doc["archetypes"]]}

    def test_nested_archive_matches_direct_definitions_and_extents(self):
        archive, spans = self.archive("nested/scene.rpf", F.scene())
        direct = self.report(archive, "--check-extents")
        nested = self.report(self.input, "--check-extents")
        self.assertEqual(direct["summary"]["archetypes"], 2)
        self.assertEqual(direct["extents"]["maps_fully_defined"], 1)
        self.assertEqual(self.without_sources(nested), self.without_sources(direct))
        self.assertEqual({r["source"] for r in nested["archetypes"]},
                         {f"nested/scene.rpf/{spans[0][0]:012x}"})

    def test_resources_in_distinct_archives_keep_distinct_source_keys(self):
        first = F.maptypes([(F.ROCK, 2, *F.ROCK_BOX)])
        second = F.maptypes([(F.CRATE, 1, *F.CRATE_BOX)])
        expected = set()
        for name in ("north/same.rpf", "south/same.rpf"):
            _, spans = self.archive(name, [first, second])
            expected.update(f"{name}/{offset:012x}" for offset, _ in spans)
        doc = self.report(self.input)
        self.assertEqual(doc["summary"]["archetypes"], 4)
        self.assertEqual(doc["summary"]["names"], 2)
        self.assertEqual({row["source"] for row in doc["archetypes"]}, expected)
        found = I.T.scan(str(self.input))
        self.assertEqual(set(found), expected)

    def test_archives_and_actual_carved_files_are_each_read_once(self):
        self.archive("scene.rpf", [F.maptypes([(F.ROCK, 2, *F.ROCK_BOX)])])
        source = self.root / "carve-source.rpf"
        source.write_bytes(F.rpf([F.maptypes([(F.CRATE, 1, *F.CRATE_BOX)])]))
        self.cli("rpf9_carve.py", source, self.input / "carved")
        doc = self.report(self.input)
        self.assertEqual(doc["summary"]["resources"], 2)
        self.assertEqual(doc["summary"]["archetypes"], 2)
        self.assertEqual({row["name"] for row in doc["archetypes"]}, {F.ROCK, F.CRATE})
        self.assertEqual(len({row["source"] for row in doc["archetypes"]}), 2)

    def test_nested_archive_uses_existing_continuation_assembly(self):
        resource = F.maptypes([(F.ROCK + i, 2, *F.ROCK_BOX) for i in range(500)])
        archive, _ = self.archive("split.rpf", [resource[:65536], resource[65536:]])
        doc = self.report(self.input)
        self.assertEqual(doc["summary"]["names"], 500)
        self.assertEqual(self.without_sources(doc), self.without_sources(self.report(archive)))

    def test_archive_windows_are_forwarded_and_do_not_filter_carved_files(self):
        first = F.maptypes([(F.ROCK, 2, *F.ROCK_BOX)])
        second = F.maptypes([(F.CRATE, 1, *F.CRATE_BOX)])
        archive, spans = self.archive("window.rpf", [first, second])
        for index, name in ((0, F.ROCK), (1, F.CRATE)):
            offset, stored = spans[index]
            args = ("--start", offset, "--limit", stored)
            with self.subTest(index=index):
                doc = self.report(self.input, *args)
                self.assertEqual([r["name"] for r in doc["archetypes"]], [name])
                self.assertEqual(self.without_sources(doc), self.without_sources(self.report(archive, *args)))
        carved = self.input / "loose.bin"
        carved.write_bytes(F.maptypes([(F.ROCK + 2, 1, *F.ROCK_BOX)]))
        doc = self.report(self.input, "--start", spans[1][0], "--limit", spans[1][1])
        self.assertEqual({row["name"] for row in doc["archetypes"]}, {F.CRATE, F.ROCK + 2})

    def test_interior_cli_retains_every_resource_in_mixed_input(self):
        for name in ("one/interiors.rpf", "two/interiors.rpf"):
            self.archive(name, [I.interiors(), I.placements()])
        (self.input / "placements.bin").write_bytes(I.placements())
        names = self.root / "names.json"
        names.write_text(json.dumps(I.NAMES), encoding="utf-8", newline="\n")
        output = self.root / "interiors.json"
        self.cli("world_interior_names.py", self.input, "--names", names, "--out", output)
        doc = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(doc["summary"]["interior_definitions"], 4)
        self.assertEqual(doc["summary"]["interior_names"], 2)
        self.assertEqual(doc["summary"]["placements"], 9)
        self.assertEqual(doc["summary"]["keys_naming_an_interior"], 2)
        self.assertEqual(doc["keys"][I.KEY_A]["placements"], 3)
        self.assertEqual(doc["keys"][I.KEY_B]["placements"], 6)

    def test_mixed_scan_order_is_preserved_including_empty_archives(self):
        def resource(index):
            return F.maptypes([(F.ROCK + index, 2, *F.ROCK_BOX)])

        for name, index in (("00-before.bin", 0), ("20-middle.bin", 3), ("40-after.bin", 5)):
            (self.input / name).write_bytes(resource(index))
        self.archive("05-empty.rpf", [])
        self.archive("10-first.rpf", [resource(1), resource(2)])
        self.archive("30-second.rpf", [resource(4)])
        doc = self.report(self.input)
        self.assertEqual([row["name"] for row in doc["archetypes"]], [F.ROCK + i for i in range(6)])

    def test_archive_hook_preserves_scanner_rows_stats_and_consumed_page_membership(self):
        real, _ = self.archive("scene.rpf", F.scene())
        resource = bytearray(F.maptypes([(F.ROCK + i, 2, *F.ROCK_BOX) for i in range(500)]))
        # Byte 65,536 falls in padding at +0x20 of a base archetype. A legal continuation can begin with
        # archive magic; only the scanner knows it was consumed by the preceding carved resource.
        resource[65536:65540] = b"9FPR"
        self.assertEqual(len(F.WA.parse_archetypes(resource)), 500)
        source = self.root / "carve-source.rpf"
        source.write_bytes(F.rpf([resource[:65536], resource[65536:]]))
        self.cli("rpf9_carve.py", source, self.input / "carved")
        before_entities, after_entities, archives = [], [], []
        before_rows, before_stats = F.WI.scan_path(str(self.input), collect=before_entities)
        rows, stats = F.WI.scan_path(str(self.input), collect=after_entities,
                                    on_archive=lambda path, selected: archives.append((path, selected)))
        self.assertEqual(rows, before_rows)
        self.assertEqual(stats, before_stats)
        self.assertEqual(after_entities, before_entities)
        self.assertEqual(len(after_entities), 2)
        self.assertEqual([path for path, _ in archives], [str(real)])
        self.assertEqual(len(archives[0][1]), 2)
        self.assertEqual([row["size"] for row in rows if "file" in row], [len(resource)])


if __name__ == "__main__":
    unittest.main()
