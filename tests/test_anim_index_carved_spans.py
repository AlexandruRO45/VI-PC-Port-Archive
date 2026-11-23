"""Carved rig continuations use archive spans, not inflated lengths. Synthetic inputs only."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import anim_index  # noqa: E402
import rbf0_tree  # noqa: E402

CHUNK = 65536


def record(index, kind, name):
    raw = name.encode("ascii")
    return bytes((index, kind)) + struct.pack("<H", len(raw)) + raw


def element(index, name, attributes=0):
    return record(index, 0, name) + struct.pack("<HHH", 0, 0, attributes)


def text(value):
    raw = value.encode("ascii") + b"\0"
    return b"\xfd\xff" + struct.pack("<I", len(raw)) + raw


def rig_resource(padding=CHUNK):
    """A complete wide RBF0 tree; its rig entry follows a long text record."""
    key = b"anim/rigset/synthetic/main"
    out = b"RBF0\xfb\xff\x01\x00" + element(0, "rage__crRigSetManager")
    out += element(1, "rigsets") + element(2, "entry", attributes=1)
    out += record(3, 0x60, "key") + struct.pack("<H", len(key)) + key
    out += text("X" * padding)
    out += element(4, "source") + text("canonical@source") + b"\xff\xff"
    out += element(5, "rig") + text("anim/rig/synthetic") + b"\xff\xff"
    out += element(6, "group") + text("body") + b"\xff\xff"
    return out + b"\xff\xff" * 3


class CarvedSpansTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = 0

    def tool(self, name, *args):
        result = subprocess.run([sys.executable, "-I", str(TOOLS / name), *map(str, args)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def carve(self, payloads, levels, gaps):
        self.case += 1
        folder = self.root / str(self.case)
        folder.mkdir()
        header = bytearray(256)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        body = bytearray()
        for payload, level, gap in zip(payloads, levels, gaps):
            body += zlib.compress(payload, level=level) + bytes(gap)
        archive = folder / "synthetic.rpf"
        archive.write_bytes(header + body)
        carved = folder / "carved"
        self.tool("rpf9_carve.py", archive, carved)
        manifest = json.loads((carved / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), len(payloads))
        return archive, carved, manifest

    def index_rigs(self, source, label):
        index = source.parent / (label + "-index.json")
        rigs = source.parent / (label + "-rigs.json")
        self.tool("anim_index.py", source, index, "--rigs", rigs, "--quiet")
        return json.loads(rigs.read_text(encoding="utf-8"))["rbf0"]

    def assert_parity(self, archive, carved, expected_bytes, expected_entries):
        archive_report = self.index_rigs(archive, "archive")
        carved_report = self.index_rigs(carved, "carved")
        self.assertEqual(archive_report["header"]["bytes"], expected_bytes)
        self.assertEqual(archive_report["summary"]["entries"], expected_entries)
        self.assertEqual(carved_report, archive_report)

    def test_expanded_zlib_page_and_allowed_padding(self):
        blob = rig_resource()
        self.assertEqual(rbf0_tree.parse_resource(blob)["status"], "ok")
        for gap in (0, 1, 16):
            with self.subTest(gap=gap):
                archive, carved, manifest = self.carve([blob[:CHUNK], blob[CHUNK:]], [0, 6], [gap, 0])
                self.assertGreater(manifest["files"][0]["stored"], CHUNK)
                self.assert_parity(archive, carved, len(blob), 1)

    def test_each_continuation_updates_the_compressed_end(self):
        blob = rig_resource(2 * CHUNK)
        self.assertEqual(rbf0_tree.parse_resource(blob)["status"], "ok")
        archive, carved, _ = self.carve([blob[:CHUNK], blob[CHUNK:2 * CHUNK], blob[2 * CHUNK:]],
                                        [6, 0, 6], [0, 1, 0])
        self.assert_parity(archive, carved, len(blob), 1)

    def test_compressed_gap_boundary(self):
        blob = rig_resource()
        for gap in (16, 17, 32):
            with self.subTest(gap=gap):
                archive, carved, manifest = self.carve([blob[:CHUNK], blob[CHUNK:]], [6, 6], [gap, 0])
                self.assertLess(manifest["files"][0]["stored"], CHUNK)
                self.assert_parity(archive, carved, len(blob) if gap <= 16 else CHUNK, int(gap <= 16))

    def test_short_pages_signatures_and_oversized_continuations_still_stop(self):
        short = rig_resource(64)
        blob = rig_resource()
        move = b"MoVE" + bytes(124)
        for label, payloads, expected_bytes, entries in (
            ("short", [short, b"X" * 128], len(short), 1),
            ("signature", [blob[:CHUNK], move, blob[CHUNK:]], CHUNK, 0),
            ("oversized", [blob[:CHUNK], blob[CHUNK:] + bytes(CHUNK)], CHUNK, 0),
        ):
            with self.subTest(case=label):
                archive, carved, _ = self.carve(payloads, [6] * len(payloads), [0] * len(payloads))
                self.assert_parity(archive, carved, expected_bytes, entries)

    def test_one_byte_overlap_cannot_bridge_an_archive_gap(self):
        blob = rig_resource()
        archive, carved, manifest = self.carve([blob[:CHUNK], blob[CHUNK:]], [0, 6], [32, 0])
        self.assert_parity(archive, carved, CHUNK, 0)
        path = carved / "index.json"
        for next_metadata in ("matching", "missing", "invalid"):
            with self.subTest(next_metadata=next_metadata):
                changed = copy.deepcopy(manifest)
                first, following = changed["files"]
                first["stored"] = following["off"] - first["off"] + 1
                if next_metadata == "missing":
                    changed["files"].pop()
                elif next_metadata == "invalid":
                    following["size"] += 1
                path.write_text(json.dumps(changed), encoding="utf-8", newline="\n")
                self.assert_parity(archive, carved, CHUNK, 0)
        path.unlink()
        self.assertEqual(anim_index.collect_rbf0_dir(carved), blob[:CHUNK])

    def test_overlap_fallback_keeps_other_pages_compressed_spans(self):
        blob = rig_resource(2 * CHUNK)
        archive, carved, manifest = self.carve([blob[:CHUNK], blob[CHUNK:2 * CHUNK], blob[2 * CHUNK:]],
                                             [0, 0, 6], [1, 32, 0])
        self.assert_parity(archive, carved, 2 * CHUNK, 0)
        first, second, following = manifest["files"]
        # The first join needs its valid compressed span: the inflated-length fallback stops too early.
        self.assertGreater(second["off"], first["off"] + CHUNK + 16)
        second["stored"] = following["off"] - second["off"] + 1
        path = carved / "index.json"
        path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
        self.assert_parity(archive, carved, 2 * CHUNK, 0)
        path.unlink()
        self.assertEqual(anim_index.collect_rbf0_dir(carved), blob[:CHUNK])

    def test_bare_folders_and_unrelated_reports_keep_the_legacy_fallback(self):
        blob = rig_resource()
        _, carved, _ = self.carve([blob[:CHUNK], blob[CHUNK:]], [6, 6], [32, 0])
        path = carved / "index.json"
        path.unlink()
        self.assertEqual(anim_index.collect_rbf0_dir(carved), blob)
        for report in ("not JSON", "[]", "{}", '{"files": {}}', '{"files": [null, 42]}'):
            with self.subTest(report=report):
                path.write_text(report, encoding="utf-8", newline="\n")
                self.assertEqual(anim_index.collect_rbf0_dir(carved), blob)

    def test_metadata_must_match_the_file_offset_and_inflated_size(self):
        blob = rig_resource()
        _, carved, manifest = self.carve([blob[:CHUNK], blob[CHUNK:]], [6, 6], [32, 0])
        path = carved / "index.json"
        self.assertEqual(anim_index.collect_rbf0_dir(carved), blob[:CHUNK])
        changes = (
            ("off", manifest["files"][0]["off"] + 1), ("off", -1), ("off", True),
            ("size", CHUNK + 1), ("size", -1), ("size", True),
            ("stored", 0), ("stored", -1), ("stored", True),
            ("stored", 187.0), ("stored", "187"), ("stored", None),
        )
        for key, value in changes:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(manifest)
                changed["files"][0][key] = value
                path.write_text(json.dumps(changed), encoding="utf-8", newline="\n")
                self.assertEqual(anim_index.collect_rbf0_dir(carved), blob)
        duplicate = copy.deepcopy(manifest)
        duplicate["files"].append(duplicate["files"][0])
        path.write_text(json.dumps(duplicate), encoding="utf-8", newline="\n")
        self.assertEqual(anim_index.collect_rbf0_dir(carved), blob)


if __name__ == "__main__":
    unittest.main()
