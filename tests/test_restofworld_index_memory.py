"""Bounded carved-video indexing with original synthetic inputs only.

Run: python -I tests/test_restofworld_index_memory.py
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import restofworld_index  # noqa: E402
import test_restofworld_index as fixtures  # noqa: E402

TOOL = ROOT / "tools" / "restofworld_index.py"
CARVER = ROOT / "tools" / "rpf9_carve.py"


def movie_header(size):
    """Build only the fixed header needed by the indexer, without decoded video."""
    head = bytearray(fixtures.bink())
    struct.pack_into("<I", head, 4, size - 8)
    return bytes(head)


def write_movie(path, size):
    with path.open("wb") as stream:
        stream.write(movie_header(size))
        stream.seek(size - 1)
        stream.write(b"\0")


def report_counts(summary):
    return {key: value for key, value in summary.items()
            if key not in ("mode", "archive", "directory", "elapsed_seconds")}


class CarvedVideoMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, tool, *args):
        proc = subprocess.run([sys.executable, "-I", str(tool), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc

    def test_actual_carve_and_cli_keep_counts_without_copying_video(self):
        size = 32 << 20
        archive = self.root / "synthetic.rpf"
        folder = self.root / "carved"
        header = bytearray(64)
        header[:4] = b"9FPR"
        struct.pack_into("<I", header, 4, len(header))
        with archive.open("wb") as stream:
            stream.write(header)
            stream.write(movie_header(size))
            stream.seek(len(header) + size - 1)
            stream.write(b"\0")
            stream.write(zlib.compress(fixtures.psin()))
        self.run_cli(CARVER, archive, folder)
        self.assertEqual(len(list(folder.glob("*.bk2"))), 1)
        self.assertEqual(len(list(folder.glob("*.bin"))), 1)

        expected_rows, expected_summary = restofworld_index.scan_rpf(
            str(archive), max_inflate=512)
        tracemalloc.start()
        try:
            rows, summary = restofworld_index.scan_dir(str(folder), max_inflate=512)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(rows, expected_rows)
        self.assertEqual(report_counts(summary), report_counts(expected_summary))
        self.assertEqual([row["signature"] for row in rows], ["bink2", "psin"])
        self.assertEqual((rows[0]["size"], rows[0]["stored"]), (size, size))
        self.assertEqual(summary["bink2_bytes"], size)
        self.assertEqual(summary["files"], 2)
        self.assertLess(peak, 2 << 20, f"32 MiB video indexing allocated {peak} bytes")

        report = self.root / "report.json"
        self.run_cli(TOOL, folder, report, "--max-inflate", 512, "--quiet")
        document = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(document["files"], expected_rows)
        self.assertEqual(report_counts(document["summary"]), report_counts(expected_summary))

    def test_bare_folder_uses_full_file_sizes_and_keeps_order(self):
        folder = self.root / "bare"
        folder.mkdir()
        sizes = (4096, 8192, 64)
        for name, size in zip(("000000000010.bk2", "000000000030.bk2", "movie.bk2"), sizes):
            write_movie(folder / name, size)
        rows, summary = restofworld_index.scan_dir(str(folder), max_inflate=1)
        self.assertEqual([row["offset"] for row in rows], [0x10, 0x30, 2])
        self.assertEqual([row["stored"] for row in rows], list(sizes))
        self.assertEqual([row["size"] for row in rows], list(sizes))
        self.assertEqual([row["signature"] for row in rows], ["bink2"] * 3)
        self.assertEqual(summary["bink2_files"], 3)
        self.assertEqual(summary["bink2_bytes"], sum(sizes))
        self.assertEqual(summary["skipped_candidates"], 0)

    def test_non_bink_suffix_inputs_keep_complete_psin_and_texture_data(self):
        # Both structured resources require bytes beyond the header probe.
        psin = bytearray(0x20 + 0x300 + 0x14)
        psin[:4] = b"PSIN"
        struct.pack_into(">I", psin, 4, 0x20)
        psin[0x20:0x24] = b"LSCH"
        struct.pack_into(">I", psin, 0x24, 0x300)
        psin[0x320:0x324] = b"CHKS"
        struct.pack_into(">I", psin, 0x324, 0x14)
        texture = bytearray(0x200)
        struct.pack_into("<Q", texture, 8, 0x50000040)
        struct.pack_into("<QHHxxxxQHH", texture, 0x20,
                         0x50000050, 1, 1, 0x50000060, 1, 1)
        texture += fixtures._synth_tex_descriptor(0, 256, 256, 71, mips=7)
        for suffix in (".bk2", ".bin"):
            with self.subTest(suffix=suffix):
                folder = self.root / suffix[1:]
                folder.mkdir()
                (folder / ("000000001000" + suffix)).write_bytes(psin)
                (folder / ("000000002000" + suffix)).write_bytes(texture)
                rows, summary = restofworld_index.scan_dir(str(folder))
                self.assertEqual([row["signature"] for row in rows], ["psin", "zero-prefix"])
                self.assertEqual([row["size"] for row in rows], [len(psin), len(texture)])
                self.assertEqual(rows[0]["count"], 2)
                self.assertIn("tags=LSCH,CHKS;", rows[0]["detail"])
                self.assertIn("complete=1", rows[0]["detail"])
                self.assertEqual(rows[1]["count"], 1)
                self.assertIn("tex980=1;dict=1;fmts=BC1_UNORM:1", rows[1]["detail"])
                self.assertEqual(summary["texture_headers"], 1)
                self.assertEqual(summary["bink2_files"], 0)

    def test_short_headers_and_signature_precedence_keep_existing_results(self):
        folder = self.root / "headers"
        folder.mkdir()
        ambiguous = bytearray(1024)
        ambiguous[:4] = b"KB2n"
        ambiguous[0x10:0x14] = b"0DRP"  # this signature precedes Bink in classify()
        struct.pack_into("<I", ambiguous, 0x14, 122)
        struct.pack_into("<Q", ambiguous, 0x20, 7)
        payloads = (b"", b"KB", b"KB2", b"KB2n" + bytes(27), bytes(ambiguous))
        for index, payload in enumerate(payloads):
            (folder / f"{index * 0x100:012x}.bk2").write_bytes(payload)
        rows, summary = restofworld_index.scan_dir(str(folder))
        self.assertEqual([row["signature"] for row in rows],
                         ["raw:", "raw:4b42", "raw:4b4232", "raw:4b42326e", "0drp"])
        self.assertEqual([row["size"] for row in rows], list(map(len, payloads)))
        self.assertEqual([row["stored"] for row in rows], list(map(len, payloads)))
        self.assertEqual(rows[-1]["count"], 7)
        self.assertEqual(rows[-1]["version"], 122)
        self.assertEqual(summary["bink2_files"], 0)


if __name__ == "__main__":
    unittest.main()
