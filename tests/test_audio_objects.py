"""tools/audio_objects.py on hand-built config streams (no game data)."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "audio_objects.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import audio_config  # noqa: E402
import audio_objects as O  # noqa: E402


def record(rid, hash_a, hash_b, duration_ms=180_000):
    """One audio-config music record as audio_config.py reads it: the duration, then the 46-byte record."""
    return (struct.pack("<I", duration_ms) + b"\x0c\x01" + struct.pack("<H", rid) + b"\x02\x00\x0c\x00"
            + struct.pack("<II", hash_a, hash_b) + b"\x00\x00\x02\x23" + bytes(22))


def entry(hash_a, value, kind=2, flags=0x800, unknown=0x12345678):
    return struct.pack("<6I", flags, unknown, kind, hash_a, value, value)


RECS = [(0xB000 + k, 0xA0000000 + k * 0x1111, 0xB0000000 + k * 0x2222) for k in range(4)]


def table(rows=RECS, **kw):
    return b"".join(entry(a, 0xC0000000 + i, **kw) for i, (_, a, _) in enumerate(rows))


class AudioObjectsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data):
        with open(os.path.join(self.dir, name), "wb") as f:
            f.write(data)

    def records_stream(self):
        return bytes(8) + b"".join(record(*r) for r in RECS) + bytes(8)

    def run_join(self):
        records = audio_config.collect(self.dir, join=True)
        self.assertEqual(len(records), len(RECS))
        return O.join(self.dir, records)

    def test_every_record_is_joined_to_its_entry(self):
        self.write("000000001000.bin", self.records_stream())
        self.write("000000002000.bin", bytes(12) + table() + bytes(12))
        rows, s = self.run_join()
        self.assertEqual(s["with_one_entry"], 4)
        self.assertEqual(s["with_no_entry"], 0)
        self.assertEqual(s["value_repeated"], 4)
        self.assertEqual(s["kinds"], {"2": 4})
        self.assertEqual(s["ascending_neighbours"], 3)
        self.assertEqual(s["values_that_are_record_hashes"], 0)
        first = rows[0]
        self.assertEqual((first["file"], first["file_offset"]), ("000000002000.bin", 12))
        self.assertEqual(first["value"], 0xC0000000)

    def test_an_entry_off_the_word_grid_is_found(self):
        # A carved stream need not start on a word boundary of the config.
        self.write("000000001000.bin", self.records_stream())
        self.write("000000002000.bin", bytes(2) + table() + bytes(6))
        rows, s = self.run_join()
        self.assertEqual(s["with_one_entry"], 4)
        self.assertEqual(rows[1]["file_offset"], 2 + 24)

    def test_words_without_the_entry_shape_are_not_entries(self):
        self.write("000000001000.bin", self.records_stream())
        bad = (entry(RECS[0][1], 1, flags=0x801)          # flags not a multiple of 0x800
               + entry(RECS[1][1], 2, flags=17 * 0x800)    # beyond the largest flags value
               + entry(RECS[2][1], 3, kind=99)             # kind is a small integer
               + entry(RECS[3][1], 4, flags=0))            # flags is never zero
        self.write("000000002000.bin", bytes(12) + bad + bytes(12))
        _, s = self.run_join()
        self.assertEqual(s["with_one_entry"], 0)
        self.assertEqual(s["with_no_entry"], 4)

    def test_a_hash_a_in_two_entries_is_reported_not_guessed(self):
        self.write("000000001000.bin", self.records_stream())
        self.write("000000002000.bin", bytes(12) + table() + entry(RECS[0][1], 0xD0000000) + bytes(12))
        rows, s = self.run_join()
        self.assertEqual(s["with_several_entries"], 1)
        self.assertNotIn("file", rows[0])
        self.assertEqual(rows[0]["entries"], 2)

    def test_the_record_itself_is_not_an_entry(self):
        # The record stores hash A at +8 after `02 00 0c 00`, which is not a valid flags/kind pair.
        self.write("000000001000.bin", self.records_stream())
        _, s = self.run_join()
        self.assertEqual(s["with_one_entry"], 0)

    def test_unequal_v_words_are_reported(self):
        # value_repeated compares the words at +16 and +20; an entry whose two V words differ must say so.
        self.write("000000001000.bin", self.records_stream())
        odd = b"".join(entry(a, 0xC0000000 + i) if i % 2 else
                       struct.pack("<6I", 0x800, 0, 2, a, 0xC0000000 + i, 0xD0000000 + i)
                       for i, (_, a, _) in enumerate(RECS))
        self.write("000000002000.bin", bytes(12) + odd + bytes(12))
        rows, s = self.run_join()
        self.assertEqual(s["with_one_entry"], 4)
        self.assertEqual(s["value_repeated"], 2)
        self.assertEqual([r["value_repeated"] for r in rows], [False, True, False, True])

    def test_shared_values_are_counted(self):
        self.write("000000001000.bin", self.records_stream())
        shared = b"".join(entry(a, 0xE0000000) for _, a, _ in RECS)
        self.write("000000002000.bin", bytes(12) + shared + bytes(12))
        _, s = self.run_join()
        self.assertEqual(s["distinct_values"], 1)
        self.assertEqual(s["values_shared"], 1)
        self.assertEqual(s["most_entries_per_value"], 4)

    def test_scan_reads_the_buffer_in_place(self):
        # A tuple of every word costs ~20 bytes per input byte; the scan must stay near the input's own size.
        import tracemalloc
        key = 0xAABBCCDD
        blob = bytearray(b"X" * (1 << 20))               # the tuple version peaked near 20 MB here
        offsets = [4096 + 65 * k for k in range(4)]      # three of them off the word grid
        for pos in offsets:
            blob[pos:pos + 24] = entry(key, 19)
        tracemalloc.start()
        try:
            rows = sorted(O.scan(blob, {key}))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual([p for p, _ in rows], offsets)
        self.assertTrue(all(e["value"] == 19 and e["value_repeated"] for _, e in rows))
        self.assertLess(peak, 256 << 10, f"scan peaked at {peak} bytes for a 1 MiB input")

    def test_join_needs_only_paths_in(self):
        # Another branch drops audio_config.files_in; the join must not depend on it.
        self.write("000000001000.bin", self.records_stream())
        self.write("000000002000.bin", bytes(12) + table() + bytes(12))
        records = audio_config.collect(self.dir, join=True)
        saved = getattr(audio_config, "files_in", None)
        if saved is not None:
            del audio_config.files_in
        try:
            _, s = O.join(self.dir, records)
        finally:
            if saved is not None:
                audio_config.files_in = saved
        self.assertEqual(s["with_one_entry"], 4)

    def test_cli_summary_and_json(self):
        self.write("000000001000.bin", self.records_stream())
        self.write("000000002000.bin", bytes(12) + table() + bytes(12))
        r = subprocess.run([sys.executable, "-I", TOOL, self.dir], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("4 records; 4 with one entry, 0 with none, 0 with several", r.stdout)
        r = subprocess.run([sys.executable, "-I", TOOL, self.dir, "--json"], capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(len(data["records"]), 4)
        self.assertEqual(data["summary"]["with_one_entry"], 4)

    def test_cli_missing_input_and_no_records(self):
        r = subprocess.run([sys.executable, "-I", TOOL, os.path.join(self.dir, "absent")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not exist", r.stderr)
        self.write("000000002000.bin", table())
        r = subprocess.run([sys.executable, "-I", TOOL, self.dir], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no audio-config records found", r.stderr)


if __name__ == "__main__":
    unittest.main()
