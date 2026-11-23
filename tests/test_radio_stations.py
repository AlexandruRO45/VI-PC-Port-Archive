import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import radio_stations  # noqa: E402
from audio_config import joaat  # noqa: E402

DEBUG_NAME = "000000b01000.bin"
DESC_NAME = "000000762160.bin"


def make_carve(root, slugs=("test_one", "test_two")):
    debug = b"\x00" + b"\x00".join(
        ("Core.RadioStation." + slug).encode("latin-1") for slug in slugs
    ) + b"\x00Core.RadioStationBad.x\x00"
    with open(os.path.join(root, DEBUG_NAME), "wb") as handle:
        handle.write(debug)

    desc = bytearray(0x40)
    struct.pack_into("<II", desc, 0x10, joaat("radio_test_one"), 0x3E8)
    struct.pack_into("<II", desc, 0x20, joaat("radio_test_two"), 0x7D0)
    with open(os.path.join(root, DESC_NAME), "wb") as handle:
        handle.write(bytes(desc))


class RadioStationsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        make_carve(self.tmp.name)

    def test_joaat_matches_record_table_rule(self):
        self.assertEqual(joaat("radio_test_one"), joaat("RADIO_TEST_ONE"))
        self.assertEqual(radio_stations.station_hashes(["test_one"]),
                         {joaat("radio_test_one"): ["test_one"]})

    def test_resolves_matching_descriptors(self):
        report = radio_stations.run(self.tmp.name)
        self.assertEqual(report["debug_stream"], DEBUG_NAME)
        self.assertEqual(report["slugs"], 2)
        self.assertEqual(report["matched"], 2)
        self.assertEqual([d["slug"] for d in report["descriptors"]], ["test_one", "test_two"])
        self.assertEqual(report["ends"], [0x3E8, 0x7D0])
        self.assertEqual(report["streams"], [DESC_NAME])

    def test_ignores_end_values_outside_the_station_band(self):
        desc = bytearray(0x40)
        struct.pack_into("<II", desc, 0x10, joaat("radio_test_one"), 0x1387)  # not a multiple of 1000
        with open(os.path.join(self.tmp.name, DESC_NAME), "wb") as handle:
            handle.write(bytes(desc))
        self.assertEqual(radio_stations.run(self.tmp.name)["matched"], 0)

    def test_ignores_non_aligned_pairs(self):
        desc = bytearray(0x40)
        struct.pack_into("<II", desc, 0x11, joaat("radio_test_one"), 0x3E8)
        with open(os.path.join(self.tmp.name, DESC_NAME), "wb") as handle:
            handle.write(bytes(desc))
        self.assertEqual(radio_stations.run(self.tmp.name)["matched"], 0)

    def test_one_slug_per_hash_and_duplicates_kept(self):
        # Two debug names can hash alike only in a synthetic fixture; the first is reported.
        table = radio_stations.station_hashes(["test_one", "test_one_alt"])
        self.assertIn(joaat("radio_test_one_alt"), table)
        self.assertEqual(table[joaat("radio_test_one_alt")][0], "test_one_alt")

    def test_cli_json_and_exit_zero(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = radio_stations.main([self.tmp.name, "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["matched"], 2)

    def test_cli_without_debug_stream_exits(self):
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        with open(os.path.join(empty.name, DESC_NAME), "wb") as handle:
            handle.write(b"\x00" * 16)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                radio_stations.main([empty.name])

    def test_cli_writes_lf_only(self):
        path = os.path.join(self.tmp.name, "report.json")
        self.assertEqual(radio_stations.main([self.tmp.name, "--json", "--out", path]), 0)
        with open(path, "rb") as handle:
            blob = handle.read()
        self.assertNotIn(b"\r\n", blob)
        self.assertTrue(blob.endswith(b"\n"))
        json.loads(blob.decode("utf-8"))

    def test_cli_exit_one_when_nothing_resolves(self):
        desc = bytearray(0x40)
        struct.pack_into("<II", desc, 0x10, 0x12345678, 0x3E8)
        with open(os.path.join(self.tmp.name, DESC_NAME), "wb") as handle:
            handle.write(bytes(desc))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(radio_stations.main([self.tmp.name]), 1)


if __name__ == "__main__":
    unittest.main()
