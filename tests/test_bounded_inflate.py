"""The archive scanners inflate each zlib candidate under an output cap (tools/bounded_inflate.py).

`zlib.decompressobj().decompress(block)` has no output limit, so one highly compressed candidate could grow in
memory until the scan died (docs/story.md: the full story000.rpf walk stopped at 72% with one candidate at
8.5 GB). This checks the helper against plain zlib, then runs every scanner that used the unbounded loop on an
archive holding a 96 MiB all-zero stream (about 96 KiB stored) under a 16 MiB cap, and measures the Python heap
peak with tracemalloc: the over-cap stream is skipped, the ordinary stream after it is still found.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
import bounded_inflate  # noqa: E402

BOMB = 96 << 20          # inflated size of the over-cap stream
CAP = 16 << 20           # the cap the scanners run with in this test
PEAK_LIMIT = 48 << 20    # under the bomb: inflating it whole would alone take 96 MiB

# Each scanner and its arguments; "{arc}" is the archive and "{out}" a fresh output directory.
SCANNERS = {
    "rpf9_carve": ["{arc}", "{out}/carve", "--index-only"],
    "story_index": ["{arc}", "{out}/story.json", "--quiet"],
    "anim_index": ["{arc}", "{out}/anim.json", "--quiet"],
    "rbf0_index": ["{arc}", "{out}/rbf0.json"],
    "page_index": ["{arc}", "{out}/pages.json", "--quiet"],
    "navmesh_index": ["{arc}", "--out", "{out}/navmesh.json"],
    "meshlet_page": ["{arc}", "--json", "{out}/meshlets.json", "--quiet"],
    "vertex_types": ["{arc}", "--json", "{out}/vertex.json"],
    "skin_streams": ["{arc}", "--json", "{out}/skin.json"],
    "analyze_4240_full": ["{arc}", "--json", "{out}/4240.json"],
}

# Runs a tool as __main__ with the cap lowered, under tracemalloc, and prints its exit code and heap peak as JSON
# on the last line. The tool imports the same bounded_inflate module object, so it sees the lowered cap.
RUNNER = r"""
import json, runpy, sys, tracemalloc
tools, cap, tool, args = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
sys.path.insert(0, tools)
import bounded_inflate
bounded_inflate.MAX_INFLATE = cap
sys.argv = [tool] + args
tracemalloc.start()
code = 0
try:
    runpy.run_path(tool, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
peak = tracemalloc.get_traced_memory()[1]
sys.stdout.flush()
print("\n" + json.dumps({"code": code, "peak": peak}))
"""


class InflateTest(unittest.TestCase):
    def test_same_bytes_and_consumed_length_as_zlib(self):
        for payload in (b"x" * 100, os.urandom(3 << 20), bytes(5 << 20)):
            stream = zlib.compress(payload)
            for trailer in (b"", b"\x00" * 7, b"tail bytes"):
                buf = b"lead" + stream + trailer
                with self.subTest(size=len(payload), trailer=len(trailer)):
                    data, used, why = bounded_inflate.inflate(buf, 4, len(buf), max_out=len(payload))
                    self.assertEqual(why, "ok")
                    self.assertEqual(data, payload)
                    self.assertEqual(used, len(stream))

    def test_output_past_the_cap_is_too_big(self):
        stream = zlib.compress(bytes(4 << 20))
        self.assertEqual(bounded_inflate.inflate(stream, 0, len(stream), max_out=(4 << 20) - 1), (None, 0, "too_big"))

    def test_the_cap_defaults_to_the_module_value_at_call_time(self):
        stream = zlib.compress(bytes(3 << 20))
        old = bounded_inflate.MAX_INFLATE
        try:
            bounded_inflate.MAX_INFLATE = 1 << 20
            self.assertEqual(bounded_inflate.inflate(stream, 0, len(stream))[2], "too_big")
        finally:
            bounded_inflate.MAX_INFLATE = old
        self.assertEqual(bounded_inflate.inflate(stream, 0, len(stream))[2], "ok")

    def test_a_stream_cut_by_the_window_is_truncated(self):
        stream = zlib.compress(os.urandom(1 << 16))
        self.assertEqual(bounded_inflate.inflate(stream, 0, len(stream) - 10), (None, 0, "truncated"))
        self.assertEqual(bounded_inflate.inflate(stream[:-10], 0, len(stream)), (None, 0, "truncated"))

    def test_invalid_bytes_raise_zlib_error(self):
        with self.assertRaises(zlib.error):
            bounded_inflate.inflate(b"\x78\x9c" + b"\xff" * 64, 0, 66)

    def test_one_call_never_returns_more_than_a_step(self):
        stream = zlib.compress(bytes(64 << 20), 9)      # about 64 KiB stored: one 1 MiB block inflates to 64 MiB
        seen = []
        real = zlib.decompressobj

        class Probe:
            def __init__(self):
                self.d = real()

            def decompress(self, data, max_length=0):
                out = self.d.decompress(data, max_length)
                seen.append((len(out), max_length))
                return out

            def __getattr__(self, name):
                return getattr(self.d, name)

        bounded_inflate.zlib.decompressobj = Probe
        try:
            data, used, why = bounded_inflate.inflate(stream, 0, len(stream), max_out=64 << 20)
        finally:
            bounded_inflate.zlib.decompressobj = real
        self.assertEqual((why, len(data), used), ("ok", 64 << 20, len(stream)))
        self.assertTrue(seen and all(0 < m <= bounded_inflate.STEP and n <= m for n, m in seen))


def make_archive(path):
    """RPF9 data area: an all-zero BOMB-byte stream, a short zero gap, then an ordinary stream."""
    bomb = zlib.compress(bytes(BOMB), 9)
    normal = zlib.compress(b"N" * 4096)
    with open(path, "wb") as f:
        f.write(b"9FPR" + struct.pack("<I", 0x40) + bytes(0x38))
        f.write(bomb)
        f.write(bytes(256))
        f.write(normal)
        f.write(bytes(4096))
    return len(bomb), len(normal)


class ScannerCapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.arc = os.path.join(cls.tmp.name, "bomb.rpf")
        cls.bomb_stored, cls.normal_stored = make_archive(cls.arc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_tool(self, name, out):
        argv = [x.format(arc=self.arc, out=out) for x in SCANNERS[name]]
        p = subprocess.run([sys.executable, "-I", "-c", RUNNER, TOOLS, str(CAP), os.path.join(TOOLS, f"{name}.py")]
                           + argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(res["code"], 0, p.stdout[-2000:] + p.stderr[-2000:])
        return res

    def test_an_over_cap_stream_is_not_inflated_whole(self):
        for name in SCANNERS:
            with self.subTest(tool=name), tempfile.TemporaryDirectory() as out:
                res = self.run_tool(name, out)
                self.assertLess(res["peak"], PEAK_LIMIT, f"{name}: heap peak {res['peak'] >> 20} MiB")

    def test_the_carve_skips_it_and_keeps_the_ordinary_stream(self):
        with tempfile.TemporaryDirectory() as out:
            self.run_tool("rpf9_carve", out)
            with open(os.path.join(out, "carve", "index.json"), encoding="utf-8") as f:
                index = json.load(f)
        self.assertEqual([(r["stored"], r["size"]) for r in index["files"]], [(self.normal_stored, 4096)])
        self.assertEqual(index["summary"]["skipped_too_big"], 1)

    def test_story_index_reports_the_skip(self):
        with tempfile.TemporaryDirectory() as out:
            self.run_tool("story_index", out)
            with open(os.path.join(out, "story.json"), encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["summary"]["skipped_too_big"], 1)
        self.assertEqual(report["summary"]["files"], 1)


if __name__ == "__main__":
    unittest.main()
