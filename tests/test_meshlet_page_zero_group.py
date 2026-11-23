"""tools/meshlet_page.py on hand-written pages: a group with no meshlets must not spin forever (#5600).

Every byte here is built in the test; no game data and no GPU.
"""
import os
import struct
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import meshlet_page as mp  # noqa: E402

# The page from #5600: a group header with 0 meshlets, 0 triangles and 0 references, then the byte 1.
EMPTY_GROUP = bytes.fromhex("00000080000000000000000001000000")


def group_page():
    """One valid group: 1 meshlet of 3 vertex refs and 1 triangle, running sums starting at 0."""
    meshlet = struct.pack("<4I", 2, 0, 0, 0)              # w0 low 7 bits: 3 vertices; next 7: 1 triangle
    triangle = struct.pack("<I", (1 << 7) | (2 << 14))     # meshlet 0; vertices 0, 1, 2
    refs = struct.pack("<3H", 0, 1, 2)
    return struct.pack("<3I", 0x80000001, 1, 3) + meshlet + triangle + refs


def run_with_timeout(call, seconds=10):
    """`call()` in a daemon thread: before #5600 parse_page never returned, and a test must not hang with it."""
    box = {}

    def work():
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise AssertionError(f"parse_page did not return within {seconds} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class EmptyGroupTest(unittest.TestCase):
    def test_empty_group_is_not_a_mesh_page(self):
        self.assertFalse(mp.is_mesh_page(EMPTY_GROUP))

    def test_parse_group_rejects_an_empty_group(self):
        with self.assertRaises(mp.PageError) as caught:
            mp.parse_group(EMPTY_GROUP, 0)
        self.assertIn("no meshlets", str(caught.exception))

    def test_a_header_without_meshlets_keeps_its_sums_message(self):
        """Counts that no meshlet accounts for are still the running-sum error, not the new one."""
        page = struct.pack("<3I", 0x80000000, 1, 3) + bytes(10)   # 0 meshlets but 1 triangle, 3 refs
        with self.assertRaises(mp.PageError) as caught:
            mp.parse_group(page, 0)
        self.assertIn("meshlet sums", str(caught.exception))

    def test_parse_page_returns_an_error(self):
        with self.assertRaises(mp.PageError):
            run_with_timeout(lambda: mp.parse_page(EMPTY_GROUP))

    def test_cli_reports_the_page_and_exits_1(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = os.path.join(scratch, "hang.bin")
            with open(path, "wb") as fh:
                fh.write(EMPTY_GROUP)
            out = StringIO()
            with redirect_stdout(out):
                code = run_with_timeout(lambda: mp.main([path]))
        self.assertEqual(code, 1)
        self.assertIn("ERROR", out.getvalue())

    def test_a_valid_group_still_parses(self):
        chunks = run_with_timeout(lambda: mp.parse_page(group_page()))
        self.assertEqual([c["kind"] for c in chunks], ["group"])
        self.assertEqual((chunks[0]["meshlets"], chunks[0]["triangles"], chunks[0]["vertex_refs"]), (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
