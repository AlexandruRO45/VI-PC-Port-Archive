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
# The same empty group header with zero padding behind it, as rows 2-4 of the #5600 review had it.
EMPTY = struct.pack("<3I", 0x80000000, 0, 0)


def group_page():
    """One valid group: 1 meshlet of 3 vertex refs and 1 triangle, running sums starting at 0."""
    meshlet = struct.pack("<4I", 2, 0, 0, 0)              # w0 low 7 bits: 3 vertices; next 7: 1 triangle
    triangle = struct.pack("<I", (1 << 7) | (2 << 14))     # meshlet 0; vertices 0, 1, 2
    refs = struct.pack("<3H", 0, 1, 2)
    return struct.pack("<3I", 0x80000001, 1, 3) + meshlet + triangle + refs


def padded(*chunks):
    """The chunks laid out as a page has them: each zero-padded to the next 16-byte boundary, padding after."""
    return b"".join(chunk + bytes(-len(chunk) % 16) for chunk in chunks) + bytes(16)


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
    def test_an_empty_group_is_a_record_with_no_counts(self):
        """It validates: the counts are what the group declares, and it ends 12 bytes in."""
        record, end = mp.parse_group(EMPTY_GROUP, 0)
        self.assertEqual((record["meshlets"], record["triangles"], record["vertex_refs"]), (0, 0, 0))
        self.assertEqual(end, 12)

    def test_a_header_without_meshlets_keeps_its_sums_message(self):
        """Counts that no meshlet accounts for are still the running-sum error, not a walk error."""
        page = struct.pack("<3I", 0x80000000, 1, 3) + bytes(10)   # 0 meshlets but 1 triangle, 3 refs
        with self.assertRaises(mp.PageError) as caught:
            mp.parse_group(page, 0)
        self.assertIn("meshlet sums", str(caught.exception))

    def test_parse_page_refuses_the_chunk_it_already_read(self):
        """The empty group ends 12 bytes in, so the alignment lands on its own start: one error, not a spin."""
        with self.assertRaises(mp.PageError) as caught:
            run_with_timeout(lambda: mp.parse_page(EMPTY_GROUP))
        self.assertIn("does not advance", str(caught.exception))

    def test_a_padded_empty_group_is_one_group(self):
        chunks = run_with_timeout(lambda: mp.parse_page(padded(EMPTY)))
        self.assertEqual([c["meshlets"] for c in chunks], [0])

    def test_groups_around_a_padded_empty_group_keep_parsing(self):
        """The review's rows 3 and 4: the empty group sits between chunks and is no reason to drop them."""
        two = run_with_timeout(lambda: mp.parse_page(padded(group_page(), EMPTY)))
        self.assertEqual([c["meshlets"] for c in two], [1, 0])
        three = run_with_timeout(lambda: mp.parse_page(padded(group_page(), EMPTY, group_page())))
        self.assertEqual([c["meshlets"] for c in three], [1, 0, 1])
        self.assertEqual([c["vertex_refs"] for c in three], [3, 0, 3])

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
