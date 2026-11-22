"""A carve folder describes one finished carve: no streams of an earlier one, and a stopped carve is refused.

    python -I tests/test_rpf9_carve_folder.py

rpf9_carve.py wrote into whatever its output folder already held and wrote index.json only at the end, while
page_index.py, anim_index.py and rbf0_index.py index a folder by listing its `<offset>.bin` files. A second archive
carved into the first one's folder, or a carve that was stopped, was therefore indexed as if its streams were the
archive's, with exit 0. These tests run the real tools on small synthetic RPF9 archives; no game data.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
import carve_folder  # noqa: E402
import rpf9_carve  # noqa: E402


def archive(path, payloads, lead=0):
    """An RPF9 archive whose data area is `lead` zero bytes and then one zlib stream per payload."""
    data = bytearray(0x40 + lead) + b"".join(zlib.compress(p) for p in payloads) + bytes(4096)
    data[0:4] = b"9FPR"
    struct.pack_into("<I", data, 4, 0x40)
    with open(path, "wb") as f:
        f.write(data)


def tool(name, *args):
    return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, name), *args], capture_output=True, text=True)


class CarveFolderTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.out = os.path.join(self.tmp, "carve")
        self.five = os.path.join(self.tmp, "five.rpf")
        self.one = os.path.join(self.tmp, "one.rpf")
        archive(self.five, [bytes([65 + i]) * (1000 + i) for i in range(5)])
        archive(self.one, [b"Z" * 3000], lead=0x20)       # its stream is at an offset the other archive has none at

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def streams(self):
        return sorted(n for n in os.listdir(self.out) if n.endswith((".bin", ".bk2")))

    def index(self):
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            return json.load(f)

    def carve(self, source, *args):
        p = tool("rpf9_carve.py", source, self.out, *args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p.stdout

    def indexed_streams(self):
        """How many streams page_index.py counts when it is given the folder."""
        report = self.path("pages.json")
        p = tool("page_index.py", self.out, report, "--quiet")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(report, encoding="utf-8") as f:
            return json.load(f)["summary"]["streams"]

    def test_a_finished_carve_leaves_its_streams_an_index_and_no_marker(self):
        self.carve(self.five)
        self.assertEqual(len(self.streams()), 5)
        self.assertEqual(sorted(os.listdir(self.out)), self.streams() + ["index.json"])
        self.assertIsNone(carve_folder.unfinished(self.out))
        self.assertEqual(self.indexed_streams(), 5)

    def test_streams_of_an_earlier_carve_are_moved_beside_the_folder(self):
        self.carve(self.five)
        earlier_streams = self.streams()
        # Files this tool did not name are not its business.
        for name in ("notes.txt", "pages.csv", "extra.bin", "00000000004g.bin"):
            with open(os.path.join(self.out, name), "w", encoding="utf-8") as f:
                f.write("mine\n")
        out = self.carve(self.one)
        self.assertEqual(self.index()["summary"]["files"], 1)
        kept = [n for n in self.streams() if n not in ("extra.bin", "00000000004g.bin")]
        self.assertEqual(kept, ["%012x.bin" % self.index()["files"][0]["off"]])
        self.assertIn("moved 5 stream file(s) of an earlier carve", out)
        # Nothing is deleted: the five are in <out>.earlier, byte for byte.
        self.assertEqual(sorted(os.listdir(self.out + ".earlier")), earlier_streams)
        with open(os.path.join(self.out + ".earlier", earlier_streams[0]), "rb") as f:
            self.assertEqual(f.read(), b"A" * 1000)
        for name in ("notes.txt", "pages.csv", "extra.bin", "00000000004g.bin"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)

    def test_the_folder_indexer_counts_only_the_new_carve(self):
        self.carve(self.five)
        self.carve(self.one)
        self.assertEqual(self.indexed_streams(), 1)       # it was 6: five stale streams and the new one

    def test_a_narrower_window_of_the_same_archive_drops_the_streams_outside_it(self):
        self.carve(self.five)
        first = self.index()["files"][0]
        self.carve(self.five, "--limit", str(first["stored"]))
        self.assertEqual(self.streams(), ["%012x.bin" % first["off"]])

    def test_carving_again_moves_nothing(self):
        self.carve(self.five)
        before = self.streams()
        self.assertNotIn("moved", self.carve(self.five))
        self.assertFalse(os.path.exists(self.out + ".earlier"))
        self.assertEqual(self.streams(), before)

    def test_index_only_writes_and_moves_no_stream(self):
        self.carve(self.five)
        before = self.streams()
        self.assertNotIn("moved", self.carve(self.one, "--index-only"))
        self.assertEqual(self.streams(), before)
        self.assertEqual(self.index()["summary"]["files"], 1)

    def stop_a_carve(self, source):
        """Run the carver in this process and fail the step that publishes index.json, as a full disk would."""
        argv = ["rpf9_carve.py", source, self.out]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(rpf9_carve, "below_normal"), \
                mock.patch.object(rpf9_carve.os, "replace", side_effect=OSError(28, "No space left on device")), \
                mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as stop:
                rpf9_carve.main()
        self.assertIn("cannot write", str(stop.exception.code))
        self.assertIn("The carve is not finished", str(stop.exception.code))

    def test_a_stopped_carve_leaves_the_marker_and_no_index(self):
        self.carve(self.one)                              # an older, finished carve: its index must not survive
        self.stop_a_carve(self.five)
        self.assertTrue(os.path.isfile(os.path.join(self.out, carve_folder.UNFINISHED)))
        self.assertFalse(os.path.exists(os.path.join(self.out, "index.json")))
        self.assertEqual([n for n in os.listdir(self.out) if n.endswith(".json")], [carve_folder.UNFINISHED])
        reason = carve_folder.unfinished(self.out)
        self.assertIn("unfinished carve of " + self.five, reason)
        self.assertNotIn("\n", reason)

    def test_the_folder_indexers_refuse_a_stopped_carve(self):
        self.stop_a_carve(self.five)
        for name, args in (("page_index.py", ()), ("anim_index.py", ()), ("rbf0_index.py", ())):
            with self.subTest(tool=name):
                report = self.path(name + ".json")
                p = tool(name, self.out, report, *args)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
                self.assertIn("unfinished carve", p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                self.assertFalse(os.path.exists(report))
        # Carving again and letting it finish makes the folder usable.
        self.carve(self.five)
        for name in ("page_index.py", "anim_index.py", "rbf0_index.py"):
            with self.subTest(tool=name, finished=True):
                p = tool(name, self.out, self.path(name + ".json"))
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_a_folder_with_no_marker_is_indexed_as_before(self):
        # Streams written by hand, by a test or by an older carver have no marker and no index.json.
        os.makedirs(self.out)
        with open(os.path.join(self.out, "000000000040.bin"), "wb") as f:
            f.write(b"A" * 1000)
        self.assertIsNone(carve_folder.unfinished(self.out))
        self.assertEqual(self.indexed_streams(), 1)

    def test_an_unreadable_marker_still_counts(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, carve_folder.UNFINISHED), "w", encoding="utf-8") as f:
            f.write("not json")
        self.assertIn("unfinished carve (", carve_folder.unfinished(self.out))

    def test_a_second_move_does_not_overwrite_the_first(self):
        # Another archive whose five streams sit at the same offsets as the first one's, with other bytes.
        again = self.path("again.rpf")
        archive(again, [bytes([97 + i]) * (1000 + i) for i in range(5)])
        self.carve(self.five)
        names = self.streams()
        self.carve(self.one)                              # moves the five to <out>.earlier
        self.carve(again)                                 # moves the one stream to <out>.earlier.2
        self.assertEqual(self.streams(), names)           # same names as the first five, other contents
        out = self.carve(self.one)                        # moves those five to <out>.earlier.3
        self.assertIn(self.out + ".earlier.3", out)
        self.assertEqual(len(os.listdir(self.out + ".earlier.2")), 1)
        for folder, first_byte in ((".earlier", b"A"), (".earlier.3", b"a")):
            self.assertEqual(sorted(os.listdir(self.out + folder)), names, folder)
            with open(os.path.join(self.out + folder, names[0]), "rb") as f:
                self.assertEqual(f.read(), first_byte * 1000, folder)

    def test_a_relative_output_folder_still_moves_beside_itself(self):
        self.carve(self.five)
        names = self.streams()
        p = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "rpf9_carve.py"), self.one, "."],
                           capture_output=True, text=True, cwd=self.out)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        # Nothing but the new stream and its index in the folder: no folder of moved streams inside it.
        self.assertEqual(sorted(os.listdir(self.out)), sorted(self.streams() + ["index.json"]))
        self.assertEqual(len(self.streams()), 1)
        self.assertEqual(sorted(os.listdir(self.out + ".earlier")), names)

    def test_a_drive_root_is_refused_before_anything_is_written(self):
        root = os.path.abspath(os.sep)
        before = set(os.listdir(root))
        p = tool("rpf9_carve.py", self.five, root)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("can't be the root of a drive", p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual({n for n in os.listdir(root) if n not in before and n.endswith((".bin", ".json"))}, set())

    def test_index_only_does_not_clear_a_stopped_carve(self):
        self.stop_a_carve(self.five)
        os.remove(os.path.join(self.out, self.streams()[-1]))     # the carve stopped one stream short
        self.carve(self.five, "--index-only")
        self.assertEqual(self.index()["summary"]["files"], 5)
        self.assertTrue(os.path.isfile(os.path.join(self.out, carve_folder.UNFINISHED)))
        p = tool("page_index.py", self.out, self.path("pages.json"))
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("unfinished carve", p.stderr)

    def test_index_only_does_not_mark_a_finished_folder(self):
        self.carve(self.five)
        self.carve(self.five, "--index-only")
        self.assertIsNone(carve_folder.unfinished(self.out))
        self.assertEqual(self.indexed_streams(), 5)

    def test_a_marker_that_cannot_be_removed_is_one_line(self):
        # A finished carve whose marker is locked (antivirus, a read-only file) used to end in a traceback.
        remove = os.remove

        def locked(path, *args, **kwargs):
            if os.path.basename(path) == carve_folder.UNFINISHED:
                raise PermissionError(13, "Permission denied")
            return remove(path, *args, **kwargs)

        argv = ["rpf9_carve.py", self.five, self.out]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(rpf9_carve, "below_normal"), \
                mock.patch.object(rpf9_carve.os, "remove", locked), mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as stop:
                rpf9_carve.main()
        message = str(stop.exception.code)
        self.assertIn("the carve finished but its marker can't be removed", message)
        self.assertIn("Delete the file", message)
        self.assertNotIn("\n", message)
        self.assertEqual(self.index()["summary"]["files"], 5)

    def test_a_move_that_fails_is_one_line_and_leaves_the_carve_unfinished(self):
        self.carve(self.five)
        argv = ["rpf9_carve.py", self.one, self.out]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(rpf9_carve, "below_normal"), \
                mock.patch.object(rpf9_carve.os, "rename", side_effect=PermissionError(13, "Permission denied")), \
                mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as stop:
                rpf9_carve.main()
        self.assertIn("can't move the stream files of an earlier carve", str(stop.exception.code))
        self.assertIn("Permission denied", str(stop.exception.code))
        self.assertTrue(os.path.isfile(os.path.join(self.out, carve_folder.UNFINISHED)))
        self.assertFalse(os.path.exists(os.path.join(self.out, "index.json")))
        self.assertEqual(len(self.streams()), 6)          # nothing was lost: the five are still there


if __name__ == "__main__":
    unittest.main()
