"""Read back OBJ primitives from a synthetic archive, independently of the writer."""
import contextlib
import io
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_navmesh_decode import build_navmesh, make_rpf, nd

TOOLS = HERE.parent / "tools"
TOOL = TOOLS / "navmesh_tile_obj.py"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import file_mode  # noqa: E402
import navmesh_tile_obj  # noqa: E402


def read_obj(path):
    vertices = []
    groups = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        op, *args = line.split()
        if op == "v":
            vertices.append(tuple(map(float, args)))
        elif op == "g":
            name, = args
            assert name not in groups
            groups[name] = []
        elif op in ("p", "l"):
            refs = list(map(int, args))
            assert all(1 <= i <= len(vertices) for i in refs)
            groups[name].append((op, [vertices[i - 1] for i in refs]))
        elif op not in ("#", "o"):
            raise AssertionError(op)
    return groups


class ExportTest(unittest.TestCase):
    def test_archive_selection_and_record_round_trip(self):
        sizes = (3, 1, 0, 2, 4, 1)
        head, tail = build_navmesh(n_verts=3, poly_sizes=sizes, tiny_bm_cont=True)
        head = bytearray(head)
        (at, _), = nd.split_array(head, 0x70, 6, 3)
        struct.pack_into("<9H", head, at, 0, 0, 0, 65535, 0, 0, 0, 65535, 65535)
        a, b, c = (-768, 512, 0), (-640, 512, 0), (-768, 640, 16)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "input.rpf"
            out = Path(directory) / "tile.obj"
            make_rpf(archive, [*build_navmesh(flags=5), head, tail, *build_navmesh(corrupt_vidx=True)])

            def run(tile, dest):
                return subprocess.run([sys.executable, "-I", TOOL, archive, "--tile", str(tile),
                                       "--out", dest], capture_output=True)

            result = run(1, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = [[("l", [a, b, c, a])], [("p", [a])], [], [("l", [b, c])],
                        [("l", [a, b, c, a, a])], [("p", [b])]]
            groups = read_obj(out)
            self.assertEqual(list(groups), [f"polygon_{i}" for i in range(len(sizes))])
            self.assertEqual(list(groups.values()), expected)
            original = out.read_bytes()
            for invalid in (-1, 0, 2, 3):
                self.assertEqual(run(invalid, out).returncode, 2)
                self.assertEqual(out.read_bytes(), original)
            self.assertEqual(run(1, archive).returncode, 2)


    def test_preflight_refuses_bad_out_before_scanning_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.rpf"
            archive.write_bytes(b"")
            blocker = root / "not_a_dir"
            blocker.write_bytes(b"x")
            with mock.patch.object(navmesh_tile_obj, "navmesh_resources",
                                   side_effect=AssertionError("archive should not be scanned")):
                for bad_out in (root, blocker / "sub" / "tile.obj"):
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err):
                        rc = navmesh_tile_obj.main([str(archive), "--tile", "0", "--out", str(bad_out)])
                    self.assertEqual(rc, 2)
                    self.assertIn("ERROR:", err.getvalue())

    def test_publish_preserves_permissions_and_cleans_up_on_replace_failure(self):
        head, tail = build_navmesh(n_verts=3, poly_sizes=(3,), tiny_bm_cont=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "input.rpf"
            out = root / "tile.obj"
            make_rpf(archive, [head, tail])

            expected_new = file_mode.new_file_mode(root)
            self.assertEqual(navmesh_tile_obj.main([str(archive), "--tile", "0", "--out", str(out)]), 0)
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), expected_new)
                os.chmod(out, 0o640)
                self.assertEqual(navmesh_tile_obj.main([str(archive), "--tile", "0", "--out", str(out)]), 0)
                self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o640)

            before = out.read_bytes()
            err = io.StringIO()
            with mock.patch.object(navmesh_tile_obj, "replace_retry", side_effect=OSError("busy")):
                with contextlib.redirect_stderr(err):
                    rc = navmesh_tile_obj.main([str(archive), "--tile", "0", "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertIn("busy", err.getvalue())
            self.assertEqual(out.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["input.rpf", "tile.obj"])


if __name__ == "__main__":
    unittest.main()
