"""Read back OBJ primitives from a synthetic archive, independently of the writer."""
import contextlib
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
import file_mode  # noqa: E402
import navmesh_tile_obj as nto  # noqa: E402
from test_navmesh_decode import build_navmesh, make_rpf, nd  # noqa: E402

TOOL = HERE.parent / "tools" / "navmesh_tile_obj.py"
SAMPLE_GEOMETRY = {"polygons": [([(0, 0, 0), (1, 0, 0), (0, 1, 0)], None, None)]}


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

    def test_directory_out_is_rejected_before_scanning_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "input.rpf"
            archive.write_bytes(b"")
            out_dir = Path(directory) / "existing_dir"
            out_dir.mkdir()
            stderr = io.StringIO()
            with mock.patch.object(nto, "navmesh_resources",
                                   side_effect=AssertionError("archive should not be scanned")), \
                    contextlib.redirect_stderr(stderr):
                rc = nto.main([str(archive), "--tile", "0", "--out", str(out_dir)])
            self.assertEqual(rc, 2)
            self.assertIn("directory", stderr.getvalue())
            self.assertEqual(list(out_dir.iterdir()), [])

    def test_publish_applies_published_mode_without_flipping_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tile.obj"
            chmod_calls = []
            real_chmod = os.chmod

            def record_chmod(path, mode):
                chmod_calls.append((Path(path), mode))
                return real_chmod(path, mode)

            with mock.patch.object(nto, "published_mode", wraps=nto.published_mode) as pub, \
                    mock.patch.object(nto.os, "chmod", side_effect=record_chmod), \
                    mock.patch("os.umask", side_effect=AssertionError("os.umask must not be called")):
                nto.publish_obj(out, SAMPLE_GEOMETRY, 0)
            self.assertEqual(pub.call_count, 1)
            self.assertEqual(pub.call_args.args, (out,))
            self.assertEqual(len(chmod_calls), 1)
            staged_path, staged_mode = chmod_calls[0]
            self.assertNotEqual(staged_path, out)
            self.assertEqual(staged_path.parent, out.parent)
            self.assertEqual(staged_mode, nto.published_mode(out))
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["tile.obj"])

    def test_failed_replace_cleans_read_only_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tile.obj"
            out.write_text("existing\n", encoding="utf-8", newline="\n")
            os.chmod(out, 0o444)
            self.addCleanup(lambda: out.exists() and os.chmod(out, 0o600))
            with mock.patch.object(nto.os, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    nto.publish_obj(out, SAMPLE_GEOMETRY, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["tile.obj"])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_new_output_gets_bits_of_a_plain_open(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tile.obj"
            nto.publish_obj(out, SAMPLE_GEOMETRY, 0)
            self.assertEqual(os.stat(out).st_mode & 0o777, file_mode.new_file_mode(directory))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_replaced_output_keeps_its_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tile.obj"
            out.write_text("old\n", encoding="utf-8", newline="\n")
            os.chmod(out, 0o640)
            nto.publish_obj(out, SAMPLE_GEOMETRY, 0)
            self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)

    def test_out_under_regular_file_is_rejected_before_scanning_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "input.rpf"
            archive.write_bytes(b"")
            blocker = Path(directory) / "blocker.txt"
            blocker.write_text("keep\n", encoding="utf-8", newline="\n")
            stderr = io.StringIO()
            with mock.patch.object(nto, "navmesh_resources",
                                   side_effect=AssertionError("archive should not be scanned")), \
                    contextlib.redirect_stderr(stderr):
                rc = nto.main([str(archive), "--tile", "0", "--out", str(blocker / "sub" / "tile.obj")])
            self.assertEqual(rc, 2)
            self.assertIn("not a directory", stderr.getvalue())
            self.assertEqual(blocker.read_text(encoding="utf-8"), "keep\n")

    def test_transient_windows_sharing_refusal_retries_and_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "tile.obj"
            attempts = []
            real_replace = os.replace

            def flaky_replace(src, dst):
                attempts.append((src, dst))
                if len(attempts) == 1:
                    err = PermissionError("sharing violation")
                    err.winerror = 32
                    raise err
                return real_replace(src, dst)

            with mock.patch("replace_retry.os.replace", side_effect=flaky_replace), \
                    mock.patch("replace_retry.time.sleep"):
                nto.publish_obj(out, SAMPLE_GEOMETRY, 0)
            self.assertEqual(len(attempts), 2)
            self.assertTrue(out.is_file())
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ["tile.obj"])


if __name__ == "__main__":
    unittest.main()
