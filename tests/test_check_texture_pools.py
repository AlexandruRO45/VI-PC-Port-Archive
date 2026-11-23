"""Synthetic DDS and manifest checks for the read-only texture pool audit."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dds_subresources import dx10  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_texture_pools as checker  # noqa: E402
from dds_png import DdsError, decode_dds, minimum_dds_size  # noqa: E402


def write_manifest(pool, files):
    pool.mkdir(parents=True, exist_ok=True)
    document = {"Count": len(files), "Textures": [{"File": name} for name in files]}
    (pool / "manifest.json").write_text(json.dumps(document), encoding="utf-8", newline="\n")


class DdsSizeTests(unittest.TestCase):
    def test_checks_every_mip_array_layer_cube_face_and_volume_slice(self):
        # RGBA8: 2x2 + 1x1 uses 20 bytes per array image. A 4x2x5 A8 volume uses 40+4+1 bytes.
        for size, kwargs in ((40, {"mips": 2, "layers": 2}),
                             (240, {"mips": 2, "layers": 2, "cube": True}),
                             (45, {"mips": 3, "dimension": 4, "depth": 5, "fmt": 65})):
            with self.subTest(**kwargs):
                dims = (4, 2) if kwargs.get("dimension") == 4 else (2, 2)
                self.assertEqual(minimum_dds_size(dx10(*dims, b"", **kwargs)), 148 + size)

    def test_size_check_covers_formats_without_a_png_decoder(self):
        for fmt, size in ((11, 240), (61, 30), (95, 64), (96, 64)):
            with self.subTest(fmt=fmt):
                data = dx10(5, 6, bytes(size), fmt=fmt)
                self.assertEqual(minimum_dds_size(data[:148]), len(data))
                with self.assertRaises(DdsError):
                    decode_dds(data)

    def test_size_check_rejects_invalid_header_sizes(self):
        for offset in (4, 76):
            with self.subTest(offset=offset):
                header = bytearray(dx10(4, 4, b""))
                struct.pack_into("<I", header, offset, 0)
                with self.assertRaisesRegex(DdsError, "structure size"):
                    minimum_dds_size(header)


class TexturePoolCheckTests(unittest.TestCase):
    def test_audit_reports_bad_outputs_and_preserves_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loose = root / "ttpf" / "loose"
            pack = root / "ttpf_pack" / "pack" / "packed"
            files = {
                "valid.dds": dx10(4, 4, bytes(64)),
                "invalid.dds": b"x",
                "truncated.dds": dx10(2, 2, bytes(39), layers=2, mips=2),
                "unsupported.dds": dx10(4, 4, bytes(64), fmt=999),
            }
            write_manifest(loose, [*files, "missing.dds", "directory.dds", None])
            write_manifest(pack, [None])
            for name, data in files.items():
                (loose / name).write_bytes(data)
            (loose / "directory.dds").mkdir()
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

            result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_texture_pools.py"), str(root)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Totals: pools=2, valid=1, not_exported=2", result.stdout)
            self.assertIn("missing=1, unreadable=1, invalid_dds=1, truncated=1, unsupported=1", result.stdout)
            self.assertEqual({path: path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)

    def test_single_pool_and_intentional_skips_exit_successfully(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            write_manifest(pool, ["valid.DDS", None])
            (pool / "valid.DDS").write_bytes(dx10(5, 6, bytes(64), fmt=95))
            self.assertEqual(checker.main([str(pool)]), 0)

    def test_invalid_manifest_is_reported_without_reading_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp)
            for names in (["../elsewhere.dds"], ["..\\elsewhere.dds"], ["other:elsewhere.dds"], [123]):
                with self.subTest(names=names):
                    write_manifest(pool, names)
                    self.assertEqual(checker.check_pool(pool / "manifest.json"), {"invalid_manifest": 1})
            (pool / "manifest.json").write_bytes(b"{")
            self.assertEqual(checker.check_pool(pool / "manifest.json"), {"invalid_manifest": 1})

    def test_missing_pool_reports_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "check_texture_pools.py"), tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("no texture pool manifests found", result.stderr)


if __name__ == "__main__":
    unittest.main()
