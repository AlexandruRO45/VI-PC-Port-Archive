"""Bound PNG-cache accounting work without changing world texture output or eviction.

All DDS files are generated here and decoded by the real standard-library reader.
"""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import dds_png  # noqa: E402
import test_world_build as fixture  # noqa: E402
import world_build  # noqa: E402


class PngCacheBudgetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        world_build._PNG_CACHE.clear()
        self.addCleanup(world_build._PNG_CACHE.clear)
        self.real_decode = dds_png.dds_file_to_png_bytes

    def texture(self, name, size=4, pixels=None):
        pixels = pixels if pixels is not None else bytes((25, 90, 175, 255)) * (size * size)
        path = self.root / name
        path.write_bytes(fixture.make_bgra8_dds(size, size, pixels))
        expected = self.real_decode(str(path), max_size=512)
        self.assertEqual(fixture.parse_and_validate_png(expected), (size, size, pixels))
        return str(path), expected

    def cached(self, path, build="synthetic-build"):
        return world_build._png_for(build, path, 512)

    def test_accounting_work_is_linear_in_new_textures(self):
        path, expected = self.texture("source.dds")
        source = Path(path).read_bytes()
        paths = []
        for i in range(128):
            target = self.root / f"{i:03d}.dds"
            target.write_bytes(source)
            paths.append(str(target))

        length_requests = 0

        class MeasuredPng(bytes):
            def __len__(self):
                nonlocal length_requests
                length_requests += 1
                return super().__len__()

        def decode(*args, **kwargs):
            # Only instrument the length of a real decoded PNG; pixels and decoder behavior are unchanged.
            return MeasuredPng(self.real_decode(*args, **kwargs))

        with mock.patch.object(dds_png, "dds_file_to_png_bytes", side_effect=decode) as decoder:
            images = [self.cached(path) for path in paths]
            for path in paths:
                self.cached(path)
        self.assertEqual(decoder.call_count, len(paths))
        self.assertTrue(all(image == expected for image in images))
        self.assertLessEqual(length_requests, 4 * len(paths),
                             "budget accounting repeatedly walked previously decoded PNGs")

    def test_exact_budget_and_hits_preserve_fifo_order(self):
        a, expected = self.texture("a.dds")
        b, _ = self.texture("b.dds")
        c, _ = self.texture("c.dds")
        with mock.patch.object(world_build, "_PNG_CACHE_BUDGET", 2 * len(expected)), \
                mock.patch.object(dds_png, "dds_file_to_png_bytes", wraps=self.real_decode) as decoder:
            for path in (a, b, a, c, b):
                self.assertEqual(self.cached(path), expected)
            # A hit does not promote a: c must evict a and keep b.
            self.assertEqual([call.args[0] for call in decoder.call_args_list], [a, b, c])
            self.assertEqual(self.cached(a), expected)
            self.assertEqual([call.args[0] for call in decoder.call_args_list], [a, b, c, a])

    def test_one_oversized_png_survives_until_the_next_entry(self):
        a, small = self.texture("a.dds")
        b, _ = self.texture("b.dds")
        c, _ = self.texture("c.dds")
        rng = random.Random(1739)
        pixels = bytes(rng.randrange(256) for _ in range(32 * 32 * 4))
        large_path, large = self.texture("large.dds", 32, pixels)
        self.assertGreater(len(large), 2 * len(small))
        for budget in (0, 2 * len(small)):
            with self.subTest(budget=budget), \
                    mock.patch.object(world_build, "_PNG_CACHE_BUDGET", budget), \
                    mock.patch.object(dds_png, "dds_file_to_png_bytes", wraps=self.real_decode) as decoder:
                world_build._PNG_CACHE.clear()
                for path, expected in ((a, small), (b, small), (large_path, large),
                                       (large_path, large), (c, small), (large_path, large)):
                    self.assertEqual(self.cached(path), expected)
                self.assertEqual([call.args[0] for call in decoder.call_args_list],
                                 [a, b, large_path, c, large_path])

    def test_failed_decodes_cost_no_png_bytes_and_remain_cached(self):
        good, expected = self.texture("good.dds")
        missing = str(self.root / "missing.dds")
        malformed = str(self.root / "malformed.dds")
        Path(malformed).write_bytes(b"synthetic unsupported DDS")
        with mock.patch.object(world_build, "_PNG_CACHE_BUDGET", len(expected)), \
                mock.patch.object(dds_png, "dds_file_to_png_bytes", wraps=self.real_decode) as decoder:
            self.assertIsNone(self.cached(missing))
            self.assertIsNone(self.cached(malformed))
            self.assertEqual(self.cached(good), expected)
            for path in (missing, malformed):
                Path(path).write_bytes(Path(good).read_bytes())
                self.assertIsNone(self.cached(path))
            self.assertEqual(self.cached(good), expected)
            self.assertEqual(decoder.call_count, 3)
            # A distinct build still gets a fresh decode, as with successful cached textures.
            self.assertEqual(self.cached(missing, build="next-build"), expected)
            self.assertEqual(decoder.call_count, 4)

    def test_clearing_the_cache_restores_its_full_budget(self):
        a, expected = self.texture("a.dds")
        b, _ = self.texture("b.dds")
        with mock.patch.object(world_build, "_PNG_CACHE_BUDGET", 2 * len(expected)), \
                mock.patch.object(dds_png, "dds_file_to_png_bytes", wraps=self.real_decode) as decoder:
            for path in (a, b):
                self.assertEqual(self.cached(path), expected)
            world_build._PNG_CACHE.clear()
            for path in (a, b, a, b):
                self.assertEqual(self.cached(path), expected)
            self.assertEqual([call.args[0] for call in decoder.call_args_list], [a, b, a, b])


if __name__ == "__main__":
    unittest.main()
