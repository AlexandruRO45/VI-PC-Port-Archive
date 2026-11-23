"""Synthetic bounded-inflation tests for tools/psocache_extract.py. No game data."""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import psocache_extract  # noqa: E402


class _CountingDecompressor:
    def __init__(self, inner, seen_limits: list[int], total_out: list[int]):
        self._inner = inner
        self._seen_limits = seen_limits
        self._total_out = total_out

    def decompress(self, data, max_length=0):
        self._seen_limits.append(max_length)
        out = self._inner.decompress(data, max_length)
        self._total_out[0] += len(out)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestPsocacheExtractBoundedInflate(unittest.TestCase):
    def test_inflate_stops_at_max_bytes_plus_one(self):
        big = zlib.compress(b"\x00" * (1 << 20))
        seen_limits: list[int] = []
        total_out = [0]
        real_decompressobj = zlib.decompressobj

        def fake_decompressobj():
            return _CountingDecompressor(real_decompressobj(), seen_limits, total_out)

        with mock.patch.object(psocache_extract.zlib, "decompressobj", side_effect=fake_decompressobj):
            used, out = psocache_extract.inflate(big, 0, len(big), keep_all=True, max_bytes=128)

        self.assertIsNotNone(used)
        self.assertEqual(len(out), 129)
        self.assertEqual(total_out[0], 129)
        self.assertEqual(seen_limits, [129])

    def test_inflate_exact_max_bytes_returns_full_stream(self):
        payload = psocache_extract.MAGIC + b"ABCD" * 31
        comp = zlib.compress(payload)
        used, out = psocache_extract.inflate(comp, 0, len(comp), keep_all=True, max_bytes=len(payload))
        self.assertEqual(used, len(comp))
        self.assertEqual(out, payload)

    def test_assemble_rejects_overshooting_first_and_continuation_without_full_inflate(self):
        first_big = zlib.compress(psocache_extract.MAGIC + b"\x00" * (1 << 20))
        total_out = [0]
        real_decompressobj = zlib.decompressobj

        def fake_decompressobj():
            return _CountingDecompressor(real_decompressobj(), [], total_out)

        with mock.patch.object(psocache_extract.zlib, "decompressobj", side_effect=fake_decompressobj):
            with self.assertRaisesRegex(ValueError, r"more than the declared 256"):
                psocache_extract.assemble(first_big, len(first_big), 0, total=256)
        self.assertLessEqual(total_out[0], 257)

        first_ok = zlib.compress(psocache_extract.MAGIC + b"\x00" * 196)  # 200 bytes
        second_big = zlib.compress(b"\x00" * (1 << 20))
        blob = first_ok + second_big
        total_out[0] = 0
        with mock.patch.object(psocache_extract.zlib, "decompressobj", side_effect=fake_decompressobj):
            with self.assertRaisesRegex(ValueError, r"past the declared 256"):
                psocache_extract.assemble(blob, len(blob), 0, total=256)
        self.assertLessEqual(total_out[0], 257)


if __name__ == "__main__":
    unittest.main()
