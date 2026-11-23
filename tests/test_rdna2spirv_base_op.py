from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402


class TestBaseOpCache(unittest.TestCase):
    def setUp(self):
        self._saved_cache = dict(rdna2spirv._BASE_OP_CACHE)
        rdna2spirv._BASE_OP_CACHE.clear()

    def tearDown(self):
        rdna2spirv._BASE_OP_CACHE.clear()
        rdna2spirv._BASE_OP_CACHE.update(self._saved_cache)

    def test_suffix_stripping(self):
        cases = [
            ("v_add_f32_e32", "v_add_f32"),
            ("v_add_f32_e64", "v_add_f32"),
            ("v_mov_b32_sdwa", "v_mov_b32"),
            ("v_mov_b32_dpp", "v_mov_b32"),
            ("v_mov_b32_dpp8", "v_mov_b32_dpp8"),
            ("v_add_f32", "v_add_f32"),
            ("s_mov_b32", "s_mov_b32"),
            ("v_foo_e32_e64", "v_foo_e32"),
            ("", ""),
        ]
        # Run twice so both cold (uncached) and warm (cached) paths are checked.
        for _ in range(2):
            for op, expected in cases:
                self.assertEqual(rdna2spirv.base_op(op), expected)

    def test_cache_is_bounded(self):
        limit = rdna2spirv._BASE_OP_CACHE_LIMIT
        for i in range(limit + 500):
            op = f"v_probe_op_{i}_e32"
            self.assertEqual(rdna2spirv.base_op(op), f"v_probe_op_{i}")
        self.assertEqual(len(rdna2spirv._BASE_OP_CACHE), limit)
        # Uncached opcodes beyond the cap still strip suffixes accurately without growing the cache.
        self.assertEqual(rdna2spirv.base_op("v_overflow_test_sdwa"), "v_overflow_test")
        self.assertEqual(rdna2spirv.base_op("v_overflow_test_dpp8"), "v_overflow_test_dpp8")
        self.assertLessEqual(len(rdna2spirv._BASE_OP_CACHE), limit)


if __name__ == "__main__":
    unittest.main()
