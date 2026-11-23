"""The three base_op() caches answer exactly what the plain suffix loop answered.

    python -I tests/test_base_op_cache.py

base_op() is memoised in rdna2spirv.py, shader_meta.py and lane_masks.py (three copies of the same four-suffix
strip). The cache is the only thing the change adds, so it is what a regression can break: a wrong hit, a stale entry
after the table is cleared, or a table that grows past its limit. Every case below is checked against a reference
implementation of the uncached strip, and the copy in each module is checked separately.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
import shader_meta  # noqa: E402

MODULES = (rdna2spirv, shader_meta, lane_masks)
SUFFIXES = ("_e32", "_e64", "_sdwa", "_dpp")


def reference(op):
    """What the strip did before it was memoised, written out the same way."""
    for suffix in SUFFIXES:
        if op.endswith(suffix):
            return op[: -len(suffix)]
    return op


class BaseOpCacheTest(unittest.TestCase):
    CASES = ("v_add_f32_e32", "v_add_f32_e64", "v_add_f32_sdwa", "v_add_f32_dpp", "v_add_f32",
             "s_endpgm", "s_cbranch_scc1", "ds_read_b32", "", "_e32", "v_add_f32_e32_dpp", "exp")

    def setUp(self):
        for module in MODULES:
            module._BASE_OP_CACHE.clear()

    def test_every_copy_matches_the_uncached_strip(self):
        for module in MODULES:
            for op in self.CASES:
                with self.subTest(module=module.__name__, op=op):
                    self.assertEqual(module.base_op(op), reference(op))

    def test_a_repeated_call_is_a_cache_hit_with_the_same_answer(self):
        for module in MODULES:
            for op in self.CASES:
                first = module.base_op(op)
                second = module.base_op(op)
                self.assertEqual(first, second, (module.__name__, op))
                self.assertEqual(first, reference(op), (module.__name__, op))
                self.assertIn(op, module._BASE_OP_CACHE, (module.__name__, op))

    def test_the_table_clears_at_its_limit_and_still_answers_correctly(self):
        for module in MODULES:
            original = module._BASE_OP_CACHE_LIMIT
            module._BASE_OP_CACHE_LIMIT = 3
            try:
                for op in self.CASES:
                    self.assertEqual(module.base_op(op), reference(op), (module.__name__, op))
                    self.assertLessEqual(len(module._BASE_OP_CACHE), 3, (module.__name__, op))
                # The clear path ran at least once, and a lookup after it is still right.
                self.assertEqual(module.base_op("v_add_f32_e32"), "v_add_f32", module.__name__)
            finally:
                module._BASE_OP_CACHE_LIMIT = original

    def test_each_copy_has_its_own_table(self):
        rdna2spirv.base_op("v_mul_f32_e32")
        self.assertNotIn("v_mul_f32_e32", shader_meta._BASE_OP_CACHE)
        self.assertNotIn("v_mul_f32_e32", lane_masks._BASE_OP_CACHE)


if __name__ == "__main__":
    unittest.main()
