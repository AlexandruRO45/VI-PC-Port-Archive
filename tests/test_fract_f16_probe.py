"""tools/fract_f16_probe.py: the measured v_fract_f16 rule (SHADER-338) against hand-worked words, and the translator
against the rule on one block of inputs. No GPU, no game files.

    python -I tests/test_fract_f16_probe.py -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fract_f16_probe as probe  # noqa: E402


class RuleTests(unittest.TestCase):
    def test_hand_worked_rows(self):
        for x, mod, ieee, flush, want in probe.ROWS:
            with self.subTest(x=hex(x), mod=mod, ieee=ieee, flush=flush):
                self.assertEqual(probe.rule(x, mod, ieee, flush), want)

    def test_rounded_fract_then_modifier(self):
        # The modifier scales the rounded f16 fract (0x3bff), not the exact 1 - 2^-15, which would round to 2.0 / 4.0.
        for x in (0x8400, 0x8401, 0x87FF, 0x9000):
            self.assertEqual(probe.rule(x, "", 0, True), 0x3BFF)
            self.assertEqual(probe.rule(x, "mul:2", 0, True), 0x3FFF)
            self.assertEqual(probe.rule(x, "mul:4", 0, True), 0x43FF)
            self.assertEqual(probe.rule(x, "div:2", 0, True), 0x37FF)

    def test_modifier_ignored_outside_ieee0_flushed(self):
        for ieee, flush in ((0, False), (1, False), (1, True)):
            for x in range(0, 0x10000, 97):
                for mod in ("mul:2", "mul:4", "div:2"):
                    self.assertEqual(probe.rule(x, mod, ieee, flush), probe.rule(x, "", ieee, flush))
                    self.assertEqual(probe.rule(x, "clamp " + mod, ieee, flush), probe.rule(x, "clamp", ieee, flush))

    def test_results_are_never_negative_or_above_one_under_clamp(self):
        for x in range(0, 0x10000, 7):
            for mod in probe.MODS:
                h = probe.rule(x, mod, 0, True)
                if "clamp" in mod:
                    self.assertLessEqual(h, 0x3C00)
                elif not probe.is_nan(h):
                    self.assertEqual(h & 0x8000, 0, f"{x:04x} {mod}")

    def test_sample_covers_the_edges(self):
        for x in (0x8000, 0x8001, 0x83FF, 0x8400, 0x87FF, 0x07FF, 0x0800, 0x7C00, 0x7C01, 0xFC00, 0xFC01):
            self.assertIn(x, probe.SAMPLE)


class TranslatorTests(unittest.TestCase):
    def test_translator_follows_the_rule_on_the_tiny_negatives(self):
        # One workgroup: -0 and the negative denormals (0x8000..0x83ff), whose fract rounds to 1.0 (0x3bff).
        inputs = set(range(0x8000, 0x8000 + probe.BLOCK))
        words = probe.run_translator(inputs, mods=["", "mul:2", "div:2", "clamp mul:4"])
        self.assertEqual(probe.check_translator(words), [])
        self.assertEqual(words["mul:2"][0x8001], probe.HIGH | 0x3BFF)


if __name__ == "__main__":
    unittest.main()
