"""tools/translator_modes.py: one order for a device-dependent translator mode at every entry point (the command-line
flag, then the environment variable, then the device). Run: python -I tests/test_translator_modes.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import translator_modes as tm  # noqa: E402

MOLTENVK, RADV = {"driver_id": 14}, {"driver_id": 3}


def no_device():
    raise AssertionError("the device was asked although the flag or the variable decides")


class Resolve(unittest.TestCase):
    def test_flag_wins_over_everything(self):
        for env in ({}, {"LEONIDA_SUBGROUP_TEMPS": "0"}, {"LEONIDA_SUBGROUP_TEMPS": "1"}):
            with self.subTest(env=env):
                self.assertTrue(tm.resolve("subgroup_temps", cli=True, environ=env, info=no_device))

    def test_variable_wins_over_the_device(self):
        for word in ("1", "true", " YES ", "on"):
            self.assertTrue(tm.resolve("subgroup_temps", environ={"LEONIDA_SUBGROUP_TEMPS": word}, info=no_device))
        for word in ("0", "false", "No", "off "):
            self.assertFalse(tm.resolve("subgroup_temps", environ={"LEONIDA_SUBGROUP_TEMPS": word}, info=no_device))

    def test_the_device_decides_last(self):
        self.assertTrue(tm.resolve("subgroup_temps", environ={}, info=MOLTENVK))
        self.assertFalse(tm.resolve("subgroup_temps", environ={}, info=RADV))
        self.assertFalse(tm.resolve("subgroup_temps", environ={}, info={}))      # can't tell: off
        self.assertFalse(tm.resolve("subgroup_temps", environ={}, info=lambda: None))
        self.assertTrue(tm.resolve("subgroup_temps", environ={}, info=lambda: MOLTENVK))

    def test_an_unknown_word_is_unset(self):
        for word in ("", "2", "maybe"):
            with self.subTest(word=word):
                self.assertIsNone(tm.env_choice("LEONIDA_SUBGROUP_TEMPS", {"LEONIDA_SUBGROUP_TEMPS": word}))
                self.assertTrue(tm.resolve("subgroup_temps", environ={"LEONIDA_SUBGROUP_TEMPS": word},
                                           info=MOLTENVK))

    def test_flag_name(self):
        self.assertEqual(tm.flag("subgroup_temps"), "--subgroup-temps")


if __name__ == "__main__":
    unittest.main()
