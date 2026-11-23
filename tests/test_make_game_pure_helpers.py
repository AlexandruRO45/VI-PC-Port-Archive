"""Unit tests for pure helper and validation functions in make_game.py.

    python -I tests/test_make_game_pure_helpers.py

Deterministic and standard library only; no game files, GPU, or build tools are needed.
"""
import argparse
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game


class PositiveIntTest(unittest.TestCase):
    """Tests for make_game.positive_int CLI validation."""

    def test_valid_positive_integers(self):
        self.assertEqual(make_game.positive_int("1"), 1)
        self.assertEqual(make_game.positive_int("8"), 8)
        self.assertEqual(make_game.positive_int("64"), 64)

    def test_rejects_zero_and_negative(self):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            make_game.positive_int("0")
        self.assertIn("must be at least 1", str(cm.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            make_game.positive_int("-1")
        self.assertIn("must be at least 1", str(cm.exception))

    def test_rejects_non_integer_strings(self):
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            make_game.positive_int("not_a_number")
        self.assertIn("invalid int value", str(cm.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            make_game.positive_int("1.5")
        self.assertIn("invalid int value", str(cm.exception))


class ExitCodeAndInterruptTest(unittest.TestCase):
    """Tests for make_game.interrupted and exit_label formatting."""

    def test_interrupted_signals_and_win32_status(self):
        # POSIX signals for SIGINT and SIGTERM
        self.assertTrue(make_game.interrupted(-2))
        self.assertTrue(make_game.interrupted(-15))
        # Windows STATUS_CONTROL_C_EXIT (unsigned and sign-extended)
        self.assertTrue(make_game.interrupted(0xC000013A))
        self.assertTrue(make_game.interrupted(-1073741510))

    def test_not_interrupted_on_other_codes(self):
        self.assertFalse(make_game.interrupted(0))
        self.assertFalse(make_game.interrupted(1))
        self.assertFalse(make_game.interrupted(2))
        self.assertFalse(make_game.interrupted(-9))
        self.assertFalse(make_game.interrupted(0xC0000005))
        self.assertFalse(make_game.interrupted(0x80000003))

    def test_exit_label_signals(self):
        self.assertEqual(make_game.exit_label(-2), "signal 2")
        self.assertEqual(make_game.exit_label(-15), "signal 15")

    def test_exit_label_standard_codes(self):
        self.assertEqual(make_game.exit_label(0), "0")
        self.assertEqual(make_game.exit_label(1), "1")
        self.assertEqual(make_game.exit_label(255), "255")

    def test_exit_label_large_and_hex_codes(self):
        self.assertEqual(make_game.exit_label(256), "256 (0x00000100)")
        self.assertEqual(make_game.exit_label(0x80000003), "2147483651 (0x80000003)")


class PathIsWithinTest(unittest.TestCase):
    """Tests for make_game.path_is_within directory confinement check."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.realpath(self.tmp.name)
        self.base = os.path.join(self.root, "base")
        self.other = os.path.join(self.root, "other")
        os.makedirs(self.base, exist_ok=True)
        os.makedirs(self.other, exist_ok=True)

    def test_path_inside_directory(self):
        child = os.path.join(self.base, "child", "file.txt")
        self.assertTrue(make_game.path_is_within(child, self.base))
        self.assertTrue(make_game.path_is_within(self.base, self.base))

    def test_path_outside_directory(self):
        self.assertFalse(make_game.path_is_within(self.other, self.base))
        self.assertFalse(make_game.path_is_within(self.root, self.base))
        outside_norm = os.path.normpath(os.path.join(self.base, "..", "other"))
        self.assertFalse(make_game.path_is_within(outside_norm, self.base))

    def test_path_different_drive_or_disjoint_returns_false(self):
        # Cross-drive or disjoint paths return False without raising ValueError
        self.assertFalse(make_game.path_is_within("D:\\different\\drive", "C:\\base\\dir"))


class ContainerMessageTest(unittest.TestCase):
    """Tests for make_game.container_message formatting."""

    def test_encrypted_container_message(self):
        hdr = {"kind": "XVD", "version": "1.0", "encrypted": True}
        msg = make_game.container_message("package.xvd", hdr)
        self.assertIn("package.xvd is a packaged XVD container (package version 1.0)", msg)
        self.assertIn("Its files are encrypted inside it", msg)
        self.assertIn("MicrosoftGame.Config", msg)

    def test_unencrypted_container_message(self):
        hdr = {"kind": "MSIXVC", "version": "2.0", "encrypted": False}
        msg = make_game.container_message("game.msixvc", hdr)
        self.assertIn("game.msixvc is a packaged MSIXVC container (package version 2.0)", msg)
        self.assertIn("Its files are packed inside it", msg)
        self.assertIn("MicrosoftGame.Config", msg)


class FmtDurationTest(unittest.TestCase):
    """Tests for make_game.fmt_duration progress timing formatting."""

    def test_seconds_formatting(self):
        self.assertEqual(make_game.fmt_duration(-1), "0s")
        self.assertEqual(make_game.fmt_duration(0), "0s")
        self.assertEqual(make_game.fmt_duration(42.4), "42s")
        self.assertEqual(make_game.fmt_duration(59), "59s")

    def test_minutes_formatting(self):
        self.assertEqual(make_game.fmt_duration(60), "1m")
        self.assertEqual(make_game.fmt_duration(75), "1m 15s")
        self.assertEqual(make_game.fmt_duration(3599), "59m 59s")

    def test_hours_formatting(self):
        self.assertEqual(make_game.fmt_duration(3600), "1h 00m")
        self.assertEqual(make_game.fmt_duration(3665), "1h 01m")
        self.assertEqual(make_game.fmt_duration(7320), "2h 02m")


if __name__ == "__main__":
    unittest.main()
