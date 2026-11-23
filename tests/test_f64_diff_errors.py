"""Unexpected f64 translator failures are harness errors, including while shrinking.

Runs synthetic programs through the real reference, translator and CPU interpreters.
No SDK, GPU or game files are needed.
"""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import f64_diff as fd  # noqa: E402


class F64HarnessErrors(unittest.TestCase):
    def campaign(self, **options):
        args = ["--no-sweep", "--programs", "2", "--seed", "1", "--no-literals",
                "--ops", options.get("op", "v_cmp_eq_f64"), "--oracle", options.get("oracle", "translator")]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fd.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_unexpected_failure_after_a_match_is_not_a_skip(self):
        for target in ("translate_assembly", "run_fragment"):
            for error in (KeyError, IndexError, ValueError, OverflowError, RuntimeError):
                with self.subTest(target=target, error=error.__name__):
                    original = getattr(fd, target)
                    calls = []

                    def fail_second(*args, **kwargs):
                        calls.append(1)
                        if len(calls) == 2:
                            raise error("synthetic unexpected failure")
                        return original(*args, **kwargs)

                    with mock.patch.object(fd, target, side_effect=fail_second):
                        rc, out, err = self.campaign()
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(rc, 2, out + err)
                    self.assertIn("harness error:", err)
                    self.assertIn("synthetic unexpected failure", err)
                    self.assertNotIn(" skips", out)

    def test_invalid_module_is_a_diagnostic_harness_error(self):
        original = fd.translate_assembly
        calls = []

        def invalid_second(*args, **kwargs):
            asm = original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                # The real mini-interpreter rejects a selection merge followed by Return.
                self.assertIn("OpReturn\n", asm)
                asm = asm.replace("OpReturn\n", "OpSelectionMerge %missing_merge None\nOpReturn\n", 1)
            return asm

        with mock.patch.object(fd, "translate_assembly", side_effect=invalid_second):
            rc, out, err = self.campaign()
        self.assertEqual(len(calls), 2)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("InvalidModule", err)
        self.assertIn("OpSelectionMerge", err)

    def test_explicit_unsupported_cases_remain_skips(self):
        for target, error in (("translate_assembly", fd.rdna2spirv.Unsupported),
                              ("run_fragment", fd.UnsupportedConstruct)):
            with self.subTest(target=target):
                original = getattr(fd, target)
                calls = []

                def unsupported_second(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 2:
                        raise error("synthetic unsupported feature")
                    return original(*args, **kwargs)

                with mock.patch.object(fd, target, side_effect=unsupported_second):
                    rc, out, err = self.campaign()
                self.assertEqual(rc, 0, out + err)
                self.assertIn("1 match, 0 mismatch; 2 total", out)
                self.assertIn("synthetic unsupported feature", out)
                self.assertEqual(err, "")

    def test_binary64_backend_limits_remain_explicit_skips(self):
        original = fd.spirv_interp.run
        for error in (NotImplementedError, ZeroDivisionError, OverflowError, ValueError):
            with self.subTest(error=error.__name__):
                calls = []

                def limited_second(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 2:
                        raise error("synthetic binary64 limit")
                    return original(*args, **kwargs)

                with mock.patch.object(fd.spirv_interp, "run", side_effect=limited_second):
                    rc, out, err = self.campaign(op="v_trunc_f64")
                self.assertEqual(len(calls), 2)
                self.assertEqual(rc, 0, out + err)
                self.assertIn("1 match, 0 mismatch; 2 total", out)
                self.assertIn("binary64 SPIR-V interpreter limitation", out)

    def test_shrinking_does_not_hide_a_harness_failure(self):
        original_assembly, original_fragment = fd.translate_assembly, fd.run_fragment
        for error in (KeyError, fd.HarnessError):
            with self.subTest(error=error.__name__):
                shrinking = []

                def fail_shrink(lines, name, tmp):
                    if name == "ps_f64_shrink":
                        shrinking.append(name)
                        raise error("synthetic shrink failure")
                    return original_assembly(lines, name, tmp)

                def mismatch(asm):
                    out = original_fragment(asm)
                    words = out["%out_mrt0"]
                    out["%out_mrt0"] = (words[0] ^ 1, *words[1:])
                    return out

                with (mock.patch.object(fd, "translate_assembly", side_effect=fail_shrink),
                      mock.patch.object(fd, "run_fragment", side_effect=mismatch)):
                    rc, out, err = self.campaign()
                self.assertTrue(shrinking)
                self.assertEqual(rc, 2, out + err)
                self.assertIn("synthetic shrink failure", err)
                self.assertNotIn("minimal program:", out)

    def test_successful_campaign_still_compares(self):
        rc, out, err = self.campaign()
        self.assertEqual(rc, 0, out + err)
        self.assertIn("2 match, 0 mismatch; 2 total", out)
        self.assertEqual(err, "")

    def test_gpu_unavailable_is_distinct_from_a_comparison_failure(self):
        for error, status in ((fd.GPUUnavailable, 77), (fd.HarnessError, 2), (KeyError, 2)):
            with self.subTest(error=error.__name__):
                calls = []

                def gpu(lines, name, tmp, context):
                    calls.append(1)
                    if len(calls) == 2:
                        raise error("synthetic GPU failure")
                    return fd.translate(lines, name, tmp)

                with (mock.patch.object(fd, "prepare_gpu", return_value={}),
                      mock.patch.object(fd, "run_gpu", side_effect=gpu)):
                    rc, out, err = self.campaign(oracle="gpu")
                self.assertEqual(len(calls), 2)
                self.assertEqual(rc, status, out + err)
                self.assertIn("synthetic GPU failure", err)
                self.assertIn("GPU oracle unavailable:" if status == 77 else "harness error:", err)


if __name__ == "__main__":
    unittest.main()
