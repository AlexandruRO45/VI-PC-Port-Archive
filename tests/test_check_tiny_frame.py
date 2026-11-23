"""The verdict logic of runtime/test/check_tiny_frame.py on synthetic output: no loader, GPU or game files."""
import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_tiny_frame", ROOT / "runtime" / "test" / "check_tiny_frame.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def good_output(extra_trace=(), drop=None, binding="shim"):
    """What a passing run prints: the binding report, the numbered trace and the guest's verdict."""
    calls = [call for call in list(check.EXPECTED) if call != drop]
    calls[10:10] = extra_trace  # between the views and the list's commands
    lines = ["tiny_frame.exe  base=0000000140000000 size=61440 entry=+0x1000  shim=7 host=40 module=0 unimplemented=0",
             f"    {binding:<13} d3d12_xs.dll!D3D12XboxCreateDevice",
             "running tiny_frame.exe"]
    lines += [f"[d3d12-trace] #{number} {call}" for number, call in enumerate(calls, 1)]
    lines += ["tiny_frame: PASS all 32 pixels are the clear colour (64, 128, 191, 255)", "tiny_frame: OK",
              "game exited with code 0", ""]
    return "\n".join(lines)


class CheckOutputTest(unittest.TestCase):
    def test_a_passing_run_has_no_problems(self):
        self.assertEqual(check.check_output(0, good_output()), [])

    def test_crlf_output_and_calls_between_the_expected_ones_are_accepted(self):
        output = good_output(extra_trace=["XboxID3D12CommandQueue::GetDesc called"]).replace("\n", "\r\n")
        self.assertEqual(check.check_output(0, output), [])

    def test_nonzero_exit_code(self):
        self.assertIn("exit code 3", check.check_output(3, good_output()))

    def test_missing_verdict_and_failed_check(self):
        output = good_output().replace("tiny_frame: OK", "tiny_frame: FAIL Signal\ntiny_frame: FAILED")
        problems = check.check_output(0, output)
        self.assertIn("the guest did not print `tiny_frame: OK`", problems)
        self.assertIn("the guest reported a failed check", problems)

    def test_missing_loader_exit_line(self):
        output = good_output().replace("game exited with code 0", "game exited with code 1")
        self.assertIn("gta6rt did not report `game exited with code 0`", check.check_output(0, output))

    def test_host_binding_is_reported(self):
        problems = check.check_output(0, good_output(binding="host"))
        self.assertEqual(problems, ["d3d12_xs.dll!D3D12XboxCreateDevice was not bound to the shim"])

    def test_missing_call_names_the_first_one_out_of_order(self):
        problems = check.check_output(0, good_output(drop="ID3D12CommandQueue::Signal ok"))
        self.assertEqual(problems, ["the trace has no `ID3D12CommandQueue::Signal ok` at its place in the "
                                    "first-frame order"])

    def test_a_call_that_did_not_succeed_is_reported(self):
        problems = check.check_output(0, good_output(extra_trace=["ID3D12Device::CreateHeap unsupported"]))
        self.assertEqual(problems, ["traced calls that did not succeed: ID3D12Device::CreateHeap unsupported"])

    def test_a_gap_in_the_numbering_is_an_invalid_trace(self):
        output = good_output().replace("[d3d12-trace] #3 ", "[d3d12-trace] #4 ")
        problems = check.check_output(0, output)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("invalid trace sequence at entry 3"))

    def test_no_trace_at_all(self):
        output = "\n".join(line for line in good_output().split("\n") if not line.startswith("[d3d12-trace]"))
        self.assertEqual(check.check_output(0, output), ["the trace has no `D3D12XboxCreateDevice ok` at its place "
                                                         "in the first-frame order"])


RUNTIME_LINE = "[rt] d3d12: LEONIDA_VK_VALIDATION: VK_LAYER_KHRONOS_validation enabled with synchronization validation"
OFF_LINE = ("[rt] d3d12: LEONIDA_VK_VALIDATION: the Vulkan loader does not list VK_LAYER_KHRONOS_validation and no "
            "validation layer is active, so validation is off")


def with_report(output, checks):
    """output with the layer's CURRENT-VALIDATION-ENABLED report listing checks, before the run's lines."""
    report = ["Information: [ CURRENT-VALIDATION-ENABLED ] | MessageID = 0x1", "Enabled checks:"]
    report += [f"  - {check}" for check in checks]
    return "\n".join([RUNTIME_LINE, *report, ""]) + output


class ValidationTest(unittest.TestCase):
    def test_a_report_with_synchronization_passes(self):
        output = with_report(good_output(), ["Core", "Synchronization"])
        self.assertEqual(check.check_output(0, output, validation=True), [])

    def test_a_report_without_synchronization_fails(self):
        output = with_report(good_output(), ["Core"])
        self.assertEqual(check.check_output(0, output, validation=True),
                         ["the validation layer ran without synchronization validation, which the device asks for"])

    def test_a_report_whose_list_is_empty_fails(self):
        output = with_report(good_output(), [])
        problems = check.check_output(0, output, validation=True)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("the validation layer ran without synchronization validation"))


class MainValidationTest(unittest.TestCase):
    """main --validation on a run with no layer report: a skip only when nothing else is wrong."""

    def main(self, code, output):
        done = subprocess.CompletedProcess([], code, stdout=output, stderr="")
        argv = ["check_tiny_frame.py", "gta6rt.exe", "tiny_frame.exe", "--validation"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(check.subprocess, "run", return_value=done), \
                contextlib.redirect_stdout(io.StringIO()):
            return check.main()

    def test_a_clean_run_with_no_layer_is_skipped(self):
        self.assertEqual(self.main(0, "\n".join([OFF_LINE, good_output()])), check.SKIP)

    def test_a_validation_error_without_a_report_fails(self):
        output = "\n".join([OFF_LINE, "Validation Error: [ VUID-x ] something", good_output()])
        self.assertEqual(self.main(0, output), 1)

    def test_a_bad_exit_code_without_a_report_fails(self):
        self.assertEqual(self.main(3, "\n".join([OFF_LINE, good_output()])), 1)


class SubsequenceTest(unittest.TestCase):
    def test_repeated_entries_need_as_many_occurrences(self):
        self.assertIsNone(check.ordered_subsequence(["a", "a", "b"], ["a", "x", "a", "b"]))
        self.assertEqual(check.ordered_subsequence(["a", "a", "b"], ["a", "b"]), "a")

    def test_order_matters(self):
        self.assertEqual(check.ordered_subsequence(["a", "b"], ["b", "a"]), "b")


if __name__ == "__main__":
    unittest.main()
