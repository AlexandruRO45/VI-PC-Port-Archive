"""Exercise the D3D12 trace checker with a synthetic child process."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHECK = Path(__file__).resolve().parents[1] / "runtime/test/check_d3d12_trace.py"
QUEUE_TRACE = """\
[d3d12-trace] #1 ID3D12CommandQueue::ExecuteCommandLists failed
[d3d12-trace] #2 ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported
[d3d12-trace] #3 ID3D12GraphicsCommandList::Close failed
[d3d12-trace] #4 ID3D12CommandQueue::Wait ok
[d3d12-trace] #5 ID3D12CommandQueue::ExecuteCommandLists ok
[d3d12-trace] #6 ID3D12CommandQueue::Signal ok
"""


class D3D12TraceOutputTests(unittest.TestCase):
    def run_checker(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.py"
            fixture.write_text(
                "import sys\n"
                f"mode = {mode!r}\n"
                f"trace = {QUEUE_TRACE!r}\n"
                "print('ordinary fixture output mentions [d3d12-trace]')\n"
                "if mode == 'process failure':\n"
                "    sys.stdout.write(trace)\n"
                "    print('synthetic child error', file=sys.stderr)\n"
                "    raise SystemExit(7)\n"
                "if '--trace-d3d12' in sys.argv:\n"
                "    sys.stderr.write(trace)\n"
                "    if mode == 'duplicated trace':\n"
                "        sys.stdout.write(trace)\n"
                "elif mode == 'stdout with tracing disabled':\n"
                "    print(trace.splitlines()[0])\n"
                "elif mode == 'stderr with tracing disabled':\n"
                "    print(trace.splitlines()[0], file=sys.stderr)\n",
                encoding="utf-8", newline="\n",
            )
            return subprocess.run(
                [sys.executable, "-I", str(CHECK), str(fixture), "--queue", "--runner", sys.executable],
                capture_output=True, text=True, timeout=10,
            )

    def test_accepts_stderr_trace_and_ordinary_stdout(self):
        result = self.run_checker("normal")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("D3D12 trace verified with 6 ordered entries", result.stdout)

    def test_rejects_trace_on_stdout(self):
        for mode in ("stdout with tracing disabled", "duplicated trace"):
            with self.subTest(mode=mode):
                result = self.run_checker(mode)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("D3D12 trace appeared on stdout", result.stderr)

    def test_rejects_stderr_trace_when_disabled(self):
        result = self.run_checker("stderr with tracing disabled")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("D3D12 trace appeared with tracing disabled", result.stderr)

    def test_preserves_process_failure_and_its_output(self):
        result = self.run_checker("process failure")
        self.assertEqual(result.returncode, 1, result.stdout)
        for text in ("exited with 7", "ordinary fixture output", "synthetic child error", QUEUE_TRACE.splitlines()[0]):
            self.assertIn(text, result.stderr)


if __name__ == "__main__":
    unittest.main()
