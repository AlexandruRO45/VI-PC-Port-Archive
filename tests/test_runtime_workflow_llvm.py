"""The CI GPU-test job must assemble fixtures with LLVM 21+ (not llvm-18)."""
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(REPO, ".github", "workflows", "runtime.yml")


class RuntimeWorkflowLlvmTests(unittest.TestCase):
    def test_gpu_tools_job_uses_llvm21(self):
        with open(WORKFLOW, encoding="utf-8", newline="\n") as f:
            text = f.read()
        self.assertNotIn("/usr/lib/llvm-18/", text)
        self.assertNotIn("pkgs+=(llvm-18)", text)
        match = re.search(r"(?ms)^  linux-gpu-tools:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)", text)
        self.assertIsNotNone(match, "linux-gpu-tools job not found")
        job = match.group(1)
        self.assertIn("llvm-21", job)
        self.assertIn("/usr/lib/llvm-21/bin/llvm-mc", job)
        # All three llvmpipe test steps must pass the same assembler explicitly.
        self.assertGreaterEqual(job.count("--llvm-mc /usr/lib/llvm-21/bin/llvm-mc"), 3)


if __name__ == "__main__":
    unittest.main()
