"""GPU fuzzers must not consume outputs from a translator process that failed.

Synthetic child processes and model readbacks; no LLVM, GPU or game files needed.
"""
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lds_fuzz  # noqa: E402
import valu_fuzz  # noqa: E402

TRANSLATOR = """
import json
from pathlib import Path
import sys
source, output = map(Path, sys.argv[1:3])
status, mode = int(sys.argv[3]), sys.argv[4]
output.mkdir(parents=True, exist_ok=True)
names = sorted(p.stem for p in source.glob('*.s'))
if mode == 'partial':
    names = names[:1]
for name in names:
    (output / (name + '.spv')).write_bytes(b'synthetic module')
if mode != 'missing':
    (output / 'index.json').write_text(json.dumps(dict.fromkeys(names, {})), encoding='utf-8', newline='\\n')
print('synthetic translator output')
if status:
    print('synthetic failure after index publication', file=sys.stderr)
sys.exit(status)
"""
CASES = ((lds_fuzz, ()), (valu_fuzz, ("--family", "mem")), (valu_fuzz, ("--family", "f16")))


class TranslatorStatus(unittest.TestCase):
    def campaign(self, module, extra, status=0, mode="complete"):
        real_run = subprocess.run
        children = []

        def translate(cmd, **kwargs):
            result = real_run([sys.executable, "-I", "-c", TRANSLATOR, cmd[3],
                               cmd[cmd.index("--out") + 1], str(status), mode], **kwargs)
            children.append(result.returncode)
            return result

        def gpu(program, spv, build, entry, emulate, dump, **kwargs):
            # A successful stand-in produces the real model's words. A translator failure must never reach it.
            regs, memory = program.model()
            words = [regs[lane][r] for lane in range(len(regs)) for r in module.POOL]
            if module is valu_fuzz:
                words = [word.v for word in words] + [0 if word is None else word for word in memory]
                memory = b""
            dump.write_bytes(struct.pack(f"<{len(words)}I", *words) + bytes(memory))
            return 0, "compute subgroup size: 32"

        with tempfile.TemporaryDirectory() as tmp:
            argv = [module.__name__ + ".py", "--programs", "2", "--seed", "1", "--wave", "32",
                    "--work", tmp, *extra]
            output = io.StringIO()
            with (mock.patch.object(sys, "argv", argv),
                  mock.patch.object(module.mc_command, "run",
                                    return_value=subprocess.CompletedProcess([], 0, "s_endpgm\n", "")),
                  mock.patch.object(module.subprocess, "run", side_effect=translate),
                  mock.patch.object(module, "run_gpu", side_effect=gpu) as run_gpu,
                  contextlib.redirect_stdout(output)):
                rc = module.main()
            self.assertEqual(children, [status], output.getvalue())
            return rc, output.getvalue(), run_gpu.call_count

    def test_nonzero_exit_with_complete_index_is_a_harness_error(self):
        for module, extra in CASES:
            for status in (1, 23):
                with self.subTest(fuzzer=module.__name__, extra=extra, status=status):
                    rc, output, gpu_calls = self.campaign(module, extra, status=status)
                    self.assertEqual(rc, 2, output)
                    self.assertIn(f"rdna2spirv.py exited {status}", output)
                    self.assertIn("synthetic failure after index publication", output)
                    self.assertEqual(gpu_calls, 0)
                    self.assertNotIn("programs that ran match", output)

    def test_successful_translation_still_runs_and_compares(self):
        for module, extra in CASES:
            with self.subTest(fuzzer=module.__name__, extra=extra):
                rc, output, gpu_calls = self.campaign(module, extra)
                self.assertEqual(rc, 0, output)
                self.assertIn("2/2 programs that ran match", output)
                self.assertEqual(gpu_calls, 2)

    def test_zero_exit_without_index_remains_a_harness_error(self):
        for module, extra in CASES:
            with self.subTest(fuzzer=module.__name__, extra=extra):
                rc, output, gpu_calls = self.campaign(module, extra, mode="missing")
                self.assertEqual(rc, 2, output)
                self.assertIn("without an index.json", output)
                self.assertEqual(gpu_calls, 0)

    def test_zero_exit_partial_index_keeps_untranslated_accounting(self):
        for module, extra in CASES:
            with self.subTest(fuzzer=module.__name__, extra=extra):
                rc, output, gpu_calls = self.campaign(module, extra, mode="partial")
                self.assertEqual(rc, 1, output)
                self.assertIn("1/1 programs that ran match", output)
                self.assertIn("1 not translated", output)
                self.assertEqual(gpu_calls, 1)


if __name__ == "__main__":
    unittest.main()
