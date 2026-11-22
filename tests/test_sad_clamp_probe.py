"""Unit tests for tools/sad_clamp_probe.py (reference model, kernel assembly, FakeOracle verdicts, CLI).

Standard library only; no GPU, no LLVM required (the clang assembly test skips when AMD clang is unavailable),
and no game files.
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sad_clamp_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class FakeOracle:
    """Simulated hw_oracle backend that parses kernel_lines() and answers from reference helpers."""

    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, *, bad_word: bool = False, wave64_diff: bool = False, error_msg: str | None = None):
        self.bad_word = bad_word
        self.wave64_diff = wave64_diff
        self.error_msg = error_msg
        self.closed = False

    def run(self, kernels):
        out = []
        for k in kernels:
            if self.error_msg is not None:
                out.append(self.error_msg)
                continue
            regs: dict[str, int] = {}
            words: list[int] = []
            for ln in k.lines:
                m_mov = re.match(r"v_mov_b32_e32 (v\d+), (0x[0-9a-fA-F]+|\d+)$", ln)
                if m_mov:
                    regs[m_mov.group(1)] = int(m_mov.group(2), 0) & pr.M32
                    continue
                m_sad = re.match(r"(v_(?:sad|msad|qsad|mqsad)\w*)\s+[^,]+,.*?( clamp)?$", ln)
                if m_sad:
                    op = m_sad.group(1)
                    clamp = m_sad.group(2) is not None
                    if op in pr.SAD32_OPS:
                        res_words = [pr.expected_sad32(op, regs["v0"], regs["v2"], regs["v4"], clamp=clamp)]
                    elif op in pr.QSAD16_OPS:
                        s0 = regs["v0"] | (regs["v1"] << 32)
                        s2 = regs["v4"] | (regs["v5"] << 32)
                        res_words = list(pr.expected_qsad_pk_u16(op, s0, regs["v2"], s2, clamp=clamp))
                    elif op in pr.MQSAD32_OPS:
                        s0 = regs["v0"] | (regs["v1"] << 32)
                        accs = (regs["v4"], regs["v5"], regs["v6"], regs["v7"])
                        res_words = list(pr.expected_mqsad_u32(s0, regs["v2"], accs, clamp=clamp))
                    else:
                        raise AssertionError(f"unexpected op in FakeOracle: {op}")
                    if not words:
                        if self.bad_word:
                            res_words[0] ^= 1
                        if self.wave64_diff and k.wave == 64:
                            res_words[0] ^= 1
                    words.extend(res_words)
            out.append(words)
        return out

    def close(self):
        self.closed = True


def _run_main(oracle, *args):
    buf = io.StringIO()
    err = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf), redirect_stderr(err):
        rc = pr.main(list(args))
    return rc, buf.getvalue(), err.getvalue()


class ReferenceModelTests(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)

    def test_critical_edge_cases(self):
        # v_sad_hi_u8 saturates the entire 32-bit word to 0xFFFFFFFF on overflow (not only the high 16 bits)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFE1234, clamp=True), 0xFFFF1234)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFF1234, clamp=False), 0x00001234)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFF1234, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0xFF00FF00, 0x00FF00FF, 0xFC03FFFF, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0xFF00FF00, 0x00FF00FF, 0xFC040000, clamp=False), 0x00000000)
        self.assertEqual(pr.expected_sad32("v_sad_hi_u8", 0xFF00FF00, 0x00FF00FF, 0xFC040000, clamp=True), 0xFFFFFFFF)

        # v_sad_u8, v_sad_u16, v_sad_u32, v_msad_u8
        self.assertEqual(pr.expected_sad32("v_sad_u8", 0xFF00FF00, 0x00FF00FF, 0xFFFFFC04, clamp=False), 0x00000000)
        self.assertEqual(pr.expected_sad32("v_sad_u8", 0xFF00FF00, 0x00FF00FF, 0xFFFFFC04, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_sad_u16", 0xFFFF0000, 0x0000FFFF, 0xFFFE0002, clamp=False), 0x00000000)
        self.assertEqual(pr.expected_sad32("v_sad_u16", 0xFFFF0000, 0x0000FFFF, 0xFFFE0002, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_sad_u32", 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, clamp=False), 0xFFFFFFFE)
        self.assertEqual(pr.expected_sad32("v_sad_u32", 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_msad_u8", 0xFF00FF00, 0x00000000, 0xFFFFFFFF, clamp=True), 0xFFFFFFFF)
        self.assertEqual(pr.expected_sad32("v_msad_u8", 0xFF00FF00, 0x01000100, 0xFFFFFE04, clamp=True), 0xFFFFFFFF)

        # Packed u16 QSAD/MQSAD saturate each u16 lane independently with zero carry into high half
        self.assertEqual(
            pr.expected_qsad_pk_u16("v_qsad_pk_u16_u8", 0, 0xFFFFFFFF, 0xFC03FC040000FFFF, clamp=False),
            (0x03FC03FB, 0xFFFF0000),
        )
        self.assertEqual(
            pr.expected_qsad_pk_u16("v_qsad_pk_u16_u8", 0, 0xFFFFFFFF, 0xFC03FC040000FFFF, clamp=True),
            (0x03FCFFFF, 0xFFFFFFFF),
        )
        self.assertEqual(
            pr.expected_qsad_pk_u16("v_mqsad_pk_u16_u8", 0, 0xFF00FFFF, 0xFD02FFFF0000FD03, clamp=False),
            (0x02FD0000, 0xFFFF02FC),
        )
        self.assertEqual(
            pr.expected_qsad_pk_u16("v_mqsad_pk_u16_u8", 0, 0xFF00FFFF, 0xFD02FFFF0000FD03, clamp=True),
            (0x02FDFFFF, 0xFFFFFFFF),
        )

        # v_mqsad_u32_u8 mixed lanes
        self.assertEqual(
            pr.expected_mqsad_u32(0x00FF00FFFF00FF00, 0x01FF01FF, (0xFFFFFF00, 0xFFFFFF00, 1, 0xFFFFFFFF), clamp=False),
            (0x000002FA, 0xFFFFFFFF, 0x000001FF, 0x000002FA),
        )
        self.assertEqual(
            pr.expected_mqsad_u32(0x00FF00FFFF00FF00, 0x01FF01FF, (0xFFFFFF00, 0xFFFFFF00, 1, 0xFFFFFFFF), clamp=True),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x000001FF, 0xFFFFFFFF),
        )


class KernelTests(unittest.TestCase):
    def test_kernel_structure_all_ops(self):
        for op in pr.OPS:
            for clamp in (False, True):
                with self.subTest(op=op, clamp=clamp):
                    lines = pr.kernel_lines(op, clamp)
                    self.assertEqual(lines[-2:], ["s_waitcnt vmcnt(0)", "s_endpgm"])
                    insns = [ln for ln in lines if ln.startswith(op + " ")]
                    self.assertEqual(len(insns), len(pr.CASES[op]))
                    if clamp:
                        self.assertTrue(all(ln.endswith(" clamp") for ln in insns))
                    else:
                        self.assertTrue(all("clamp" not in ln for ln in insns))

    def test_kernels_assemble_when_amd_clang_available(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        for wave in (32, 64):
            kernels, _ = pr.build_jobs(list(pr.OPS), [wave])
            text, _, _ = asm.module_text(kernels, "gfx1036", wave=wave)
            with tempfile.TemporaryDirectory() as d:
                s_path = Path(d) / "k.s"
                o_path = Path(d) / "k.o"
                with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                flags = ["-mwavefrontsize64"] if wave == 64 else []
                res = subprocess.run(
                    [clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-c", str(s_path), "-o", str(o_path)],
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(res.returncode, 0, res.stderr[:800])


class ProbeCliTests(unittest.TestCase):
    def test_fake_oracle_pass_and_json_lf(self):
        oracle = FakeOracle()
        with tempfile.TemporaryDirectory() as d:
            report = os.path.join(d, "nested", "report.json")
            rc, out, err = _run_main(oracle, "--json", report)
            self.assertEqual(rc, 0, out + err)
            self.assertTrue(oracle.closed)
            self.assertIn("mismatches=0, wave_diffs=0", out)
            with open(report, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["arch"], "gfx1036")
            self.assertEqual(doc["waves"], [32, 64])
            self.assertEqual(doc["ops"], list(pr.OPS))
            self.assertEqual(doc["mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)
            self.assertEqual(doc["total_words"], 2 * sum(len(pr.CASES[op]) * pr.words_per_case(op) for op in pr.OPS) * 2)

    def test_fake_oracle_word_mismatch_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(bad_word=True), "--ops", "v_sad_u8", "--waves", "32")
        self.assertEqual(rc, 1, out)
        self.assertIn("MISMATCH", out)

    def test_fake_oracle_wave_diff_exits_1(self):
        rc, out, _ = _run_main(FakeOracle(wave64_diff=True), "--ops", "v_sad_hi_u8", "--waves", "32,64")
        self.assertEqual(rc, 1, out)
        self.assertIn("WAVE DIFF", out)

    def test_fake_oracle_harness_error_exits_2(self):
        rc, out, _ = _run_main(FakeOracle(error_msg="assembler rejected: test"), "--ops", "v_sad_u8", "--waves", "32")
        self.assertEqual(rc, 2, out)
        self.assertIn("harness error", out)

    def test_bad_cli_args_exit_2(self):
        rc, _, err = _run_main(FakeOracle(), "--waves", "16")
        self.assertEqual(rc, 2)
        self.assertIn("--waves", err)
        rc, _, err = _run_main(FakeOracle(), "--ops", "v_not_an_op")
        self.assertEqual(rc, 2)
        self.assertIn("--ops", err)

    def test_json_directory_refused_and_unwritable_path_exits_2(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = _run_main(FakeOracle(), "--ops", "v_sad_u8", "--waves", "32", "--json", d)
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", err)
            self.assertNotIn("Traceback", err)
            blocker = os.path.join(d, "file_not_dir")
            with open(blocker, "wb") as f:
                f.write(b"keep")
            rc, _, err = _run_main(
                FakeOracle(), "--ops", "v_sad_u8", "--waves", "32", "--json", os.path.join(blocker, "r.json")
            )
            self.assertEqual(rc, 2)
            self.assertIn("ERROR: cannot write", err)
            self.assertNotIn("Traceback", err)

    def test_non_gfx10_arch_exits_77(self):
        oracle = FakeOracle()
        oracle.arch = "gfx1100"
        rc, out, _ = _run_main(oracle)
        self.assertEqual(rc, 77, out)
        self.assertTrue(oracle.closed)

    def test_unavailable_oracle_exits_77(self):
        buf = io.StringIO()
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("missing")), redirect_stdout(buf):
            rc = pr.main(["--ops", "v_sad_u8", "--waves", "32"])
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
