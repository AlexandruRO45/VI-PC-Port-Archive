"""Unit tests for tools/dot2_i16_opsel_probe.py (reference models, FakeOracle verdicts, CLI).

Standard library only; no GPU or LLVM required (optional live gfx1036 test skips cleanly when hw_oracle is
unavailable) and no game files.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import dot2_i16_opsel_probe as pr  # noqa: E402
from hw_oracle import runner  # noqa: E402


class FakeOracle:
    """Simulated hw_oracle backend that parses probe kernels and answers from ref_dot2 / ref_dot2_neg."""

    def __init__(self, *, arch="gfx1036", bad_word=False, wave64_diff=False, error_msg=None):
        self.arch = arch
        self.device = f"fake {arch}"
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
            regs = {}
            words = []
            for ln in k.lines:
                m_v = re.match(r"v_mov_b32_e32 (v\d+), (0x[0-9a-fA-F]+)$", ln)
                if m_v:
                    regs[m_v.group(1)] = int(m_v.group(2), 16)
                    continue
                m_s = re.match(r"s_mov_b32 (s\d+), (0x[0-9a-fA-F]+)$", ln)
                if m_s:
                    regs[m_s.group(1)] = int(m_s.group(2), 16)
                    continue
                m_dot = re.match(r"(v_dot2_[iu]32_[iu]16) v8, ([^,]+), ([^,]+), ([^ ]+)(?:\s+(.*))?$", ln)
                if m_dot:
                    op, t0, t1, t2, mod_str = m_dot.groups()
                    mod_str = (mod_str or "").strip()

                    def val_of(tok):
                        tok = tok.strip()
                        if tok in regs:
                            return regs[tok]
                        return int(tok, 0) & 0xFFFFFFFF

                    def parse_bits(name, default):
                        m = re.search(rf"\b{name}:\[(\d+),(\d+),(\d+)\]", mod_str)
                        return tuple(int(g) for g in m.groups()) if m else default

                    sel = parse_bits("op_sel", (0, 0, 0))
                    selhi = parse_bits("op_sel_hi", (1, 1, 1))
                    neg_lo = parse_bits("neg_lo", (0, 0, 0))
                    neg_hi = parse_bits("neg_hi", (0, 0, 0))
                    clamp = "clamp" in mod_str.split()
                    w = pr.ref_dot2_neg(
                        op, val_of(t0), val_of(t1), val_of(t2),
                        neg_lo=neg_lo, neg_hi=neg_hi, sel=sel, selhi=selhi, clamp=clamp,
                    )
                    if self.bad_word and not words:
                        w ^= 1
                    if self.wave64_diff and k.wave == 64 and not words:
                        w ^= 1
                    words.append(w)
            out.append(words)
        return out

    def close(self):
        self.closed = True


def _run_main(oracle, *args):
    buf, err = io.StringIO(), io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf), redirect_stderr(err):
        rc = pr.main(list(args))
    return rc, buf.getvalue(), err.getvalue()


class ReferenceModelTests(unittest.TestCase):
    def test_self_test_direct_and_subprocess(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)
        proc = subprocess.run(
            [sys.executable, "-I", str(TOOLS / "dot2_i16_opsel_probe.py"), "--self-test"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("self-test ok", proc.stdout)

    def test_ref_dot2_all_16_opsel_and_clamp_edges(self):
        s0, s1, s2 = 0x80007FFF, 0xFFFF0002, 0x00000123
        for s0_lo, s1_lo, s0_hi, s1_hi in itertools.product((0, 1), repeat=4):
            sel = (s0_lo, s1_lo, 0)
            selhi = (s0_hi, s1_hi, 1)
            for op in pr.OPS:
                signed = op == "v_dot2_i32_i16"
                def h(w, hi):
                    v = (w >> (16 if hi else 0)) & 0xFFFF
                    return v - 0x10000 if (signed and (v & 0x8000)) else v
                want = (h(s0, s0_lo) * h(s1, s1_lo) + h(s0, s0_hi) * h(s1, s1_hi) + s2) & 0xFFFFFFFF
                self.assertEqual(pr.ref_dot2(op, s0, s1, s2, sel=sel, selhi=selhi, clamp=False), want)

        # (-32768)^2 + (-32768)^2 + (-1) == 0x7FFFFFFF cancellation vs +2^31 overflow
        self.assertEqual(
            pr.ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0xFFFFFFFF,
                        sel=(0, 1, 0), selhi=(0, 1, 1), clamp=True),
            0x7FFFFFFF,
        )
        self.assertEqual(
            pr.ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0,
                        sel=(0, 1, 0), selhi=(0, 1, 1), clamp=False),
            0x80000000,
        )
        self.assertEqual(
            pr.ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0,
                        sel=(0, 1, 0), selhi=(0, 1, 1), clamp=True),
            0x7FFFFFFF,
        )
        # Signed negative saturation & unsigned overflow saturation
        self.assertEqual(
            pr.ref_dot2("v_dot2_i32_i16", 0x80007FFF, 0x80007FFF, 0x80000000,
                        sel=(1, 0, 0), selhi=(0, 1, 1), clamp=True),
            0x80000000,
        )
        self.assertEqual(
            pr.ref_dot2("v_dot2_u32_u16", 0xFFFF0001, 0x0001FFFF, 0x80000000,
                        sel=(1, 0, 0), selhi=(0, 1, 1), clamp=True),
            0xFFFFFFFF,
        )

    def test_ref_dot2_neg_characterization(self):
        # Plain: 2*5 + 3*7 + 0x87654321 = 0x87654340
        self.assertEqual(pr.ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321), 0x87654340)
        # S0 neg_hi flips bit 31 of S0 (0x0003 -> 0x8003 = -32765)
        self.assertEqual(
            pr.ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321, neg_hi=(1, 0, 0)),
            (0x87654321 + 2 * 5 + (-32765) * 7) & 0xFFFFFFFF,
        )
        # S2 neg_hi clears bit 31 before neg_lo flips bit 31
        self.assertEqual(
            pr.ref_dot2_neg("v_dot2_u32_u16", 0x00030002, 0x00070005, 0x87654321, neg_hi=(0, 0, 1)),
            0x07654340,
        )
        self.assertEqual(
            pr.ref_dot2_neg("v_dot2_u32_u16", 0x00030002, 0x00070005, 0x07654321, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1)),
            0x87654340,
        )


class ProbeCliAndFakeOracleTests(unittest.TestCase):
    def test_fake_oracle_passes_and_writes_lf_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = Path(tmp) / "sub" / "report.json"
            oracle = FakeOracle()
            rc, out, err = _run_main(oracle, "--json", str(rep))
            self.assertEqual(rc, 0, err or out)
            self.assertTrue(oracle.closed)
            self.assertIn("total_words=856 (grid=768, sources=48, controls=40)", out)
            self.assertIn("mismatches=0, wave_diffs=0", out)
            raw = rep.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            doc = json.loads(raw.decode("utf-8"))
            self.assertEqual(doc["total_words"], 856)
            self.assertEqual(doc["mismatches"], 0)
            self.assertEqual(doc["wave_diffs"], 0)

    def test_fake_oracle_mismatch_and_wavediff_exit_1(self):
        oracle_bad = FakeOracle(bad_word=True)
        rc, out, _ = _run_main(oracle_bad, "--waves", "32", "--ops", "v_dot2_i32_i16")
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out)

        oracle_wd = FakeOracle(wave64_diff=True)
        rc, out, _ = _run_main(oracle_wd, "--ops", "v_dot2_u32_u16")
        self.assertEqual(rc, 1)
        self.assertIn("WAVE DIFF", out)

    def test_cli_validation_errors_exit_2(self):
        for args in (["--waves", "16"], ["--waves", ""], ["--ops", "v_dot4_u32_u8"], ["--ops", ""]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    pr.main(args)
                self.assertEqual(ctx.exception.code, 2)

    def test_unavailable_and_nongfx10_exit_77(self):
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no gpu")), redirect_stdout(io.StringIO()):
            self.assertEqual(pr.main([]), 77)

        oracle_rdna3 = FakeOracle(arch="gfx1100")
        rc, out, _ = _run_main(oracle_rdna3)
        self.assertEqual(rc, 77)
        self.assertTrue(oracle_rdna3.closed)
        self.assertIn("not GFX10", out)

    def test_harness_error_string_exits_2(self):
        oracle_err = FakeOracle(error_msg="assembler rejected: invalid operand")
        rc, out, _ = _run_main(oracle_err, "--waves", "32", "--ops", "v_dot2_i32_i16")
        self.assertEqual(rc, 2)
        self.assertIn("harness error:", out)

    def test_optional_live_gfx1036(self):
        try:
            oracle = runner.open_oracle(arch="gfx1036")
        except runner.Unavailable as e:
            self.skipTest(f"hw_oracle unavailable: {e}")
        oracle.close()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pr.main(["--hw-arch", "gfx1036"])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("total_words=856", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
