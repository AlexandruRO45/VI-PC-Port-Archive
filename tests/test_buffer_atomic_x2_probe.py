"""Unit tests for tools/buffer_atomic_x2_probe.py (no GPU, no game files, standard library only)."""
from contextlib import redirect_stderr, redirect_stdout
import io
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
sys.path.insert(0, str(TOOLS))

import buffer_atomic_x2_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


def _parse_kernel(kernel):
    """Reconstruct a Case from generated kernel lines so FakeOracle exercises kernel_lines() directly."""
    base_add = stride = num_records = oob_select = soff = idx = voff = inst_off = 0
    idxen = False
    op = ""
    glc = False
    has_v7 = False
    v6 = 0
    for ln in kernel.lines:
        if ln.startswith("s_add_u32 s8, s2, "):
            base_add = int(ln.split()[-1], 0)
        elif ln.startswith("s_or_b32 s9, s9, "):
            stride = int(ln.split()[-1], 0) >> 16
        elif ln.startswith("s_mov_b32 s10, "):
            num_records = int(ln.split()[-1], 0)
        elif ln.startswith("s_mov_b32 s11, "):
            oob_select = (int(ln.split()[-1], 0) >> 28) & 3
        elif ln.startswith("s_mov_b32 s14, "):
            soff = int(ln.split()[-1], 0)
        elif ln.startswith("v_mov_b32 v6, "):
            v6 = int(ln.split()[-1], 0)
        elif ln.startswith("v_mov_b32 v7, "):
            voff = int(ln.split()[-1], 0)
            has_v7 = True
        elif ln.startswith("buffer_atomic_"):
            op = ln.split()[0]
            glc = " glc" in ln
            m = re.search(r"offset:(\d+)", ln)
            if m:
                inst_off = int(m.group(1))
    if has_v7:
        idxen = True
        idx = v6
    else:
        voff = v6
    return pr.Case(
        name="parsed", op=op, glc=glc, oob_select=oob_select, voff=voff,
        num_records=num_records, stride=stride, idx=idx, idxen=idxen,
        base_add=base_add, soff=soff, inst_off=inst_off,
    )


class FakeOracle:
    """Simulates a GFX10 GPU running buffer_atomic_x2_probe kernels."""
    arch = "gfx1036"
    device = "fake gfx1036"

    def __init__(self, policy="all_or_nothing", wave64_off_idx=None, corrupt_inactive=False, error_at=None):
        self.policy = policy
        self.wave64_off_idx = wave64_off_idx
        self.corrupt_inactive = corrupt_inactive
        self.error_at = error_at
        self.closed = False

    def run(self, kernels, words=4):
        out = []
        for idx, k in enumerate(kernels):
            if self.error_at is not None and idx == self.error_at:
                out.append("assembler rejected: simulated error")
                continue
            parsed = _parse_kernel(k)
            row = list(pr.expected_words(parsed, policy=self.policy))
            if self.wave64_off_idx is not None and k.wave == 64 and idx == self.wave64_off_idx:
                row[16] ^= 1
            if self.corrupt_inactive and idx == 0:
                row[18] ^= 0x10
            out.append(row)
        return out

    def close(self):
        self.closed = True


def _run_main(oracle, *args):
    out_buf, err_buf = io.StringIO(), io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), \
            redirect_stdout(out_buf), redirect_stderr(err_buf):
        rc = pr.main(list(args))
    return rc, out_buf.getvalue(), err_buf.getvalue()


class ProbeTests(unittest.TestCase):
    def test_self_test(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = pr.main(["--self-test"])
        self.assertEqual(rc, 0)
        self.assertIn("self-test ok", buf.getvalue())

    def test_descriptor_word3(self):
        for mode in range(4):
            w = pr.descriptor_word3(mode)
            self.assertEqual(w & 0xFFF, 0xFAC)
            self.assertEqual((w >> 12) & 0x7F, 22)
            self.assertEqual((w >> 24) & 1, 1)
            self.assertEqual((w >> 28) & 3, mode)
        self.assertEqual(pr.descriptor_word3(3), 0x31016FAC)

    def test_cases_cover_all_ops_modes_and_aligned_straddles(self):
        cases = pr.build_cases()
        self.assertEqual({c.oob_select for c in cases}, {0, 1, 2, 3})
        self.assertEqual({c.glc for c in cases}, {True, False})
        groups = {c.group for c in cases}
        self.assertEqual(
            groups,
            {"raw_straddle", "raw_base4", "raw_offset_terms", "mode0_straddle",
             "mode1", "mode2", "align_control", "ctrl32"},
        )
        for op in pr.OPS:
            op_name = f"buffer_atomic_{op}_x2"
            straddling = [
                c for c in cases
                if c.op == op_name and pr.expected_words(c, "all_or_nothing") != pr.expected_words(c, "per_dword")
            ]
            self.assertGreaterEqual(len(straddling), 10, op_name)
            for c in straddling:
                eff = c.base_add + c.inst_off + c.voff + c.soff + (c.idx * c.stride if c.idxen else 0)
                self.assertEqual(eff % 8, 0, c)

    def test_kernel_lines_wave32_and_wave64(self):
        case = pr.build_cases(("xor",))[0]
        w32 = pr.kernel_lines(case, wave=32)
        w64 = pr.kernel_lines(case, wave=64)
        self.assertIn("s_and_saveexec_b32 s12, vcc_lo", w32)
        self.assertIn("s_mov_b32 exec_lo, s12", w32)
        self.assertIn("v_cmp_eq_u32 vcc_lo, 0, v0", w32)
        self.assertIn("s_and_saveexec_b64 s[12:13], vcc", w64)
        self.assertIn("s_mov_b64 exec, s[12:13]", w64)
        self.assertIn("v_cmp_eq_u32 vcc, 37, v0", w64)

    def test_kernels_assemble_when_amd_clang_available(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        cases = pr.build_cases(("xor",))[:4]
        for wave in (32, 64):
            kernels = [asm.Kernel(pr.kernel_lines(c, wave), wave, wave, pr.OUT_WORDS, raw=True, vgprs=16, sgprs=24)
                       for c in cases]
            text, _, _ = asm.module_text(kernels, "gfx1036", wave=wave)
            flags = ["-mwavefrontsize64"] if wave == 64 else []
            with tempfile.TemporaryDirectory() as d:
                s = Path(d) / "k.s"
                s.write_text(text, encoding="utf-8", newline="\n")
                r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1036", *flags, "-c", str(s), "-o",
                                    str(Path(d) / "k.o")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:800])

    def test_main_with_fake_oracle_matches(self):
        oracle = FakeOracle(policy="all_or_nothing")
        with tempfile.TemporaryDirectory() as d:
            json_path = Path(d) / "nested" / "report.json"
            rc, out, err = _run_main(oracle, "--json", str(json_path))
            self.assertEqual(rc, 0, out + err)
            self.assertTrue(oracle.closed)
            self.assertIn("all_or_nothing 53/53", out)
            self.assertIn("mismatches 0, wave32/64 diffs 0", out)
            raw = json_path.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            rep = json.loads(raw.decode("utf-8"))
        self.assertEqual(rep["summary"]["mismatches"], 0)
        self.assertEqual(rep["summary"]["wave_diffs"], 0)
        self.assertEqual(rep["summary"]["straddle_distinguishing"], 53)
        self.assertEqual(rep["summary"]["straddle_matched_all_or_nothing"], 53)

    def test_main_detects_per_dword_divergence(self):
        oracle = FakeOracle(policy="per_dword")
        rc, out, err = _run_main(oracle, "--ops", "xor", "--waves", "32")
        self.assertEqual(rc, 1, out + err)
        self.assertIn("per_dword=True", out)
        self.assertIn("mismatches ", out)

    def test_main_detects_wave_divergence(self):
        oracle = FakeOracle(policy="all_or_nothing", wave64_off_idx=0)
        rc, out, err = _run_main(oracle, "--ops", "xor")
        self.assertEqual(rc, 1, out + err)
        self.assertIn("wave32/64 diffs 1", out)

    def test_main_detects_inactive_lane_corruption_and_kernel_error(self):
        rc1, out1, _ = _run_main(FakeOracle(corrupt_inactive=True), "--ops", "xor", "--waves", "32")
        self.assertEqual(rc1, 2, out1)
        self.assertIn("inactive lane vdata changed", out1)

        rc2, out2, _ = _run_main(FakeOracle(error_at=0), "--ops", "xor", "--waves", "32")
        self.assertEqual(rc2, 2, out2)
        self.assertIn("assembler rejected", out2)

    def test_main_unavailable_and_non_gfx10_exit_77(self):
        out_buf = io.StringIO()
        with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no gpu")), \
                redirect_stdout(out_buf):
            rc = pr.main(["--ops", "xor", "--waves", "32"])
        self.assertEqual(rc, 77)
        self.assertIn("unavailable", out_buf.getvalue())

        non_gfx10 = FakeOracle()
        non_gfx10.arch = "gfx1150"
        rc2, out2, _ = _run_main(non_gfx10, "--ops", "xor", "--waves", "32")
        self.assertEqual(rc2, 77)
        self.assertTrue(non_gfx10.closed)
        self.assertIn("not GFX10", out2)

    def test_main_json_directory_error_exits_2(self):
        oracle = FakeOracle()
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = _run_main(oracle, "--ops", "xor", "--waves", "32", "--json", d)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("ERROR:", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
