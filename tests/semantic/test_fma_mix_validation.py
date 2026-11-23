"""Operand and modifier validation tests for v_fma_mix_f32, v_fma_mixlo_f16, and v_fma_mixhi_f16.

Verifies that tools/rdna2spirv.py rejects malformed operands, non-VGPR destinations,
malformed/duplicate/out-of-range op_sel/op_sel_hi lists, VOP3 omod (mul:*/div:*),
VOP3P neg_lo/neg_hi, _dpp/_sdwa/_e32 variants, and unknown modifiers, while
accepting valid gfx1030 VOP3P forms (op_sel, op_sel_hi, clamp, source -/| |)
verified through llvm-mc decode round-trip, spirv-val, and spirv_mini_interp.
Standard library only.
"""
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)
import rdna2spirv  # noqa: E402

OPS = ("v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16")


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def h2u(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def pack_h(lo, hi):
    return h2u(lo) | (h2u(hi) << 16)


def llvm_mc_cmd():
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    exe = shutil.which("llvm-mc")
    return [exe] if exe else None


def spirv_tools():
    return shutil.which("spirv-as"), shutil.which("spirv-val")


def validate_spirv(spvasm):
    spirv_as, spirv_val = spirv_tools()
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        sa = os.path.join(tmp, "fma_mix.spvasm")
        sv = os.path.join(tmp, "fma_mix.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        as_res = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", sa, "-o", sv],
            capture_output=True, text=True, check=False,
        )
        if as_res.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{as_res.stderr}{as_res.stdout}")
        val_res = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", sv],
            capture_output=True, text=True, check=False,
        )
        if val_res.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{val_res.stderr}{val_res.stdout}")


def llvm_mc_roundtrip(ins_line):
    """Assemble ins_line for gfx1030 and disassemble the encoded bytes back to text."""
    mc = llvm_mc_cmd()
    assert mc is not None
    enc = subprocess.run(
        mc + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
        input=ins_line + "\n", capture_output=True, text=True, check=False,
    )
    if enc.returncode != 0:
        raise AssertionError(f"llvm-mc failed to assemble {ins_line!r}:\n{enc.stderr}{enc.stdout}")
    m = re.search(r"encoding:\s*\[([^\]]+)\]", enc.stdout)
    if not m:
        raise AssertionError(f"missing encoding in llvm-mc output for {ins_line!r}:\n{enc.stdout}")
    hex_bytes = " ".join(b.strip() for b in m.group(1).split(","))
    dec = subprocess.run(
        mc + ["-arch=amdgcn", "-mcpu=gfx1030", "-disassemble"],
        input=hex_bytes + "\n", capture_output=True, text=True, check=False,
    )
    if dec.returncode != 0:
        raise AssertionError(f"llvm-mc failed to disassemble {hex_bytes!r}:\n{dec.stderr}{dec.stdout}")
    lines = [ln.strip() for ln in dec.stdout.splitlines() if ln.strip() and not ln.strip().startswith(".")]
    if len(lines) != 1:
        raise AssertionError(f"expected 1 decoded instruction for {ins_line!r}, got {lines!r}")
    return lines[0]


def translate_single(ins_line):
    return translate_lines("ps_fma_mix_val", [
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        ins_line,
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])


def run_fma_mix(ins_line, v1, v2, v3, v0_init=0):
    spvasm = translate_lines("ps_fma_mix_run", [
        f"v_mov_b32 v0, {v0_init:#010x}",
        f"v_mov_b32 v1, {v1:#010x}",
        f"v_mov_b32 v2, {v2:#010x}",
        f"v_mov_b32 v3, {v3:#010x}",
        ins_line,
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])
    validate_spirv(spvasm)
    return mrt0_words(spvasm)[0]


class FmaMixNegativeValidationTests(unittest.TestCase):
    def assert_rejected(self, ins_line):
        with self.subTest(ins=ins_line), self.assertRaises(rdna2spirv.Unsupported):
            translate_single(ins_line)

    def test_reject_non_vgpr_and_invalid_destinations(self):
        bad_dsts = ("s0", "vcc_lo", "exec_lo", "v[0:1]", "-v0", "|v0|")
        for op in OPS:
            for dst in bad_dsts:
                self.assert_rejected(f"{op} {dst}, v1, v2, v3")

    def test_reject_wrong_operand_count(self):
        bad_arg_lists = (
            "v0, v1",
            "v0, v1, v2",
            "v0, v1, v2, v3, v4",
        )
        for op in OPS:
            for args in bad_arg_lists:
                self.assert_rejected(f"{op} {args}")

    def test_reject_malformed_and_out_of_range_op_sel(self):
        bad_mods = (
            "op_sel:[]",
            "op_sel_hi:[]",
            "op_sel:[0,0,0,0]",
            "op_sel_hi:[0,0,0,0]",
            "op_sel:[0,0,2]",
            "op_sel_hi:[2,0,0]",
            "op_sel:[0,-1,0]",
            "op_sel_hi:[0,-1,0]",
            "op_sel:[0,x,0]",
            "op_sel_hi:[x,0,0]",
            "op_sel:0,0,0",
            "op_sel:[0,0,0",
            "op_sel:[0,0,]",
            "op_sel_hi:0,0,0",
            "op_sel_hi:[0,0,0",
            "op_sel_hi:[0,0,]",
        )
        for op in OPS:
            for mod in bad_mods:
                self.assert_rejected(f"{op} v0, v1, v2, v3 {mod}")

    def test_reject_duplicate_modifiers(self):
        dup_mods = (
            "op_sel:[0,0,0] op_sel:[1,0,0]",
            "op_sel_hi:[1,0,0] op_sel_hi:[0,1,0]",
            "clamp clamp",
        )
        for op in OPS:
            for mod in dup_mods:
                self.assert_rejected(f"{op} v0, v1, v2, v3 {mod}")

    def test_reject_omod_vop3p_and_unknown_modifiers(self):
        bad_mods = (
            "mul:2",
            "mul:4",
            "div:2",
            "mul:2 clamp",
            "neg_lo:[1,0,0]",
            "neg_hi:[1,0,0]",
            "abs:[1,0,0]",
            "bogus_mod",
        )
        for op in OPS:
            for mod in bad_mods:
                self.assert_rejected(f"{op} v0, v1, v2, v3 {mod}")

    def test_reject_unsupported_encoding_suffixes(self):
        for op in OPS:
            self.assert_rejected(f"{op}_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3]")
            self.assert_rejected(f"{op}_sdwa v0, v1, v2, v3")
            self.assert_rejected(f"{op}_e32 v0, v1, v2, v3")

    @unittest.skipUnless(llvm_mc_cmd() is not None, "needs llvm-mc or LLVM_MC")
    def test_llvm_mc_also_rejects_disallowed_forms(self):
        mc = llvm_mc_cmd()
        disallowed = (
            "v_fma_mix_f32 v0, v1, v2, v3 mul:2",
            "v_fma_mix_f32 v0, v1, v2, v3 mul:4",
            "v_fma_mix_f32 v0, v1, v2, v3 div:2",
            "v_fma_mixlo_f16 v0, v1, v2, v3 mul:2",
            "v_fma_mixhi_f16 v0, v1, v2, v3 div:2",
            "v_fma_mix_f32 v0, v1, v2, v3 neg_lo:[1,0,0]",
            "v_fma_mix_f32 v0, v1, v2, v3 neg_hi:[1,0,0]",
            "v_fma_mix_f32_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3]",
            "v_fma_mix_f32_sdwa v0, v1, v2, v3",
            "v_fma_mix_f32_e32 v0, v1, v2, v3",
            "v_fma_mix_f32 v0, v1, v2, v3 op_sel:[0,0,2]",
        )
        for line in disallowed:
            with self.subTest(ins=line):
                res = subprocess.run(
                    mc + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                    input=line + "\n", capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(res.returncode, 0, f"expected llvm-mc to reject {line!r}")


class FmaMixPositiveControlTests(unittest.TestCase):
    def test_fma_mix_f32_mods_and_clamp(self):
        # src0 = f16 lo (-2.0) -> -|-2.0| = -2.0; src1 = f32 (-3.0) -> -(-3.0) = 3.0;
        # src2 = f16 hi (-1.5) -> |-1.5| = 1.5; result = -2.0 * 3.0 + 1.5 = -4.5.
        v1 = pack_h(-2.0, 9.0)
        v2 = f2u(-3.0)
        v3 = pack_h(7.0, -1.5)
        got = run_fma_mix("v_fma_mix_f32 v0, -|v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1]", v1, v2, v3)
        self.assertEqual(got, f2u(-4.5))

        # Clamp: -4.5 -> 0.0; +2.5 -> 1.0; +0.375 -> 0.375.
        got_lo = run_fma_mix("v_fma_mix_f32 v0, -|v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1] clamp", v1, v2, v3)
        self.assertEqual(got_lo, f2u(0.0))
        got_hi = run_fma_mix("v_fma_mix_f32 v0, |v1|, |v2|, v3 op_sel:[0,0,1] op_sel_hi:[1,0,1] clamp", v1, v2, v3)
        self.assertEqual(got_hi, f2u(1.0))
        got_mid = run_fma_mix(
            "v_fma_mix_f32 v0, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] clamp",
            pack_h(9.0, 0.5), pack_h(0.5, 9.0), f2u(0.125),
        )
        self.assertEqual(got_mid, f2u(0.375))

        # Short op_sel / op_sel_hi lists (padded with 0 to 3 entries, matching llvm-mc).
        got_short = run_fma_mix(
            "v_fma_mix_f32 v0, v1, v2, v3 op_sel:[1,0] op_sel_hi:[1,1]",
            pack_h(9.0, 0.5), pack_h(0.5, 9.0), f2u(0.125),
        )
        self.assertEqual(got_short, f2u(0.375))

    def test_fma_mixlo_and_mixhi_f16_mods_and_clamp(self):
        old = 0xCAFEBEEF
        # src0 = f16 lo (-2.0) -> -|-2.0| = -2.0; src1 = f16 hi (-0.25) -> -(-0.25) = 0.25;
        # src2 = f32 (0.25) -> |0.25| = 0.25; result = -2.0 * 0.25 + 0.25 = -0.25.
        v1 = pack_h(-2.0, 5.0)
        v2 = pack_h(9.0, -0.25)
        v3 = f2u(0.25)

        got_lo = run_fma_mix(
            "v_fma_mixlo_f16 v0, -|v1|, -v2, |v3| op_sel:[0,1,0] op_sel_hi:[1,1,0]",
            v1, v2, v3, v0_init=old,
        )
        self.assertEqual(got_lo, (old & 0xFFFF0000) | h2u(-0.25))

        got_lo_clamp = run_fma_mix(
            "v_fma_mixlo_f16 v0, -|v1|, -v2, |v3| op_sel:[0,1,0] op_sel_hi:[1,1,0] clamp",
            v1, v2, v3, v0_init=old,
        )
        self.assertEqual(got_lo_clamp, (old & 0xFFFF0000) | h2u(0.0))

        # src0 = f16 hi (1.5); src1 = f16 lo (0.5); src2 = f32 (0.125) -> 1.5*0.5 + 0.125 = 0.875.
        got_hi = run_fma_mix(
            "v_fma_mixhi_f16 v0, |v1|, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,1,0] clamp",
            pack_h(9.0, -1.5), pack_h(0.5, 9.0), f2u(0.125), v0_init=old,
        )
        self.assertEqual(got_hi, (h2u(0.875) << 16) | (old & 0xFFFF))

    @unittest.skipUnless(llvm_mc_cmd() is not None, "needs llvm-mc or LLVM_MC")
    def test_llvm_mc_decode_roundtrip_and_spirv_val(self):
        old = 0x12345678
        cases = (
            (
                "v_fma_mix_f32 v0, -|v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1] clamp",
                pack_h(-0.5, 9.0), f2u(-1.5), pack_h(7.0, -0.125),
                f2u(0.0),  # -|-0.5| * -(-1.5) + |-0.125| = -0.625 -> clamped to 0.0
            ),
            (
                "v_fma_mix_f32 v0, |v1|, -v2, |v3| op_sel:[0,0,1] op_sel_hi:[1,0,1] clamp",
                pack_h(-0.5, 9.0), f2u(-1.5), pack_h(7.0, -0.125),
                f2u(0.875),  # |-0.5| * -(-1.5) + |-0.125| = 0.75 + 0.125 = 0.875
            ),
            (
                "v_fma_mixlo_f16 v0, -|v1|, v2, |v3| op_sel:[0,1,0] op_sel_hi:[1,1,0] clamp",
                pack_h(-0.5, 9.0), pack_h(7.0, -1.0), f2u(-0.25),
                (old & 0xFFFF0000) | h2u(0.75),  # -0.5 * -1.0 + 0.25 = 0.75
            ),
            (
                "v_fma_mixhi_f16 v0, v1, -v2, |v3| op_sel:[1,0,1] op_sel_hi:[1,1,1] clamp",
                pack_h(9.0, 2.0), pack_h(-0.25, 7.0), pack_h(9.0, -0.125),
                (h2u(0.625) << 16) | (old & 0xFFFF),  # 2.0 * 0.25 + 0.125 = 0.625
            ),
        )
        for line, v1, v2, v3, want in cases:
            with self.subTest(ins=line):
                decoded = llvm_mc_roundtrip(line)
                got = run_fma_mix(decoded, v1, v2, v3, v0_init=old)
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
