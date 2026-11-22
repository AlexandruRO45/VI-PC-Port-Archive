"""Semantic tests for `null` (`0.0`) source operands across implemented binary64 (`_f64`) VALU ops.

Every test assembles RDNA2 instructions with `llvm-mc -mcpu=gfx1013 -show-encoding`, translates the
disassembly with `tools/rdna2spirv.py`, assembles and validates with `spirv-as` + `spirv-val
--target-env vulkan1.2`, and executes the resulting module in `tests/semantic/spirv_mini_interp.py`.

Implemented `_f64` VALU instructions:
  - `v_add_f64`, `v_cvt_f32_f64`, `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64`
  - `v_frexp_mant_f64`, `v_frexp_exp_i32_f64`
  - `v_cmp_f_f64`, `v_cmp_tru_f64`, `v_cmpx_f_f64`, `v_cmpx_tru_f64`
"""
import gc
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000


def d2u(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _find_llvm_mc() -> str | None:
    env = os.environ.get("LLVM_MC")
    if env:
        return env if os.path.isfile(env) else shutil.which(env)
    return shutil.which("llvm-mc")


def assemble_gfx1013(lines: list[str]) -> list[str]:
    """Assemble with llvm-mc -mcpu=gfx1013 -show-encoding and return the printed disassembly lines."""
    llvm_mc = _find_llvm_mc()
    if not llvm_mc:
        raise unittest.SkipTest("llvm-mc not found; set LLVM_MC or add it to PATH")
    src = "\n".join(lines) + "\n"
    res = subprocess.run(
        [llvm_mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"],
        input=src,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{src}\n{res.stderr}")
    out = []
    for raw in res.stdout.splitlines():
        s = raw.strip()
        if s and not s.startswith("."):
            out.append(s)
    if len(out) != len(lines):
        raise AssertionError(f"expected {len(lines)} assembled lines, got {len(out)}:\n{res.stdout}")
    return out


def validate_spirv(spv_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}\n{spv_text}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}\n{spv_text}")


def translate_and_validate(lines: list[str]) -> str:
    assembled = assemble_gfx1013(list(lines) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    with tempfile.TemporaryDirectory() as d:
        disasm_dir = os.path.join(d, "disasm")
        os.mkdir(disasm_dir)
        src_path = os.path.join(disasm_dir, "ps_f64_null.s")
        with open(os.path.join(d, "ps_f64_null.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("; ps_f64_null\n" + "".join(f"\t{ln}\n" for ln in assembled))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src_path)
            gc.collect()
    validate_spirv(spv)
    return spv


def run_f64_pair(lines: list[str]) -> tuple[int, str]:
    spv = translate_and_validate(lines)
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[1] << 32) | out[0], spv


def run_u32(lines: list[str]) -> tuple[int, str]:
    spv = translate_and_validate(lines)
    return si.run_fragment(spv)["%out_mrt0"][0], spv


def mov_f64(lo_reg: int, hi_reg: int, bits: int) -> list[str]:
    return [
        f"v_mov_b32 v{lo_reg}, {bits & 0xFFFFFFFF:#x}",
        f"v_mov_b32 v{hi_reg}, {(bits >> 32) & 0xFFFFFFFF:#x}",
    ]


@unittest.skipUnless(_find_llvm_mc(), "needs llvm-mc")
class F64NullSrcTests(unittest.TestCase):
    def test_v_add_f64_null_and_modifiers(self):
        cases = (
            ("v_add_f64 v[0:1], null, v[2:3]", d2u(1.5), d2u(1.5)),
            ("v_add_f64 v[0:1], v[2:3], null", d2u(-3.25), d2u(-3.25)),
            ("v_add_f64 v[0:1], null, v[2:3]", NEG_ZERO_F64, POS_ZERO_F64),
            ("v_add_f64 v[0:1], v[2:3], null", NEG_ZERO_F64, POS_ZERO_F64),
            ("v_add_f64 v[0:1], null, null", POS_ZERO_F64, POS_ZERO_F64),
            # -null is -0.0: adding -0.0 preserves -0.0 under IEEE-754 round-to-nearest-even.
            ("v_add_f64 v[0:1], -null, v[2:3]", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], v[2:3], -null", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], -null, -null", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], abs(null), v[2:3]", NEG_ZERO_F64, POS_ZERO_F64),
            ("v_add_f64 v[0:1], -abs(null), v[2:3]", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], v[2:3], -|null|", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], -null, v[2:3]", d2u(2.25), d2u(2.25)),
            # Output modifiers on -null + -0.0: mul:2 keeps -0.0, clamp gives +0.0.
            ("v_add_f64 v[0:1], -null, v[2:3] mul:2", NEG_ZERO_F64, NEG_ZERO_F64),
            ("v_add_f64 v[0:1], -null, v[2:3] clamp", NEG_ZERO_F64, POS_ZERO_F64),
        )
        for insn, src_bits, want in cases:
            with self.subTest(insn=insn, src=hex(src_bits)):
                got, spv = run_f64_pair(mov_f64(2, 3, src_bits) + [insn])
                self.assertIn("OpCapability Float64", spv)
                self.assertEqual(got, want)

    def test_v_add_f64_null_with_round_mode(self):
        rtz = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
        got_neg, _ = run_f64_pair(mov_f64(2, 3, NEG_ZERO_F64) + [rtz, "v_add_f64 v[0:1], -null, v[2:3]"])
        self.assertEqual(got_neg, NEG_ZERO_F64)
        got_pos, _ = run_f64_pair(mov_f64(2, 3, NEG_ZERO_F64) + [rtz, "v_add_f64 v[0:1], null, v[2:3]"])
        self.assertEqual(got_pos, POS_ZERO_F64)

    def test_v_cvt_f32_f64_null_and_modifiers(self):
        cases = (
            ("v_cvt_f32_f64 v0, null", POS_ZERO_F32),
            ("v_cvt_f32_f64_e64 v0, -null", NEG_ZERO_F32),
            ("v_cvt_f32_f64_e64 v0, abs(null)", POS_ZERO_F32),
            ("v_cvt_f32_f64_e64 v0, -abs(null)", NEG_ZERO_F32),
            ("v_cvt_f32_f64_e64 v0, -null mul:2", NEG_ZERO_F32),
            ("v_cvt_f32_f64_e64 v0, -null clamp", POS_ZERO_F32),
        )
        for insn, want in cases:
            with self.subTest(insn=insn):
                got, spv = run_u32([insn])
                self.assertIn("OpCapability Float64", spv)
                self.assertEqual(got, want)

    def test_v_sqrt_f64_null_and_modifiers(self):
        cases = (
            ("v_sqrt_f64 v[0:1], null", POS_ZERO_F64),
            ("v_sqrt_f64_e64 v[0:1], -null", NEG_ZERO_F64),
            ("v_sqrt_f64_e64 v[0:1], abs(null)", POS_ZERO_F64),
            ("v_sqrt_f64_e64 v[0:1], -abs(null)", NEG_ZERO_F64),
            ("v_sqrt_f64_e64 v[0:1], -null mul:2", NEG_ZERO_F64),
            ("v_sqrt_f64_e64 v[0:1], -null clamp", POS_ZERO_F64),
        )
        for insn, want in cases:
            with self.subTest(insn=insn):
                got, spv = run_f64_pair([insn])
                self.assertIn("OpCapability Float64", spv)
                self.assertEqual(got, want)

    def test_v_rcp_and_rsq_f64_null_and_modifiers(self):
        # OpFDiv by 0.0 and InverseSqrt(0.0) are undefined in SPIR-V, so spirv_mini_interp rejects execution while
        # translation + spirv-val succeed and the expected FAbs/OpFNegate modifiers are emitted on the (0, 0) double.
        for op, spv_pattern, err_re in (
            ("v_rcp_f64", r"OpFDiv %double", r"OpFDiv by zero"),
            ("v_rsq_f64", r"InverseSqrt", r"InverseSqrt"),
        ):
            for suffix, expect_abs, expect_neg in (
                (f"{op} v[0:1], null", False, False),
                (f"{op}_e64 v[0:1], -null", False, True),
                (f"{op}_e64 v[0:1], abs(null)", True, False),
                (f"{op}_e64 v[0:1], -abs(null)", True, True),
            ):
                with self.subTest(insn=suffix):
                    spv = translate_and_validate([suffix])
                    self.assertIn("OpCapability Float64", spv)
                    self.assertRegex(spv, spv_pattern)
                    self.assertEqual("FAbs" in spv, expect_abs)
                    self.assertEqual("OpFNegate %double" in spv, expect_neg)
                    with self.assertRaisesRegex(si.UnsupportedConstruct, err_re):
                        si.run_fragment(spv)

    def test_v_frexp_mant_f64_null_and_modifiers(self):
        plain_cases = (
            ("v_frexp_mant_f64 v[0:1], null", POS_ZERO_F64),
            ("v_frexp_mant_f64_e64 v[0:1], -null", NEG_ZERO_F64),
            ("v_frexp_mant_f64_e64 v[0:1], abs(null)", POS_ZERO_F64),
            ("v_frexp_mant_f64_e64 v[0:1], -abs(null)", NEG_ZERO_F64),
        )
        for insn, want in plain_cases:
            with self.subTest(insn=insn):
                got, spv = run_f64_pair([insn])
                self.assertNotIn("OpCapability Float64", spv)
                self.assertEqual(got, want)

        mod_cases = (
            ("v_frexp_mant_f64_e64 v[0:1], -null mul:2", NEG_ZERO_F64),
            ("v_frexp_mant_f64_e64 v[0:1], -null clamp", POS_ZERO_F64),
        )
        for insn, want in mod_cases:
            with self.subTest(insn=insn):
                got, spv = run_f64_pair([insn])
                self.assertIn("OpCapability Float64", spv)
                self.assertEqual(got, want)

    def test_v_frexp_exp_i32_f64_null_and_modifiers(self):
        for insn in (
            "v_frexp_exp_i32_f64 v0, null",
            "v_frexp_exp_i32_f64_e64 v0, -null",
            "v_frexp_exp_i32_f64_e64 v0, abs(null)",
            "v_frexp_exp_i32_f64_e64 v0, -abs(null)",
        ):
            with self.subTest(insn=insn):
                got, spv = run_u32([insn])
                self.assertNotIn("OpCapability Float64", spv)
                self.assertEqual(got, 0)

    def test_v_cmp_and_cmpx_f64_constant_predicates_with_null(self):
        cases = (
            ("v_cmp_f_f64 vcc_lo, null, v[2:3]", "vcc_lo", 0),
            ("v_cmp_tru_f64 vcc_lo, v[2:3], null", "vcc_lo", 1),
            ("v_cmp_f_f64_e64 s0, -null, v[2:3]", "s0", 0),
            ("v_cmp_tru_f64_e64 s0, abs(null), -null", "s0", 1),
        )
        for insn, mask, want in cases:
            with self.subTest(insn=insn):
                got, _ = run_u32([
                    f"s_mov_b32 {mask}, {-1 if not want else 0}",
                    insn,
                    f"v_cndmask_b32 v0, 0x40000000, 0x40000001, {mask}",
                ])
                self.assertEqual(got & 1, want)

        # v_cmpx_tru_f64 keeps EXEC active; v_cmpx_f_f64 disables EXEC (no export).
        got_tru, _ = run_u32([
            "v_mov_b32 v0, 0x40000001",
            "v_cmpx_tru_f64_e64 v[2:3], -abs(null)",
        ])
        self.assertEqual(got_tru, 0x40000001)

        spv_f = translate_and_validate(["v_cmpx_f_f64 null, v[2:3]"])
        self.assertEqual(si.run_fragment(spv_f), {})


if __name__ == "__main__":
    unittest.main()
