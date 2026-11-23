"""Validation and positive-control tests for 32-bit bit/misc VALU instructions.

Covers:
  - VOP1/VOP3: v_not_b32, v_bfrev_b32, v_ffbh_u32, v_ffbl_b32, v_ffbh_i32
  - VOP1-only: v_swap_b32
  - VOP3-only: v_bfm_b32, v_lerp_u8

Verifies that tools/rdna2spirv.py rejects invalid arity, non-VGPR or multi-dword
destinations, multi-dword sources, invalid source/destination modifiers (neg, abs,
non-SDWA sext), unsupported encoding suffixes (_e32/_e64/_sdwa/_dpp), and unsupported
modifiers (clamp, mul:*, div:*, op_sel:*, invalid SDWA selectors, unknown modifiers),
while preserving all valid forms including plain negative integer immediates and valid
VOP1 SDWA (with sext) and DPP forms.

Positive controls are assembled with spirv-as, validated with
`spirv-val --target-env vulkan1.2`, and executed in spirv_mini_interp.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
VOP1_VOP3_OPS = ("v_not_b32", "v_bfrev_b32", "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32")


def _validate_spvasm(spvasm: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise unittest.SkipTest("spirv-as / spirv-val not found on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        res_as = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
            capture_output=True,
            text=True,
        )
        if res_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{res_as.stderr}\n{res_as.stdout}")
        res_val = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", spv_path],
            capture_output=True,
            text=True,
        )
        if res_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{res_val.stderr}\n{res_val.stdout}")


def _translate_ps(name: str, body_lines: list[str], export_regs=("v0", "v0", "v0", "v0")) -> str:
    lines = list(body_lines) + [f"exp mrt0 {', '.join(export_regs)} done vm", "s_endpgm"]
    return translate_lines(name, lines)


class BitMiscValuNegativeValidationTests(unittest.TestCase):
    def assertUnsupported(self, insn: str, setup: list[str] | None = None) -> None:
        lines = list(setup or []) + [insn]
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {insn}"):
            _translate_ps("ps_neg_check", lines)

    def test_vop1_vop3_arity_dst_src_and_modifiers(self):
        for op in VOP1_VOP3_OPS:
            cases = [
                # Arity
                f"{op} v0",
                f"{op} v0, v1, v2",
                f"{op}_e32 v0",
                f"{op}_e64 v0, v1, v2",
                # Destination kind / width / modifiers
                f"{op} s0, v1",
                f"{op} vcc_lo, v1",
                f"{op} null, v1",
                f"{op} v[0:1], v2",
                f"{op} -v0, v1",
                f"{op} |v0|, v1",
                f"{op} sext(v0), v1",
                f"{op}_sdwa sext(v0), v1 dst_sel:DWORD src0_sel:BYTE_0",
                # Source width
                f"{op} v0, v[1:2]",
                f"{op} v0, s[0:1]",
                # Source modifiers on non-SDWA
                f"{op} v0, -v1",
                f"{op} v0, |v1|",
                f"{op} v0, sext(v1)",
                f"{op}_e64 v0, neg(1)",
                f"{op}_e64 v0, abs(1)",
                f"{op}_e64 v0, -|v1|",
                # Source modifiers on SDWA / DPP
                f"{op}_sdwa v0, -v1 dst_sel:DWORD src0_sel:WORD_0",
                f"{op}_sdwa v0, |v1| dst_sel:DWORD src0_sel:WORD_0",
                f"{op}_dpp v0, -v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, |v1| quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, sext(v1) quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                # Unsupported modifiers on bare / _e32 / _e64
                f"{op} v0, v1 clamp",
                f"{op}_e64 v0, v1 clamp",
                f"{op}_e64 v0, v1 mul:2",
                f"{op}_e64 v0, v1 div:2",
                f"{op}_e64 v0, v1 op_sel:[0,0]",
                f"{op}_e32 v0, v1 clamp",
                f"{op} v0, v1 bogus_mod:1",
                # Unsupported modifiers on _sdwa
                f"{op}_sdwa v0, v1 dst_sel:DWORD src0_sel:WORD_0 clamp",
                f"{op}_sdwa v0, v1 dst_sel:DWORD src0_sel:WORD_0 mul:2",
                f"{op}_sdwa v0, v1 dst_sel:DWORD src1_sel:WORD_0",
                f"{op}_sdwa v0, v1 dst_sel:INVALID src0_sel:WORD_0",
                f"{op}_sdwa v0, v1 dst_sel:DWORD src0_sel:INVALID",
                f"{op}_sdwa v0, v1 dst_sel:DWORD dst_unused:INVALID",
                f"{op}_sdwa v0, v1 dst_sel:DWORD unknown_mod",
                # Unsupported modifiers on _dpp
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf clamp",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf mul:2",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf op_sel:[0,0]",
            ]
            for insn in cases:
                with self.subTest(op=op, insn=insn):
                    self.assertUnsupported(insn, ["v_mov_b32 v1, 0x12345678", "v_mov_b32 v2, 0x9abcdef0"])

    def test_v_swap_b32_negative(self):
        cases = [
            # Arity
            "v_swap_b32 v0",
            "v_swap_b32 v0, v1, v2",
            # Both operands must be single VGPRs
            "v_swap_b32 s0, v1",
            "v_swap_b32 v0, s0",
            "v_swap_b32 v0, 1",
            "v_swap_b32 v0, -1",
            "v_swap_b32 v0, null",
            "v_swap_b32 v[0:1], v2",
            "v_swap_b32 v0, v[1:2]",
            # Operand modifiers
            "v_swap_b32 -v0, v1",
            "v_swap_b32 v0, -v1",
            "v_swap_b32 |v0|, v1",
            "v_swap_b32 v0, |v1|",
            "v_swap_b32 sext(v0), v1",
            "v_swap_b32 v0, sext(v1)",
            # Unsupported encodings (VOP1-only)
            "v_swap_b32_e64 v0, v1",
            "v_swap_b32_sdwa v0, v1 dst_sel:DWORD src0_sel:DWORD",
            "v_swap_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            # Unsupported modifiers
            "v_swap_b32 v0, v1 clamp",
            "v_swap_b32_e32 v0, v1 mul:2",
            "v_swap_b32 v0, v1 div:2",
            "v_swap_b32 v0, v1 op_sel:[0,0]",
            "v_swap_b32 v0, v1 bogus_mod",
        ]
        for insn in cases:
            with self.subTest(insn=insn):
                self.assertUnsupported(insn, ["v_mov_b32 v0, 1", "v_mov_b32 v1, 2", "v_mov_b32 v2, 3"])

    def test_v_bfm_b32_negative(self):
        cases = [
            # Arity
            "v_bfm_b32 v0, v1",
            "v_bfm_b32 v0, v1, v2, v3",
            # Destination
            "v_bfm_b32 s0, v1, v2",
            "v_bfm_b32 v[0:1], v2, v3",
            "v_bfm_b32 -v0, v1, v2",
            "v_bfm_b32 |v0|, v1, v2",
            "v_bfm_b32 sext(v0), v1, v2",
            # Source width
            "v_bfm_b32 v0, v[1:2], v3",
            "v_bfm_b32 v0, v1, v[2:3]",
            "v_bfm_b32 v0, s[0:1], v2",
            "v_bfm_b32 v0, v1, s[0:1]",
            # Source modifiers
            "v_bfm_b32 v0, -v1, v2",
            "v_bfm_b32 v0, v1, -v2",
            "v_bfm_b32 v0, |v1|, v2",
            "v_bfm_b32 v0, v1, |v2|",
            "v_bfm_b32 v0, sext(v1), v2",
            "v_bfm_b32 v0, v1, sext(v2)",
            "v_bfm_b32 v0, neg(4), 2",
            "v_bfm_b32 v0, 4, abs(2)",
            # Unsupported encodings (VOP3-only)
            "v_bfm_b32_e32 v0, v1, v2",
            "v_bfm_b32_sdwa v0, v1, v2 dst_sel:DWORD src0_sel:DWORD src1_sel:DWORD",
            "v_bfm_b32_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            # Unsupported modifiers
            "v_bfm_b32 v0, v1, v2 clamp",
            "v_bfm_b32_e64 v0, v1, v2 mul:2",
            "v_bfm_b32_e64 v0, v1, v2 div:2",
            "v_bfm_b32_e64 v0, v1, v2 op_sel:[0,0,0]",
            "v_bfm_b32 v0, v1, v2 bogus_mod",
        ]
        for insn in cases:
            with self.subTest(insn=insn):
                self.assertUnsupported(insn, ["v_mov_b32 v1, 4", "v_mov_b32 v2, 8", "v_mov_b32 v3, 1"])

    def test_v_lerp_u8_negative(self):
        cases = [
            # Arity
            "v_lerp_u8 v0, v1, v2",
            "v_lerp_u8 v0, v1, v2, v3, v4",
            # Destination
            "v_lerp_u8 s0, v1, v2, v3",
            "v_lerp_u8 v[0:1], v2, v3, v4",
            "v_lerp_u8 -v0, v1, v2, v3",
            "v_lerp_u8 |v0|, v1, v2, v3",
            "v_lerp_u8 sext(v0), v1, v2, v3",
            # Source width
            "v_lerp_u8 v0, v[1:2], v3, v4",
            "v_lerp_u8 v0, v1, v[2:3], v4",
            "v_lerp_u8 v0, v1, v2, v[3:4]",
            "v_lerp_u8 v0, s[0:1], v2, v3",
            # Source modifiers
            "v_lerp_u8 v0, -v1, v2, v3",
            "v_lerp_u8 v0, v1, |v2|, v3",
            "v_lerp_u8 v0, v1, v2, sext(v3)",
            "v_lerp_u8 v0, neg(1), v2, v3",
            "v_lerp_u8 v0, v1, abs(1), v3",
            # Unsupported encodings (VOP3-only)
            "v_lerp_u8_e32 v0, v1, v2, v3",
            "v_lerp_u8_sdwa v0, v1, v2, v3 dst_sel:DWORD src0_sel:DWORD",
            "v_lerp_u8_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            # Unsupported modifiers
            "v_lerp_u8 v0, v1, v2, v3 clamp",
            "v_lerp_u8_e64 v0, v1, v2, v3 clamp",
            "v_lerp_u8_e64 v0, v1, v2, v3 mul:2",
            "v_lerp_u8_e64 v0, v1, v2, v3 div:2",
            "v_lerp_u8_e64 v0, v1, v2, v3 op_sel:[0,0,0,0]",
            "v_lerp_u8 v0, v1, v2, v3 bogus_mod",
        ]
        for insn in cases:
            with self.subTest(insn=insn):
                self.assertUnsupported(
                    insn,
                    ["v_mov_b32 v1, 0x10203040", "v_mov_b32 v2, 0x50607080", "v_mov_b32 v3, 0x01000100"],
                )


class BitMiscValuPositiveControlTests(unittest.TestCase):
    def test_vop1_vop3_forms_and_negative_immediates(self):
        # Expected outputs for src=0x00000010 and src=-1 (0xFFFFFFFF)
        expected_16 = {
            "v_not_b32": (~0x10) & MASK32,
            "v_bfrev_b32": 0x08000000,
            "v_ffbh_u32": 27,
            "v_ffbl_b32": 4,
            "v_ffbh_i32": 27,
        }
        expected_neg1 = {
            "v_not_b32": 0x00000000,
            "v_bfrev_b32": 0xFFFFFFFF,
            "v_ffbh_u32": 0,
            "v_ffbl_b32": 0,
            "v_ffbh_i32": 0xFFFFFFFF,
        }
        for op in VOP1_VOP3_OPS:
            for suffix in ("", "_e32", "_e64"):
                with self.subTest(op=op, suffix=suffix, kind="vgpr"):
                    spv = _translate_ps(
                        f"ps_pos_{op}{suffix}",
                        ["v_mov_b32 v1, 0x10", f"{op}{suffix} v0, v1"],
                    )
                    _validate_spvasm(spv)
                    self.assertEqual(mrt0_words(spv)[0] & MASK32, expected_16[op])
                with self.subTest(op=op, suffix=suffix, kind="neg_imm"):
                    spv = _translate_ps(
                        f"ps_negimm_{op}{suffix}",
                        [f"{op}{suffix} v0, -1"],
                    )
                    _validate_spvasm(spv)
                    self.assertEqual(mrt0_words(spv)[0] & MASK32, expected_neg1[op])

    def test_vop1_sdwa_and_sext_positive(self):
        # v1 = 0x12345680 -> BYTE_0 = 0x80; sext(BYTE_0) = 0xFFFFFF80
        expected_sext = {
            "v_not_b32": (~0xFFFFFF80) & MASK32,   # 0x0000007F
            "v_bfrev_b32": 0x01FFFFFF,
            "v_ffbh_u32": 0,
            "v_ffbl_b32": 7,
            "v_ffbh_i32": 25,                       # sign=1, first 0-bit from MSB is at bit 6 -> 31-6 = 25
        }
        for op in VOP1_VOP3_OPS:
            with self.subTest(op=op):
                spv = _translate_ps(
                    f"ps_sdwa_{op}",
                    [
                        "v_mov_b32 v1, 0x12345680",
                        f"{op}_sdwa v0, sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0",
                    ],
                )
                _validate_spvasm(spv)
                self.assertEqual(mrt0_words(spv)[0] & MASK32, expected_sext[op])

    def test_vop1_dpp_positive(self):
        for op in VOP1_VOP3_OPS:
            with self.subTest(op=op):
                spv = _translate_ps(
                    f"ps_dpp_{op}",
                    [
                        "v_mov_b32 v0, 0",
                        "v_mov_b32 v1, 0x12345678",
                        f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                    ],
                )
                _validate_spvasm(spv)

    def test_v_swap_b32_positive(self):
        for mnemonic in ("v_swap_b32", "v_swap_b32_e32"):
            with self.subTest(mnemonic=mnemonic):
                spv = _translate_ps(
                    f"ps_pos_{mnemonic}",
                    [
                        "v_mov_b32 v1, 0x11223344",
                        "v_mov_b32 v2, 0xAABBCCDD",
                        f"{mnemonic} v1, v2",
                    ],
                    export_regs=("v1", "v2", "v1", "v2"),
                )
                _validate_spvasm(spv)
                self.assertEqual(
                    [w & MASK32 for w in mrt0_words(spv)],
                    [0xAABBCCDD, 0x11223344, 0xAABBCCDD, 0x11223344],
                )

    def test_v_bfm_b32_positive(self):
        for mnemonic in ("v_bfm_b32", "v_bfm_b32_e64"):
            with self.subTest(mnemonic=mnemonic, kind="vgpr"):
                spv = _translate_ps(
                    f"ps_pos_{mnemonic}",
                    [
                        "v_mov_b32 v1, 8",
                        "v_mov_b32 v2, 4",
                        f"{mnemonic} v0, v1, v2",
                    ],
                )
                _validate_spvasm(spv)
                self.assertEqual(mrt0_words(spv)[0] & MASK32, 0x00000FF0)
            with self.subTest(mnemonic=mnemonic, kind="neg_imm"):
                # width = -1 -> masked to 31 -> 0x7FFFFFFF; offset = 1 -> 0xFFFFFFFE
                spv = _translate_ps(
                    f"ps_negimm_{mnemonic}",
                    [f"{mnemonic} v0, -1, 1"],
                )
                _validate_spvasm(spv)
                self.assertEqual(mrt0_words(spv)[0] & MASK32, 0xFFFFFFFE)

    def test_v_lerp_u8_positive(self):
        for mnemonic in ("v_lerp_u8", "v_lerp_u8_e64"):
            with self.subTest(mnemonic=mnemonic, kind="vgpr"):
                spv = _translate_ps(
                    f"ps_pos_{mnemonic}",
                    [
                        "v_mov_b32 v1, 0x00102030",
                        "v_mov_b32 v2, 0x01112131",
                        "v_mov_b32 v3, 0x01000100",
                        f"{mnemonic} v0, v1, v2, v3",
                    ],
                )
                _validate_spvasm(spv)
                # Byte0: (0x30+0x31+0)>>1=0x30; Byte1: (0x20+0x21+1)>>1=0x21;
                # Byte2: (0x10+0x11+0)>>1=0x10; Byte3: (0x00+0x01+1)>>1=0x01
                self.assertEqual(mrt0_words(spv)[0] & MASK32, 0x01102130)
            with self.subTest(mnemonic=mnemonic, kind="neg_imm"):
                # src0 = -1 (0xFFFFFFFF), src1 = 0x00000000, src2 = 0x01010101 -> (255+0+1)>>1 = 128 (0x80) per byte
                spv = _translate_ps(
                    f"ps_negimm_{mnemonic}",
                    [
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0x01010101",
                        f"{mnemonic} v0, -1, v2, v3",
                    ],
                )
                _validate_spvasm(spv)
                self.assertEqual(mrt0_words(spv)[0] & MASK32, 0x80808080)


if __name__ == "__main__":
    unittest.main()
