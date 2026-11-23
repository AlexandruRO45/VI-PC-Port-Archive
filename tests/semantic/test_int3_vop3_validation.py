"""Operand and modifier validation tests for 3-input 32-bit integer VOP3 ops.

Covers all 8 ops handled in tools/rdna2spirv.py:
  v_bfi_b32, v_lshl_or_b32, v_lshl_add_u32, v_add_lshl_u32,
  v_or3_b32, v_xor3_b32, v_add3_u32, v_and_or_b32

Verifies that:
  - unmodelled modifiers (clamp, mul:2/4, div:2, op_sel, op_sel_hi, neg_lo, unknown),
    _dpp / _sdwa encodings, integer operand neg/abs/sext, non-VGPR / multi-VGPR
    destinations, multi-register sources, and wrong operand counts all raise Unsupported;
  - valid plain, _e64, and negative-inline-immediate forms translate, pass
    spirv-as + spirv-val --target-env vulkan1.2, and produce exact 32-bit results
    in spirv_mini_interp.
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

OPS = (
    "v_bfi_b32",
    "v_lshl_or_b32",
    "v_lshl_add_u32",
    "v_add_lshl_u32",
    "v_or3_b32",
    "v_xor3_b32",
    "v_add3_u32",
    "v_and_or_b32",
)

REFERENCE = {
    "v_bfi_b32": lambda x, y, z: ((x & y) | ((~x & MASK32) & z)) & MASK32,
    "v_lshl_or_b32": lambda x, y, z: (((x << (y & 31)) & MASK32) | z) & MASK32,
    "v_lshl_add_u32": lambda x, y, z: (((x << (y & 31)) & MASK32) + z) & MASK32,
    "v_add_lshl_u32": lambda x, y, z: (((x + y) & MASK32) << (z & 31)) & MASK32,
    "v_or3_b32": lambda x, y, z: (x | y | z) & MASK32,
    "v_xor3_b32": lambda x, y, z: (x ^ y ^ z) & MASK32,
    "v_add3_u32": lambda x, y, z: (x + y + z) & MASK32,
    "v_and_or_b32": lambda x, y, z: ((x & y) | z) & MASK32,
}


def _shader(insn_line, s0=0x13579BDF, s1=0x00000005, s2=0xFEDCBA98):
    return [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        f"v_mov_b32 v4, 0x12345678",
        "s_mov_b32 s0, 0x00FF00FF",
        insn_line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]


def _validate_spirv(spvasm, tmpdir):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise unittest.SkipTest("spirv-as / spirv-val not on PATH")
    asm_path = os.path.join(tmpdir, "shader.spvasm")
    spv_path = os.path.join(tmpdir, "shader.spv")
    with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(spvasm)
    res_as = subprocess.run(
        [spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
        capture_output=True, text=True, check=False,
    )
    if res_as.returncode != 0:
        raise AssertionError(f"spirv-as failed:\n{res_as.stderr}")
    res_val = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.2", spv_path],
        capture_output=True, text=True, check=False,
    )
    if res_val.returncode != 0:
        raise AssertionError(f"spirv-val failed:\n{res_val.stderr}")


class TestInt3Vop3Validation(unittest.TestCase):
    def _assert_unsupported(self, line):
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {line}"):
            translate_lines("ps_int3_neg", _shader(line))

    def test_reject_modifiers(self):
        bad_mods = (
            "clamp",
            "mul:2",
            "mul:4",
            "div:2",
            "op_sel:[0,0,0]",
            "op_sel_hi:[0,0,0]",
            "neg_lo:[0,0,0]",
            "unknown_mod:1",
        )
        for op in OPS:
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self._assert_unsupported(f"{op} v0, v1, v2, v3 {mod}")

    def test_reject_dpp_and_sdwa(self):
        for op in OPS:
            cases = (
                f"{op}_dpp v0, v1, v2, v3 quad_perm:[0,1,2,3]",
                f"{op}_dpp v0, v1, v2, v3",
                f"{op}_sdwa v0, v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0",
                f"{op}_sdwa v0, v1, v2, v3",
            )
            for line in cases:
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_reject_operand_modifiers(self):
        for op in OPS:
            cases = (
                f"{op} -v0, v1, v2, v3",
                f"{op} |v0|, v1, v2, v3",
                f"{op} sext(v0), v1, v2, v3",
                f"{op} v0, -v1, v2, v3",
                f"{op} v0, |v1|, v2, v3",
                f"{op} v0, sext(v1), v2, v3",
                f"{op} v0, v1, -v2, v3",
                f"{op} v0, v1, |v2|, v3",
                f"{op} v0, v1, sext(v2), v3",
                f"{op} v0, v1, v2, -v3",
                f"{op} v0, v1, v2, |v3|",
                f"{op} v0, v1, v2, sext(v3)",
                f"{op} v0, neg(1), v2, v3",
                f"{op} v0, abs(1), v2, v3",
            )
            for line in cases:
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_reject_bad_dst_and_operand_count(self):
        for op in OPS:
            cases = (
                f"{op} s0, v1, v2, v3",
                f"{op} null, v1, v2, v3",
                f"{op} exec_lo, v1, v2, v3",
                f"{op} v[0:1], v1, v2, v3",
                f"{op} v0, v[1:2], v2, v3",
                f"{op} v0, v1, v2",
                f"{op} v0, v1, v2, v3, v4",
            )
            for line in cases:
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_dead_sentinel_v_and_or_b32_underscore_rejected(self):
        self._assert_unsupported("v_and_or_b32_ v0, v1, v2, v3")

    def test_positive_controls_spirv_val_and_semantics(self):
        s0, s1, s2 = 0x13579BDF, 0x00000005, 0xFEDCBA98
        with tempfile.TemporaryDirectory() as tmpdir:
            for op in OPS:
                expected = REFERENCE[op](s0, s1, s2)
                for suffix in ("", "_e64"):
                    with self.subTest(op=op + suffix, form="vgpr"):
                        spvasm = translate_lines("ps_int3_pos", _shader(f"{op}{suffix} v0, v1, v2, v3", s0, s1, s2))
                        _validate_spirv(spvasm, tmpdir)
                        words = mrt0_words(spvasm)
                        self.assertEqual(words, [expected] * 4)

                # Negative inline constant (-1 == 0xFFFFFFFF) is valid integer syntax, not VOP3 neg().
                exp_neg1 = REFERENCE[op](0xFFFFFFFF, s1, s2)
                with self.subTest(op=op, form="neg1_imm"):
                    spvasm = translate_lines("ps_int3_neg1", _shader(f"{op} v0, -1, v2, v3", s0, s1, s2))
                    _validate_spirv(spvasm, tmpdir)
                    words = mrt0_words(spvasm)
                    self.assertEqual(words, [exp_neg1] * 4)

                # Mixed SGPR + VGPR + inline constant.
                exp_mix = REFERENCE[op](0x00FF00FF, s1, 7)
                with self.subTest(op=op, form="sgpr_mix"):
                    spvasm = translate_lines("ps_int3_mix", _shader(f"{op} v0, s0, v2, 7", s0, s1, s2))
                    _validate_spirv(spvasm, tmpdir)
                    words = mrt0_words(spvasm)
                    self.assertEqual(words, [exp_mix] * 4)

    def test_v_bfi_b32_known_mask_paths(self):
        y, z = 0xAABBCCDD, 0x11223344
        cases = (
            ("0", 0),
            ("-1", 0xFFFFFFFF),
            ("0x0000ff00", 0x0000FF00),
            ("0x0f0f0f0f", 0x0F0F0F0F),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for mask_tok, mask_val in cases:
                expected = REFERENCE["v_bfi_b32"](mask_val, y, z)
                with self.subTest(mask=mask_tok):
                    spvasm = translate_lines("ps_bfi_mask", _shader(f"v_bfi_b32 v0, {mask_tok}, v2, v3", 0, y, z))
                    _validate_spirv(spvasm, tmpdir)
                    words = mrt0_words(spvasm)
                    self.assertEqual(words, [expected] * 4)


if __name__ == "__main__":
    unittest.main()
