"""Validation and positive-control regression tests for v_mad_u16 and v_mad_i16.

Verifies that tools/rdna2spirv.py rejects invalid operand counts, non-VGPR or multi-VGPR destinations
(including op_sel dhi=1 where missing 'r' previously raised KeyError), integer source/destination
neg/abs/sext modifiers, _dpp/_sdwa forms, and unsupported or duplicate instruction modifiers with
rdna2spirv.Unsupported, while valid v_mad_u16/v_mad_i16 forms (with op_sel dhi=0/1 and clamp)
assemble, validate under spirv-val --target-env vulkan1.2, and match the reference semantics under
spirv_mini_interp.
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

from support import mrt0_words, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402

OPS = ("v_mad_u16", "v_mad_i16")
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _translate_single(line):
    return translate_lines(
        "ps_mad16_val",
        [
            "v_mov_b32 v0, 0xaaaa5555",
            "v_mov_b32 v1, 0x00030007",
            "v_mov_b32 v2, 0x0005000b",
            "v_mov_b32 v3, 0x0002000d",
            line,
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ],
    )


def _validate_spvasm(spvasm):
    if not SPIRV_AS or not SPIRV_VAL:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        res_as = subprocess.run(
            [SPIRV_AS, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{res_as.stderr}\n{res_as.stdout}")
        res_val = subprocess.run(
            [SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{res_val.stderr}\n{res_val.stdout}")
    return True


class Mad16ValidationTests(unittest.TestCase):
    def test_reject_unsupported_and_duplicate_modifiers(self):
        bad_mods = [
            "mul:2",
            "mul:4",
            "div:2",
            "op_sel_hi:[0,0,0]",
            "neg_lo:[0,0,0]",
            "neg_hi:[0,0,0]",
            "clamp clamp",
            "clamp op_sel:[0,0,0,0] clamp",
            "op_sel:[0,0,0,0] op_sel:[1,0,0,0]",
            "unknown_mod",
            "foo:1",
            "op_sel:[0,0,0,0] mul:2",
        ]
        for op in OPS:
            for mod in bad_mods:
                line = f"{op} v0, v1, v2, v3 {mod}"
                with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                    _translate_single(line)

    def test_reject_wrong_arity(self):
        bad_args = [
            "v0",
            "v0, v1",
            "v0, v1, v2",
            "v0, v1, v2, v3, v4",
        ]
        for op in OPS:
            for args in bad_args:
                line = f"{op} {args}"
                with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                    _translate_single(line)

    def test_reject_integer_operand_neg_abs_sext(self):
        bad_operands = [
            "-v0, v1, v2, v3",
            "|v0|, v1, v2, v3",
            "neg(v0), v1, v2, v3",
            "abs(v0), v1, v2, v3",
            "sext(v0), v1, v2, v3",
            "v0, -v1, v2, v3",
            "v0, |v1|, v2, v3",
            "v0, neg(v1), v2, v3",
            "v0, abs(v1), v2, v3",
            "v0, sext(v1), v2, v3",
            "v0, v1, -v2, v3",
            "v0, v1, |v2|, v3",
            "v0, v1, sext(v2), v3",
            "v0, v1, v2, -v3",
            "v0, v1, v2, |v3|",
            "v0, v1, v2, sext(v3)",
            "v0, neg(1), v2, v3",
            "v0, abs(1), v2, v3",
        ]
        for op in OPS:
            for args in bad_operands:
                line = f"{op} {args}"
                with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                    _translate_single(line)

    def test_reject_non_vgpr_and_multi_vgpr_dst(self):
        # Includes op_sel dhi=1 where non-VGPR dsts without 'r' previously crashed with KeyError('r').
        bad_dsts = [
            ("s0", ""),
            ("s0", " op_sel:[0,0,0,1]"),
            ("vcc_lo", ""),
            ("vcc_lo", " op_sel:[0,0,0,1]"),
            ("exec_lo", " op_sel:[0,0,0,1]"),
            ("null", ""),
            ("null", " op_sel:[0,0,0,1]"),
            ("0", " op_sel:[0,0,0,1]"),
            ("v[0:1]", ""),
            ("v[0:1]", " op_sel:[0,0,0,0]"),
            ("v[0:1]", " op_sel:[0,0,0,1]"),
        ]
        for op in OPS:
            for dst, mod in bad_dsts:
                line = f"{op} {dst}, v1, v2, v3{mod}"
                with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                    _translate_single(line)

    def test_reject_dpp_and_sdwa_forms(self):
        bad_lines = [
            "v_mad_u16_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_mad_i16_dpp v0, v1, v2, v3 row_shl:1 row_mask:0xf bank_mask:0xf",
            "v_mad_u16_sdwa v0, v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_mad_i16_sdwa v0, v1, v2, v3 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
        ]
        for line in bad_lines:
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                _translate_single(line)

    def test_positive_controls_opsel_clamp_and_spirv_val(self):
        # v0 old = 0xaaaa5555
        # v1 = 0x0003_0007 (hi=3, lo=7)
        # v2 = 0x0005_000b (hi=5, lo=11)
        # v3 = 0x0002_000d (hi=2, lo=13)
        cases = [
            # (setup, inst, expected_v0)
            ([], "v_mad_u16 v0, v1, v2, v3", 0xAAAA0000 | (7 * 11 + 13)),
            ([], "v_mad_u16 v0, v1, v2, v3 op_sel:[1,1,1,0]", 0xAAAA0000 | (3 * 5 + 2)),
            ([], "v_mad_u16 v0, v1, v2, v3 op_sel:[1,0,1,1]", ((3 * 11 + 2) << 16) | 0x5555),
            (
                ["v_mov_b32 v1, 0x01000100", "v_mov_b32 v2, 0x01000100", "v_mov_b32 v3, 0x00050005"],
                "v_mad_u16 v0, v1, v2, v3 op_sel:[0,0,0,0] clamp",
                0xAAAAFFFF,
            ),
            (
                ["v_mov_b32 v1, 0x01000001", "v_mov_b32 v2, 0x01000001", "v_mov_b32 v3, 0x00050000"],
                "v_mad_u16 v0, v1, v2, v3 clamp op_sel:[1,1,1,1]",
                0xFFFF5555,
            ),
            (
                ["v_mov_b32 v1, 0xfffe0007", "v_mov_b32 v2, 0x0003000b", "v_mov_b32 v3, 0x0005000d"],
                "v_mad_i16 v0, v1, v2, v3 op_sel:[1,1,1,0]",
                0xAAAA0000 | (-1 & 0xFFFF),  # -2 * 3 + 5 = -1
            ),
            (
                ["v_mov_b32 v1, 0x7fff0001", "v_mov_b32 v2, 0x00020001", "v_mov_b32 v3, 0x00000000"],
                "v_mad_i16 v0, v1, v2, v3 op_sel:[1,1,1,1] clamp",
                0x7FFF5555,  # 32767 * 2 -> clamped to 0x7fff in high half
            ),
            (
                ["v_mov_b32 v1, 0x00007fff", "v_mov_b32 v2, 0x0000fffe", "v_mov_b32 v3, 0x00000000"],
                "v_mad_i16 v0, v1, v2, v3 clamp",
                0xAAAA8000,  # 32767 * -2 -> clamped to -32768 (0x8000) in low half
            ),
            # Plain negative integer immediate is a signed constant, not a VOP3 neg() modifier.
            (
                ["v_mov_b32 v1, 4", "v_mov_b32 v2, 3"],
                "v_mad_i16 v0, v1, v2, -1",
                0xAAAA000B,  # 4 * 3 + (-1) = 11
            ),
        ]
        validated = False
        for setup, inst, expected in cases:
            with self.subTest(inst=inst):
                spvasm = translate_lines(
                    "ps_mad16_pos",
                    [
                        "v_mov_b32 v0, 0xaaaa5555",
                        "v_mov_b32 v1, 0x00030007",
                        "v_mov_b32 v2, 0x0005000b",
                        "v_mov_b32 v3, 0x0002000d",
                        *setup,
                        inst,
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ],
                )
                self.assertEqual(mrt0_words(spvasm)[0], expected)
                validated = _validate_spvasm(spvasm) or validated
        if not validated:
            self.skipTest("spirv-as / spirv-val not found on PATH")


if __name__ == "__main__":
    unittest.main()
