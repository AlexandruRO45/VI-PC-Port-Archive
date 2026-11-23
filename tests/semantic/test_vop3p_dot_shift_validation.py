"""Synthetic validation regression tests for VOP3P/VOP2 dot and packed-16-bit shift handlers.

Covers operand count, destination register kind/width/modifiers, _dpp/_sdwa rejection,
and strict bracketed bit-list modifier parsing for:
  1. Packed integer dot products:
     v_dot2_i32_i16, v_dot2_u32_u16, v_dot4_u32_u8, v_dot8_i32_i4, v_dot8_u32_u4
     and accumulating VOP2 forms v_dot2c_i32_i16, v_dot2c_u32_u16, v_dot4c_u32_u8,
     v_dot8c_i32_i4, v_dot8c_u32_u4.
  2. Half-to-float dot products:
     v_dot2_f32_f16 and v_dot2c_f32_f16.
  3. Packed 16-bit integer shifts:
     v_pk_lshlrev_b16, v_pk_lshrrev_b16, v_pk_ashrrev_i16.

Positive controls verify that valid forms still translate, pass spirv-val
(--target-env vulkan1.2), and evaluate to exact expected words in spirv_mini_interp.
Standard library only; no GPU, no game files.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for path in (ROOT, TOOLS_DIR, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

INT_DOT_VOP3P = (
    "v_dot2_i32_i16",
    "v_dot2_u32_u16",
    "v_dot4_u32_u8",
    "v_dot8_i32_i4",
    "v_dot8_u32_u4",
)
INT_DOT_VOP2 = (
    "v_dot2c_i32_i16",
    "v_dot2c_u32_u16",
    "v_dot4c_u32_u8",
    "v_dot8c_i32_i4",
    "v_dot8c_u32_u4",
)
PK_SHIFT_OPS = (
    "v_pk_lshlrev_b16",
    "v_pk_lshrrev_b16",
    "v_pk_ashrrev_i16",
)


def vgpr(r: int, n: int = 1, *, neg: bool = False, abs_: bool = False) -> dict:
    d = {"k": "vgpr", "r": r, "n": n}
    if neg:
        d["neg"] = True
    if abs_:
        d["abs"] = True
    return d


def sgpr(r: int, n: int = 1) -> dict:
    return {"k": "sgpr", "r": r, "n": n}


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def validate_spirv(spv_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "shader.spvasm")
        bin_p = os.path.join(d, "shader.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p],
            capture_output=True,
            text=True,
        )
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", bin_p],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def translate_prog(name: str, body: list[str]) -> str:
    return support.translate_lines(
        name,
        body + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


def emit_synthetic(op: str, args: list[dict], mods: list[str]) -> None:
    gen = rdna2spirv.Gen.__new__(rdna2spirv.Gen)
    gen._insn({"op": op, "args": args, "mods": mods})


class IntDotValidationTests(unittest.TestCase):
    def test_vop3p_rejects_bad_operand_counts_and_destinations(self):
        for op in INT_DOT_VOP3P:
            bad_ir = (
                ([], []),
                ([vgpr(0), vgpr(1), vgpr(2)], []),
                ([vgpr(0), vgpr(1), vgpr(2), vgpr(3), vgpr(4)], []),
                ([sgpr(0), vgpr(1), vgpr(2), vgpr(3)], []),
                ([{"k": "special", "t": "vcc_lo"}, vgpr(1), vgpr(2), vgpr(3)], []),
                ([vgpr(0, 2), vgpr(1), vgpr(2), vgpr(3)], []),
                ([vgpr(0, neg=True), vgpr(1), vgpr(2), vgpr(3)], []),
                ([vgpr(0, abs_=True), vgpr(1), vgpr(2), vgpr(3)], []),
            )
            for args, mods in bad_ir:
                with self.subTest(op=op, args=args):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op, args, mods)

        for line in (
            "v_dot2_i32_i16 v0, v1, v2",
            "v_dot2_u32_u16 v0, v1, v2, v3, v4",
            "v_dot4_u32_u8 s0, v1, v2, v3",
            "v_dot8_i32_i4 v[0:1], v1, v2, v3",
            "v_dot8_u32_u4 -v0, v1, v2, v3",
            "v_dot2_i32_i16 |v0|, v1, v2, v3",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_prog("ps_int_dot_bad_dst", [
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        line,
                    ])

    def test_vop3p_rejects_malformed_duplicate_and_unknown_modifiers(self):
        good_args = [vgpr(0), vgpr(1), vgpr(2), vgpr(3)]
        bad_mods = (
            ["op_sel:bad"],
            ["op_sel:foo[0,0,0]"],
            ["op_sel:[0,0,0"],
            ["op_sel:[0,0,0]]"],
            ["op_sel:[]"],
            ["op_sel:[0,x,0]"],
            ["op_sel:[0,2,0]"],
            ["op_sel:[+0,0,0]"],
            ["op_sel:[0,0]"],
            ["op_sel:[0,0,0,0]"],
            ["op_sel_hi:foo[1,1,1]"],
            ["op_sel_hi:[1,1,1]]"],
            ["op_sel_hi:[1,2,1]"],
            ["op_sel_hi:[1,1]"],
            ["op_sel_hi:[1,1,1,1]"],
            ["op_sel:[0,0,0]", "op_sel:[0,0,0]"],
            ["op_sel:[0,0,0]", "op_sel:[1,0,0]"],
            ["op_sel_hi:[1,1,1]", "op_sel_hi:[1,1,1]"],
            ["clamp", "clamp"],
            ["neg_lo:[0,0,0]"],
            ["neg_hi:[0,0,0]"],
            ["mul:2"],
            ["div:2"],
            ["op_sel_extra:[0,0,0]"],
        )
        for op in INT_DOT_VOP3P:
            for mods in bad_mods:
                with self.subTest(op=op, mods=mods):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op, good_args, mods)
            for suffix in ("_dpp", "_sdwa"):
                with self.subTest(op=op + suffix):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op + suffix, good_args, [])

        # End-to-end assembly text forms that previously crashed with ValueError or slipped past next().
        for line in (
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,x,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 op_sel:[]",
            "v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[0,0,0] op_sel:[1,0,0]",
            "v_dot8_i32_i4 v0, v1, v2, v3 op_sel_hi:[1,1,1] op_sel_hi:[0,1,1]",
            "v_dot8_u32_u4 v0, v1, v2, v3 clamp clamp",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_prog("ps_int_dot_bad_mods", [
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 2",
                        "v_mov_b32 v3, 3",
                        line,
                    ])

    def test_vop2_dotc_rejects_bad_operands_destinations_and_modifiers(self):
        for op in INT_DOT_VOP2:
            for args, mods in (
                ([], []),
                ([vgpr(0), vgpr(1)], []),
                ([vgpr(0), vgpr(1), vgpr(2), vgpr(3)], []),
                ([sgpr(0), vgpr(1), vgpr(2)], []),
                ([vgpr(0, 2), vgpr(1), vgpr(2)], []),
                ([vgpr(0, neg=True), vgpr(1), vgpr(2)], []),
                ([vgpr(0, abs_=True), vgpr(1), vgpr(2)], []),
                ([vgpr(0), vgpr(1), vgpr(2)], ["clamp", "clamp"]),
                ([vgpr(0), vgpr(1), vgpr(2)], ["op_sel:[0,0,0]"]),
                ([vgpr(0), vgpr(1), vgpr(2)], ["mul:2"]),
            ):
                with self.subTest(op=op, args=args, mods=mods):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op, args, mods)
            for suffix in ("_dpp", "_sdwa"):
                with self.subTest(op=op + suffix):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op + suffix, [vgpr(0), vgpr(1), vgpr(2)], [])

    def test_int_dot_positive_controls(self):
        spv = translate_prog("ps_int_dot_pos", [
            "v_mov_b32 v1, 0x00020003",
            "v_mov_b32 v2, 0x00040005",
            "v_mov_b32 v3, 10",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[1,1,1]",
            "v_dot2c_u32_u16_e64 v0, v1, v2 clamp",
            "v_dot4_u32_u8 v1, 0x01020304, 0x01010101, 5 op_sel:[0,0,0] op_sel_hi:[1,1,1]",
            "v_dot8_i32_i4 v2, 0x11111111, 0x22222222, 4",
            "v_dot8c_u32_u4_e32 v3, 0x11111111, 0x11111111",
        ])
        validate_spirv(spv)
        words = support.mrt0_words(spv)
        # v0: (2*4 + 3*5 + 10) = 33, then + (2*4 + 3*5) = 56
        # v1: (1+2+3+4) + 5 = 15
        # v2: 8*(1*2) + 4 = 20
        # v3: 8*(1*1) + 10 = 18
        self.assertEqual(words, [56, 15, 20, 18])


class HalfDotF32ValidationTests(unittest.TestCase):
    def test_v_dot2_f32_f16_rejects_bad_operands_destinations_and_modifiers(self):
        good_args = [vgpr(0), vgpr(1), vgpr(2), vgpr(3)]
        for args in (
            [],
            [vgpr(0), vgpr(1), vgpr(2)],
            [vgpr(0), vgpr(1), vgpr(2), vgpr(3), vgpr(4)],
            [sgpr(0), vgpr(1), vgpr(2), vgpr(3)],
            [vgpr(0, 2), vgpr(1), vgpr(2), vgpr(3)],
            [vgpr(0, neg=True), vgpr(1), vgpr(2), vgpr(3)],
            [vgpr(0, abs_=True), vgpr(1), vgpr(2), vgpr(3)],
        ):
            with self.subTest(args=args):
                with self.assertRaises(rdna2spirv.Unsupported):
                    emit_synthetic("v_dot2_f32_f16", args, [])

        bad_mods = (
            ["op_sel:bad"],
            ["op_sel:foo[0,0,0]"],
            ["op_sel:[0,0,0"],
            ["op_sel:[0,0,0]]"],
            ["op_sel:[]"],
            ["op_sel:[0,x,0]"],
            ["op_sel:[0,2,0]"],
            ["op_sel:[+0,0,0]"],
            ["op_sel:[0,0]"],
            ["op_sel:[0,0,0,0]"],
            ["op_sel:[0,0,1]"],
            ["op_sel_hi:[1,1,0]"],
            ["neg_hi:[0,0,1]"],
            ["neg_lo:foo[0,0,0]"],
            ["neg_lo:[0,0,0]]"],
            ["neg_hi:[0,2,0]"],
            ["op_sel:[0,0,0]", "op_sel:[1,0,0]"],
            ["op_sel_hi:[1,1,1]", "op_sel_hi:[0,1,1]"],
            ["neg_lo:[0,0,0]", "neg_lo:[1,0,0]"],
            ["neg_hi:[0,0,0]", "neg_hi:[0,1,0]"],
            ["clamp", "clamp"],
            ["mul:2"],
            ["div:2"],
        )
        for mods in bad_mods:
            with self.subTest(mods=mods):
                with self.assertRaises(rdna2spirv.Unsupported):
                    emit_synthetic("v_dot2_f32_f16", good_args, mods)
        for suffix in ("_dpp", "_sdwa"):
            with self.subTest(op="v_dot2_f32_f16" + suffix):
                with self.assertRaises(rdna2spirv.Unsupported):
                    emit_synthetic("v_dot2_f32_f16" + suffix, good_args, [])

        for line in (
            "v_dot2_f32_f16 v[0:1], v1, v2, v3",
            "v_dot2_f32_f16 v0, v1, v2",
            "v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[0,x,0]",
            "v_dot2_f32_f16 v0, v1, v2, v3 neg_lo:[0,0,0] neg_lo:[1,0,0]",
            "v_dot2_f32_f16 v0, v1, v2, v3 clamp clamp",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_prog("ps_fdot_bad", [
                        "v_mov_b32 v1, 0x3c003c00",
                        "v_mov_b32 v2, 0x40004000",
                        "v_mov_b32 v3, 0",
                        line,
                    ])

    def test_v_dot2c_f32_f16_rejects_bad_operands_destinations_and_modifiers(self):
        for args, mods in (
            ([], []),
            ([vgpr(0), vgpr(1)], []),
            ([vgpr(0), vgpr(1), vgpr(2), vgpr(3)], []),
            ([sgpr(0), vgpr(1), vgpr(2)], []),
            ([vgpr(0, 2), vgpr(1), vgpr(2)], []),
            ([vgpr(0, neg=True), vgpr(1), vgpr(2)], []),
            ([vgpr(0), vgpr(1), vgpr(2)], ["clamp", "clamp"]),
            ([vgpr(0), vgpr(1), vgpr(2)], ["op_sel:[0,0,0]"]),
            ([vgpr(0), vgpr(1), vgpr(2)], ["neg_lo:[0,0,0]"]),
        ):
            with self.subTest(args=args, mods=mods):
                with self.assertRaises(rdna2spirv.Unsupported):
                    emit_synthetic("v_dot2c_f32_f16", args, mods)
        for suffix in ("_dpp", "_sdwa"):
            with self.subTest(op="v_dot2c_f32_f16" + suffix):
                with self.assertRaises(rdna2spirv.Unsupported):
                    emit_synthetic("v_dot2c_f32_f16" + suffix, [vgpr(0), vgpr(1), vgpr(2)], [])

    def test_half_dot_positive_controls(self):
        # 0x40003c00 = (lo=1.0, hi=2.0), 0x42004000 = (lo=2.0, hi=3.0)
        spv = translate_prog("ps_fdot_pos", [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x42004000",
            "v_mov_b32 v3, 0x3f800000",  # 1.0f
            # op_sel:[1,0,0] -> lo pair (2.0 * 2.0), op_sel_hi:[0,1,1] -> hi pair (1.0 * 3.0)
            # neg_lo:[1,0,1] -> -(2.0)*2.0 + 1.0*3.0 + (-1.0) = -2.0f
            "v_dot2_f32_f16 v0, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,1] neg_hi:[0,0,0]",
            # v_dot2c_f32_f16: 1.0*2.0 + 2.0*3.0 + 1.0 = 9.0f -> clamp to 1.0f
            "v_dot2c_f32_f16_e64 v3, v1, v2 clamp",
        ])
        validate_spirv(spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], f32_bits(-2.0))
        self.assertEqual(words[3], f32_bits(1.0))


class PkShiftValidationTests(unittest.TestCase):
    def test_rejects_bad_operands_destinations_dpp_sdwa_and_clamp(self):
        good_args = [vgpr(0), vgpr(1), vgpr(2)]
        for op in PK_SHIFT_OPS:
            for args in (
                [],
                [vgpr(0), vgpr(1)],
                [vgpr(0), vgpr(1), vgpr(2), vgpr(3)],
                [sgpr(0), vgpr(1), vgpr(2)],
                [{"k": "special", "t": "null"}, vgpr(1), vgpr(2)],
                [vgpr(0, 2), vgpr(1), vgpr(2)],
                [vgpr(0, neg=True), vgpr(1), vgpr(2)],
                [vgpr(0, abs_=True), vgpr(1), vgpr(2)],
            ):
                with self.subTest(op=op, args=args):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op, args, [])
            for suffix in ("_dpp", "_sdwa"):
                with self.subTest(op=op + suffix):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op + suffix, good_args, [])
            with self.subTest(op=op, mod="clamp"):
                with self.assertRaises(rdna2spirv.Unsupported) as caught:
                    emit_synthetic(op, good_args, ["clamp"])
                self.assertIn("clamp", str(caught.exception))

        for line in (
            "v_pk_lshlrev_b16 s0, v1, v2",
            "v_pk_lshrrev_b16 v[0:1], v1, v2",
            "v_pk_ashrrev_i16 -v0, v1, v2",
            "v_pk_lshlrev_b16 v0, v1",
            "v_pk_lshrrev_b16 v0, v1, v2, v3",
            "v_pk_lshlrev_b16_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_pk_ashrrev_i16_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_prog("ps_pk_shift_bad_op", [
                        "v_mov_b32 v1, 0x00020001",
                        "v_mov_b32 v2, 0x8004fff0",
                        line,
                    ])

    def test_rejects_malformed_duplicate_and_third_slot_invalid_modifiers(self):
        good_args = [vgpr(0), vgpr(1), vgpr(2)]
        bad_mods = (
            ["op_sel:bad"],
            ["op_sel:foo[0,0]"],
            ["op_sel:[0,0"],
            ["op_sel:[0,0]]"],
            ["op_sel:[]"],
            ["op_sel:[0]"],
            ["op_sel:[0,0,0,0]"],
            ["op_sel:[0,x]"],
            ["op_sel:[+0,0]"],
            ["op_sel:[0,2]"],
            ["op_sel:[0,0,2]"],
            ["op_sel:[0,0,-1]"],
            ["op_sel_hi:[1,1,2]"],
            ["op_sel_hi:foo[1,1]"],
            ["op_sel_hi:[1,1]]"],
            ["neg_lo:[1,0]"],
            ["neg_lo:[0,1]"],
            ["neg_lo:[0,0,1]"],
            ["neg_lo:[0,0,2]"],
            ["neg_hi:[1,0]"],
            ["neg_hi:[0,1]"],
            ["neg_hi:[0,0,1]"],
            ["neg_hi:[0,0,2]"],
            ["op_sel:[0,0]", "op_sel:[1,0]"],
            ["op_sel_hi:[1,1]", "op_sel_hi:[0,1]"],
            ["neg_lo:[0,0]", "neg_lo:[0,0]"],
            ["neg_hi:[0,0]", "neg_hi:[0,0]"],
            ["mul:2"],
            ["div:2"],
            ["unknown_mod"],
        )
        for op in PK_SHIFT_OPS:
            for mods in bad_mods:
                with self.subTest(op=op, mods=mods):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        emit_synthetic(op, good_args, mods)

        for line in (
            "v_pk_lshlrev_b16 v0, v1, v2 op_sel:[0,0,2]",
            "v_pk_lshrrev_b16 v0, v1, v2 op_sel_hi:[1,1,2]",
            "v_pk_ashrrev_i16 v0, v1, v2 neg_lo:[0,0,1]",
            "v_pk_lshlrev_b16 v0, v1, v2 neg_hi:[0,0,1]",
            "v_pk_lshrrev_b16 v0, v1, v2 op_sel:[0,0] op_sel:[1,0]",
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_prog("ps_pk_shift_bad_mods", [
                        "v_mov_b32 v1, 0x00020001",
                        "v_mov_b32 v2, 0x8004fff0",
                        line,
                    ])

    def test_pk_shift_positive_controls_two_and_three_entry_lists(self):
        spv = translate_prog("ps_pk_shift_pos", [
            "v_mov_b32 v1, 0x00030002",
            "v_mov_b32 v2, 0x8004fff0",
            # Plain 2-entry default: lo = (0xfff0 << 2) & 0xffff = 0xffc0, hi = (0x8004 << 3) & 0xffff = 0x0020
            "v_pk_lshlrev_b16 v0, v1, v2 op_sel:[0,0] op_sel_hi:[1,1] neg_lo:[0,0] neg_hi:[0,0]",
            # 3-entry llvm-mc gfx1013/gfx1030 style (ps_ops6.s):
            # op_sel:[1,0,0] -> amt=3, val=0xfff0 -> 0xfff0 >> 3 = 0x1ffe
            # op_sel_hi:[0,1,0] -> amt=2, val=0x8004 -> 0x8004 >> 2 = 0x2001
            "v_pk_lshrrev_b16 v3, v1, v2 op_sel:[1,0,0] op_sel_hi:[0,1,0] neg_lo:[0,0,0] neg_hi:[0,0,0]",
            # Signed arithmetic right shift with 3-entry op_sel_hi:[1,1,1]:
            # lo: ashr(0xfff0, 2) = 0xfffc; hi: ashr(0x8004, 3) = 0xf000
            "v_pk_ashrrev_i16 v2, v1, v2 op_sel:[0,0,0] op_sel_hi:[1,1,1]",
        ])
        validate_spirv(spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x0020FFC0)
        self.assertEqual(words[2], 0xF000FFFC)
        self.assertEqual(words[3], 0x20011FFE)


if __name__ == "__main__":
    unittest.main()
