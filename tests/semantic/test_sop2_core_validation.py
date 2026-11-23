"""Operand, encoding, and modifier validation tests for 16 core 32-bit SOP2 SALU ops.

Covers:
  s_pack_ll_b32_b16, s_pack_lh_b32_b16, s_pack_hh_b32_b16, s_bfm_b32,
  s_min_i32, s_max_i32, s_add_i32, s_sub_i32, s_min_u32, s_max_u32,
  s_cselect_b32, s_bfe_u32, s_bfe_i32, s_lshl_b32, s_lshr_b32, s_mul_i32.

Each negative test asserts tools/rdna2spirv.py raises Unsupported on malformed arity,
non-SOP2 encoding suffixes, instruction modifiers, invalid destinations/sources, or
operand modifiers (neg/abs/sext). Positive controls verify valid SGPR, special-register,
and immediate forms (including plain negative integer and float immediates) through
spirv-as, spirv-val --target-env vulkan1.2, and tests/semantic/spirv_mini_interp.py.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

CORE_SOP2_OPS = (
    "s_pack_ll_b32_b16",
    "s_pack_lh_b32_b16",
    "s_pack_hh_b32_b16",
    "s_bfm_b32",
    "s_min_i32",
    "s_max_i32",
    "s_add_i32",
    "s_sub_i32",
    "s_min_u32",
    "s_max_u32",
    "s_cselect_b32",
    "s_bfe_u32",
    "s_bfe_i32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_mul_i32",
)


def translate_ps(lines):
    full = list(lines) + [
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    return support.translate_lines("ps_sop2_core_val", full)


def validate_spirv(spv_text: str):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        bin_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", bin_path],
            capture_output=True,
            text=True,
        )
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", bin_path],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ref_op(op: str, x: int, y: int, scc_in: int = 1) -> int:
    x &= 0xFFFFFFFF
    y &= 0xFFFFFFFF
    if op == "s_pack_ll_b32_b16":
        return ((y & 0xFFFF) << 16) | (x & 0xFFFF)
    if op == "s_pack_lh_b32_b16":
        return (y & 0xFFFF0000) | (x & 0xFFFF)
    if op == "s_pack_hh_b32_b16":
        return (y & 0xFFFF0000) | (x >> 16)
    if op == "s_bfm_b32":
        return (((1 << (x & 31)) - 1) << (y & 31)) & 0xFFFFFFFF
    if op == "s_min_i32":
        return (x if s32(x) < s32(y) else y) & 0xFFFFFFFF
    if op == "s_max_i32":
        return (x if s32(x) > s32(y) else y) & 0xFFFFFFFF
    if op == "s_add_i32":
        return (x + y) & 0xFFFFFFFF
    if op == "s_sub_i32":
        return (x - y) & 0xFFFFFFFF
    if op == "s_min_u32":
        return x if x < y else y
    if op == "s_max_u32":
        return x if x > y else y
    if op == "s_cselect_b32":
        return x if scc_in else y
    if op == "s_bfe_u32":
        off = y & 31
        wid = min((y >> 16) & 0x7F, 32 - off)
        return 0 if wid == 0 else ((x >> off) & ((1 << wid) - 1))
    if op == "s_bfe_i32":
        off = y & 31
        wid = min((y >> 16) & 0x7F, 32 - off)
        if wid == 0:
            return 0
        raw = (x >> off) & ((1 << wid) - 1)
        sign = 1 << (wid - 1)
        return ((raw ^ sign) - sign) & 0xFFFFFFFF
    if op == "s_lshl_b32":
        return (x << (y & 31)) & 0xFFFFFFFF
    if op == "s_lshr_b32":
        return (x >> (y & 31)) & 0xFFFFFFFF
    if op == "s_mul_i32":
        return (x * y) & 0xFFFFFFFF
    raise ValueError(op)


class Sop2CoreValidationTests(unittest.TestCase):
    def assertUnsupported(self, lines):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_ps(lines)

    def test_reject_wrong_arity(self):
        bad_forms = (
            "{op}",
            "{op} s4",
            "{op} s4, s0",
            "{op} s4, s0, s1, s2",
        )
        for op in CORE_SOP2_OPS:
            for form in bad_forms:
                with self.subTest(op=op, form=form):
                    self.assertUnsupported([form.format(op=op)])

    def test_reject_encoding_suffixes(self):
        bad_forms = (
            "{op}_e32 s4, s0, s1",
            "{op}_e64 s4, s0, s1",
            "{op}_sdwa s4, s0, s1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "{op}_dpp s4, s0, s1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        )
        for op in CORE_SOP2_OPS:
            for form in bad_forms:
                with self.subTest(op=op, form=form):
                    self.assertUnsupported([form.format(op=op)])

    def test_reject_instruction_modifiers(self):
        bad_mods = (
            "clamp",
            "mul:2",
            "div:2",
            "op_sel:[0,0,0]",
            "offset:4",
            "glc",
        )
        for op in CORE_SOP2_OPS:
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self.assertUnsupported([f"{op} s4, s0, s1 {mod}"])

    def test_reject_invalid_destinations(self):
        bad_dsts = (
            "v0",
            "v[0:1]",
            "s[4:5]",
            "vcc",
            "exec",
            "[v0, v1]",
            "[s4, s5]",
            "42",
            "1.0",
            "scc",
            "src_scc",
            "off",
            "src_pops_exiting_wave_id",
        )
        for op in CORE_SOP2_OPS:
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    self.assertUnsupported([f"{op} {dst}, s0, s1"])

    def test_reject_invalid_sources(self):
        bad_srcs = (
            "v0",
            "v[0:1]",
            "s[0:1]",
            "vcc",
            "exec",
            "[v0, v1]",
            "[s0, s1]",
            "off",
        )
        for op in CORE_SOP2_OPS:
            for src in bad_srcs:
                with self.subTest(op=op, pos="src0", src=src):
                    self.assertUnsupported([f"{op} s4, {src}, s1"])
                with self.subTest(op=op, pos="src1", src=src):
                    self.assertUnsupported([f"{op} s4, s0, {src}"])

    def test_reject_operand_modifiers(self):
        bad_src_mods = (
            "-s0",
            "neg(s0)",
            "|s0|",
            "abs(s0)",
            "sext(s0)",
            "neg(5)",
            "|5|",
            "abs(5)",
            "neg(1.0)",
            "|1.0|",
        )
        bad_dst_mods = (
            "-s4",
            "|s4|",
            "sext(s4)",
        )
        for op in CORE_SOP2_OPS:
            for mod_dst in bad_dst_mods:
                with self.subTest(op=op, pos="dst", arg=mod_dst):
                    self.assertUnsupported([f"{op} {mod_dst}, s0, s1"])
            for mod_src in bad_src_mods:
                with self.subTest(op=op, pos="src0", arg=mod_src):
                    self.assertUnsupported([f"{op} s4, {mod_src}, s1"])
                with self.subTest(op=op, pos="src1", arg=mod_src):
                    self.assertUnsupported([f"{op} s4, s0, {mod_src}"])

    def test_positive_all_16_ops_spirv_val_and_exec(self):
        samples = (
            (0x12345678, 0x00080004),
            (0xFFFFFFFB, 0x00040002),
        )
        for op in CORE_SOP2_OPS:
            spv = translate_ps([
                "s_cmp_eq_u32 s2, 1",
                f"{op} s4, s0, s1",
                "s_cselect_b32 s5, 1, 0",
            ])
            validate_spirv(spv)
            for x, y in samples:
                for scc_in in (0, 1):
                    with self.subTest(op=op, x=hex(x), y=hex(y), scc_in=scc_in):
                        out = support.mrt0_words(spv, push_constants=[x, y, scc_in])
                        self.assertEqual(out[0], ref_op(op, x, y, scc_in))

    def test_positive_plain_negative_and_float_immediates(self):
        # Plain negative integer immediates (-5, -1) and float immediates (1.0, -1.0) must stay accepted.
        spv_neg = translate_ps([
            "s_add_i32 s4, s0, -5",
            "s_sub_i32 s5, -1, s1",
        ])
        validate_spirv(spv_neg)
        out = support.mrt0_words(spv_neg, push_constants=[12, 3])
        self.assertEqual((out[0], out[1]), (7, 0xFFFFFFFC))

        spv_neg_fold = translate_ps([
            "s_min_i32 s4, -10, 5",
            "s_mul_i32 s5, -3, -7",
        ])
        validate_spirv(spv_neg_fold)
        out = support.mrt0_words(spv_neg_fold)
        self.assertEqual((out[0], out[1]), (0xFFFFFFF6, 21))

        spv_fimm = translate_ps([
            "s_pack_ll_b32_b16 s4, 1.0, -1.0",
            "s_pack_hh_b32_b16 s5, 1.0, -1.0",
        ])
        validate_spirv(spv_fimm)
        out = support.mrt0_words(spv_fimm)
        pos1, neg1 = f32_bits(1.0), f32_bits(-1.0)
        self.assertEqual(out[0], ref_op("s_pack_ll_b32_b16", pos1, neg1))
        self.assertEqual(out[1], ref_op("s_pack_hh_b32_b16", pos1, neg1))

    def test_positive_valid_special_destinations(self):
        # Writable 32-bit scalar special destinations: null, m0, vcc_lo, vcc_hi, exec_lo, exec_hi.
        spv = translate_ps([
            "s_add_i32 null, s0, s1",
            "s_add_i32 m0, s0, 3",
            "s_add_i32 vcc_lo, m0, 4",
            "s_add_i32 vcc_hi, vcc_lo, 5",
            "s_mov_b32 s4, vcc_hi",
            "s_cselect_b32 exec_lo, exec_lo, exec_lo",
            "s_mov_b32 s5, m0",
        ])
        validate_spirv(spv)
        out = support.mrt0_words(spv, push_constants=[10, 20])
        self.assertEqual((out[0], out[1]), (22, 13))

        # exec_hi destination in a valid 32-bit SOP2 form.
        spv_exec_hi = translate_ps([
            "s_add_i32 exec_hi, s0, 1",
            "s_mov_b32 s4, s0",
            "s_mov_b32 s5, s1",
        ])
        validate_spirv(spv_exec_hi)
        out = support.mrt0_words(spv_exec_hi, push_constants=[7, 9])
        self.assertEqual((out[0], out[1]), (7, 9))

    def test_positive_valid_special_sources(self):
        # Readable 32-bit scalar special sources:
        # null, m0, vcc_lo, vcc_hi, exec_lo, exec_hi, scc, src_scc, src_pops_exiting_wave_id.
        spv = translate_ps([
            "s_mov_b32 m0, 10",
            "s_mov_b32 vcc_lo, 20",
            "s_mov_b32 vcc_hi, 30",
            "s_add_i32 s4, m0, null",
            "s_add_i32 s4, s4, vcc_lo",
            "s_add_i32 s4, s4, vcc_hi",
            "s_cmp_eq_u32 s0, 1",
            "s_add_i32 s4, s4, scc",
            "s_cmp_eq_u32 s0, 1",
            "s_add_i32 s4, s4, src_scc",
            "s_sub_i32 s5, exec_lo, exec_hi",
        ])
        validate_spirv(spv)
        out = support.mrt0_words(spv, push_constants=[1])
        # s4 = 10 + 0 + 20 + 30 + 1 + 1 = 62; s5 = exec_lo - exec_hi = 0 in wave32 single-lane model.
        self.assertEqual((out[0], out[1]), (62, 0))

        # src_pops_exiting_wave_id enables POPS fragment interlock, which spirv-val checks directly.
        spv_pops = translate_ps([
            "s_add_i32 s4, s0, src_pops_exiting_wave_id",
            "s_mov_b32 s5, 0",
        ])
        validate_spirv(spv_pops)


if __name__ == "__main__":
    unittest.main()
