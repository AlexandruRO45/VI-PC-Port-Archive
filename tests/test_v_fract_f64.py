"""Check SPIR-V lowering, guards, modifiers, literals and gfx1036 reference behavior for v_fract_f64.

Covers both v_fract_f64_e32 (VOP1) and v_fract_f64_e64 (VOP3). Expected 64-bit boundary results were measured
natively on AMD Radeon(TM) Graphics (gfx1036, Windows driver 32.0.21042.62, tools/hw_oracle; wave32 and wave64
identical on every case).
"""

import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

import rdna2spirv
import shader_ir
from listing import translate_listing


def words(x):
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits & 0xFFFFFFFF, bits >> 32


def load_double(lo_reg, hi_reg, value):
    lo, hi = words(value)
    return f"v_mov_b32 v{lo_reg}, {lo:#x}\nv_mov_b32 v{hi_reg}, {hi:#x}"


def run_spirv_tools(test, directory, name, spv):
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        spv_bin = Path(directory) / f"{name}.spv"
        p_as = subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.2", "-o", str(spv_bin), "-"],
            input=spv, text=True, capture_output=True,
        )
        test.assertEqual(p_as.returncode, 0, p_as.stderr)
        p_val = subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.2", str(spv_bin)],
            text=True, capture_output=True,
        )
        test.assertEqual(p_val.returncode, 0, p_val.stderr)


class VFractF64Test(unittest.TestCase):
    def _result(self, spv, instruction):
        match = re.search(rf"^ *(%\w+) = {instruction}$", spv, re.M)
        self.assertIsNotNone(match, f"Missing SPIR-V instruction: {instruction}")
        return match.group(1)

    def _const_u32(self, spv, value):
        return self._result(spv, rf"OpConstant %uint {value}")

    def _double_from_words(self, spv, lo, hi):
        c_lo = self._const_u32(spv, lo)
        c_hi = self._const_u32(spv, hi)
        pair = self._result(spv, rf"OpCompositeConstruct %v2uint_d {c_lo} {c_hi}")
        return self._result(spv, rf"OpBitcast %double {pair}")

    def _check_store_pair(self, spv, value_id, dst_lo, dst_hi):
        pair = self._result(spv, rf"OpBitcast %v2uint_d {value_id}")
        for index, reg in ((0, f"%c{dst_lo}"), (1, f"%c{dst_hi}")):
            word = self._result(spv, rf"OpCompositeExtract %uint {pair} {index}")
            dest = self._result(spv, rf"OpAccessChain %_ptr_Function_uint %vgpr {reg}")
            old_word = self._result(spv, rf"OpLoad %uint {dest}")
            masked = self._result(spv, rf"OpSelect %uint %\w+ {word} {old_word}")
            self.assertRegex(spv, rf"(?m)^ *OpStore {dest} {masked}$")

    def test_v_fract_f64_guards_and_output_modifiers(self):
        cases = (
            ("v_fract_f64_e32", "", None, False),
            ("v_fract_f64_e64", "", None, False),
            ("v_fract_f64_e64", "mul:2", 2.0, False),
            ("v_fract_f64_e64", "mul:4 clamp", 4.0, True),
            ("v_fract_f64_e64", "clamp div:2", 0.5, True),
        )
        prefix = load_double(0, 1, -1e-20)
        with tempfile.TemporaryDirectory() as directory:
            for op, mods, scale, clamp in cases:
                with self.subTest(op=op, mods=mods):
                    line = f"{op} v[4:5], v[0:1] {mods}".rstrip()
                    spv = translate_listing(Path(directory), f"{prefix}\n{line}")
                    self.assertIn("OpCapability Float64", spv)
                    m_raw = re.search(r"^ *(%\w+) = OpExtInst %double %glsl Fract (%\w+)$", spv, re.M)
                    self.assertIsNotNone(m_raw, spv)
                    raw, src = m_raw.groups()
                    one = self._result(spv, rf"OpFOrdEqual %bool {raw} {self._double_from_words(spv, *words(1.0))}")
                    below = self._double_from_words(spv, 0xFFFFFFFF, 0x3FEFFFFF)
                    clamped = self._result(spv, rf"OpSelect %double {one} {below} {raw}")
                    is_inf = self._result(spv, rf"OpIsInf %bool {src}")
                    qnan = self._double_from_words(spv, 0, 0xFFF80000)
                    val = self._result(spv, rf"OpSelect %double {is_inf} {qnan} {clamped}")
                    if scale is not None:
                        val = self._result(spv, rf"OpFMul %double {val} {self._double_from_words(spv, *words(scale))}")
                        self.assertRegex(spv, rf"(?m)^ *OpDecorate {val} NoContraction$")
                    else:
                        self.assertNotRegex(spv, r"OpFMul %double")
                    if clamp:
                        val = self._result(spv, rf"OpExtInst %double %glsl FClamp {val} %\w+ %\w+")
                    else:
                        self.assertNotIn("FClamp", spv)
                    self._check_store_pair(spv, val, 4, 5)
                    run_spirv_tools(self, directory, f"{op}_{clamp}_{scale}", spv)

    def test_input_modifiers_sgpr_inline_and_f64_literal(self):
        ins = shader_ir.parse_line("\tv_fract_f64_e32 v[4:5], 0x3ff80000")
        self.assertTrue(ins["args"][1].get("f64_literal"))
        self.assertEqual(ins["args"][1]["v"], 0x3FF80000)

        body = "\n".join((
            load_double(0, 1, -2.5),
            "s_mov_b32 s2, 0",
            "s_mov_b32 s3, 0x3ff80000",
            "v_fract_f64_e64 v[2:3], |v[0:1]|",
            "v_fract_f64_e64 v[4:5], -v[0:1]",
            "v_fract_f64_e64 v[6:7], -|s[2:3]|",
            "v_fract_f64_e64 v[8:9], 1.5",
            "v_fract_f64_e32 v[10:11], 0x3ff80000",
        ))
        with tempfile.TemporaryDirectory() as directory:
            spv = translate_listing(Path(directory), body)
            self.assertIn("OpExtInst %double %glsl FAbs", spv)
            self.assertIn("OpFNegate %double", spv)
            self.assertEqual(len(re.findall(r"OpExtInst %double %glsl Fract", spv)), 5)
            # 0x3ff80000 -> lo=0, hi=0x3ff80000 (1.5 in binary64).
            self._double_from_words(spv, 0, 0x3FF80000)
            run_spirv_tools(self, directory, "v_fract_f64_inputs", spv)

    def test_non_pair_destination_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(rdna2spirv.Unsupported):
                translate_listing(Path(directory), f"{load_double(0, 1, 1.5)}\nv_fract_f64_e32 v4, v[0:1]")

    def test_guard_formula_matches_gfx1036_measured_words(self):
        """Verify the lowered SPIR-V formula reproduces every gfx1036-measured v_fract_f64 edge word."""
        def lowered_fract_f64(bits):
            x = struct.unpack("<d", struct.pack("<Q", bits))[0]
            if math.isinf(x):
                return 0xFFF8000000000000
            if math.isnan(x):
                return bits
            fl = math.copysign(float(math.floor(x)), x)
            raw = x - fl
            if raw == 1.0:
                return 0x3FEFFFFFFFFFFFFF
            return struct.unpack("<Q", struct.pack("<d", raw))[0]

        # Native gfx1036 measurements (wave32 == wave64).
        measured = (
            ("pos_zero",    0x0000000000000000, 0x0000000000000000),
            ("neg_zero",    0x8000000000000000, 0x0000000000000000),
            ("pos_sub_min", 0x0000000000000001, 0x0000000000000001),
            ("neg_sub_min", 0x8000000000000001, 0x3FEFFFFFFFFFFFFF),
            ("pos_sub_max", 0x000FFFFFFFFFFFFF, 0x000FFFFFFFFFFFFF),
            ("neg_sub_max", 0x800FFFFFFFFFFFFF, 0x3FEFFFFFFFFFFFFF),
            ("neg_norm",    0x8010000000000000, 0x3FEFFFFFFFFFFFFF),
            ("neg_1e_20",   0xBBD4D285620A2E4B, 0x3FEFFFFFFFFFFFFF),
            ("pos_half",    0x3FE0000000000000, 0x3FE0000000000000),
            ("neg_half",    0xBFE0000000000000, 0x3FE0000000000000),
            ("pos_1p5",     0x3FF8000000000000, 0x3FE0000000000000),
            ("neg_1p5",     0xBFF8000000000000, 0x3FE0000000000000),
            ("neg_1_p_ulp", 0xBFEFFFFFFFFFFFFF, 0x3CA0000000000000),
            ("pos_one",     0x3FF0000000000000, 0x0000000000000000),
            ("neg_one",     0xBFF0000000000000, 0x0000000000000000),
            ("neg_2p52h",   0xC32FFFFFFFFFFFFF, 0x3FE0000000000000),
            ("pos_2p53",    0x4340000000000000, 0x0000000000000000),
            ("pos_1e300",   0x7E37E43C8800759C, 0x0000000000000000),
            ("pos_inf",     0x7FF0000000000000, 0xFFF8000000000000),
            ("neg_inf",     0xFFF0000000000000, 0xFFF8000000000000),
            ("pos_qnan",    0x7FF8000000000000, 0x7FF8000000000000),
            ("neg_qnan",    0xFFF8000000000000, 0xFFF8000000000000),
        )
        for name, x_bits, want in measured:
            with self.subTest(case=name):
                self.assertEqual(lowered_fract_f64(x_bits), want)


if __name__ == "__main__":
    unittest.main()
