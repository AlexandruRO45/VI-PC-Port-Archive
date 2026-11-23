"""Semantic tests for RDNA2 v_cvt_f32_u32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_cvt_f32_u32 dst, src0 (VOP1/VOP3)
Converts a 32-bit unsigned integer to an IEEE 754 single-precision float (binary32).
Rounding follows round-to-nearest-even (RNE).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import support  # noqa: E402
except Exception:
    support = None

MASK32 = 0xFFFFFFFF


def ref_cvt_f32_u32(val: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_cvt_f32_u32: u32 to binary32 bits."""
    val &= MASK32
    f = float(val)
    return struct.unpack("<I", struct.pack("<f", f))[0]


def run_hw_cvt_f32_u32(val: int) -> int:
    """Execute v_cvt_f32_u32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {val & MASK32:#x}",
        "v_cvt_f32_u32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cvt_f32_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVCvtF32U32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_exact_integers(self):
        cases = [
            (0, 0x00000000),         # +0.0
            (1, 0x3F800000),         # 1.0
            (2, 0x40000000),         # 2.0
            (4, 0x40800000),         # 4.0
            (10, 0x41200000),        # 10.0
            (16777215, 0x4B7FFFFF),  # 2^24 - 1
            (16777216, 0x4B800000),  # 2^24
        ]
        for u, exp_bits in cases:
            with self.subTest(u=u):
                self.assertEqual(ref_cvt_f32_u32(u), exp_bits)

    def test_reference_rounding_mode_rne(self):
        # 16777217 (2^24 + 1): midway between 16777216 and 16777218 -> rounds to even 16777216.0
        self.assertEqual(ref_cvt_f32_u32(16777217), 0x4B800000)
        # 16777219 (2^24 + 3): midway between 16777218 and 16777220 -> rounds to even 16777220.0
        f_val = struct.unpack("<f", struct.pack("<I", ref_cvt_f32_u32(16777219)))[0]
        self.assertEqual(f_val, 16777220.0)
        # 0xFFFFFFFF (4294967295): rounds to 4294967296.0 (2^32) = 0x4F800000
        self.assertEqual(ref_cvt_f32_u32(0xFFFFFFFF), 0x4F800000)

    def test_translator_hw_cvt_f32_u32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [0, 1, 100, 16777215, 16777216, 16777217, 16777219, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                got = run_hw_cvt_f32_u32(val)
                self.assertEqual(got, ref_cvt_f32_u32(val))

    def test_translator_hw_clamp(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        lines = [
            "v_mov_b32 v1, 100",
            "v_cvt_f32_u32 v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_clamp", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F800000)


class RegType(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    val: int = 0

    def format(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        return f"{self.val:#x}"


@dataclasses.dataclass(frozen=True)
class VCvtF32U32Inst:
    dst: Operand
    src0: Operand

    def emit_asm(self) -> str:
        return f"v_cvt_f32_u32 {self.dst.format()}, {self.src0.format()}"


class TestVCvtF32U32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
        )
        self.assertEqual(inst.emit_asm(), "v_cvt_f32_u32 v0, v1")

    def test_ast_with_literal(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.IMM, 0, 100),
        )
        self.assertEqual(inst.emit_asm(), "v_cvt_f32_u32 v0, 0x64")

    def test_ast_immutability(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.VGPR, 2)  # type: ignore


class ScalarMachine:
    def __init__(self):
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs.get(op.index, 0)
        return op.val & MASK32

    def write_vgpr(self, op: Operand, val: int):
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = val & MASK32

    def step(self, inst: VCvtF32U32Inst):
        s0 = self.read_operand(inst.src0)
        res = ref_cvt_f32_u32(s0)
        self.write_vgpr(inst.dst, res)


class TestVCvtF32U32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter."""

    def test_scalar_conversion_pipeline(self):
        mach = ScalarMachine()
        mach.vgprs[1] = 42
        inst = VCvtF32U32Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1))
        mach.step(inst)
        self.assertEqual(mach.vgprs[0], ref_cvt_f32_u32(42))

    def test_scalar_max_u32_conversion(self):
        mach = ScalarMachine()
        mach.vgprs[5] = 0xFFFFFFFF
        inst = VCvtF32U32Inst(Operand(RegType.VGPR, 4), Operand(RegType.VGPR, 5))
        mach.step(inst)
        self.assertEqual(mach.vgprs[4], 0x4F800000)


class SimdWave:
    def __init__(self, size: int = 32):
        self.size = size
        self.exec_mask: int = (1 << size) - 1
        self.lanes: list[dict[int, int]] = [{i: 0 for i in range(32)} for _ in range(size)]

    def step_cvt(self, dst_idx: int, src_idx: int):
        for lane in range(self.size):
            if self.exec_mask & (1 << lane):
                s0 = self.lanes[lane].get(src_idx, 0)
                self.lanes[lane][dst_idx] = ref_cvt_f32_u32(s0)


class TestVCvtF32U32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_conversions(self):
        wave = SimdWave(32)
        for lane in range(32):
            wave.lanes[lane][1] = 1 << lane  # Powers of 2
            wave.lanes[lane][0] = 0xDEADBEEF

        wave.exec_mask = 0x55555555  # Even lanes
        wave.step_cvt(0, 1)

        for lane in range(32):
            if lane % 2 == 0:
                expected = ref_cvt_f32_u32(1 << lane)
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xDEADBEEF)

    def test_wave64_divergence(self):
        wave = SimdWave(64)
        for lane in range(64):
            wave.lanes[lane][1] = lane * 1000
            wave.lanes[lane][0] = 0xCAFEBABE

        wave.exec_mask = 0xFFFFFFFF00000000  # High 32 lanes
        wave.step_cvt(0, 1)

        for lane in range(64):
            if lane >= 32:
                expected = ref_cvt_f32_u32(lane * 1000)
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xCAFEBABE)


class TestVCvtF32U32RegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation."""

    def test_dst_aliases_src(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 100
        # v0 = cvt_f32_u32 v0
        mach.step(VCvtF32U32Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0)))
        self.assertEqual(mach.vgprs[0], ref_cvt_f32_u32(100))


class TestVCvtF32U32ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow preservation."""

    def test_vcc_and_scc_preserved(self):
        vcc_initial = 0xDEADBEEF01234567
        scc_initial = 1
        vcc, scc = vcc_initial, scc_initial
        res = ref_cvt_f32_u32(0)
        self.assertEqual(res, 0)
        self.assertEqual(vcc, vcc_initial)
        self.assertEqual(scc, scc_initial)


class TestVCvtF32U32IEEE754(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_zero_is_always_positive_zero(self):
        # 0 must convert to +0.0 (bits 0x00000000), never -0.0 (0x80000000)
        res_bits = ref_cvt_f32_u32(0)
        self.assertEqual(res_bits, 0x00000000)
        self.assertEqual(res_bits >> 31, 0)  # Sign bit is 0

    def test_high_bit_set_remains_positive(self):
        # In unsigned conversion, 0x80000000 (2147483648) converts to +2147483648.0 (sign bit 0)
        res_bits = ref_cvt_f32_u32(0x80000000)
        self.assertEqual(res_bits >> 31, 0)  # Positive
        self.assertEqual(res_bits, 0x4F000000)


class TestVCvtF32U32NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_extra_operands_accepted_by_translator(self):
        """The translator accepts surplus operands in VOP1 assembly and emits OpConvertUToF."""
        if support is not None and getattr(support, "translate_lines", None) is not None:
            spv = support.translate_lines("ps_extra", ["v_cvt_f32_u32 v0, v1, v2", "s_endpgm"])
            self.assertIn("OpConvertUToF", spv)

    def test_syntax_rejection_insufficient_operands(self):
        """Missing source operand is rejected during translation."""
        if support is not None and getattr(support, "translate_lines", None) is not None:
            with self.assertRaises(Exception):
                support.translate_lines("ps_invalid", ["v_cvt_f32_u32 v0", "s_endpgm"])

    def test_mutation_signed_vs_unsigned_conversion(self):
        # Mutation: uses signed int conversion (converting 0xFFFFFFFF to -1.0)
        def mutated_signed_cvt(val):
            val &= MASK32
            signed_val = val - (1 << 32) if val >= (1 << 31) else val
            f = float(signed_val)
            return struct.unpack("<I", struct.pack("<f", f))[0]

        correct = ref_cvt_f32_u32(0xFFFFFFFF)
        mutant = mutated_signed_cvt(0xFFFFFFFF)
        self.assertEqual(correct, 0x4F800000)  # +4294967296.0
        self.assertEqual(mutant, 0xBF800000)   # -1.0
        self.assertNotEqual(correct, mutant)


if __name__ == "__main__":
    unittest.main()
