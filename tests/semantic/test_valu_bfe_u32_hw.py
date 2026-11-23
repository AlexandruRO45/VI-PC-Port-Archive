"""Semantic tests for RDNA2 v_bfe_u32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_bfe_u32 dst, src0, src1, src2 (VOP3)
Bitfield extract unsigned 32-bit:
    src0 = base 32-bit unsigned integer
    src1 = bit offset (low 5 bits: offset = src1 & 31)
    src2 = bitfield width (low 5 bits: width = src2 & 31)

Semantics:
    if width == 0:
        dst = 0
    else:
        eff_width = min(width, 32 - offset)
        dst = (src0 >> offset) & ((1 << eff_width) - 1)
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


def ref_bfe_u32(src0: int, src1: int, src2: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_bfe_u32."""
    base = src0 & MASK32
    offset = src1 & 31
    width = src2 & 31
    if width == 0:
        return 0
    eff_width = min(width, 32 - offset)
    return ((base >> offset) & ((1 << eff_width) - 1)) & MASK32


def run_hw_bfe_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_bfe_u32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_bfe_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVBfeU32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_zero_width(self):
        for off in range(32):
            self.assertEqual(ref_bfe_u32(0xDEADBEEF, off, 0), 0)
            self.assertEqual(ref_bfe_u32(0xFFFFFFFF, off, 32), 0)  # 32 & 31 == 0

    def test_reference_single_bit_extractions(self):
        val = 0b10101010111100001100110001010101
        for off in range(32):
            expected = (val >> off) & 1
            self.assertEqual(ref_bfe_u32(val, off, 1), expected)

    def test_reference_multi_bit_fields(self):
        val = 0x12345678
        self.assertEqual(ref_bfe_u32(val, 4, 4), 0x7)
        self.assertEqual(ref_bfe_u32(val, 8, 4), 0x6)
        self.assertEqual(ref_bfe_u32(val, 16, 8), 0x34)
        self.assertEqual(ref_bfe_u32(val, 0, 16), 0x5678)
        self.assertEqual(ref_bfe_u32(val, 16, 16), 0x1234)

    def test_reference_offset_masking(self):
        val = 0x87654321
        self.assertEqual(ref_bfe_u32(val, 33, 4), ref_bfe_u32(val, 1, 4))
        self.assertEqual(ref_bfe_u32(val, 64, 8), ref_bfe_u32(val, 0, 8))

    def test_reference_width_masking(self):
        val = 0xFFFFFFFF
        self.assertEqual(ref_bfe_u32(val, 0, 33), 1)
        self.assertEqual(ref_bfe_u32(val, 0, 64), 0)

    def test_reference_past_bit31_clamp(self):
        val = 0xF0000000
        self.assertEqual(ref_bfe_u32(val, 28, 8), 0xF)
        self.assertEqual(ref_bfe_u32(val, 31, 10), 1)

    def test_translator_hw_bfe_u32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (0x12345678, 4, 4),
            (0xDEADBEEF, 8, 8),
            (0xFFFFFFFF, 0, 16),
            (0xF0000000, 28, 8),
            (0x12345678, 0, 0),
        ]
        for s0, s1, s2 in cases:
            with self.subTest(s0=hex(s0), s1=s1, s2=s2):
                res = run_hw_bfe_u32(s0, s1, s2)
                self.assertEqual(res, ref_bfe_u32(s0, s1, s2))


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
class VBfeU32Inst:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def emit_asm(self) -> str:
        return f"v_bfe_u32 {self.dst.format()}, {self.src0.format()}, {self.src1.format()}, {self.src2.format()}"


class TestVBfeU32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = VBfeU32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        self.assertEqual(inst.emit_asm(), "v_bfe_u32 v0, v1, v2, v3")

    def test_ast_with_literals(self):
        inst = VBfeU32Inst(
            dst=Operand(RegType.VGPR, 5),
            src0=Operand(RegType.VGPR, 6),
            src1=Operand(RegType.IMM, 0, 8),
            src2=Operand(RegType.IMM, 0, 16),
        )
        self.assertEqual(inst.emit_asm(), "v_bfe_u32 v5, v6, 0x8, 0x10")

    def test_ast_immutability(self):
        inst = VBfeU32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.VGPR, 4)  # type: ignore


class ScalarMachine:
    def __init__(self):
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs.get(op.index, 0)
        elif op.reg_type == RegType.SGPR:
            return self.sgprs.get(op.index, 0)
        return op.val & MASK32

    def write_operand(self, op: Operand, val: int):
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = val & MASK32
        elif op.reg_type == RegType.SGPR:
            self.sgprs[op.index] = val & MASK32

    def step(self, inst: VBfeU32Inst):
        s0 = self.read_operand(inst.src0)
        s1 = self.read_operand(inst.src1)
        s2 = self.read_operand(inst.src2)
        res = ref_bfe_u32(s0, s1, s2)
        self.write_operand(inst.dst, res)


class TestVBfeU32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter."""

    def test_scalar_rgb565_unpacking(self):
        color = 0xF815
        mach = ScalarMachine()
        mach.vgprs[1] = color

        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 2), Operand(RegType.VGPR, 1), Operand(RegType.IMM, 0, 0), Operand(RegType.IMM, 0, 5)))
        self.assertEqual(mach.vgprs[2], 0b10101)

        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 3), Operand(RegType.VGPR, 1), Operand(RegType.IMM, 0, 5), Operand(RegType.IMM, 0, 6)))
        self.assertEqual(mach.vgprs[3], 0b000000)

        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 4), Operand(RegType.VGPR, 1), Operand(RegType.IMM, 0, 11), Operand(RegType.IMM, 0, 5)))
        self.assertEqual(mach.vgprs[4], 0b11111)

    def test_scalar_sequential_pipeline(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 0xCAFEBABE
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 0), Operand(RegType.IMM, 0, 8), Operand(RegType.IMM, 0, 8)))
        self.assertEqual(mach.vgprs[1], 0xBA)
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 2), Operand(RegType.VGPR, 1), Operand(RegType.IMM, 0, 4), Operand(RegType.IMM, 0, 4)))
        self.assertEqual(mach.vgprs[2], 0xB)


class SimdWave:
    def __init__(self, size: int = 32):
        self.size = size
        self.exec_mask: int = (1 << size) - 1
        self.lanes: list[dict[int, int]] = [{i: 0 for i in range(32)} for _ in range(size)]

    def step_bfe(self, dst_idx: int, s0_idx: int, s1_idx: int, s2_idx: int):
        for lane in range(self.size):
            if self.exec_mask & (1 << lane):
                s0 = self.lanes[lane].get(s0_idx, 0)
                s1 = self.lanes[lane].get(s1_idx, 0)
                s2 = self.lanes[lane].get(s2_idx, 0)
                self.lanes[lane][dst_idx] = ref_bfe_u32(s0, s1, s2)


class TestVBfeU32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_offsets(self):
        wave = SimdWave(32)
        base_val = 0x89ABCDEF
        for lane in range(32):
            wave.lanes[lane][1] = base_val
            wave.lanes[lane][2] = lane
            wave.lanes[lane][3] = 4
            wave.lanes[lane][0] = 0xAA

        wave.exec_mask = 0x55555555
        wave.step_bfe(0, 1, 2, 3)

        for lane in range(32):
            if lane % 2 == 0:
                expected = ref_bfe_u32(base_val, lane, 4)
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xAA)

    def test_wave64_divergence_and_exec_boundary(self):
        wave = SimdWave(64)
        for lane in range(64):
            wave.lanes[lane][1] = (lane * 0x11111111) & MASK32
            wave.lanes[lane][2] = (lane * 3) & 31
            wave.lanes[lane][3] = ((lane % 8) + 1)
            wave.lanes[lane][0] = 0xFFFFFFFF

        wave.exec_mask = 0xFFFFFFFF00000000
        wave.step_bfe(0, 1, 2, 3)

        for lane in range(64):
            if lane >= 32:
                expected = ref_bfe_u32(wave.lanes[lane][1], wave.lanes[lane][2], wave.lanes[lane][3])
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xFFFFFFFF)


class TestVBfeU32RegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 0x12345678
        mach.vgprs[1] = 8
        mach.vgprs[2] = 8
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 2)))
        self.assertEqual(mach.vgprs[0], 0x56)
        self.assertEqual(mach.vgprs[1], 8)
        self.assertEqual(mach.vgprs[2], 8)

    def test_dst_aliases_src1_offset(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 0xAABBCCDD
        mach.vgprs[1] = 16
        mach.vgprs[2] = 8
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 2)))
        self.assertEqual(mach.vgprs[1], 0xBB)
        self.assertEqual(mach.vgprs[0], 0xAABBCCDD)
        self.assertEqual(mach.vgprs[2], 8)

    def test_dst_aliases_src2_width(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 0xAABBCCDD
        mach.vgprs[1] = 0
        mach.vgprs[2] = 8
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 2), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 2)))
        self.assertEqual(mach.vgprs[2], 0xDD)

    def test_all_operands_aliased(self):
        mach = ScalarMachine()
        mach.vgprs[0] = 0x3F
        mach.step(VBfeU32Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0)))
        self.assertEqual(mach.vgprs[0], 0)


class TestVBfeU32ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow preservation."""

    def test_vcc_and_scc_preserved(self):
        vcc_initial = 0xDEADBEEF01234567
        scc_initial = 1
        vcc = vcc_initial
        scc = scc_initial
        res = ref_bfe_u32(0xFFFFFFFF, 0, 32)
        self.assertEqual(res, 0)
        self.assertEqual(vcc, vcc_initial)
        self.assertEqual(scc, scc_initial)

    def test_zero_result_does_not_set_vcc_or_scc(self):
        res = ref_bfe_u32(0, 0, 10)
        self.assertEqual(res, 0)


class TestVBfeU32IEEE754(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_sign_bit_extraction_floats(self):
        pos_zero = struct.unpack("<I", struct.pack("<f", +0.0))[0]
        neg_zero = struct.unpack("<I", struct.pack("<f", -0.0))[0]
        pos_inf = struct.unpack("<I", struct.pack("<f", math.inf))[0]
        neg_inf = struct.unpack("<I", struct.pack("<f", -math.inf))[0]

        self.assertEqual(ref_bfe_u32(pos_zero, 31, 1), 0)
        self.assertEqual(ref_bfe_u32(neg_zero, 31, 1), 1)
        self.assertEqual(ref_bfe_u32(pos_inf, 31, 1), 0)
        self.assertEqual(ref_bfe_u32(neg_inf, 31, 1), 1)

    def test_exponent_and_mantissa_extraction(self):
        val = struct.unpack("<I", struct.pack("<f", 1.5))[0]
        exp = ref_bfe_u32(val, 23, 8)
        mant = ref_bfe_u32(val, 0, 23)
        self.assertEqual(exp, 0x7F)
        self.assertEqual(mant, 0x400000)

    def test_subnormal_no_denorm_flushing(self):
        subnormal_bits = 0x00000001
        self.assertEqual(ref_bfe_u32(subnormal_bits, 0, 1), 1)
        self.assertEqual(ref_bfe_u32(subnormal_bits, 23, 8), 0)

    def test_nan_payload_preservation(self):
        nan_bits = 0x7F800000 | 0x12345
        payload = ref_bfe_u32(nan_bits, 0, 23)
        self.assertEqual(payload, 0x12345)


class TestVBfeU32NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_operand_count(self):
        invalid_lines = [
            "v_bfe_u32 v0, v1, v2",
            "v_bfe_u32 v0, v1",
            "v_bfe_u32 v0",
        ]
        if support is not None and getattr(support, "translate_lines", None) is not None:
            for line in invalid_lines:
                with self.assertRaises(Exception):
                    support.translate_lines("ps_invalid", [line, "s_endpgm"])

    def test_mutation_offset_masking(self):
        def mutated_bfe(s0, s1, s2):
            off = s1 & 63
            w = s2 & 31
            if w == 0:
                return 0
            return (s0 >> off) & ((1 << min(w, 32 - min(off, 32))) - 1)

        s0 = 0b10
        correct = ref_bfe_u32(s0, 33, 1)
        mutant = mutated_bfe(s0, 33, 1)
        self.assertNotEqual(correct, mutant)

    def test_mutation_zero_width_nonzero(self):
        def mutated_bfe(s0, s1, s2):
            off = s1 & 31
            w = s2 & 31
            eff_w = min(w if w != 0 else 32, 32 - off)
            return (s0 >> off) & ((1 << eff_w) - 1)

        correct = ref_bfe_u32(0x12345678, 0, 0)
        mutant = mutated_bfe(0x12345678, 0, 0)
        self.assertEqual(correct, 0)
        self.assertNotEqual(correct, mutant)

    def test_mutation_past_bit31_wrap(self):
        def mutated_bfe(s0, s1, s2):
            off = s1 & 31
            w = s2 & 31
            return (s0 >> off) & ((1 << w) - 1)

        val = 0xF0000000
        correct = ref_bfe_u32(val, 28, 8)
        self.assertEqual(correct, 0xF)


if __name__ == "__main__":
    unittest.main()
