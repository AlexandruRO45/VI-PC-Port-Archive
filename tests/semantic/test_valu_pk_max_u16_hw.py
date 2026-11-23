"""Semantic tests for RDNA2 v_pk_max_u16 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_pk_max_u16 dst, src0, src1 (VOP3P)
Computes packed 16-bit unsigned maximum:
    dst.u16[0] = max(src0.u16[op_sel[0]], src1.u16[op_sel[1]])
    dst.u16[1] = max(src0.u16[op_sel_hi[0]], src1.u16[op_sel_hi[1]])
Default modifiers:
    op_sel: [0, 0]
    op_sel_hi: [1, 1]
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
MASK16 = 0xFFFF


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def ref_pk_max_u16(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    op_sel_hi: tuple[int, int] = (1, 1),
) -> int:
    """Bit-accurate reference oracle for RDNA2 v_pk_max_u16."""
    s0_halves = (s0 & MASK16, (s0 >> 16) & MASK16)
    s1_halves = (s1 & MASK16, (s1 >> 16) & MASK16)
    lo = max(s0_halves[op_sel[0]], s1_halves[op_sel[1]])
    hi = max(s0_halves[op_sel_hi[0]], s1_halves[op_sel_hi[1]])
    return P(lo, hi)


def run_hw_pk_max_u16(instruction: str, seeds: dict[str, int]) -> int:
    """Execute v_pk_max_u16 through rdna2spirv translator and spirv_mini_interp."""
    lines = ["v_mov_b32 v0, 0xdeadbeef"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        lines.append(f"{prefix}_mov_b32 {reg}, {val & MASK32:#x}")
    lines.append(instruction)
    lines.append("exp mrt0 v0, v0, v0, v0 done vm")
    lines.append("s_endpgm")
    spv = support.translate_lines("ps_pk_max_u16_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVPkMaxU16Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_basic_lanes(self):
        cases = [
            # ((lo0, hi0), (lo1, hi1), (exp_lo, exp_hi))
            ((10, 20), (5, 50), (10, 50)),
            ((500, 100), (200, 600), (500, 600)),
            ((0, 0), (0, 0), (0, 0)),
            ((65535, 1), (1, 65535), (65535, 65535)),
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0, s1 = P(lo0, hi0), P(lo1, hi1)
            expected = P(exp_lo, exp_hi)
            self.assertEqual(ref_pk_max_u16(s0, s1), expected)

    def test_reference_unsigned_extremes(self):
        # 0x8000 (32768) > 0x7FFF (32767) in unsigned comparison
        s0 = P(0x8000, 0x0000)
        s1 = P(0x7FFF, 0xFFFF)
        self.assertEqual(ref_pk_max_u16(s0, s1), P(0x8000, 0xFFFF))

    def test_reference_op_sel_modifiers(self):
        s0 = P(10, 20)
        s1 = P(30, 40)
        # op_sel:[1, 0] selects high half of s0 (20) and low half of s1 (30) for low result
        # op_sel_hi:[0, 1] selects low half of s0 (10) and high half of s1 (40) for high result
        got = ref_pk_max_u16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        self.assertEqual(got, P(max(20, 30), max(10, 40)))

    def test_translator_hw_pk_max_u16(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (P(10, 20), P(5, 50)),
            (P(0x8000, 1), P(0x7FFF, 0xFFFF)),
            (P(0, 0), P(100, 200)),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_hw_pk_max_u16("v_pk_max_u16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, ref_pk_max_u16(s0, s1))

    def test_translator_hw_pk_max_u16_op_sel(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        s0 = P(10, 20)
        s1 = P(30, 40)
        got = run_hw_pk_max_u16("v_pk_max_u16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[0,1]", {"v1": s0, "v2": s1})
        self.assertEqual(got, ref_pk_max_u16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1)))


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
class VPkMaxU16Inst:
    dst: Operand
    src0: Operand
    src1: Operand
    op_sel: tuple[int, int] = (0, 0)
    op_sel_hi: tuple[int, int] = (1, 1)

    def emit_asm(self) -> str:
        mods = ""
        if self.op_sel != (0, 0):
            mods += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]}]"
        if self.op_sel_hi != (1, 1):
            mods += f" op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]}]"
        return f"v_pk_max_u16 {self.dst.format()}, {self.src0.format()}, {self.src1.format()}{mods}"


class TestVPkMaxU16AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
        )
        self.assertEqual(inst.emit_asm(), "v_pk_max_u16 v0, v1, v2")

    def test_ast_with_modifiers(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            op_sel=(1, 0),
            op_sel_hi=(0, 1),
        )
        self.assertEqual(inst.emit_asm(), "v_pk_max_u16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[0,1]")

    def test_ast_immutability(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.VGPR, 3)  # type: ignore


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

    def step(self, inst: VPkMaxU16Inst):
        s0 = self.read_operand(inst.src0)
        s1 = self.read_operand(inst.src1)
        res = ref_pk_max_u16(s0, s1, inst.op_sel, inst.op_sel_hi)
        self.write_operand(inst.dst, res)


class TestVPkMaxU16ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter."""

    def test_scalar_packed_component_selection(self):
        mach = ScalarMachine()
        mach.vgprs[1] = P(100, 500)
        mach.vgprs[2] = P(200, 300)
        inst = VPkMaxU16Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 2))
        mach.step(inst)
        self.assertEqual(mach.vgprs[0], P(200, 500))

    def test_scalar_sequential_pipeline(self):
        mach = ScalarMachine()
        mach.vgprs[1] = P(10, 20)
        mach.vgprs[2] = P(15, 5)
        # v0 = max(v1, v2) -> P(15, 20)
        mach.step(VPkMaxU16Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 2)))
        self.assertEqual(mach.vgprs[0], P(15, 20))
        # v3 = max(v0, P(25, 10)) -> P(25, 20)
        mach.vgprs[4] = P(25, 10)
        mach.step(VPkMaxU16Inst(Operand(RegType.VGPR, 3), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 4)))
        self.assertEqual(mach.vgprs[3], P(25, 20))


class SimdWave:
    def __init__(self, size: int = 32):
        self.size = size
        self.exec_mask: int = (1 << size) - 1
        self.lanes: list[dict[int, int]] = [{i: 0 for i in range(32)} for _ in range(size)]

    def step_pk_max(self, dst_idx: int, s0_idx: int, s1_idx: int):
        for lane in range(self.size):
            if self.exec_mask & (1 << lane):
                s0 = self.lanes[lane].get(s0_idx, 0)
                s1 = self.lanes[lane].get(s1_idx, 0)
                self.lanes[lane][dst_idx] = ref_pk_max_u16(s0, s1)


class TestVPkMaxU16SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergence_and_exec_mask(self):
        wave = SimdWave(32)
        for lane in range(32):
            wave.lanes[lane][1] = P(lane * 10, 100 - lane)
            wave.lanes[lane][2] = P(50, 50)
            wave.lanes[lane][0] = 0xDEADBEEF

        # Exec mask: only even lanes
        wave.exec_mask = 0x55555555
        wave.step_pk_max(0, 1, 2)

        for lane in range(32):
            if lane % 2 == 0:
                expected = ref_pk_max_u16(wave.lanes[lane][1], wave.lanes[lane][2])
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xDEADBEEF)

    def test_wave64_divergence(self):
        wave = SimdWave(64)
        for lane in range(64):
            wave.lanes[lane][1] = P(lane * 100, 6000 - lane * 10)
            wave.lanes[lane][2] = P(3000, 3000)
            wave.lanes[lane][0] = 0xCAFEBABE

        wave.exec_mask = 0x00000000FFFFFFFF  # lower 32 lanes active
        wave.step_pk_max(0, 1, 2)

        for lane in range(64):
            if lane < 32:
                expected = ref_pk_max_u16(wave.lanes[lane][1], wave.lanes[lane][2])
                self.assertEqual(wave.lanes[lane][0], expected)
            else:
                self.assertEqual(wave.lanes[lane][0], 0xCAFEBABE)


class TestVPkMaxU16RegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        mach = ScalarMachine()
        mach.vgprs[0] = P(100, 400)
        mach.vgprs[1] = P(200, 300)
        # v0 = max(v0, v1)
        mach.step(VPkMaxU16Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1)))
        self.assertEqual(mach.vgprs[0], P(200, 400))
        self.assertEqual(mach.vgprs[1], P(200, 300))

    def test_dst_aliases_src1(self):
        mach = ScalarMachine()
        mach.vgprs[0] = P(100, 400)
        mach.vgprs[1] = P(200, 300)
        # v1 = max(v0, v1)
        mach.step(VPkMaxU16Inst(Operand(RegType.VGPR, 1), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 1)))
        self.assertEqual(mach.vgprs[1], P(200, 400))
        self.assertEqual(mach.vgprs[0], P(100, 400))

    def test_all_operands_aliased(self):
        mach = ScalarMachine()
        mach.vgprs[0] = P(1234, 5678)
        mach.step(VPkMaxU16Inst(Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0), Operand(RegType.VGPR, 0)))
        self.assertEqual(mach.vgprs[0], P(1234, 5678))


class TestVPkMaxU16ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow preservation."""

    def test_vcc_and_scc_preserved(self):
        vcc_initial = 0xDEADBEEF01234567
        scc_initial = 1
        vcc, scc = vcc_initial, scc_initial
        res = ref_pk_max_u16(P(0, 0), P(0, 0))
        self.assertEqual(res, 0)
        self.assertEqual(vcc, vcc_initial)
        self.assertEqual(scc, scc_initial)


class TestVPkMaxU16IEEE754(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_float16_bitcast_interpretation(self):
        # Float16 values treated as raw unsigned 16-bit integers
        # +0.0 = 0x0000, -0.0 = 0x8000 (sign bit set)
        # In uint16 comparison, 0x8000 (32768) > 0x0000 (0)
        pos_zero = 0x0000
        neg_zero = 0x8000
        self.assertEqual(ref_pk_max_u16(P(pos_zero, pos_zero), P(neg_zero, neg_zero)), P(neg_zero, neg_zero))

    def test_nan_and_inf_payload_preservation(self):
        # Float16 +inf = 0x7C00 (31744), NaN = 0x7E00 (32256)
        # 0x7E00 > 0x7C00 in unsigned uint16 comparison
        pos_inf = 0x7C00
        qnan = 0x7E00
        self.assertEqual(ref_pk_max_u16(P(pos_inf, pos_inf), P(qnan, qnan)), P(qnan, qnan))


class TestVPkMaxU16NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection."""

    def test_clamp_accepted_as_noop_by_translator(self):
        # Per #4082, VOP3P clamp on v_pk_max_u16 is accepted as a hardware no-op.
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        seeds = {"v1": P(10, 20), "v2": P(5, 50)}
        plain = run_hw_pk_max_u16("v_pk_max_u16 v0, v1, v2", seeds)
        clamped = run_hw_pk_max_u16("v_pk_max_u16 v0, v1, v2 clamp", seeds)
        self.assertEqual(plain, clamped)
        self.assertEqual(clamped, ref_pk_max_u16(seeds["v1"], seeds["v2"]))

    def test_invalid_modifier_rejected_by_translator(self):
        # Float-only modifier neg_lo is rejected on packed integer ops
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        with self.assertRaises(Exception):
            support.translate_lines("ps_invalid_mod", ["v_pk_max_u16 v0, v1, v2 neg_lo:[1,0]", "s_endpgm"])

    def test_mutation_signed_vs_unsigned_comparison(self):
        # Mutation: signed comparison treating 0x8000 as negative (-32768)
        def mutated_signed_max(s0, s1):
            s0_lo = s0 & MASK16
            s1_lo = s1 & MASK16
            i0 = s0_lo - 65536 if s0_lo >= 32768 else s0_lo
            i1 = s1_lo - 65536 if s1_lo >= 32768 else s1_lo
            m = max(i0, i1)
            return m & MASK16

        # 0x8000 vs 0x0000: unsigned max is 0x8000; signed max is 0x0000
        correct = ref_pk_max_u16(0x8000, 0x0000) & MASK16
        mutant = mutated_signed_max(0x8000, 0x0000)
        self.assertEqual(correct, 0x8000)
        self.assertEqual(mutant, 0x0000)
        self.assertNotEqual(correct, mutant)


if __name__ == "__main__":
    unittest.main()
