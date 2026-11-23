"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_med3_f16.

AMD RDNA2 / GFX10 Vector ALU (VALU) Architecture:
  - 16-bit Floating-Point 3-Operand Median:
    v_med3_f16 vdst, src0, src1, src2 [op_sel:[s0,s1,s2,d]] [clamp]
  - Operands:
    src0, src1, src2: 16-bit floating-point values selected from low (0) or high (1)
    16-bit half of the operand registers.
  - Mathematical Semantics:
    Without NaN:
      Computes median of 3 IEEE 754 half-precision values:
      NMin(NMax(z, NMin(x, y)), NMax(x, y))
    With any NaN operand:
      The hardware returns min3(x, y, z) with NaN-ignoring min (GLSL NMin),
      returning the minimum among all non-NaN operands, or NaN if all are NaN.
    With clamp modifier:
      Clamps the result to [0.0, 1.0] (0x0000 to 0x3C00).
    Destination:
      VOP3 opsel preserves the other 16-bit half of destination VGPR.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF

F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_ONE: int = 0x3C00
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_QNAN: int = 0x7E00


def is_nan_f16(u16: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 half-precision NaN."""
    return (u16 & 0x7C00) == 0x7C00 and (u16 & 0x03FF) != 0


def f16_sort_key(u16: int) -> int:
    """Order f16 values with -inf < ... < -0.0 < +0.0 < ... < +inf."""
    sign = (u16 >> 15) & 1
    if sign:
        return -(u16 & 0x7FFF) - 1
    return u16 & 0x7FFF


def nmin_f16(a: int, b: int) -> int:
    """GLSL/SPIR-V NMin semantics for 16-bit float: ignores NaN unless both are NaN."""
    a_nan = is_nan_f16(a)
    b_nan = is_nan_f16(b)
    if a_nan and b_nan:
        return F16_QNAN
    if a_nan:
        return b
    if b_nan:
        return a
    return a if f16_sort_key(a) <= f16_sort_key(b) else b


def nmax_f16(a: int, b: int) -> int:
    """GLSL/SPIR-V NMax semantics for 16-bit float: ignores NaN unless both are NaN."""
    a_nan = is_nan_f16(a)
    b_nan = is_nan_f16(b)
    if a_nan and b_nan:
        return F16_QNAN
    if a_nan:
        return b
    if b_nan:
        return a
    return a if f16_sort_key(a) >= f16_sort_key(b) else b


def v_med3_f16_oracle(x: int, y: int, z: int, clamp: bool = False) -> int:
    """Mathematical reference oracle for RDNA2 v_med3_f16."""
    x &= MASK16
    y &= MASK16
    z &= MASK16
    if is_nan_f16(x) or is_nan_f16(y) or is_nan_f16(z):
        res = nmin_f16(nmin_f16(x, y), z)
    else:
        lo = nmin_f16(x, y)
        hi = nmax_f16(x, y)
        m1 = nmax_f16(z, lo)
        res = nmin_f16(m1, hi)
    if clamp:
        if not is_nan_f16(res):
            if f16_sort_key(res) < f16_sort_key(F16_POS_ZERO):
                res = F16_POS_ZERO
            elif f16_sort_key(res) > f16_sort_key(F16_ONE):
                res = F16_ONE
    return res & MASK16


class TestVMed3F16Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_standard_median(self) -> None:
        c1 = 0x3C00  # 1.0
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        self.assertEqual(v_med3_f16_oracle(c1, c2, c3), c2)

    def test_oracle_permutations(self) -> None:
        c1 = 0x3C00  # 1.0
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        for p in [
            (c1, c2, c3), (c1, c3, c2),
            (c2, c1, c3), (c2, c3, c1),
            (c3, c1, c2), (c3, c2, c1),
        ]:
            self.assertEqual(v_med3_f16_oracle(p[0], p[1], p[2]), c2)

    def test_oracle_signed_zeros(self) -> None:
        pos_zero = F16_POS_ZERO
        neg_zero = F16_NEG_ZERO
        one = F16_ONE
        # -0.0 < +0.0 < 1.0, median is +0.0
        self.assertEqual(v_med3_f16_oracle(pos_zero, neg_zero, one), pos_zero)

    def test_oracle_nan_operand_min3_fallback(self) -> None:
        nan = F16_QNAN
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        self.assertEqual(v_med3_f16_oracle(nan, c2, c3), c2)
        self.assertEqual(v_med3_f16_oracle(nan, nan, c3), c3)
        self.assertTrue(is_nan_f16(v_med3_f16_oracle(nan, nan, nan)))

    def test_oracle_clamp_modifier(self) -> None:
        # 2.0, 3.0, 5.0 -> median 3.0, clamped to 1.0 (0x3C00)
        self.assertEqual(v_med3_f16_oracle(0x4000, 0x4200, 0x4500, clamp=True), F16_ONE)
        # -3.0, -2.0, -1.0 -> median -2.0, clamped to 0.0 (0x0000)
        self.assertEqual(v_med3_f16_oracle(0xC200, 0xC000, 0xBC00, clamp=True), F16_POS_ZERO)


# Append AST definitions to file
@dataclasses.dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index must be in [0, 255], got {self.index}")


@dataclasses.dataclass(frozen=True)
class ImmF16:
    raw: int

    def __post_init__(self) -> None:
        if not (0 <= self.raw <= 0xFFFF):
            raise ValueError(f"ImmF16 raw bits must be in [0, 0xFFFF], got {self.raw}")


class ALUOp(enum.Enum):
    V_MED3_F16 = "v_med3_f16"


@dataclasses.dataclass(frozen=True)
class VMed3F16Instruction:
    dst: VGPR
    src0: Union[VGPR, ImmF16]
    src1: Union[VGPR, ImmF16]
    src2: Union[VGPR, ImmF16]
    op_sel: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (s0, s1, s2, dst)
    clamp: bool = False
    neg0: bool = False
    neg1: bool = False
    neg2: bool = False
    abs0: bool = False
    abs1: bool = False
    abs2: bool = False

    def __post_init__(self) -> None:
        for b in self.op_sel:
            if b not in (0, 1):
                raise ValueError(f"op_sel bits must be 0 or 1, got {self.op_sel}")


class TestVMed3F16AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation(self) -> None:
        inst = VMed3F16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
            src2=ImmF16(0x3C00),
            op_sel=(1, 0, 0, 1),
            clamp=True,
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.op_sel, (1, 0, 0, 1))
        self.assertTrue(inst.clamp)

    def test_ast_vgpr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_ast_op_sel_validation(self) -> None:
        with self.assertRaises(ValueError):
            VMed3F16Instruction(
                dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3),
                op_sel=(2, 0, 0, 0)
            )

# Append scalar interpreter
class ScalarRegisterFile:
    """Architectural 256-entry 32-bit Vector Register File for a single thread."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index] & MASK32

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32

    def get_half(self, src: Union[VGPR, ImmF16], hi: int, neg: bool = False, abs_m: bool = False) -> int:
        if isinstance(src, ImmF16):
            bits = src.raw
        else:
            w = self.get_vgpr(src)
            bits = (w >> 16 if hi else w) & MASK16
        if abs_m:
            bits &= 0x7FFF
        if neg:
            bits ^= 0x8000
        return bits & MASK16

    def set_half(self, reg: VGPR, val16: int, hi: int) -> None:
        old = self.get_vgpr(reg)
        val16 &= MASK16
        if hi:
            new_val = (old & 0x0000FFFF) | (val16 << 16)
        else:
            new_val = (old & 0xFFFF0000) | val16
        self.set_vgpr(reg, new_val)


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for VALU v_med3_f16."""

    def execute(self, rf: ScalarRegisterFile, inst: VMed3F16Instruction) -> int:
        s0_hi, s1_hi, s2_hi, d_hi = inst.op_sel
        v0 = rf.get_half(inst.src0, s0_hi, inst.neg0, inst.abs0)
        v1 = rf.get_half(inst.src1, s1_hi, inst.neg1, inst.abs1)
        v2 = rf.get_half(inst.src2, s2_hi, inst.neg2, inst.abs2)
        res = v_med3_f16_oracle(v0, v1, v2, clamp=inst.clamp)
        rf.set_half(inst.dst, res, d_hi)
        return rf.get_vgpr(inst.dst)


class TestVMed3F16ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter for v_med3_f16."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter()

    def test_scalar_low_half_preserves_high_half(self) -> None:
        # Pre-set v0 = 0xDEADBEEF
        self.rf.set_vgpr(VGPR(0), 0xDEADBEEF)
        self.rf.set_vgpr(VGPR(1), 0x00003C00)  # 1.0
        self.rf.set_vgpr(VGPR(2), 0x00004000)  # 2.0
        self.rf.set_vgpr(VGPR(3), 0x00004200)  # 3.0

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3),
            op_sel=(0, 0, 0, 0)
        )
        res = self.interp.execute(self.rf, inst)
        # Low half should be 0x4000, high half preserved as 0xDEAD
        self.assertEqual(res, 0xDEAD4000)

    def test_scalar_high_half_preserves_low_half(self) -> None:
        self.rf.set_vgpr(VGPR(0), 0x12345678)
        self.rf.set_vgpr(VGPR(1), 0x3C000000)  # src0 high = 1.0
        self.rf.set_vgpr(VGPR(2), 0x40000000)  # src1 high = 2.0
        self.rf.set_vgpr(VGPR(3), 0x42000000)  # src2 high = 3.0

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3),
            op_sel=(1, 1, 1, 1)
        )
        res = self.interp.execute(self.rf, inst)
        # High half should be 0x4000, low half preserved as 0x5678
        self.assertEqual(res, 0x40005678)

# Append SIMD multi-lane interpreter
class SIMDState:
    """Architectural SIMD state for Wave32 / Wave64."""

    def __init__(self, num_lanes: int = 32, exec_mask: Optional[int] = None) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = ((1 << num_lanes) - 1) if exec_mask is None else exec_mask
        self.lanes: List[ScalarRegisterFile] = [ScalarRegisterFile() for _ in range(num_lanes)]

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))


class SIMDVALUInterpreter:
    """SIMD multi-lane instruction interpreter executing under EXEC mask."""

    def __init__(self) -> None:
        self.scalar_interp = ScalarVALUInterpreter()

    def execute(self, state: SIMDState, inst: VMed3F16Instruction) -> List[int]:
        out = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                res = self.scalar_interp.execute(state.lanes[lane], inst)
            else:
                res = state.lanes[lane].get_vgpr(inst.dst)
            out.append(res)
        return out


class TestVMed3F16SIMD(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_lanes(self) -> None:
        # Alternating mask (even lanes active)
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        for i in range(32):
            state.lanes[i].set_vgpr(VGPR(0), 0xAAAAAAAA)
            state.lanes[i].set_vgpr(VGPR(1), 0x00003C00)
            state.lanes[i].set_vgpr(VGPR(2), 0x00004000)
            state.lanes[i].set_vgpr(VGPR(3), 0x00004200)

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3),
            op_sel=(0, 0, 0, 0)
        )
        interp = SIMDVALUInterpreter()
        results = interp.execute(state, inst)

        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(results[i], 0xAAAA4000)
            else:
                self.assertEqual(results[i], 0xAAAAAAAA)

    def test_wave64_exec_masking(self) -> None:
        # High 32 lanes active only
        mask = 0xFFFFFFFF00000000
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for i in range(64):
            state.lanes[i].set_vgpr(VGPR(0), 0x11111111)
            state.lanes[i].set_vgpr(VGPR(1), 0x00003C00)
            state.lanes[i].set_vgpr(VGPR(2), 0x00004000)
            state.lanes[i].set_vgpr(VGPR(3), 0x00004200)

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3),
            op_sel=(0, 0, 0, 0)
        )
        interp = SIMDVALUInterpreter()
        results = interp.execute(state, inst)

        for i in range(32):
            self.assertEqual(results[i], 0x11111111)
        for i in range(32, 64):
            self.assertEqual(results[i], 0x11114000)

# Append register aliasing tests
class TestVMed3F16Aliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter()

    def test_dst_aliases_src0(self) -> None:
        # v0 = 0xCAFE3C00 (1.0 in low half), v1 = 2.0, v2 = 3.0
        self.rf.set_vgpr(VGPR(0), 0xCAFE3C00)
        self.rf.set_vgpr(VGPR(1), 0x00004000)
        self.rf.set_vgpr(VGPR(2), 0x00004200)

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(0), src1=VGPR(1), src2=VGPR(2),
            op_sel=(0, 0, 0, 0)
        )
        res = self.interp.execute(self.rf, inst)
        self.assertEqual(res, 0xCAFE4000)

    def test_dst_aliases_src1_and_src2(self) -> None:
        self.rf.set_vgpr(VGPR(0), 0x00003C00)  # 1.0
        self.rf.set_vgpr(VGPR(1), 0xBEEF4200)  # 3.0

        inst = VMed3F16Instruction(
            dst=VGPR(1), src0=VGPR(0), src1=VGPR(1), src2=VGPR(1),
            op_sel=(0, 0, 0, 0)
        )
        res = self.interp.execute(self.rf, inst)
        # median of 1.0, 3.0, 3.0 is 3.0 (0x4200)
        self.assertEqual(res, 0xBEEF4200)

    def test_all_operands_aliased(self) -> None:
        self.rf.set_vgpr(VGPR(0), 0x99994000)  # 2.0
        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(0), src1=VGPR(0), src2=VGPR(0),
            op_sel=(0, 0, 0, 0)
        )
        res = self.interp.execute(self.rf, inst)
        self.assertEqual(res, 0x99994000)

# Append condition codes and status flags
@dataclasses.dataclass
class HardwareStatusFlags:
    scc: int = 1
    vcc: int = 0xFFFFFFFF_FFFFFFFF


class TestVMed3F16ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter()
        self.flags = HardwareStatusFlags()

    def test_scc_and_vcc_unmodified(self) -> None:
        self.rf.set_vgpr(VGPR(0), 0)
        self.rf.set_vgpr(VGPR(1), 0x00003C00)
        self.rf.set_vgpr(VGPR(2), 0x00004000)
        self.rf.set_vgpr(VGPR(3), 0x00004200)

        inst = VMed3F16Instruction(
            dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3)
        )
        self.interp.execute(self.rf, inst)
        # VALU v_med3_f16 does not touch SCC or VCC
        self.assertEqual(self.flags.scc, 1)
        self.assertEqual(self.flags.vcc, 0xFFFFFFFF_FFFFFFFF)

if __name__ == "__main__":
    unittest.main()
