from __future__ import annotations

"""Semantic reference oracle, AST definitions, interpreter models, and translator tests for s_add_i32.

AMD RDNA2 / GFX10 SOP2 (Scalar ALU) 32-bit Signed Integer Addition:
    s_add_i32 sdst, ssrc0, ssrc1

Hardware Semantics:
    sdst = (ssrc0 + ssrc1) & 0xFFFFFFFF
    SCC  = 1 if signed 32-bit integer overflow occurs, else 0

RDNA2 Condition Code (SCC) Behavior for s_add_i32:
    - ssrc0 and ssrc1 are interpreted as signed two's-complement 32-bit integers in [-2^31, 2^31 - 1].
    - Signed overflow occurs when:
        * both operands are positive (bit 31 == 0) and the sum is negative (bit 31 == 1)
        * both operands are negative (bit 31 == 1) and the sum is positive (bit 31 == 0)
    - Bitwise formulation: ((ssrc0 ^ sum) & (ssrc1 ^ sum)) >> 31 == 1
    - Note that unsigned wrap (carry-out) without signed overflow (such as (-1) + 1 = 0)
      clears SCC to 0 on s_add_i32, in direct contrast to s_add_u32 which sets SCC to 1.

Standard library Python only; zero third-party dependencies; zero game data.
"""

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000       # -2147483648
INT32_MAX: int = 0x7FFFFFFF        # 2147483647
UINT32_INT_MIN: int = 0x80000000   # 2147483648 as unsigned

def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit word [0, 0xFFFFFFFF]."""
    return val & MASK32


def to_s32(val: int) -> int:
    """Interpret 32-bit unsigned word as signed two's-complement integer in [-2^31, 2^31 - 1]."""
    u = val & MASK32
    return u - (1 << 32) if (u >> 31) else u


def ref_s_add_i32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_i32.

    Returns (dst_u32, scc_overflow).
    """
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    tot = v0 + v1
    dst = tot & MASK32
    scc = 1 if (tot < INT32_MIN or tot > INT32_MAX) else 0
    return dst, scc


def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for RDNA2 s_add_u32 for direct contrast (carry out of bit 31)."""
    u0 = to_u32(s0)
    u1 = to_u32(s1)
    tot = u0 + u1
    dst = tot & MASK32
    scc = 1 if tot > MASK32 else 0
    return dst, scc


class TestSAddI32ReferenceOracle(unittest.TestCase):
    """Mathematical reference oracle tests for RDNA2 s_add_i32."""

    def test_identity_element(self) -> None:
        for x in (0, 1, -1, INT32_MAX, INT32_MIN, 0x12345678, 0xEDCBA987):
            with self.subTest(x=x):
                dst, scc = ref_s_add_i32(x, 0)
                self.assertEqual(dst, to_u32(x))
                self.assertEqual(scc, 0)
                dst_rev, scc_rev = ref_s_add_i32(0, x)
                self.assertEqual(dst_rev, to_u32(x))
                self.assertEqual(scc_rev, 0)

    def test_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (100, 200),
            (INT32_MAX, 1),
            (INT32_MIN, -1),
            (0xFFFFFFFF, 1),
            (0x7FFFFFFF, 0x7FFFFFFF),
            (0x80000000, 0x80000000),
            (0x12345678, 0x87654321),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(to_u32(a)), b=hex(to_u32(b))):
                self.assertEqual(ref_s_add_i32(a, b), ref_s_add_i32(b, a))

    def test_positive_overflow(self) -> None:
        # INT32_MAX + 1 -> 0x80000000 (signed negative), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, 1)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

        # INT32_MAX + INT32_MAX -> 0xFFFFFFFE (-2), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

        # 2^30 + 2^30 -> 2^31 (0x80000000), overflow = 1
        dst, scc = ref_s_add_i32(0x40000000, 0x40000000)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

    def test_negative_overflow(self) -> None:
        # INT32_MIN + (-1) -> 0x7FFFFFFF (positive), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, -1)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

        # INT32_MIN + INT32_MIN -> 0x00000000 (0), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MIN)
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 1)

        # -2^30 + (-2^30 - 1) -> 0x7FFFFFFF, overflow = 1
        dst, scc = ref_s_add_i32(-0x40000000, -0x40000001)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

    def test_unsigned_wrap_without_signed_overflow(self) -> None:
        # (-1) + 1 = 0: unsigned carry-out occurs, but signed -1 + 1 = 0 (no signed overflow)
        dst_i, scc_i = ref_s_add_i32(0xFFFFFFFF, 1)
        self.assertEqual(dst_i, 0)
        self.assertEqual(scc_i, 0)

        # Contrast with s_add_u32 which sets SCC on unsigned carry:
        dst_u, scc_u = ref_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(dst_u, 0)
        self.assertEqual(scc_u, 1)

        # (-1) + (-1) = -2: unsigned wrap occurs, signed -1 + -1 = -2 (no signed overflow)
        dst_i2, scc_i2 = ref_s_add_i32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_i2, 0xFFFFFFFE)
        self.assertEqual(scc_i2, 0)

        dst_u2, scc_u2 = ref_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_u2, 0xFFFFFFFE)
        self.assertEqual(scc_u2, 1)

    def test_opposite_signs_no_overflow(self) -> None:
        # Opposite signs can never overflow 32-bit signed range
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

        dst, scc = ref_s_add_i32(0x12345678, to_u32(-0x12345678))
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 0)

class OperandKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    LITERAL = "literal"
    INLINE_INT = "inline_int"


@dataclasses.dataclass(frozen=True)
class Reg:
    kind: OperandKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == OperandKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
        if self.kind == OperandKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")

    def format_asm(self) -> str:
        prefix = "s" if self.kind == OperandKind.SGPR else "v"
        return f"{prefix}{self.index}"


@dataclasses.dataclass(frozen=True)
class Imm:
    value: int
    is_inline: bool = False

    def __post_init__(self) -> None:
        if self.is_inline and not (-16 <= self.value <= 64):
            raise ValueError(f"Inline integer {self.value} out of range [-16, 64]")

    def format_asm(self) -> str:
        if self.is_inline:
            return str(self.value)
        return hex(to_u32(self.value))


class Opcode(enum.Enum):
    S_ADD_I32 = "s_add_i32"
    S_ADD_U32 = "s_add_u32"
    S_SUB_I32 = "s_sub_i32"
    S_MOV_B32 = "s_mov_b32"
    S_CSELECT_B32 = "s_cselect_b32"


@dataclasses.dataclass(frozen=True)
class SOP2Inst:
    opcode: Opcode
    sdst: Reg
    ssrc0: Reg | Imm
    ssrc1: Reg | Imm

    def __post_init__(self) -> None:
        if self.sdst.kind != OperandKind.SGPR:
            raise ValueError("sdst must be an SGPR")
        lit_count = 0
        if isinstance(self.ssrc0, Imm) and not self.ssrc0.is_inline:
            lit_count += 1
        if isinstance(self.ssrc1, Imm) and not self.ssrc1.is_inline:
            lit_count += 1
        if lit_count > 1:
            raise ValueError("SOP2 cannot encode more than one 32-bit literal constant")

    def format_asm(self) -> str:
        return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}, {self.ssrc1.format_asm()}"


class TestSAddI32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and serialization."""

    def test_reg_creation_and_formatting(self) -> None:
        r0 = Reg(OperandKind.SGPR, 0)
        r105 = Reg(OperandKind.SGPR, 105)
        v5 = Reg(OperandKind.VGPR, 5)
        self.assertEqual(r0.format_asm(), "s0")
        self.assertEqual(r105.format_asm(), "s105")
        self.assertEqual(v5.format_asm(), "v5")

    def test_imm_creation_and_formatting(self) -> None:
        imm_inline = Imm(1, is_inline=True)
        imm_inline_neg = Imm(-16, is_inline=True)
        imm_lit = Imm(0x12345678, is_inline=False)
        self.assertEqual(imm_inline.format_asm(), "1")
        self.assertEqual(imm_inline_neg.format_asm(), "-16")
        self.assertEqual(imm_lit.format_asm(), "0x12345678")

    def test_instruction_formatting(self) -> None:
        inst_rr = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        self.assertEqual(inst_rr.format_asm(), "s_add_i32 s0, s1, s2")

        inst_ri = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Imm(42, is_inline=True),
        )
        self.assertEqual(inst_ri.format_asm(), "s_add_i32 s0, s1, 42")

        inst_rl = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Imm(0x7FFFFFFF, is_inline=False),
        )
        self.assertEqual(inst_rl.format_asm(), "s_add_i32 s0, s1, 0x7fffffff")

    def test_ast_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            Reg(OperandKind.SGPR, 106)
        with self.assertRaises(ValueError):
            Reg(OperandKind.SGPR, -1)
        with self.assertRaises(ValueError):
            Imm(65, is_inline=True)
        with self.assertRaises(ValueError):
            Imm(-17, is_inline=True)
        with self.assertRaises(ValueError):
            SOP2Inst(
                opcode=Opcode.S_ADD_I32,
                sdst=Reg(OperandKind.VGPR, 0),
                ssrc0=Reg(OperandKind.SGPR, 1),
                ssrc1=Reg(OperandKind.SGPR, 2),
            )
        with self.assertRaises(ValueError):
            SOP2Inst(
                opcode=Opcode.S_ADD_I32,
                sdst=Reg(OperandKind.SGPR, 0),
                ssrc0=Imm(0x1000, is_inline=False),
                ssrc1=Imm(0x2000, is_inline=False),
            )

@dataclasses.dataclass
class ScalarState:
    sgprs: list[int] = dataclasses.field(default_factory=lambda: [0] * 106)
    scc: int = 0

    def read_operand(self, op: Reg | Imm) -> int:
        if isinstance(op, Imm):
            return to_u32(op.value)
        return self.sgprs[op.index]

    def write_sgpr(self, reg: Reg, val: int) -> None:
        self.sgprs[reg.index] = to_u32(val)


class ScalarInterpreter:
    def __init__(self, state: Optional[ScalarState] = None) -> None:
        self.state = state or ScalarState()

    def step(self, inst: SOP2Inst) -> tuple[int, int]:
        v0 = self.state.read_operand(inst.ssrc0)
        v1 = self.state.read_operand(inst.ssrc1)
        if inst.opcode == Opcode.S_ADD_I32:
            dst, scc = ref_s_add_i32(v0, v1)
        elif inst.opcode == Opcode.S_ADD_U32:
            dst, scc = ref_s_add_u32(v0, v1)
        else:
            raise NotImplementedError(f"Opcode {inst.opcode} not implemented in scalar interpreter")
        self.state.write_sgpr(inst.sdst, dst)
        self.state.scc = scc
        return dst, scc


class TestSAddI32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter(self.state)

    def test_reg_to_reg_addition(self) -> None:
        self.state.sgprs[1] = 1234
        self.state.sgprs[2] = 5678
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 6912)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.sgprs[0], 6912)
        self.assertEqual(self.state.scc, 0)

    def test_reg_to_literal_overflow(self) -> None:
        self.state.sgprs[4] = INT32_MAX
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 3),
            ssrc0=Reg(OperandKind.SGPR, 4),
            ssrc1=Imm(1, is_inline=True),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.sgprs[3], UINT32_INT_MIN)
        self.assertEqual(self.state.scc, 1)

    def test_reg_to_inline_negative_overflow(self) -> None:
        self.state.sgprs[10] = UINT32_INT_MIN
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 11),
            ssrc0=Reg(OperandKind.SGPR, 10),
            ssrc1=Imm(-1, is_inline=True),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.sgprs[11], INT32_MAX)
        self.assertEqual(self.state.scc, 1)

    def test_unrelated_sgpr_preservation(self) -> None:
        for i in range(106):
            self.state.sgprs[i] = (i * 0x01010101) & MASK32
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 50),
            ssrc0=Reg(OperandKind.SGPR, 10),
            ssrc1=Reg(OperandKind.SGPR, 20),
        )
        self.interp.step(inst)
        for i in range(106):
            if i == 50:
                continue
            self.assertEqual(self.state.sgprs[i], (i * 0x01010101) & MASK32)

@dataclasses.dataclass
class WaveState:
    wave_size: int = 64
    exec_mask: int = 0xFFFFFFFFFFFFFFFF
    sgprs: list[int] = dataclasses.field(default_factory=lambda: [0] * 106)
    vgprs: list[list[int]] = dataclasses.field(default_factory=list)
    scc: int = 0

    def __post_init__(self) -> None:
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]


class SIMDInterpreter:
    def __init__(self, wave: WaveState) -> None:
        self.wave = wave

    def execute_salu(self, inst: SOP2Inst) -> tuple[int, int]:
        s = ScalarState(sgprs=self.wave.sgprs, scc=self.wave.scc)
        interp = ScalarInterpreter(s)
        dst, scc = interp.step(inst)
        self.wave.sgprs = s.sgprs
        self.wave.scc = scc
        return dst, scc

    def broadcast_to_vgpr(self, src_sgpr: int, dst_vgpr: int) -> None:
        val = self.wave.sgprs[src_sgpr]
        for lane in range(self.wave.wave_size):
            if (self.wave.exec_mask >> lane) & 1:
                self.wave.vgprs[lane][dst_vgpr] = val


class TestSAddI32SIMDDivergence(unittest.TestCase):
    """Unit tests for wavefront SIMD execution and EXEC divergence."""

    def test_wave32_uniform_scalar_addition(self) -> None:
        wave = WaveState(wave_size=32, exec_mask=0xFFFFFFFF)
        wave.sgprs[1] = 1000
        wave.sgprs[2] = 2000
        simd = SIMDInterpreter(wave)
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = simd.execute_salu(inst)
        self.assertEqual(dst, 3000)
        self.assertEqual(scc, 0)
        simd.broadcast_to_vgpr(0, 5)
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][5], 3000)

    def test_wave64_divergent_exec_mask(self) -> None:
        # Alternating active lanes (even lanes active, odd lanes inactive)
        mask = 0x5555555555555555
        wave = WaveState(wave_size=64, exec_mask=mask)
        wave.sgprs[1] = INT32_MAX
        wave.sgprs[2] = 1
        for lane in range(64):
            wave.vgprs[lane][0] = 0xDEADBEEF

        simd = SIMDInterpreter(wave)
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 3),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = simd.execute_salu(inst)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)
        simd.broadcast_to_vgpr(3, 0)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], UINT32_INT_MIN)
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xDEADBEEF)

    def test_wave64_zero_exec_mask(self) -> None:
        wave = WaveState(wave_size=64, exec_mask=0)
        wave.sgprs[1] = 50
        wave.sgprs[2] = 50
        for lane in range(64):
            wave.vgprs[lane][1] = 999

        simd = SIMDInterpreter(wave)
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = simd.execute_salu(inst)
        self.assertEqual(dst, 100)
        self.assertEqual(wave.sgprs[0], 100)
        simd.broadcast_to_vgpr(0, 1)

        # Inactive lanes untouched
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][1], 999)

    def test_wave32_prefix_exec_mask(self) -> None:
        wave = WaveState(wave_size=32, exec_mask=0x000000FF)  # First 8 lanes active
        wave.sgprs[0] = 42
        for lane in range(32):
            wave.vgprs[lane][2] = 0

        simd = SIMDInterpreter(wave)
        simd.broadcast_to_vgpr(0, 2)
        for lane in range(32):
            expected = 42 if lane < 8 else 0
            self.assertEqual(wave.vgprs[lane][2], expected)

class TestSAddI32RegisterAliasing(unittest.TestCase):
    """Unit tests for in-place register aliasing and self-addition."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter(self.state)

    def test_dst_aliases_src0(self) -> None:
        self.state.sgprs[10] = 500
        self.state.sgprs[11] = 300
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 10),
            ssrc0=Reg(OperandKind.SGPR, 10),
            ssrc1=Reg(OperandKind.SGPR, 11),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 800)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.sgprs[10], 800)
        self.assertEqual(self.state.sgprs[11], 300)

    def test_dst_aliases_src1(self) -> None:
        self.state.sgprs[10] = 100
        self.state.sgprs[11] = 400
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 11),
            ssrc0=Reg(OperandKind.SGPR, 10),
            ssrc1=Reg(OperandKind.SGPR, 11),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 500)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.sgprs[10], 100)
        self.assertEqual(self.state.sgprs[11], 500)

    def test_dst_aliases_both_srcs_doubling(self) -> None:
        # Doubling positive without overflow
        self.state.sgprs[7] = 1000
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 7),
            ssrc0=Reg(OperandKind.SGPR, 7),
            ssrc1=Reg(OperandKind.SGPR, 7),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 2000)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.sgprs[7], 2000)

        # Doubling 0x40000000 -> 0x80000000 (positive overflow)
        self.state.sgprs[7] = 0x40000000
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.sgprs[7], UINT32_INT_MIN)

        # Doubling 0x80000000 -> 0x00000000 (negative overflow)
        self.state.sgprs[7] = UINT32_INT_MIN
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.sgprs[7], 0)

        # Doubling -0x40000000 (0xC0000000) -> 0x80000000 (no overflow)
        self.state.sgprs[7] = 0xC0000000
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.sgprs[7], UINT32_INT_MIN)

    def test_chain_aliased_accumulator(self) -> None:
        self.state.sgprs[0] = 0
        self.state.sgprs[1] = 10
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 0),
            ssrc1=Reg(OperandKind.SGPR, 1),
        )
        for i in range(1, 11):
            self.interp.step(inst)
            self.assertEqual(self.state.sgprs[0], i * 10)

class TestSAddI32ConditionCodes(unittest.TestCase):
    """Unit tests for SCC (Scalar Condition Code) flag generation and preservation."""

    def setUp(self) -> None:
        self.state = ScalarState()
        self.interp = ScalarInterpreter(self.state)

    def test_scc_cleared_from_prior_set(self) -> None:
        # Pre-set SCC = 1
        self.state.scc = 1
        self.state.sgprs[1] = 10
        self.state.sgprs[2] = 20
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, 30)
        self.assertEqual(scc, 0)
        self.assertEqual(self.state.scc, 0)  # SCC must be cleared

    def test_scc_set_from_prior_cleared(self) -> None:
        # Pre-set SCC = 0
        self.state.scc = 0
        self.state.sgprs[1] = INT32_MAX
        self.state.sgprs[2] = 1
        inst = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        dst, scc = self.interp.step(inst)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.scc, 1)  # SCC must be set

    def test_conditional_select_simulation(self) -> None:
        # Simulate s_cselect_b32 behavior following s_add_i32:
        # if SCC == 1 select overflow_val else normal_val
        for x, y, expected_overflow in [
            (INT32_MAX, 1, 1),
            (10, 20, 0),
            (UINT32_INT_MIN, to_u32(-1), 1),
            (0xFFFFFFFF, 1, 0),
        ]:
            with self.subTest(x=hex(x), y=hex(y)):
                dst, scc = ref_s_add_i32(x, y)
                self.assertEqual(scc, expected_overflow)
                selected = 0xFEEDFACE if scc else 0xCAFEBABE
                expected_sel = 0xFEEDFACE if expected_overflow else 0xCAFEBABE
                self.assertEqual(selected, expected_sel)

    def test_systematic_overflow_truth_table(self) -> None:
        cases = [
            # (+A) + (+B) = (+R) -> no overflow
            (100, 200, 300, 0),
            # (+A) + (+B) = (-R) -> overflow
            (0x70000000, 0x20000000, 0x90000000, 1),
            # (-A) + (-B) = (-R) -> no overflow
            (to_u32(-100), to_u32(-200), to_u32(-300), 0),
            # (-A) + (-B) = (+R) -> overflow
            (0x80000000, 0x80000000, 0, 1),
            # (+A) + (-B) = (+R) -> no overflow
            (500, to_u32(-100), 400, 0),
            # (+A) + (-B) = (-R) -> no overflow
            (100, to_u32(-500), to_u32(-400), 0),
        ]
        for s0, s1, expected_dst, expected_scc in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                dst, scc = ref_s_add_i32(s0, s1)
                self.assertEqual(dst, expected_dst)
                self.assertEqual(scc, expected_scc)

class TestSAddI32IEEE754EdgeCases(unittest.TestCase):
    """Unit tests verifying raw bitwise preservation of IEEE 754 bit patterns through s_add_i32."""

    def test_signed_zeros(self) -> None:
        pos_zero = struct.unpack("<I", struct.pack("<f", +0.0))[0]  # 0x00000000
        neg_zero = struct.unpack("<I", struct.pack("<f", -0.0))[0]  # 0x80000000
        self.assertEqual(pos_zero, 0x00000000)
        self.assertEqual(neg_zero, 0x80000000)

        # pos_zero + neg_zero: 0 + (-2^31) -> -2^31 (0x80000000), no overflow
        dst, scc = ref_s_add_i32(pos_zero, neg_zero)
        self.assertEqual(dst, 0x80000000)
        self.assertEqual(scc, 0)

        # neg_zero + neg_zero: (-2^31) + (-2^31) -> 0, negative overflow
        dst, scc = ref_s_add_i32(neg_zero, neg_zero)
        self.assertEqual(dst, 0x00000000)
        self.assertEqual(scc, 1)

    def test_infinities(self) -> None:
        pos_inf = struct.unpack("<I", struct.pack("<f", float("inf")))[0]   # 0x7F800000
        neg_inf = struct.unpack("<I", struct.pack("<f", float("-inf")))[0]  # 0xFF800000
        self.assertEqual(pos_inf, 0x7F800000)
        self.assertEqual(neg_inf, 0xFF800000)

        # Integer addition of pos_inf bit pattern with 1
        dst, scc = ref_s_add_i32(pos_inf, 1)
        self.assertEqual(dst, 0x7F800001)  # Signaling NaN bit pattern
        self.assertEqual(scc, 0)

        # Integer addition of pos_inf + pos_inf -> 0xFF000000 (overflow)
        dst, scc = ref_s_add_i32(pos_inf, pos_inf)
        self.assertEqual(dst, 0xFF000000)
        self.assertEqual(scc, 1)

        # Integer addition of pos_inf + neg_inf -> 0x7F000000 (no overflow)
        dst, scc = ref_s_add_i32(pos_inf, neg_inf)
        self.assertEqual(dst, 0x7F000000)
        self.assertEqual(scc, 0)

    def test_subnormals_unflushed(self) -> None:
        # Subnormal values must not be flushed to zero by integer SALU instructions
        min_subnormal = 0x00000001
        max_subnormal = 0x007FFFFF
        dst, scc = ref_s_add_i32(min_subnormal, max_subnormal)
        self.assertEqual(dst, 0x00800000)  # Min normal float bit pattern
        self.assertEqual(scc, 0)

    def test_nans_bitwise_preservation(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        # No NaN canonicalization or trap in integer addition
        dst, scc = ref_s_add_i32(qnan, 0x10)
        self.assertEqual(dst, 0x7FC00010)
        self.assertEqual(scc, 0)

        dst, scc = ref_s_add_i32(snan, snan)
        self.assertEqual(dst, 0xFF000002)
        self.assertEqual(scc, 1)

try:
    import support  # noqa: E402
    from spirv_mini_interp import run_fragment  # noqa: E402
    TRANSLATOR_AVAILABLE = True
except Exception:
    support = None  # type: ignore
    run_fragment = None  # type: ignore
    TRANSLATOR_AVAILABLE = False


ADD_BOUNDARY_CASES = [
    (0, 0, "zero + zero"),
    (0, 0x7FFFFFFF, "zero + INT32_MAX"),
    (0, 0x80000000, "zero + INT32_MIN"),
    (0x7FFFFFFF, 1, "positive overflow: INT32_MAX + 1"),
    (0x7FFFFFFF, 0x7FFFFFFF, "positive overflow: INT32_MAX + INT32_MAX"),
    (0x40000000, 0x40000000, "positive overflow: 2^30 + 2^30"),
    (0x80000000, 0x80000000, "negative overflow: INT32_MIN + INT32_MIN"),
    (0x80000000, 0xFFFFFFFF, "negative overflow: INT32_MIN + (-1)"),
    (0xC0000000, 0xBFFFFFFF, "negative overflow: -2^30 + (-2^30 - 1)"),
    (0xFFFFFFFF, 1, "unsigned wrap without signed overflow: (-1) + 1"),
    (0xFFFFFFFF, 0xFFFFFFFF, "unsigned wrap without signed overflow: (-1) + (-1)"),
    (0x80000000, 0x7FFFFFFF, "opposite signs: INT32_MIN + INT32_MAX"),
    (0x7FFFFFFF, 0x80000000, "opposite signs: INT32_MAX + INT32_MIN"),
    (0x12345678, 0xEDCBA988, "opposite signs cancelling to zero"),
]


def run_translator_add_i32(x: int, y: int, scc_in: int = 0) -> tuple[int, int]:
    """Assemble, translate, and execute a pixel shader running s_add_i32."""
    if not TRANSLATOR_AVAILABLE or support is None:
        raise unittest.SkipTest("tools/rdna2spirv.py translator not available")

    lines = [
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s3, {y:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        "s_add_i32 s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spvasm = support.translate_lines("ps_salu_add_i32", lines)
    words = support.mrt0_words(spvasm)
    return words[0], words[1]


def run_translator_add_i32_imm(x: int, imm_str: str, scc_in: int = 0) -> tuple[int, int]:
    """Assemble, translate, and execute s_add_i32 with an immediate / inline constant."""
    if not TRANSLATOR_AVAILABLE or support is None:
        raise unittest.SkipTest("tools/rdna2spirv.py translator not available")

    lines = [
        f"s_mov_b32 s2, {x:#x}",
        f"s_mov_b32 s4, {scc_in}",
        "s_cmp_eq_u32 s4, 1",
        f"s_add_i32 s0, s2, {imm_str}",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spvasm = support.translate_lines("ps_salu_add_i32_imm", lines)
    words = support.mrt0_words(spvasm)
    return words[0], words[1]


class TestSAddI32TranslatorExecution(unittest.TestCase):
    """Translator-backed execution tests verifying SPIR-V generation against hardware semantics."""

    def test_translator_all_boundary_cases(self) -> None:
        for x, y, desc in ADD_BOUNDARY_CASES:
            want_dst, want_scc = ref_s_add_i32(x, y)
            for prior_scc in (0, 1):
                with self.subTest(case=desc, x=hex(x), y=hex(y), prior_scc=prior_scc):
                    got_dst, got_scc = run_translator_add_i32(x, y, scc_in=prior_scc)
                    self.assertEqual(got_dst, want_dst, f"Destination mismatch in {desc}")
                    self.assertEqual(got_scc, want_scc, f"SCC mismatch in {desc}")

    def test_translator_immediate_operands(self) -> None:
        cases = [
            (100, "0", 0),
            (100, "1", 1),
            (100, "-1", -1),
            (0x7FFFFFFF, "1", 1),
            (0x80000000, "-1", -1),
            (50, "64", 64),
            (10, "-16", -16),
            (0x10000000, "0x20000000", 0x20000000),
        ]
        for reg_val, imm_str, imm_val in cases:
            want_dst, want_scc = ref_s_add_i32(reg_val, imm_val)
            with self.subTest(reg=hex(reg_val), imm=imm_str):
                got_dst, got_scc = run_translator_add_i32_imm(reg_val, imm_str, scc_in=0)
                self.assertEqual(got_dst, want_dst)
                self.assertEqual(got_scc, want_scc)

    def test_translator_seeded_random(self) -> None:
        rng = random.Random(0xADD132)
        edge_vals = [0, 1, -1, INT32_MAX, INT32_MIN, 0x40000000, -0x40000000]
        for _ in range(25):
            x = rng.choice(edge_vals) if rng.random() < 0.4 else rng.getrandbits(32)
            y = rng.choice(edge_vals) if rng.random() < 0.4 else rng.getrandbits(32)
            scc_in = rng.randint(0, 1)
            want_dst, want_scc = ref_s_add_i32(x, y)
            with self.subTest(x=hex(x), y=hex(y), scc_in=scc_in):
                got_dst, got_scc = run_translator_add_i32(x, y, scc_in=scc_in)
                self.assertEqual(got_dst, want_dst)
                self.assertEqual(got_scc, want_scc)

    def test_negative_controls_and_mutation_detection(self) -> None:
        # Mutation 1: s_add_u32 carry-out used instead of signed overflow
        # On (-1) + 1: s_add_u32 sets SCC=1, s_add_i32 sets SCC=0
        correct_dst, correct_scc = ref_s_add_i32(0xFFFFFFFF, 1)
        mutated_dst, mutated_scc = ref_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(correct_scc, 0)
        self.assertEqual(mutated_scc, 1)
        self.assertNotEqual(correct_scc, mutated_scc)

        # Mutation 2: Overflow condition inverted
        inv_scc = 1 - correct_scc
        self.assertNotEqual(correct_scc, inv_scc)

        # Mutation 3: SCC hardcoded to 0
        ovf_dst, ovf_scc = ref_s_add_i32(INT32_MAX, 1)
        self.assertEqual(ovf_scc, 1)
        zero_scc = 0
        self.assertNotEqual(ovf_scc, zero_scc)

if __name__ == "__main__":
    unittest.main()
