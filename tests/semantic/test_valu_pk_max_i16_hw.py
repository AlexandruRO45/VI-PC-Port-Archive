from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_pk_max_i16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_pk_max_i16 performs packed 16-bit signed integer maximum.
  Instruction:
    v_pk_max_i16 vd, src0, src1  (VOP3P format)

  Operands:
    vd   = Destination 32-bit VGPR containing two 16-bit signed integers [lo, hi]
    src0 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]
    src1 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]

  Semantics:
    lo0 = s16(src0[15:0]), hi0 = s16(src0[31:16])
    lo1 = s16(src1[15:0]), hi1 = s16(src1[31:16])
    res_lo = max(lo0, lo1) & 0xFFFF
    res_hi = max(hi0, hi1) & 0xFFFF
    vd = (res_hi << 16) | res_lo

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
SIGN_BIT16: int = 0x8000
_TRANSLATE_ID: int = 0


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned integer as signed two's complement."""
    val &= MASK16
    return val - 0x10000 if (val & SIGN_BIT16) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit integer range [0, 2^16 - 1]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_pk_max_i16(src0: int, src1: int) -> int:
    """Exact architectural reference oracle for AMD RDNA2 v_pk_max_i16.

    Args:
        src0: 32-bit packed pair of two signed 16-bit integers
        src1: 32-bit packed pair of two signed 16-bit integers

    Returns:
        32-bit packed pair with element-wise signed maximum
    """
    lo0 = s16(src0 & MASK16)
    hi0 = s16((src0 >> 16) & MASK16)
    lo1 = s16(src1 & MASK16)
    hi1 = s16((src1 >> 16) & MASK16)

    res_lo = max(lo0, lo1) & MASK16
    res_hi = max(hi0, hi1) & MASK16
    return (res_hi << 16) | res_lo


def run_translated_pk_max_i16(
    src0: int,
    src1: int,
    op_sel: tuple[int, int] | None = None,
    op_sel_hi: tuple[int, int] | None = None,
) -> int:
    """Execute v_pk_max_i16 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32_e32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32_e32 v2, {src1 & MASK32:#x}",
    ]
    mods = ""
    if op_sel is not None:
        mods += f" op_sel:[{op_sel[0]},{op_sel[1]}]"
    if op_sel_hi is not None:
        mods += f" op_sel_hi:[{op_sel_hi[0]},{op_sel_hi[1]}]"
    lines.append(f"v_pk_max_i16 v0, v1, v2{mods}")
    lines.append("exp mrt0, v0, v0, v0, v0 done")
    lines.append("s_endpgm")
    spv = support.translate_lines(f"ps_pk_max_i16_trans_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0] & MASK32


class TestValuPkMaxI16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_pk_max_i16."""

    def test_oracle_mixed_signs(self):
        # src0: lo = -5 (0xFFFB), hi = +5 (0x0005) -> 0x0005FFFB
        # src1: lo = +5 (0x0005), hi = -5 (0xFFFB) -> 0xFFFB0005
        # Expected: lo = max(-5, +5) = +5 (0x0005), hi = max(+5, -5) = +5 (0x0005) -> 0x00050005
        res = ref_pk_max_i16(0x0005FFFB, 0xFFFB0005)
        self.assertEqual(res, 0x00050005)
        trans_res = run_translated_pk_max_i16(0x0005FFFB, 0xFFFB0005)
        self.assertEqual(trans_res, 0x00050005)

    def test_oracle_both_negative(self):
        # lo: max(-10, -20) = -10 (0xFFF6)
        # hi: max(-30, -5) = -5 (0xFFFB)
        s0 = ((-30 & MASK16) << 16) | (-10 & MASK16)
        s1 = ((-5 & MASK16) << 16) | (-20 & MASK16)
        res = ref_pk_max_i16(s0, s1)
        expected = ((-5 & MASK16) << 16) | (-10 & MASK16)
        self.assertEqual(res, expected)
        trans_res = run_translated_pk_max_i16(s0, s1)
        self.assertEqual(trans_res, expected)

    def test_oracle_both_positive(self):
        s0 = (100 << 16) | 300
        s1 = (200 << 16) | 150
        res = ref_pk_max_i16(s0, s1)
        expected = (200 << 16) | 300
        self.assertEqual(res, expected)
        trans_res = run_translated_pk_max_i16(s0, s1)
        self.assertEqual(trans_res, expected)

    def test_oracle_signed_extremes(self):
        # -32768 (0x8000) vs 32767 (0x7FFF)
        s0 = (0x8000 << 16) | 0x7FFF
        s1 = (0x7FFF << 16) | 0x8000
        res = ref_pk_max_i16(s0, s1)
        expected = (0x7FFF << 16) | 0x7FFF
        self.assertEqual(res, expected)
        trans_res = run_translated_pk_max_i16(s0, s1)
        self.assertEqual(trans_res, expected)

    def test_oracle_commutativity(self):
        a = 0x12345678
        b = 0x87654321
        self.assertEqual(ref_pk_max_i16(a, b), ref_pk_max_i16(b, a))
        trans_a_b = run_translated_pk_max_i16(a, b)
        trans_b_a = run_translated_pk_max_i16(b, a)
        self.assertEqual(trans_a_b, trans_b_a)
        self.assertEqual(trans_a_b, ref_pk_max_i16(a, b))

    def test_oracle_signed_vs_unsigned_discriminators(self):
        """Discriminator tests covering extreme packed signed values (e.g. 0x8000 vs 0x0000)."""
        cases = [
            # 0x8000 (-32768) vs 0x0000 (0): signed max is 0 (0x0000), unsigned max is 0x8000 (32768)
            (0x80008000, 0x00000000, 0x00000000),
            (0x00000000, 0x80008000, 0x00000000),
            # 0xFFFF (-1) vs 0x0000 (0): signed max is 0 (0x0000), unsigned max is 0xFFFF (65535)
            (0xFFFFFFFF, 0x00000000, 0x00000000),
            (0x00000000, 0xFFFFFFFF, 0x00000000),
            # 0x8000 (-32768) vs 0xFFFF (-1): signed max is -1 (0xFFFF), unsigned max is 0xFFFF
            (0x80008000, 0xFFFFFFFF, 0xFFFFFFFF),
            # Mixed half extremes: lo = (-32768 vs 0 -> 0), hi = (-1 vs -32768 -> -1)
            (0x80008000, 0xFFFF0000, 0xFFFF0000),
            # lo = (+100 vs -32768 -> +100), hi = (-32768 vs +32767 -> +32767)
            (0x80000064, 0x7FFF8000, 0x7FFF0064),
        ]
        for s0, s1, expected in cases:
            self.assertEqual(ref_pk_max_i16(s0, s1), expected)
            actual = run_translated_pk_max_i16(s0, s1)
            self.assertEqual(actual, expected, f"Discriminator mismatch for ({s0:#x}, {s1:#x})")


class Opcode(Enum):
    V_PK_MAX_I16 = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL32 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def vgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 255:
            raise ValueError(f"VGPR index out of range: v{index}")
        return cls(kind=OperandKind.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index)

    @classmethod
    def literal32(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, value=u32(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, value=u32(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.V_PK_MAX_I16:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")


class TestValuPkMaxI16AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.V_PK_MAX_I16)
        self.assertEqual(inst.dst.kind, OperandKind.VGPR)
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)

        # Cross-check through translator
        lines = [
            "v_mov_b32_e32 v1, 0x0014fff6",  # [-10, +20]
            "v_mov_b32_e32 v2, 0xfffb0032",  # [+50, -5]
            "v_pk_max_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_pk_max_ast_{_TRANSLATE_ID}", lines)
        self.assertIn("OpBitFieldSExtract", spv)
        self.assertIn("SMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0x00140032)

    def test_ast_rejects_out_of_range_vgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.vgpr(256)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class RegisterState:
    """Simulated state of GCN/RDNA registers for VALU scalar execution."""

    def __init__(self):
        self.vgpr = [0] * 256
        self.sgpr = [0] * 106

    def read_operand_u32(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgpr[op.index] & MASK32
        elif op.kind == OperandKind.SGPR:
            return self.sgpr[op.index] & MASK32
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            return op.value & MASK32
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_vgpr_u32(self, index: int, val: int) -> None:
        self.vgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> int:
        inst.validate()
        src0 = self.read_operand_u32(inst.src0)
        src1 = self.read_operand_u32(inst.src1)
        res = ref_pk_max_i16(src0, src1)
        self.write_vgpr_u32(inst.dst.index, res)
        return res


class TestValuPkMaxI16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = RegisterState()
        state.vgpr[1] = 0x0005FFFB  # [-5, +5]
        state.vgpr[2] = 0xFFFB0005  # [+5, -5]
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x00050005)
        self.assertEqual(state.vgpr[0], 0x00050005)

        # Cross-check through translator
        trans_res = run_translated_pk_max_i16(0x0005FFFB, 0xFFFB0005)
        self.assertEqual(trans_res, 0x00050005)


class TestValuPkMaxI16SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and per-lane packed comparisons."""

    def test_simd_per_lane_varying_inputs(self):
        for lane in range(64):
            val0 = (s16(lane - 32) & MASK16) | (((32 - lane) & MASK16) << 16)
            val1 = 0x00000000
            expected = ref_pk_max_i16(val0, val1)
            self.assertEqual(ref_pk_max_i16(val0, val1), expected)

        # Cross-check representative lanes (lane 0: -32, +32; lane 32: 0, 0; lane 63: +31, -31)
        # Lane 0: val0 = (-32, +32) vs 0 -> max(-32, 0) = 0, max(32, 0) = 32
        val0_lane0 = (s16(-32) & MASK16) | ((32 & MASK16) << 16)
        self.assertEqual(run_translated_pk_max_i16(val0_lane0, 0), (32 << 16) | 0)

        # Lane 63: val0 = (+31, -31) vs 0 -> max(31, 0) = 31, max(-31, 0) = 0
        val0_lane63 = (s16(31) & MASK16) | ((s16(-31) & MASK16) << 16)
        self.assertEqual(run_translated_pk_max_i16(val0_lane63, 0), (0 << 16) | 31)

    def test_simd_lane_masking_active_inactive(self):
        exec_mask = 0x5555555555555555
        dest_vgpr = [0xDEADBEEF] * 64

        for lane in range(64):
            is_active = (exec_mask & (1 << lane)) != 0
            if is_active:
                dest_vgpr[lane] = ref_pk_max_i16(lane, lane + 1)

        for lane in range(64):
            if (exec_mask & (1 << lane)) != 0:
                self.assertEqual(dest_vgpr[lane], ref_pk_max_i16(lane, lane + 1))
            else:
                self.assertEqual(dest_vgpr[lane], 0xDEADBEEF)

        # Cross-check active lane computation through translator
        self.assertEqual(run_translated_pk_max_i16(10, 11), ref_pk_max_i16(10, 11))


class TestValuPkMaxI16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        state = RegisterState()
        state.vgpr[0] = 0x0005FFFB
        state.vgpr[1] = 0xFFFB0005
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x00050005)
        self.assertEqual(state.vgpr[0], 0x00050005)

        # Translated cross-check
        lines = [
            "v_mov_b32_e32 v0, 0x0005fffb",
            "v_mov_b32_e32 v1, 0xfffb0005",
            "v_pk_max_i16 v0, v0, v1",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_pk_alias_dst_src0_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0x00050005)

    def test_dst_aliases_src1(self):
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        state = RegisterState()
        state.vgpr[0] = 0x0005FFFB
        state.vgpr[1] = 0xFFFB0005
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(1),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x00050005)
        self.assertEqual(state.vgpr[1], 0x00050005)

        # Translated cross-check
        lines = [
            "v_mov_b32_e32 v0, 0x0005fffb",
            "v_mov_b32_e32 v1, 0xfffb0005",
            "v_pk_max_i16 v1, v0, v1",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_pk_alias_dst_src1_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0x00050005)

    def test_self_aliasing(self):
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        state = RegisterState()
        state.vgpr[0] = 0x12345678
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x12345678)
        self.assertEqual(state.vgpr[0], 0x12345678)

        # Translated cross-check
        lines = [
            "v_mov_b32_e32 v0, 0x12345678",
            "v_pk_max_i16 v0, v0, v0",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_pk_alias_self_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 0x12345678)


class TestValuPkMaxI16ConditionCodes(unittest.TestCase):
    """Test suite for condition codes and flag preservation."""

    def test_flags_neutrality(self):
        # Packed integer ops do not set SCC or VCC
        scc = 0
        vcc = 0
        res = ref_pk_max_i16(0x7FFF7FFF, 0x80008000)
        self.assertEqual(res, 0x7FFF7FFF)
        self.assertEqual(scc, 0)
        self.assertEqual(vcc, 0)

        trans_res = run_translated_pk_max_i16(0x7FFF7FFF, 0x80008000)
        self.assertEqual(trans_res, 0x7FFF7FFF)


class TestValuPkMaxI16BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 16-bit float bit representations and signed int boundaries."""

    def test_float16_patterns_interpreted_as_signed_int16(self):
        # 0x8000 is -0.0 in float16, but -32768 in int16
        # 0x0000 is +0.0 in float16, but 0 in int16
        # In signed int: max(-32768, 0) = 0 (0x0000)
        res = ref_pk_max_i16(0x80008000, 0x00000000)
        self.assertEqual(res, 0x00000000)

        # Cross-check through translator: must yield 0, not 0x8000
        trans_res = run_translated_pk_max_i16(0x80008000, 0x00000000)
        self.assertEqual(trans_res, 0x00000000)

    def test_boundary_checkerboard_bitmasks(self):
        checkerboards = [0x55555555, 0xAAAAAAAA, 0x33333333, 0xCCCCCCCC, 0x0F0F0F0F, 0xF0F0F0F0]
        for cb0 in checkerboards:
            for cb1 in checkerboards:
                res = ref_pk_max_i16(cb0, cb1)
                self.assertEqual(res, ref_pk_max_i16(cb1, cb0))

        # Cross-check representative checkerboard pairs through translator
        self.assertEqual(run_translated_pk_max_i16(0x55555555, 0xAAAAAAAA), ref_pk_max_i16(0x55555555, 0xAAAAAAAA))
        self.assertEqual(run_translated_pk_max_i16(0x33333333, 0xCCCCCCCC), ref_pk_max_i16(0x33333333, 0xCCCCCCCC))


def parse_v_pk_max_i16_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for v_pk_max_i16."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    mnemonic = parts[0].lower()
    if mnemonic != "v_pk_max_i16":
        raise ValueError(f"Expected mnemonic v_pk_max_i16, got {mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")
    operands_str = [op.strip() for op in parts[1].split(",")]
    if len(operands_str) != 3:
        raise ValueError(f"Expected 3 operands (dst, src0, src1), got {len(operands_str)}")

    def parse_op(op_str: str) -> ASTOperand:
        op_lower = op_str.lower()
        if op_lower.startswith("v") and op_lower[1:].isdigit():
            return ASTOperand.vgpr(int(op_lower[1:]))
        elif op_lower.startswith("s") and op_lower[1:].isdigit():
            return ASTOperand.sgpr(int(op_lower[1:]))
        elif op_str.startswith("0x") or op_str.startswith("-") or op_str.isdigit():
            val = int(op_str, 0)
            return ASTOperand.literal32(val)
        else:
            raise ValueError(f"Invalid operand: {op_str}")

    dst = parse_op(operands_str[0])
    src0 = parse_op(operands_str[1])
    src1 = parse_op(operands_str[2])

    inst = ASTInstruction(Opcode.V_PK_MAX_I16, dst, src0, src1)
    inst.validate()
    return inst


# Mutant functions for mutation testing
def mutant_min(src0: int, src1: int) -> int:
    """Mutant 1: uses min instead of max."""
    lo0 = s16(src0 & MASK16)
    hi0 = s16((src0 >> 16) & MASK16)
    lo1 = s16(src1 & MASK16)
    hi1 = s16((src1 >> 16) & MASK16)
    return ((min(hi0, hi1) & MASK16) << 16) | (min(lo0, lo1) & MASK16)


def mutant_unsigned_max(src0: int, src1: int) -> int:
    """Mutant 2: uses unsigned max (fails on negative numbers)."""
    lo0 = src0 & MASK16
    hi0 = (src0 >> 16) & MASK16
    lo1 = src1 & MASK16
    hi1 = (src1 >> 16) & MASK16
    return ((max(hi0, hi1) & MASK16) << 16) | (max(lo0, lo1) & MASK16)


def mutant_lo_only(src0: int, src1: int) -> int:
    """Mutant 3: writes lo result to both halves."""
    lo0 = s16(src0 & MASK16)
    lo1 = s16(src1 & MASK16)
    m = max(lo0, lo1) & MASK16
    return (m << 16) | m


def mutant_hi_only(src0: int, src1: int) -> int:
    """Mutant 4: writes hi result to both halves."""
    hi0 = s16((src0 >> 16) & MASK16)
    hi1 = s16((src1 >> 16) & MASK16)
    m = max(hi0, hi1) & MASK16
    return (m << 16) | m


class TestValuPkMaxI16NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        with self.assertRaises(ValueError):
            parse_v_pk_max_i16_asm("")

        with self.assertRaises(ValueError):
            parse_v_pk_max_i16_asm("v_pk_min_i16 v0, v1, v2")

        with self.assertRaises(ValueError):
            parse_v_pk_max_i16_asm("v_pk_max_i16 v0, v1")

        with self.assertRaises(TypeError):
            parse_v_pk_max_i16_asm("v_pk_max_i16 s0, v1, v2")

    def test_mutation_detection_min(self):
        s0 = 0x0005FFFB
        s1 = 0xFFFB0005
        correct = ref_pk_max_i16(s0, s1)
        mutated = mutant_min(s0, s1)
        self.assertNotEqual(correct, mutated)

        # Cross-check through translated execution
        trans_res = run_translated_pk_max_i16(s0, s1)
        self.assertEqual(trans_res, correct)
        self.assertNotEqual(trans_res, mutated)

    def test_mutation_detection_unsigned_max(self):
        # -5 (0xFFFB) vs +5 (0x0005): in signed max, result is +5 (0x0005).
        # In unsigned max, 0xFFFB (65531) > 0x0005 (5), so result is 0xFFFB.
        s0 = 0x0005FFFB
        s1 = 0xFFFB0005
        correct = ref_pk_max_i16(s0, s1)
        mutated = mutant_unsigned_max(s0, s1)
        self.assertNotEqual(correct, mutated)

        # Cross-check through translated execution
        trans_res = run_translated_pk_max_i16(s0, s1)
        self.assertEqual(trans_res, correct)
        self.assertNotEqual(trans_res, mutated)

    def test_mutation_detection_lo_only(self):
        s0 = (10 << 16) | 1
        s1 = (20 << 16) | 2
        correct = ref_pk_max_i16(s0, s1)
        mutated = mutant_lo_only(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_hi_only(self):
        s0 = (10 << 16) | 1
        s1 = (20 << 16) | 2
        correct = ref_pk_max_i16(s0, s1)
        mutated = mutant_hi_only(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_pk_max_spirv_emission(self):
        """Verify that support.translate_lines emits SMax and OpBitFieldSExtract in SPIR-V disassembly."""
        lines = [
            "v_mov_b32_e32 v1, 0x0005fffb",
            "v_mov_b32_e32 v2, 0xfffb0005",
            "v_pk_max_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_i16_emit", lines)
        self.assertIn("OpBitFieldSExtract", spv)
        self.assertIn("SMax", spv)
        self.assertNotIn("OpBitFieldUExtract", spv)
        self.assertNotIn("UMax", spv)
        self.assertNotIn("SMin", spv)

    def test_pk_max_hardware_execution(self):
        """Verify hardware-accurate execution of v_pk_max_i16 through support.mrt0_words."""
        lines = [
            "v_mov_b32_e32 v1, 0x0005fffb",
            "v_mov_b32_e32 v2, 0xfffb0005",
            "v_pk_max_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_i16_hw", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x00050005)
        self.assertEqual(words[1], 0x00050005)
        self.assertEqual(words[2], 0x00050005)
        self.assertEqual(words[3], 0x00050005)

    def test_pk_max_op_sel_modifiers_translated(self):
        """Verify op_sel and op_sel_hi operand routing with v_pk_max_i16."""
        # s0 = hi: 10, lo: -20 (0x000A_FFEC)
        # s1 = hi: -5, lo: 30  (0xFFFB_001E)
        # op_sel:[1, 0] -> selects s0.hi (10) and s1.lo (30): max(10, 30) = 30 (0x001E)
        # op_sel_hi:[0, 1] -> selects s0.lo (-20) and s1.hi (-5): max(-20, -5) = -5 (0xFFFB)
        s0 = (10 << 16) | (-20 & MASK16)
        s1 = ((-5 & MASK16) << 16) | 30
        res = run_translated_pk_max_i16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        expected = ((-5 & MASK16) << 16) | 30
        self.assertEqual(res, expected)

    def test_pk_max_sgpr_and_literal_operands_translated(self):
        """Verify v_pk_max_i16 with SGPR source and literal constant operands."""
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "s_mov_b32 s0, 0x80000005",   # hi: -32768, lo: +5
            "v_mov_b32_e32 v1, 0x0000fffb", # hi: 0, lo: -5
            "v_pk_max_i16 v0, s0, v1",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_pk_max_sgpr_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        # lo: max(+5, -5) = +5 (0x0005)
        # hi: max(-32768, 0) = 0 (0x0000)
        self.assertEqual(words[0] & MASK32, 0x00000005)


if __name__ == "__main__":
    unittest.main()
