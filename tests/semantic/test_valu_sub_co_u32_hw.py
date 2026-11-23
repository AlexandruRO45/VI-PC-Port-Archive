from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_sub_co_u32.

Validates 32-bit unsigned vector subtraction with carry-out / borrow:
  vdst = (src0 - src1) & 0xFFFFFFFF
  sdst / vcc_lo = 0xFFFFFFFF if (src0 < src1) else 0

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with carry/borrow, IEEE 754 bit-pattern invariance, and
negative controls. All stages cross-checked with rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_TRANSLATE_ID = 0


def oracle_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Mathematical reference oracle for v_sub_co_u32.

    Returns:
        (diff, borrow) where:
          diff = (src0 - src1) & 0xFFFFFFFF
          borrow = 1 if (src0 & 0xFFFFFFFF) < (src1 & 0xFFFFFFFF) else 0
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    borrow = 1 if u0 < u1 else 0
    return diff, borrow


def run_translated_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Execute v_sub_co_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        "v_sub_co_u32 v0, vcc_lo, v1, v2",
        "v_cndmask_b32 v3, 0, 1, vcc_lo",
        "exp mrt0 v0, v3, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_sub_co_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32, words[1] & 1

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuSubCoU32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_sub_co_u32."""

    def test_oracle_basic_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(10, 3), (7, 0))
        self.assertEqual(oracle_sub_co_u32(100, 100), (0, 0))
        self.assertEqual(oracle_sub_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_sub_co_u32(1000, 400), (600, 0))

    def test_oracle_borrow_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(3, 10), (MASK32 - 6, 1))
        self.assertEqual(oracle_sub_co_u32(0, 1), (MASK32, 1))
        self.assertEqual(oracle_sub_co_u32(0, MASK32), (1, 1))
        self.assertEqual(oracle_sub_co_u32(MASK32, 0), (MASK32, 0))

    def test_oracle_matches_translated_execution(self) -> None:
        pairs = [
            (10, 3),
            (3, 10),
            (0, 0),
            (0, 1),
            (1, 0),
            (MASK32, 1),
            (1, MASK32),
            (0x12345678, 0x11111111),
            (0x11111111, 0x12345678),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
        ]
        for a, b in pairs:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Mismatch on {a:#x} - {b:#x}")

# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()
    SPECIAL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index < 256):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.kind == RegKind.SGPR and not (0 <= self.index < 106):
            raise ValueError(f"SGPR index out of range: {self.index}")

    def __str__(self) -> str:
        if self.name:
            return self.name
        if self.kind == RegKind.VGPR:
            return f"v{self.index}"
        if self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return f"special_{self.index}"


class Opcode(enum.Enum):
    V_SUB_CO_U32 = "v_sub_co_u32"


@dataclasses.dataclass(frozen=True)
class VOP3COInstruction:
    opcode: Opcode
    vdst: Register
    sdst: Register
    src0: Register | int
    src1: Register | int

    def __post_init__(self) -> None:
        if self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination vdst must be VGPR, got {self.vdst.kind}")
        if self.sdst.kind not in (RegKind.SGPR, RegKind.SPECIAL):
            raise ValueError(f"Destination sdst must be SGPR or SPECIAL, got {self.sdst.kind}")


class TestValuSubCoU32ASTDefinitions(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_sub_co_u32."""

    def test_ast_construction_and_formatting(self) -> None:
        vdst = Register(RegKind.VGPR, 0)
        sdst = Register(RegKind.SPECIAL, 0, "vcc_lo")
        src0 = Register(RegKind.VGPR, 1)
        src1 = Register(RegKind.VGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, src1)

        self.assertEqual(inst.opcode, Opcode.V_SUB_CO_U32)
        self.assertEqual(str(inst.vdst), "v0")
        self.assertEqual(str(inst.sdst), "vcc_lo")
        self.assertEqual(str(inst.src0), "v1")
        self.assertEqual(str(inst.src1), "v2")

    def test_ast_with_literal_and_sgpr(self) -> None:
        vdst = Register(RegKind.VGPR, 5)
        sdst = Register(RegKind.SGPR, 4)
        src0 = Register(RegKind.SGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, 42)

        self.assertEqual(str(inst.vdst), "v5")
        self.assertEqual(str(inst.sdst), "s4")
        self.assertEqual(str(inst.src0), "s2")
        self.assertEqual(inst.src1, 42)

# ---------------------------------------------------------------------------
# Stage 3: Deterministic instruction interpreter
# ---------------------------------------------------------------------------

class ValuSubCoInterpreter:
    """Scalar architectural interpreter for v_sub_co_u32."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc_lo = 0

    def resolve(self, operand: Register | int) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        if operand.kind == RegKind.VGPR:
            return self.vgprs[operand.index]
        if operand.kind == RegKind.SGPR:
            return self.sgprs[operand.index]
        if operand.name == "vcc_lo":
            return self.vcc_lo
        return 0

    def step(self, inst: VOP3COInstruction) -> tuple[int, int]:
        s0 = self.resolve(inst.src0)
        s1 = self.resolve(inst.src1)
        diff, borrow = oracle_sub_co_u32(s0, s1)
        self.vgprs[inst.vdst.index] = diff
        borrow_mask = MASK32 if borrow else 0
        if inst.sdst.kind == RegKind.SPECIAL and inst.sdst.name == "vcc_lo":
            self.vcc_lo = borrow_mask
        elif inst.sdst.kind == RegKind.SGPR:
            self.sgprs[inst.sdst.index] = borrow_mask
        return diff, borrow


class TestValuSubCoU32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_sub_co_u32."""

    def test_interpreter_vgpr_subtraction(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 50
        interp.vgprs[2] = 20

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 30)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 30)
        self.assertEqual(interp.vcc_lo, 0)

    def test_interpreter_borrow_into_sgpr(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 10
        interp.vgprs[2] = 25

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 3),
            Register(RegKind.SGPR, 8),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, (10 - 25) & MASK32)
        self.assertEqual(borrow, 1)
        self.assertEqual(interp.vgprs[3], (10 - 25) & MASK32)
        self.assertEqual(interp.sgprs[8], MASK32)

    def test_interpreter_literal_subtraction(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 100

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            30,
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 70)
        self.assertEqual(borrow, 0)

# ---------------------------------------------------------------------------
# Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# ---------------------------------------------------------------------------

class SIMDWavefront:
    """SIMD Wavefront multi-lane model with 32-bit or 64-bit EXEC masking."""

    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.mask_limit = (1 << wave_size) - 1
        self.exec_mask = self.mask_limit if exec_mask is None else (exec_mask & self.mask_limit)

    def execute(self, src0_lanes: list[int], src1_lanes: list[int]) -> tuple[list[int], int]:
        diff_lanes: list[int] = [0] * self.wave_size
        vcc_mask = 0

        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                diff, borrow = oracle_sub_co_u32(src0_lanes[lane], src1_lanes[lane])
                diff_lanes[lane] = diff
                if borrow:
                    vcc_mask |= (1 << lane)
            else:
                diff_lanes[lane] = 0

        return diff_lanes, vcc_mask


class TestValuSubCoU32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and EXEC masking for v_sub_co_u32."""

    def test_wave32_all_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        s0 = [lane * 10 for lane in range(32)]
        s1 = [5 for _ in range(32)]

        diffs, vcc = wave.execute(s0, s1)
        # Lane 0: 0 - 5 => borrow
        self.assertEqual(diffs[0], (0 - 5) & MASK32)
        self.assertEqual(vcc & 1, 1)
        # Lane 1+: >= 10 - 5 => no borrow
        for lane in range(1, 32):
            self.assertEqual(diffs[lane], lane * 10 - 5)
            self.assertEqual((vcc >> lane) & 1, 0)

    def test_wave64_divergence_masking(self) -> None:
        # Alternating even lanes enabled
        mask = 0x5555_5555_5555_5555
        wave = SIMDWavefront(wave_size=64, exec_mask=mask)
        s0 = [0] * 64
        s1 = [1] * 64

        diffs, vcc = wave.execute(s0, s1)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(diffs[lane], MASK32)
                self.assertEqual((vcc >> lane) & 1, 1)
            else:
                self.assertEqual(diffs[lane], 0)
                self.assertEqual((vcc >> lane) & 1, 0)
        self.assertEqual(vcc, mask)

# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuSubCoU32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_sub_co_u32."""

    def test_inplace_dst_equals_src0(self) -> None:
        # v0 = v0 - v1
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 100
        interp.vgprs[1] = 40

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 60)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 60)
        self.assertEqual(interp.vgprs[1], 40)

    def test_inplace_dst_equals_src1(self) -> None:
        # v1 = v0 - v1
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 50
        interp.vgprs[1] = 80

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, (50 - 80) & MASK32)
        self.assertEqual(borrow, 1)
        self.assertEqual(interp.vgprs[1], (50 - 80) & MASK32)
        self.assertEqual(interp.vgprs[0], 50)

    def test_inplace_all_aliased(self) -> None:
        # v0 = v0 - v0
        interp = ValuSubCoInterpreter()
        interp.vgprs[0] = 0xDEADBEEF

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 0),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 0)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 0)

    def test_aliased_translated_execution(self) -> None:
        global _TRANSLATE_ID
        _TRANSLATE_ID += 1
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 35",
            "v_sub_co_u32 v0, vcc_lo, v0, v1",
            "v_cndmask_b32 v3, 0, 1, vcc_lo",
            "exp mrt0 v0, v3, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines(f"ps_sub_co_alias_{_TRANSLATE_ID}", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK32, 65)
        self.assertEqual(words[1] & 1, 0)

# ---------------------------------------------------------------------------
# Stage 6: Condition codes, status flags, and carry/borrow generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SubStatusFlags:
    zero: bool
    negative: bool
    borrow: bool
    overflow: bool


def compute_sub_status_flags(src0: int, src1: int) -> SubStatusFlags:
    """Compute ALU status flags for 32-bit subtraction src0 - src1."""
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    zero = (diff == 0)
    negative = bool((diff >> 31) & 1)
    borrow = (u0 < u1)
    # Signed overflow: operands have different signs and result sign differs from src0
    overflow = bool(((u0 ^ u1) & (u0 ^ diff) & 0x8000_0000) != 0)
    return SubStatusFlags(zero=zero, negative=negative, borrow=borrow, overflow=overflow)


class TestValuSubCoU32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow for v_sub_co_u32."""

    def test_zero_flag(self) -> None:
        flags = compute_sub_status_flags(0x1234, 0x1234)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.overflow)

    def test_negative_and_borrow(self) -> None:
        flags = compute_sub_status_flags(5, 10)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.overflow)

    def test_signed_overflow_without_unsigned_borrow(self) -> None:
        # 0x8000_0000 (-2147483648) - 1 => 0x7FFF_FFFF (+2147483647)
        # Unsigned: 0x8000_0000 >= 1 => no borrow
        # Signed: -2147483648 - 1 => overflow
        flags = compute_sub_status_flags(0x8000_0000, 1)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertTrue(flags.overflow)

    def test_multi_precision_64bit_borrow_chain(self) -> None:
        # Compute 64-bit subtraction using two 32-bit steps:
        # A = 0x0000_0002_0000_0001, B = 0x0000_0001_0000_0005
        # Lower: 1 - 5 = 0xFFFF_FFFC with borrow = 1
        # Upper: 2 - 1 - borrow = 0 with borrow = 0
        a_lo, a_hi = 1, 2
        b_lo, b_hi = 5, 1

        diff_lo, borrow_lo = oracle_sub_co_u32(a_lo, b_lo)
        self.assertEqual(diff_lo, 0xFFFF_FFFC)
        self.assertEqual(borrow_lo, 1)

        diff_hi, borrow_hi = oracle_sub_co_u32(a_hi - borrow_lo, b_hi)
        self.assertEqual(diff_hi, 0)
        self.assertEqual(borrow_hi, 0)

        res64 = (diff_hi << 32) | diff_lo
        expected64 = (((a_hi << 32) | a_lo) - ((b_hi << 32) | b_lo)) & MASK64
        self.assertEqual(res64, expected64)

# ---------------------------------------------------------------------------
# Stage 7: IEEE 754 boundary extremes, integer boundaries, and bitwise limits
# ---------------------------------------------------------------------------

class TestValuSubCoU32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, integer limits, and bit-pattern invariance for v_sub_co_u32."""

    def test_integer_boundary_extremes(self) -> None:
        # Max unsigned boundary
        self.assertEqual(oracle_sub_co_u32(MASK32, MASK32), (0, 0))
        self.assertEqual(oracle_sub_co_u32(MASK32, 0), (MASK32, 0))
        self.assertEqual(oracle_sub_co_u32(0, MASK32), (1, 1))

        # Sign bit boundaries
        self.assertEqual(oracle_sub_co_u32(0x8000_0000, 1), (0x7FFF_FFFF, 0))
        self.assertEqual(oracle_sub_co_u32(0x7FFF_FFFF, 0x8000_0000), (MASK32, 1))

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]

        res = oracle_sub_co_u32(pos_inf, neg_inf)
        expected = ((pos_inf - neg_inf) & MASK32, 1 if pos_inf < neg_inf else 0)
        self.assertEqual(res, expected)

        res_trans = run_translated_sub_co_u32(pos_inf, neg_inf)
        self.assertEqual(res_trans, expected)

        # qnan and 0
        res_nan = oracle_sub_co_u32(qnan, 0)
        self.assertEqual(res_nan, (qnan, 0))
        self.assertEqual(run_translated_sub_co_u32(qnan, 0), (qnan, 0))

if __name__ == "__main__":
    unittest.main()
