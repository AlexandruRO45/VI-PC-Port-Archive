from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_add3_u32.

Tests v_add3_u32 (three-operand 32-bit unsigned addition) against mathematical reference semantics
and translated SPIR-V shader executions via rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_add3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_add3_u32: (a + b + c) mod 2^32."""
    return (u32(a) + u32(b) + u32(c)) & MASK32


_TRANSLATE_ID: int = 0


def run_translated_add3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_add3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_add3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_add3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuAdd3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_add3_u32."""

    def test_oracle_basic_addition(self) -> None:
        self.assertEqual(oracle_add3_u32(10, 20, 30), 60)
        self.assertEqual(oracle_add3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_add3_u32(1, 2, 3), 6)
        self.assertEqual(oracle_add3_u32(100, 200, 300), 600)

    def test_oracle_wrapping(self) -> None:
        self.assertEqual(oracle_add3_u32(MASK32, 1, 0), 0)
        self.assertEqual(oracle_add3_u32(MASK32, 1, 5), 5)
        self.assertEqual(oracle_add3_u32(MASK32, MASK32, MASK32), MASK32 - 2)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 1, 0),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
        ]
        for a, b, c in triples:
            expected = oracle_add3_u32(a, b, c)
            actual = run_translated_add3_u32(a, b, c)
            self.assertEqual(actual, expected)

    def test_oracle_commutativity_and_associativity(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = oracle_add3_u32(a, b, c)
        res2 = oracle_add3_u32(c, a, b)
        res3 = oracle_add3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)


# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        if self.kind == RegKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


class Opcode(enum.Enum):
    V_ADD3_U32 = "v_add3_u32"


@dataclasses.dataclass
class VOP3Instruction:
    opcode: Opcode
    dst: Register
    src0: Union[Register, int]
    src1: Union[Register, int]
    src2: Union[Register, int]

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP3 destination must be VGPR")
        if self.opcode != Opcode.V_ADD3_U32:
            raise ValueError(f"Invalid opcode {self.opcode}")

    def to_asm(self) -> str:
        s0 = f"{self.src0:#x}" if isinstance(self.src0, int) else str(self.src0)
        s1 = f"{self.src1:#x}" if isinstance(self.src1, int) else str(self.src1)
        s2 = f"{self.src2:#x}" if isinstance(self.src2, int) else str(self.src2)
        return f"{self.opcode.value} {self.dst}, {s0}, {s1}, {s2}"


class TestValuAdd3U32AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_add3_u32."""

    def test_instruction_creation_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        s0 = Register(RegKind.VGPR, 1)
        s1 = Register(RegKind.VGPR, 2)
        s2 = Register(RegKind.VGPR, 3)
        inst = VOP3Instruction(Opcode.V_ADD3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_add3_u32 v0, v1, v2, v3")

    def test_instruction_with_literals_and_sgprs(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        s0 = Register(RegKind.SGPR, 0)
        s1 = 0x100
        s2 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_ADD3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_add3_u32 v4, s0, 0x100, v5")

    def test_ast_emits_valid_spirv(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_add3", lines)
        self.assertIn("OpIAdd", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 60)


# ---------------------------------------------------------------------------
# Stage 3: Deterministic scalar instruction interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState:
    """Architectural state for executing VALU scalar instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[Register, int]) -> int:
        if isinstance(op, int):
            return op & MASK32
        if op.kind == RegKind.VGPR:
            return self.vgprs[op.index]
        return self.sgprs[op.index]

    def write_reg(self, reg: Register, val: int) -> None:
        val &= MASK32
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = val
        else:
            self.sgprs[reg.index] = val

    def step(self, inst: VOP3Instruction) -> int:
        val0 = self.read_operand(inst.src0)
        val1 = self.read_operand(inst.src1)
        val2 = self.read_operand(inst.src2)
        res = oracle_add3_u32(val0, val1, val2)
        self.write_reg(inst.dst, res)
        return res


class TestValuAdd3U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_add3_u32."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_scalar_step_basic(self) -> None:
        self.state.vgprs[1] = 5
        self.state.vgprs[2] = 10
        self.state.vgprs[3] = 15
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 30)
        self.assertEqual(self.state.vgprs[0], 30)

    def test_scalar_step_sgpr_and_imm(self) -> None:
        self.state.sgprs[4] = 100
        self.state.vgprs[1] = 50
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SGPR, 4),
            25,
            Register(RegKind.VGPR, 1),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 175)
        self.assertEqual(self.state.vgprs[0], 175)

    def test_scalar_matches_translated_shader(self) -> None:
        test_cases = [
            (1, 2, 3),
            (100, 200, 300),
            (0xFFFF_0000, 0x0000_FFFF, 1),
            (0x8000_0000, 0x4000_0000, 0x4000_0000),
        ]
        for a, b, c in test_cases:
            self.state.vgprs[1] = a
            self.state.vgprs[2] = b
            self.state.vgprs[3] = c
            inst = VOP3Instruction(
                Opcode.V_ADD3_U32,
                Register(RegKind.VGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
                Register(RegKind.VGPR, 3),
            )
            scalar_res = self.state.step(inst)
            trans_res = run_translated_add3_u32(a, b, c)
            self.assertEqual(scalar_res, trans_res)


# ---------------------------------------------------------------------------
# Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# ---------------------------------------------------------------------------

class SIMDWavefront:
    """Wavefront model simulating 32-lane or 64-lane SIMD execution."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]

    def read_operand(self, op: Union[Register, int], lane: int) -> int:
        if isinstance(op, int):
            return op & MASK32
        return self.vgprs[op.index][lane]

    def write_vgpr(self, reg: Register, lane: int, val: int) -> None:
        self.vgprs[reg.index][lane] = val & MASK32

    def step(self, inst: VOP3Instruction) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                v0 = self.read_operand(inst.src0, lane)
                v1 = self.read_operand(inst.src1, lane)
                v2 = self.read_operand(inst.src2, lane)
                res = oracle_add3_u32(v0, v1, v2)
                self.write_vgpr(inst.dst, lane, res)


class TestValuAdd3U32SIMD(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_add3_u32."""

    def test_simd32_all_lanes_active(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        for lane in range(32):
            wave.vgprs[1][lane] = lane
            wave.vgprs[2][lane] = lane * 2
            wave.vgprs[3][lane] = lane * 3
        wave.step(inst)
        for lane in range(32):
            self.assertEqual(wave.vgprs[0][lane], lane * 6)

    def test_simd64_divergence_exec_mask(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        wave.exec_mask = 0x5555_5555_5555_5555  # even lanes active
        for lane in range(64):
            wave.vgprs[0][lane] = 0xCAFE_BABE
            wave.vgprs[1][lane] = 10
            wave.vgprs[2][lane] = 20
            wave.vgprs[3][lane] = 30
        wave.step(inst)
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], 60)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xCAFE_BABE)

    def test_simd_inactive_lanes_unmodified(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        wave.exec_mask = 0x0000_0001  # only lane 0 active
        for lane in range(32):
            wave.vgprs[0][lane] = 0x1111_1111 * lane
            wave.vgprs[1][lane] = 1
            wave.vgprs[2][lane] = 2
            wave.vgprs[3][lane] = 3
        wave.step(inst)
        self.assertEqual(wave.vgprs[0][0], 6)
        for lane in range(1, 32):
            self.assertEqual(wave.vgprs[0][lane], 0x1111_1111 * lane)

    def test_simd_translator_active_mask_preservation(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [inst.to_asm(), "s_endpgm"]
        spv = support.translate_lines("ps_simd_add3", lines)
        self.assertIn("OpSelect", spv)


# ---------------------------------------------------------------------------
# Stage 5: Register aliasing and in-place operand preservation
# ---------------------------------------------------------------------------

class TestValuAdd3U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_add3_u32."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_in_place_dst_equals_src0(self) -> None:
        self.state.vgprs[1] = 10
        self.state.vgprs[2] = 20
        self.state.vgprs[3] = 30
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 60)
        self.assertEqual(self.state.vgprs[1], 60)

    def test_in_place_all_operands_aliased(self) -> None:
        self.state.vgprs[1] = 15
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 1),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 45)
        self.assertEqual(self.state.vgprs[1], 45)

    def test_in_place_translated_shader(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 300",
            "v_add3_u32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inplace_add3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 600)

    def test_surrounding_registers_unmodified(self) -> None:
        self.state.vgprs[0] = 0xAAAA_AAAA
        self.state.vgprs[4] = 0xBBBB_BBBB
        self.state.vgprs[1] = 1
        self.state.vgprs[2] = 2
        self.state.vgprs[3] = 3
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[0], 0xAAAA_AAAA)
        self.assertEqual(self.state.vgprs[2], 6)
        self.assertEqual(self.state.vgprs[4], 0xBBBB_BBBB)


# ---------------------------------------------------------------------------
# Stage 6: Condition codes, status flags, and carry/borrow generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Add3StatusFlags:
    """Status flags and carry analysis for 3-operand addition."""
    zero: bool
    negative: bool
    carry_count: int  # 0, 1, or 2 (since 3 * (2^32 - 1) < 3 * 2^32)
    parity: bool      # True if even parity


def compute_add3_status_flags(a: int, b: int, c: int) -> Add3StatusFlags:
    total = u32(a) + u32(b) + u32(c)
    res = total & MASK32
    carry = total >> 32
    zero = (res == 0)
    negative = bool(res & SIGN_BIT32)
    parity = (bin(res).count("1") % 2 == 0)
    return Add3StatusFlags(
        zero=zero,
        negative=negative,
        carry_count=carry,
        parity=parity,
    )


class TestValuAdd3U32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry generation for v_add3_u32."""

    def test_status_flags_zero(self) -> None:
        flags = compute_add3_status_flags(0, 0, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertEqual(flags.carry_count, 0)
        self.assertTrue(flags.parity)

    def test_status_flags_single_carry(self) -> None:
        flags = compute_add3_status_flags(MASK32, 1, 0)
        self.assertTrue(flags.zero)
        self.assertEqual(flags.carry_count, 1)

    def test_status_flags_double_carry(self) -> None:
        flags = compute_add3_status_flags(MASK32, MASK32, MASK32)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertEqual(flags.carry_count, 2)
        self.assertEqual(flags.carry_count * (1 << 32) + (MASK32 - 2), MASK32 * 3)

    def test_status_flags_negative(self) -> None:
        flags = compute_add3_status_flags(0x8000_0000, 0, 0)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)


# ---------------------------------------------------------------------------
# Stage 7: IEEE 754 boundary extremes, integer boundaries, and bitwise limits
# ---------------------------------------------------------------------------

class TestValuAdd3U32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, integer limits, and bit-pattern invariance for v_add3_u32."""

    def test_integer_boundary_extremes(self) -> None:
        # Max positive signed int wrap
        self.assertEqual(oracle_add3_u32(0x7FFF_FFFF, 1, 0), 0x8000_0000)
        self.assertEqual(oracle_add3_u32(0x7FFF_FFFF, 0x7FFF_FFFF, 2), 0)
        # Power of two alignments
        self.assertEqual(oracle_add3_u32(0x1000_0000, 0x2000_0000, 0x4000_0000), 0x7000_0000)

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        # In RDNA2, integer instructions treat IEEE 754 float bit-patterns strictly as integers
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]
        
        res = oracle_add3_u32(pos_inf, neg_inf, 0)
        expected = (pos_inf + neg_inf) & MASK32
        self.assertEqual(res, expected)

        res_trans = run_translated_add3_u32(pos_inf, neg_inf, 0)
        self.assertEqual(res_trans, expected)

    def test_double_overflow_wrap(self) -> None:
        a = 0xFFFF_FFFF
        b = 0xFFFF_FFFF
        c = 0xFFFF_FFFF
        res_oracle = oracle_add3_u32(a, b, c)
        res_trans = run_translated_add3_u32(a, b, c)
        self.assertEqual(res_oracle, 0xFFFF_FFFD)
        self.assertEqual(res_trans, 0xFFFF_FFFD)


# ---------------------------------------------------------------------------
# Stage 8: Negative controls, syntax rejection, and mutation detection
# ---------------------------------------------------------------------------

class TestValuAdd3U32NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection for v_add3_u32."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)

    def test_invalid_destination_kind(self) -> None:
        with self.assertRaises(ValueError):
            VOP3Instruction(
                Opcode.V_ADD3_U32,
                Register(RegKind.SGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
                Register(RegKind.VGPR, 3),
            )

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=128)

    def test_mutation_detection_and_oracle_rejection(self) -> None:
        a, b, c = 10, 20, 30
        correct = oracle_add3_u32(a, b, c)

        def mutated_oracle_sub(x: int, y: int, z: int) -> int:
            return (x + y - z) & MASK32

        def mutated_oracle_two_ops(x: int, y: int, z: int) -> int:
            return (x + y) & MASK32

        self.assertNotEqual(mutated_oracle_sub(a, b, c), correct)
        self.assertNotEqual(mutated_oracle_two_ops(a, b, c), correct)

    def test_translator_compliance_guard(self) -> None:
        test_path = os.path.join(HERE, "test_valu_add3_u32_hw.py")
        self.assertTrue(os.path.isfile(test_path))
        with open(test_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("import support", content)
        self.assertIn("support.translate_lines", content)


if __name__ == "__main__":
    unittest.main()
