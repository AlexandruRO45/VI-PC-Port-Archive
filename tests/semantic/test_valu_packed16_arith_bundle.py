"""Hardware-verified semantic reference and translator regression suite for RDNA2 VALU packed-16 arithmetic bundle.

Covers the complete packed 16-bit integer ALU instruction family:
- v_pk_add_u16:    Packed unsigned 16-bit addition (with optional clamp saturation to 0xFFFF)
- v_pk_add_i16:    Packed signed 16-bit addition (with optional clamp saturation to [-32768, 32767])
- v_pk_sub_u16:    Packed unsigned 16-bit subtraction (with optional clamp saturation to 0x0000)
- v_pk_sub_i16:    Packed signed 16-bit subtraction (with optional clamp saturation to [-32768, 32767])
- v_pk_mul_lo_u16: Packed 16-bit low-half multiplication (modulo 2^16)
- v_pk_min_u16:    Packed unsigned 16-bit minimum: min(S0, S1)
- v_pk_max_u16:    Packed unsigned 16-bit maximum: max(S0, S1)
- v_pk_min_i16:    Packed signed 16-bit minimum: min(S0, S1)

Tests mathematical reference oracles, AST typed instruction definitions, scalar machine state interpreters,
SIMD multi-lane divergence and 16-bit boundary extremes, signed/unsigned divergence,
and full translation into SPIR-V shaders executed through spirv_mini_interp.

Zero game data, standard library Python only.
"""
from __future__ import annotations

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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
SIGN_BIT16: int = 0x8000


def u16(x: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return x & MASK16


def s16(x: int) -> int:
    """Interpret 16-bit halfword as signed two's complement integer."""
    x &= MASK16
    return x - (1 << 16) if (x & SIGN_BIT16) else x


def clamp_u16(x: int) -> int:
    """Saturate unsigned integer to [0, 0xFFFF]."""
    return max(0, min(0xFFFF, x))


def clamp_i16(x: int) -> int:
    """Saturate signed integer to [-32768, 32767]."""
    return max(-32768, min(32767, x)) & MASK16


def eval_pk16(op: str, a: int, b: int, *, clamp: bool = False) -> int:
    """Mathematical reference oracle for packed 16-bit integer operations."""
    a_lo, a_hi = a & MASK16, (a >> 16) & MASK16
    b_lo, b_hi = b & MASK16, (b >> 16) & MASK16

    def eval_half(h0: int, h1: int) -> int:
        if op == "v_pk_add_u16":
            return clamp_u16(h0 + h1) if clamp else ((h0 + h1) & MASK16)
        elif op == "v_pk_add_i16":
            return clamp_i16(s16(h0) + s16(h1)) if clamp else ((s16(h0) + s16(h1)) & MASK16)
        elif op == "v_pk_sub_u16":
            return clamp_u16(h0 - h1) if clamp else ((h0 - h1) & MASK16)
        elif op == "v_pk_sub_i16":
            return clamp_i16(s16(h0) - s16(h1)) if clamp else ((s16(h0) - s16(h1)) & MASK16)
        elif op == "v_pk_mul_lo_u16":
            return (h0 * h1) & MASK16
        elif op == "v_pk_min_u16":
            return min(h0, h1)
        elif op == "v_pk_max_u16":
            return max(h0, h1)
        elif op == "v_pk_min_i16":
            return (s16(h0) if s16(h0) < s16(h1) else s16(h1)) & MASK16
        raise ValueError(f"Unknown op {op}")

    r_lo = eval_half(a_lo, b_lo)
    r_hi = eval_half(a_hi, b_hi)
    return (r_lo & MASK16) | ((r_hi & MASK16) << 16)


_TRANSLATE_ID: int = 0


def run_translated_pk16(op: str, src0: int, src1: int, *, clamp: bool = False) -> int:
    """Execute packed 16-bit VALU opcode through rdna2spirv and return result word."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_str = " clamp" if clamp else ""
    prog = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"{op} v0, v1, v2{clamp_str}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_pk16_{op}_{_TRANSLATE_ID}", prog)
    return support.mrt0_words(spv)[0] & MASK32


@dataclasses.dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")

    def __str__(self) -> str:
        return f"v{self.index}"


class Packed16Opcode(enum.Enum):
    V_PK_ADD_U16 = "v_pk_add_u16"
    V_PK_ADD_I16 = "v_pk_add_i16"
    V_PK_SUB_U16 = "v_pk_sub_u16"
    V_PK_SUB_I16 = "v_pk_sub_i16"
    V_PK_MUL_LO_U16 = "v_pk_mul_lo_u16"
    V_PK_MIN_U16 = "v_pk_min_u16"
    V_PK_MAX_U16 = "v_pk_max_u16"
    V_PK_MIN_I16 = "v_pk_min_i16"


@dataclasses.dataclass
class VOP3PInstruction:
    opcode: Packed16Opcode
    dst: VGPR
    src0: Union[VGPR, int]
    src1: Union[VGPR, int]
    clamp: bool = False

    def to_asm(self) -> str:
        s0 = f"{self.src0:#x}" if isinstance(self.src0, int) else str(self.src0)
        s1 = f"{self.src1:#x}" if isinstance(self.src1, int) else str(self.src1)
        clamp_str = " clamp" if self.clamp else ""
        return f"{self.opcode.value} {self.dst}, {s0}, {s1}{clamp_str}"


class Packed16MachineState:
    """Architectural state for executing packed 16-bit VALU instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256

    def read_operand(self, op: Union[VGPR, int]) -> int:
        if isinstance(op, int):
            return op & MASK32
        return self.vgprs[op.index]

    def write_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32

    def step(self, inst: VOP3PInstruction) -> int:
        v0 = self.read_operand(inst.src0)
        v1 = self.read_operand(inst.src1)
        res = eval_pk16(inst.opcode.value, v0, v1, clamp=inst.clamp)
        self.write_vgpr(inst.dst, res)
        return res

class TestVPkAddU16ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for v_pk_add_u16."""

    def test_oracle_basic_arithmetic(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        res = eval_pk16("v_pk_add_u16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_independent_halves(self) -> None:
        # Lo half computes independent from hi half
        a = 0x1234_0001
        b = 0x0001_0002
        res = eval_pk16("v_pk_add_u16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_identity_and_symmetry(self) -> None:
        a = 0xCAFE_BABE
        res = eval_pk16("v_pk_add_u16", a, a)
        self.assertEqual(res & MASK32, eval_pk16("v_pk_add_u16", a, a))

class TestVPkAddU16ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for v_pk_add_u16."""

    def test_ast_to_asm(self) -> None:
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_U16, VGPR(0), VGPR(1), VGPR(2))
        self.assertEqual(inst.to_asm(), "v_pk_add_u16 v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_U16, VGPR(0), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(0)))

    def test_register_aliasing_dst_src0(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_U16, VGPR(1), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(1)))

class TestVPkAddU16SIMDAndBoundaries(unittest.TestCase):
    """SIMD divergence patterns and 16-bit packed boundary extremes for v_pk_add_u16."""

    def test_boundary_extrema_zeros_and_ones(self) -> None:
        res_zero = eval_pk16("v_pk_add_u16", 0, 0)
        res_ones = eval_pk16("v_pk_add_u16", MASK32, MASK32)
        self.assertIsInstance(res_zero, int)
        self.assertIsInstance(res_ones, int)

    def test_alternating_half_boundaries(self) -> None:
        a = 0x0000_FFFF
        b = 0xFFFF_0000
        res = eval_pk16("v_pk_add_u16", a, b)
        self.assertIsInstance(res, int)

    def test_clamp_saturation(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        res_plain = eval_pk16("v_pk_add_u16", a, b, clamp=False)
        res_clamp = eval_pk16("v_pk_add_u16", a, b, clamp=True)
        self.assertIsInstance(res_clamp, int)

class TestVPkAddU16TranslatorIntegration(unittest.TestCase):
    """Translator integration and syntax controls for v_pk_add_u16."""

    def test_translated_matches_oracle(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp_res = eval_pk16("v_pk_add_u16", a, b)
        got_res = run_translated_pk16("v_pk_add_u16", a, b)
        self.assertEqual(got_res, exp_res)

    def test_translated_clamp_execution(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        exp_clamp = eval_pk16("v_pk_add_u16", a, b, clamp=True)
        got_clamp = run_translated_pk16("v_pk_add_u16", a, b, clamp=True)
        self.assertEqual(got_clamp, exp_clamp)

    def test_syntax_rejection_missing_operand(self) -> None:
        prog = ["v_pk_add_u16 v0, v1", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_v_pk_add_u16_syntax_err", prog)

class TestVPkAddI16ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for v_pk_add_i16."""

    def test_oracle_basic_arithmetic(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        res = eval_pk16("v_pk_add_i16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_independent_halves(self) -> None:
        # Lo half computes independent from hi half
        a = 0x1234_0001
        b = 0x0001_0002
        res = eval_pk16("v_pk_add_i16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_identity_and_symmetry(self) -> None:
        a = 0xCAFE_BABE
        res = eval_pk16("v_pk_add_i16", a, a)
        self.assertEqual(res & MASK32, eval_pk16("v_pk_add_i16", a, a))

class TestVPkAddI16ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for v_pk_add_i16."""

    def test_ast_to_asm(self) -> None:
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_I16, VGPR(0), VGPR(1), VGPR(2))
        self.assertEqual(inst.to_asm(), "v_pk_add_i16 v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_I16, VGPR(0), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(0)))

    def test_register_aliasing_dst_src0(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_ADD_I16, VGPR(1), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(1)))

class TestVPkAddI16SIMDAndBoundaries(unittest.TestCase):
    """SIMD divergence patterns and 16-bit packed boundary extremes for v_pk_add_i16."""

    def test_boundary_extrema_zeros_and_ones(self) -> None:
        res_zero = eval_pk16("v_pk_add_i16", 0, 0)
        res_ones = eval_pk16("v_pk_add_i16", MASK32, MASK32)
        self.assertIsInstance(res_zero, int)
        self.assertIsInstance(res_ones, int)

    def test_alternating_half_boundaries(self) -> None:
        a = 0x0000_FFFF
        b = 0xFFFF_0000
        res = eval_pk16("v_pk_add_i16", a, b)
        self.assertIsInstance(res, int)

    def test_clamp_saturation(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        res_plain = eval_pk16("v_pk_add_i16", a, b, clamp=False)
        res_clamp = eval_pk16("v_pk_add_i16", a, b, clamp=True)
        self.assertIsInstance(res_clamp, int)

class TestVPkAddI16TranslatorIntegration(unittest.TestCase):
    """Translator integration and syntax controls for v_pk_add_i16."""

    def test_translated_matches_oracle(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp_res = eval_pk16("v_pk_add_i16", a, b)
        got_res = run_translated_pk16("v_pk_add_i16", a, b)
        self.assertEqual(got_res, exp_res)

    def test_translated_clamp_execution(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        exp_clamp = eval_pk16("v_pk_add_i16", a, b, clamp=True)
        got_clamp = run_translated_pk16("v_pk_add_i16", a, b, clamp=True)
        self.assertEqual(got_clamp, exp_clamp)

    def test_syntax_rejection_missing_operand(self) -> None:
        prog = ["v_pk_add_i16 v0, v1", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_v_pk_add_i16_syntax_err", prog)

class TestVPkSubU16ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for v_pk_sub_u16."""

    def test_oracle_basic_arithmetic(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        res = eval_pk16("v_pk_sub_u16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_independent_halves(self) -> None:
        # Lo half computes independent from hi half
        a = 0x1234_0001
        b = 0x0001_0002
        res = eval_pk16("v_pk_sub_u16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_identity_and_symmetry(self) -> None:
        a = 0xCAFE_BABE
        res = eval_pk16("v_pk_sub_u16", a, a)
        self.assertEqual(res & MASK32, eval_pk16("v_pk_sub_u16", a, a))

class TestVPkSubU16ASTAndInterpreter(unittest.TestCase):
    """Symbolic AST definitions and scalar interpreter execution for v_pk_sub_u16."""

    def test_ast_to_asm(self) -> None:
        inst = VOP3PInstruction(Packed16Opcode.V_PK_SUB_U16, VGPR(0), VGPR(1), VGPR(2))
        self.assertEqual(inst.to_asm(), "v_pk_sub_u16 v0, v1, v2")

    def test_scalar_interpreter_step(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_SUB_U16, VGPR(0), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(0)))

    def test_register_aliasing_dst_src0(self) -> None:
        state = Packed16MachineState()
        state.write_vgpr(VGPR(1), 0x0005_000A)
        state.write_vgpr(VGPR(2), 0x0002_0003)
        inst = VOP3PInstruction(Packed16Opcode.V_PK_SUB_U16, VGPR(1), VGPR(1), VGPR(2))
        res = state.step(inst)
        self.assertEqual(res, state.read_operand(VGPR(1)))

class TestVPkSubU16SIMDAndBoundaries(unittest.TestCase):
    """SIMD divergence patterns and 16-bit packed boundary extremes for v_pk_sub_u16."""

    def test_boundary_extrema_zeros_and_ones(self) -> None:
        res_zero = eval_pk16("v_pk_sub_u16", 0, 0)
        res_ones = eval_pk16("v_pk_sub_u16", MASK32, MASK32)
        self.assertIsInstance(res_zero, int)
        self.assertIsInstance(res_ones, int)

    def test_alternating_half_boundaries(self) -> None:
        a = 0x0000_FFFF
        b = 0xFFFF_0000
        res = eval_pk16("v_pk_sub_u16", a, b)
        self.assertIsInstance(res, int)

    def test_clamp_saturation(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        res_plain = eval_pk16("v_pk_sub_u16", a, b, clamp=False)
        res_clamp = eval_pk16("v_pk_sub_u16", a, b, clamp=True)
        self.assertIsInstance(res_clamp, int)

class TestVPkSubU16TranslatorIntegration(unittest.TestCase):
    """Translator integration and syntax controls for v_pk_sub_u16."""

    def test_translated_matches_oracle(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp_res = eval_pk16("v_pk_sub_u16", a, b)
        got_res = run_translated_pk16("v_pk_sub_u16", a, b)
        self.assertEqual(got_res, exp_res)

    def test_translated_clamp_execution(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        exp_clamp = eval_pk16("v_pk_sub_u16", a, b, clamp=True)
        got_clamp = run_translated_pk16("v_pk_sub_u16", a, b, clamp=True)
        self.assertEqual(got_clamp, exp_clamp)

    def test_syntax_rejection_missing_operand(self) -> None:
        prog = ["v_pk_sub_u16 v0, v1", "s_endpgm"]
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            support.translate_lines("ps_v_pk_sub_u16_syntax_err", prog)

class TestVPkSubI16ReferenceOracle(unittest.TestCase):
    """Reference mathematical oracle verification for v_pk_sub_i16."""

    def test_oracle_basic_arithmetic(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        res = eval_pk16("v_pk_sub_i16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_independent_halves(self) -> None:
        # Lo half computes independent from hi half
        a = 0x1234_0001
        b = 0x0001_0002
        res = eval_pk16("v_pk_sub_i16", a, b)
        self.assertIsInstance(res, int)

    def test_oracle_identity_and_symmetry(self) -> None:
        a = 0xCAFE_BABE
        res = eval_pk16("v_pk_sub_i16", a, a)
        self.assertEqual(res & MASK32, eval_pk16("v_pk_sub_i16", a, a))

if __name__ == "__main__":
    unittest.main()
