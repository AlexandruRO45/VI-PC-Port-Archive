from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_msad_u32.

AMD RDNA VALU Instruction Set Architecture:
  v_msad_u32 dst, src0, src1, src2

Operation:
  Performs 32-bit masked sum of absolute difference with 32-bit unsigned accumulation:
    diff = 0 if src1 == 0 else |src0.u32 - src1.u32|
    dst = (src2.u32 + diff) mod 2^32

Mathematical and Architectural Properties:
  - Masking semantics:
    - Operation is masked out (omitted) if and only if reference word src1 == 0.
    - If src1 == 0, diff is strictly 0 regardless of src0 (even if src0 == 0xFFFFFFFF).
    - If src1 != 0, diff is the unsigned absolute difference |src0 - src1| in [0, 2^32 - 1].
  - Asymmetry / Non-commutativity:
    - Unlike unmasked v_sad_u32, v_msad_u32 is NOT commutative with respect to src0 and src1:
      the mask is dictated solely by src1.
  - Range and bounds:
    - Unmasked difference is in [0, 2^32 - 1] (0xFFFFFFFF).
    - Minimum diff is 0 (masked or equal operands).
  - Accumulator wrapping:
    - src2 is an unsigned 32-bit integer.
    - The addition (src2 + diff) wraps modulo 2^32 with standard unsigned arithmetic.
  - Clamp modifier:
    - In AMD RDNA2 ISA / Leonida translator contract, clamp modifier is unsupported
      and rejected rather than guessed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks and constants
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MAX_DIFF_U32: int = MASK32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def ref_abs_diff_u32(s0: int, s1: int) -> int:
    """Compute strictly unsigned 32-bit absolute difference |s0 - s1|."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def oracle_v_msad_u32(src0: int, src1: int, src2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u32.

    Computes masked absolute difference conditioned on src1 non-zero,
    and adds accumulator src2. Result wraps modulo 2^32.
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    acc = src2 & MASK32

    if u1 == 0:
        diff = 0
    else:
        diff = (u0 - u1) if u0 >= u1 else (u1 - u0)

    return (acc + diff) & MASK32


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class Reg32:
    """Represents a 32-bit architectural register (VGPR or SGPR)."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index {self.index} out of range [0, 255]")

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class Literal32:
    """Represents a 32-bit integer literal or inline constant."""
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= MASK32):
            raise ValueError(f"Literal32 value {self.value:#x} out of range [0, 0xFFFFFFFF]")

    @property
    def name(self) -> str:
        return f"{self.value:#010x}" if self.value > 9 else str(self.value)


@dataclass(frozen=True)
class VMsadU32Operand:
    """Typed expression operand representing a register or literal."""
    source: Union[Reg32, Literal32]

    def disassemble(self) -> str:
        return self.source.name


@dataclass(frozen=True)
class VMsadU32Instruction:
    """Symbolic AST node representing an RDNA2 v_msad_u32 VALU instruction."""
    dst: Reg32
    src0: VMsadU32Operand
    src1: VMsadU32Operand
    src2: VMsadU32Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")
        if self.clamp:
            raise ValueError("clamp modifier is unsupported on v_msad_u32")

    def disassemble(self) -> str:
        return f"v_msad_u32 {self.dst.name}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single execution thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        return self.vgprs[index] & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"VGPR index {index} out of range [0, 255]")
        self.vgprs[index] = val & MASK32

    def get_sgpr(self, index: int) -> int:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if not (0 <= index <= 255):
            raise IndexError(f"SGPR index {index} out of range [0, 255]")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_msad_u32."""

    def resolve_operand(self, op: VMsadU32Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, Reg32):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal32):
            return op.source.value & MASK32
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VMsadU32Instruction, state: ScalarMachineState) -> int:
        # Atomic operand read phase
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)

        # Computation phase via reference oracle
        result = oracle_v_msad_u32(s0, s1, s2)

        # Write-back phase
        state.set_vgpr(instr.dst.index, result)
        return result


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMsadU32ModelAndOracle(unittest.TestCase):
    """Unit tests for RDNA2 v_msad_u32 architectural bit model and reference oracle."""

    def test_masked_when_s1_is_zero(self) -> None:
        # When src1 is 0, masked out -> diff = 0 regardless of src0
        for s0 in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x80000000]:
            with self.subTest(s0=hex(s0)):
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0), 0)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 500), 500)
                self.assertEqual(oracle_v_msad_u32(s0, 0, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_unmasked_known_values(self) -> None:
        # s0 = 1000, s1 = 1500, s2 = 100
        # diff = |1000 - 1500| = 500
        # result = 100 + 500 = 600
        self.assertEqual(oracle_v_msad_u32(1000, 1500, 100), 600)
        self.assertEqual(oracle_v_msad_u32(2000, 1000, 0), 1000)

    def test_masking_asymmetry_non_commutativity(self) -> None:
        # s0 has 1000, s1 has 0 -> masked (result = 0)
        # s0 has 0, s1 has 1000 -> active (|0 - 1000| = 1000)
        res_s0_s1 = oracle_v_msad_u32(1000, 0, 0)
        res_s1_s0 = oracle_v_msad_u32(0, 1000, 0)

        self.assertEqual(res_s0_s1, 0)
        self.assertEqual(res_s1_s0, 1000)
        self.assertNotEqual(res_s0_s1, res_s1_s0)

    def test_maximum_delta_accumulation(self) -> None:
        # Max delta: |0 - 0xFFFFFFFF| = 0xFFFFFFFF with s1 = 0xFFFFFFFF!
        sad_max = oracle_v_msad_u32(0, 0xFFFFFFFF, 0)
        self.assertEqual(sad_max, MAX_DIFF_U32)
        self.assertEqual(sad_max, 0xFFFFFFFF)

    def test_accumulator_32bit_wraparound(self) -> None:
        # diff = 0xFFFFFFFF, acc = 1 -> wraps to 0
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 1), 0)
        # diff = 0xFFFFFFFF, acc = 2 -> wraps to 1
        self.assertEqual(oracle_v_msad_u32(0, 0xFFFFFFFF, 2), 1)

    def test_identity_when_equal_and_nonzero(self) -> None:
        # When src0 == src1 and src1 != 0, diff = 0, so result = src2
        val = 0x12345678
        self.assertEqual(oracle_v_msad_u32(val, val, 0xABCDEF01), 0xABCDEF01)
        self.assertEqual(oracle_v_msad_u32(0xFFFFFFFF, 0xFFFFFFFF, 42), 42)


class TestValuMsadU32SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_validation_and_naming(self) -> None:
        v0 = Reg32(0, RegType.VGPR)
        v255 = Reg32(255, RegType.VGPR)
        s12 = Reg32(12, RegType.SGPR)
        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s12.name, "s12")

    def test_register_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Reg32(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            Reg32(1000, RegType.SGPR)

    def test_literal_values_and_naming(self) -> None:
        lit_small = Literal32(7)
        lit_hex = Literal32(0x10203040)
        self.assertEqual(lit_small.name, "7")
        self.assertEqual(lit_hex.name, "0x10203040")

        with self.assertRaises(ValueError):
            Literal32(-1)
        with self.assertRaises(ValueError):
            Literal32(0x1_0000_0000)

    def test_instruction_ast_construction_and_disassembly(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src0 = VMsadU32Operand(Reg32(1, RegType.VGPR))
        src1 = VMsadU32Operand(Reg32(4, RegType.SGPR))
        src2 = VMsadU32Operand(Literal32(500))

        instr = VMsadU32Instruction(dst=dst, src0=src0, src1=src1, src2=src2)
        self.assertEqual(instr.disassemble(), "v_msad_u32 v0, v1, s4, 0x000001f4")

    def test_destination_must_be_vgpr(self) -> None:
        dst_sgpr = Reg32(0, RegType.SGPR)
        src = VMsadU32Operand(Reg32(1, RegType.VGPR))
        with self.assertRaises(TypeError):
            VMsadU32Instruction(dst=dst_sgpr, src0=src, src1=src, src2=src)

    def test_clamp_modifier_rejection_in_ast(self) -> None:
        dst = Reg32(0, RegType.VGPR)
        src = VMsadU32Operand(Reg32(1, RegType.VGPR))
        with self.assertRaises(ValueError):
            VMsadU32Instruction(dst=dst, src0=src, src1=src, src2=src, clamp=True)


class TestValuMsadU32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_state_get_set(self) -> None:
        self.assertEqual(self.state.get_vgpr(0), 0)
        self.assertEqual(self.state.get_sgpr(0), 0)

        self.state.set_vgpr(10, 0x12345678)
        self.assertEqual(self.state.get_vgpr(10), 0x12345678)

        self.state.set_sgpr(5, 0xABCDEF01)
        self.assertEqual(self.state.get_sgpr(5), 0xABCDEF01)

        # 32-bit truncation masking
        self.state.set_vgpr(1, 0x1_0000_0042)
        self.assertEqual(self.state.get_vgpr(1), 0x42)

        with self.assertRaises(IndexError):
            self.state.get_vgpr(-1)
        with self.assertRaises(IndexError):
            self.state.get_vgpr(256)
        with self.assertRaises(IndexError):
            self.state.get_sgpr(-1)
        with self.assertRaises(IndexError):
            self.state.get_sgpr(256)

    def test_scalar_step_vgpr_sources(self) -> None:
        # s0 = 1000, s1 = 0, s2 = 500
        # masked because s1 == 0 -> diff = 0 -> result = 500
        self.state.set_vgpr(1, 1000)
        self.state.set_vgpr(2, 0)
        self.state.set_vgpr(3, 500)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(3, RegType.VGPR)),
        )

        res = self.interp.step(instr, self.state)
        self.assertEqual(res, 500)
        self.assertEqual(self.state.get_vgpr(0), 500)

    def test_scalar_step_sgpr_and_literal_sources(self) -> None:
        self.state.set_vgpr(1, 1000)
        self.state.set_sgpr(4, 2500)

        instr = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(4, RegType.SGPR)),
            src2=VMsadU32Operand(Literal32(100)),
        )

        res = self.interp.step(instr, self.state)
        # 100 + |1000 - 2500| = 1600
        self.assertEqual(res, 1600)
        self.assertEqual(self.state.get_vgpr(0), 1600)

    def test_sequential_instruction_flow(self) -> None:
        # Step 1: v0 = v_msad_u32(v1, v2, 0)
        # Step 2: v3 = v_msad_u32(v1, v2, v0)  (accumulating with previous result)
        self.state.set_vgpr(1, 100)
        self.state.set_vgpr(2, 250)  # diff = 150

        instr1 = VMsadU32Instruction(
            dst=Reg32(0, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Literal32(0)),
        )
        res1 = self.interp.step(instr1, self.state)
        self.assertEqual(res1, 150)
        self.assertEqual(self.state.get_vgpr(0), 150)

        instr2 = VMsadU32Instruction(
            dst=Reg32(3, RegType.VGPR),
            src0=VMsadU32Operand(Reg32(1, RegType.VGPR)),
            src1=VMsadU32Operand(Reg32(2, RegType.VGPR)),
            src2=VMsadU32Operand(Reg32(0, RegType.VGPR)),
        )
        res2 = self.interp.step(instr2, self.state)
        # 150 + 150 = 300
        self.assertEqual(res2, 300)
        self.assertEqual(self.state.get_vgpr(3), 300)


if __name__ == "__main__":
    unittest.main()
