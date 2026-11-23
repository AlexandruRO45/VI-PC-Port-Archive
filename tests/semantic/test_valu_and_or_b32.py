"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_and_or_b32.

Models RDNA2 / GFX10 VALU 3-operand bitwise operation:
  v_and_or_b32 vd, src0, src1, src2

RDNA2 VALU Architecture & Semantics:
  1. Operands & Computation:
     - vd: 32-bit Destination VGPR
     - src0, src1, src2: 32-bit input sources (VGPR, SGPR, or literal/inline constant)
     - Operation: dst = (src0 & src1) | src2
  2. Condition Codes & Flags:
     - Neither SCC nor VCC is modified by this bitwise instruction.
  3. Execution Masking (EXEC):
     - Inactive lanes under EXEC mask do not write to their destination VGPR.
  4. Translator Integration:
     - Validates end-to-end SPIR-V emission via support.translate_lines and execution under spirv_mini_interp.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned integer as signed two's-complement."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_and_or_b32(src0: int, src1: int, src2: int) -> int:
    """Pure mathematical reference oracle for RDNA2 v_and_or_b32.

    Computes: (src0 & src1) | src2
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    s2 = src2 & MASK32
    return ((s0 & s1) | s2) & MASK32

# ---------------------------------------------------------------------------
# Symbolic AST Definitions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    SGPR = "s"
    VGPR = "v"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
    kind: RegKind
    index: int
    count: int = 1

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"


class Opcode(enum.Enum):
    V_AND_OR_B32 = "v_and_or_b32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a v_and_or_b32 instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | int
    src1: Register | int
    src2: Register | int

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr or self.dst.count != 1:
            raise ValueError(f"dst must be a single 32-bit VGPR: {self.dst}")
        if self.dst.index < 0 or self.dst.index > 255:
            raise ValueError(f"dst register index out of range [0, 255]: {self.dst.index}")

    def format_operand(self, op: Register | int) -> str:
        if isinstance(op, Register):
            return op.format_asm()
        if op < 0:
            return f"{s32(op)}"
        if op > 9:
            return f"{hex(op)}"
        return str(op)

    def to_assembly(self) -> str:
        d_str = self.dst.format_asm()
        s0_str = self.format_operand(self.src0)
        s1_str = self.format_operand(self.src1)
        s2_str = self.format_operand(self.src2)
        return f"{self.opcode.value} {d_str}, {s0_str}, {s1_str}, {s2_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestValuAndOrB32AST(unittest.TestCase):
    """Unit tests for symbolic AST nodes and assembly formatting."""

    def test_instruction_creation_defaults(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        self.assertEqual(inst.opcode, Opcode.V_AND_OR_B32)
        self.assertEqual(inst.to_assembly(), "v_and_or_b32 v0, v1, v2, v3")

    def test_instruction_with_sgpr_and_immediates(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 4),
            src0=Register(RegKind.SGPR, 0),
            src1=Register(RegKind.VGPR, 5),
            src2=0x1234,
        )
        self.assertEqual(inst.to_assembly(), "v_and_or_b32 v4, s0, v5, 0x1234")

    def test_program_representation(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=0xF0,
            src2=0x0F,
        ))
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.SGPR, 4),
            src2=0,
        ))
        lines = prog.to_assembly().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("v_and_or_b32 v0", lines[0])
        self.assertIn("v_and_or_b32 v2", lines[1])

# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Single-lane scalar interpreter for v_and_or_b32."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.scc: int = 0
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def resolve_operand(self, op: Register | int) -> int:
        if isinstance(op, Register):
            return self.read_reg_u32(op.kind, op.index)
        return op & MASK32

    def step(self, inst: Instruction) -> int:
        s0 = self.resolve_operand(inst.src0)
        s1 = self.resolve_operand(inst.src1)
        s2 = self.resolve_operand(inst.src2)

        res = ref_and_or_b32(s0, s1, s2)
        self.write_reg_u32(inst.dst.kind, inst.dst.index, res)
        self.trace.append(f"{inst.to_assembly()} => {res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        return [self.step(inst) for inst in prog.instructions]


class TestValuAndOrB32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()

    def test_scalar_execution_all_vgprs(self) -> None:
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0xFF00)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0xF0F0)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 0x000F)

        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        # (0xFF00 & 0xF0F0) | 0x000F = 0xF000 | 0x000F = 0xF00F
        res = self.interp.step(inst)
        self.assertEqual(res, 0xF00F)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 0), 0xF00F)

    def test_scalar_mixed_sgpr_and_immediate(self) -> None:
        self.interp.write_reg_u32(RegKind.SGPR, 4, 0xAAAA0000)
        self.interp.write_reg_u32(RegKind.VGPR, 5, 0xFFFF0000)

        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 6),
            src0=Register(RegKind.SGPR, 4),
            src1=Register(RegKind.VGPR, 5),
            src2=0x00005555,
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0xAAAA5555)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 6), 0xAAAA5555)

    def test_scalar_program_chaining(self) -> None:
        prog = Program()
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0xFF)
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=0x0F,
            src2=0x10,
        ))
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=0x1F,
            src2=0xA0,
        ))
        results = self.interp.run_program(prog)
        self.assertEqual(results, [0x1F, 0xBF])
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 2), 0xBF)

# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDVALUWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for v_and_or_b32."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def resolve_lane_operand(self, lane: int, op: Register | int) -> int:
        if isinstance(op, Register):
            return self.read_lane_reg_u32(lane, op.kind, op.index)
        return op & MASK32

    def step(self, inst: Instruction) -> list[int | None]:
        lane_inputs: list[tuple[int, int, int]] = []
        for lane in range(self.wave_size):
            s0 = self.resolve_lane_operand(lane, inst.src0)
            s1 = self.resolve_lane_operand(lane, inst.src1)
            s2 = self.resolve_lane_operand(lane, inst.src2)
            lane_inputs.append((s0, s1, s2))

        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            s0, s1, s2 = lane_inputs[lane]
            res = ref_and_or_b32(s0, s1, s2)
            self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, res)
            lane_results.append(res)

        return lane_results


class TestValuAndOrB32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        simd = SIMDVALUWavefrontInterpreter(wave_size=32)
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 1 << lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0xFFFFFFFF)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 3, 0)

        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        results = simd.step(inst)

        for lane in range(32):
            self.assertEqual(results[lane], 1 << lane)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 1 << lane)

    def test_wave64_alternating_exec_mask(self) -> None:
        simd = SIMDVALUWavefrontInterpreter(wave_size=64)
        exec_mask = 0x5555555555555555  # Even lanes active
        simd.set_exec_mask(exec_mask)

        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x11111111)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 0xFFFFFFFF)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0xAAAAAAAA)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 3, 0x05050505)

        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        results = simd.step(inst)

        expected_val = (0xFFFFFFFF & 0xAAAAAAAA) | 0x05050505  # 0xAF05AF05
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(results[lane], expected_val)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected_val)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane must retain initial destination value
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0x11111111)

    def test_all_inactive_lanes(self) -> None:
        simd = SIMDVALUWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x77777777)

        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=0xFFFFFFFF,
            src1=0xFFFFFFFF,
            src2=0xFFFFFFFF,
        )
        results = simd.step(inst)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), 0x77777777)

class TestValuAndOrB32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and bitwise truth tables."""

    def test_basic_and_or_truth_table(self) -> None:
        # 4-bit exhaustive truth table patterns replicated across dwords
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    src0 = 0xFFFFFFFF if a else 0
                    src1 = 0xFFFFFFFF if b else 0
                    src2 = 0xFFFFFFFF if c else 0
                    expected = 0xFFFFFFFF if ((a and b) or c) else 0
                    self.assertEqual(ref_and_or_b32(src0, src1, src2), expected)

    def test_annihilators_and_identities(self) -> None:
        # src2 = 0xFFFFFFFF -> result always 0xFFFFFFFF
        self.assertEqual(ref_and_or_b32(0, 0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_and_or_b32(0x12345678, 0x9ABCDEF0, 0xFFFFFFFF), 0xFFFFFFFF)

        # src2 = 0 -> result is src0 & src1
        self.assertEqual(ref_and_or_b32(0x0F0F0F0F, 0x33333333, 0), 0x03030303)

        # src0 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0, 0xFFFFFFFF, 0xCAFEBABE), 0xCAFEBABE)

        # src1 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0xFFFFFFFF, 0, 0xDEADBEEF), 0xDEADBEEF)

    def test_commutativity_of_and_operands(self) -> None:
        val_a = 0xA5A5A5A5
        val_b = 0x5A5A5A5A
        val_c = 0x12345678
        self.assertEqual(
            ref_and_or_b32(val_a, val_b, val_c),
            ref_and_or_b32(val_b, val_a, val_c),
        )

    def test_bitfield_manipulation_masks(self) -> None:
        # Clear field (AND mask) then set field (OR mask)
        orig = 0x12345678
        clear_mask = 0xFFFF00FF
        insert_val = 0x0000AB00
        res = ref_and_or_b32(orig, clear_mask, insert_val)
        self.assertEqual(res, 0x1234AB78)

if __name__ == "__main__":
    unittest.main()
