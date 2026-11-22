from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_src2_f32.

RDNA2 / GFX10 Local Data Share (LDS) Dual-Source Atomic Float Maximum:
Performs an in-place atomic single-precision IEEE 754 floating-point maximum operation
between two LDS memory locations (A and B) addressed by a common base VGPR and two
independent 8-bit dword offsets (offset0 and offset1).

Syntax:
    ds_max_src2_f32 vaddr [offset0:offset0] [offset1:offset1] [gds]

Mathematical Specification:
    Address A = (vaddr + offset0 * 4) & 0xFFFFFFFF
    Address B = (vaddr + offset1 * 4) & 0xFFFFFFFF
    dword_index_a = (Address A >> 2) & (lds_dwords - 1)
    dword_index_b = (Address B >> 2) & (lds_dwords - 1)
    val_a = LDS[dword_index_a]
    val_b = LDS[dword_index_b]
    new_a = oracle_max_f32(val_a, val_b)
    LDS[dword_index_a] = new_a
    # LDS[dword_index_b] is unchanged unless dword_index_a == dword_index_b
    # Non-returning instruction: VGPRs are not modified.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

# Architectural Bitmasks and Constants
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default capacity in 32-bit dwords
DEFAULT_NUM_LANES: int = 64     # Wave64 default width
DEFAULT_NUM_VGPRS: int = 256    # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106    # Standard GFX10 scalar register file size
NUM_LDS_BANKS: int = 32         # 32 memory banks

SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000
CANONICAL_QNAN: int = 0x7FC00000

# Canonical IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
NEG_MAX_NORMAL_F32: int = 0xFF7FFFFF
NEG_MIN_NORMAL_F32: int = 0x80800000
NEG_MIN_SUBNORMAL_F32: int = 0x80000001
NEG_MAX_SUBNORMAL_F32: int = 0x807FFFFF

ONE_F32: int = 0x3F800000       # 1.0f
NEG_ONE_F32: int = 0xBF800000   # -1.0f
TWO_F32: int = 0x40000000       # 2.0f
NEG_TWO_F32: int = 0xC0000000   # -2.0f
HALF_F32: int = 0x3F000000      # 0.5f
NEG_HALF_F32: int = 0xBF000000  # -0.5f
TEN_F32: int = 0x41200000       # 10.0f
NEG_TEN_F32: int = 0xC1200000   # -10.0f


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_f32_qnan(bits: int) -> bool:
    """Test if 32-bit bit pattern is a Quiet NaN."""
    return is_f32_nan(bits) and bool(bits & QNAN_BIT)


def is_f32_snan(bits: int) -> bool:
    """Test if 32-bit bit pattern is a Signaling NaN."""
    return is_f32_nan(bits) and not bool(bits & QNAN_BIT)


def is_f32_inf(bits: int) -> bool:
    """Test if 32-bit bit pattern is positive or negative infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is a subnormal float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_f32_zero(bits: int) -> bool:
    """Test if 32-bit bit pattern is positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for address A and address B.

    Address A = (vaddr + offset0 * 4) & MASK32
    Address B = (vaddr + offset1 * 4) & MASK32
    Dword index = (addr >> 2) & (lds_dwords - 1)

    Returns:
        (byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b)
    """
    byte_addr_a = u32(vaddr + (offset0 * 4))
    byte_addr_b = u32(vaddr + (offset1 * 4))
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


def calc_lds_bank(dword_idx: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for a dword index."""
    return dword_idx % NUM_LDS_BANKS


def oracle_max_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_src2_f32 atomic operation.

    Follows IEEE 754-2008 / IEEE 754-2019 maxNum:
      - NaN Suppression: If one operand is NaN, returns the valid numerical operand.
      - If both operands are NaN, returns a quiet NaN (first NaN with quiet bit set).
      - Signed zero ordering: -0.0 < +0.0:
        max(+0.0, -0.0) -> +0.0, max(-0.0, +0.0) -> +0.0, max(-0.0, -0.0) -> -0.0.
      - Subnormals and infinities preserved without flushing to zero.
    """
    a = a_bits & MASK32
    b = b_bits & MASK32

    a_is_nan = is_f32_nan(a)
    b_is_nan = is_f32_nan(b)

    if a_is_nan and b_is_nan:
        return a | QNAN_BIT
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    if is_f32_zero(a) and is_f32_zero(b):
        return POS_ZERO_F32 if ((a == POS_ZERO_F32) or (b == POS_ZERO_F32)) else NEG_ZERO_F32

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Equal magnitude: positive sign bit (0) > negative sign bit (1)
        if (a & SIGN_MASK) == 0:
            return a
        if (b & SIGN_MASK) == 0:
            return b
        return a


def ref_ds_max_src2_f32(val_a: int, val_b: int) -> int:
    """Reference calculation for ds_max_src2_f32: LDS[A] = max(LDS[A], LDS[B])."""
    return oracle_max_f32(val_a, val_b)


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_dwords}")
        self.size_dwords: int = size_dwords
        self.data: List[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)

    def read_by_byte_addr(self, byte_addr: int) -> int:
        idx = (byte_addr >> 2) & (self.size_dwords - 1)
        return self.read_dword(idx)

    def write_by_byte_addr(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr >> 2) & (self.size_dwords - 1)
        self.write_dword(idx, val)

    def atomic_max_src2_f32(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> int:
        """Execute ds_max_src2_f32 on LDS: LDS[A] = max(LDS[A], LDS[B]), returning new A."""
        _, _, idx_a, idx_b = calc_lds_src2_addresses(vaddr, offset0, offset1, self.size_dwords)
        val_a = self.read_dword(idx_a)
        val_b = self.read_dword(idx_b)
        new_a = oracle_max_f32(val_a, val_b)
        self.write_dword(idx_a, new_a)
        return new_a


class TestReferenceModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS model."""

    def test_oracle_standard_floats(self) -> None:
        self.assertEqual(oracle_max_f32(ONE_F32, TWO_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(TWO_F32, ONE_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(NEG_TEN_F32, ONE_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(NEG_ONE_F32, NEG_TWO_F32), NEG_ONE_F32)

    def test_oracle_signed_zeros(self) -> None:
        self.assertEqual(oracle_max_f32(POS_ZERO_F32, NEG_ZERO_F32), POS_ZERO_F32)
        self.assertEqual(oracle_max_f32(NEG_ZERO_F32, POS_ZERO_F32), POS_ZERO_F32)
        self.assertEqual(oracle_max_f32(NEG_ZERO_F32, NEG_ZERO_F32), NEG_ZERO_F32)

    def test_oracle_infinities(self) -> None:
        self.assertEqual(oracle_max_f32(MAX_NORMAL_F32, POS_INF_F32), POS_INF_F32)
        self.assertEqual(oracle_max_f32(NEG_INF_F32, NEG_MAX_NORMAL_F32), NEG_MAX_NORMAL_F32)
        self.assertEqual(oracle_max_f32(POS_INF_F32, NEG_INF_F32), POS_INF_F32)

    def test_oracle_nan_suppression(self) -> None:
        self.assertEqual(oracle_max_f32(ONE_F32, QNAN_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(QNAN_F32, ONE_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(NEG_TEN_F32, SNAN_F32), NEG_TEN_F32)
        both_nan = oracle_max_f32(SNAN_F32, QNAN_F32)
        self.assertTrue(is_f32_qnan(both_nan))

    def test_calc_lds_src2_addresses(self) -> None:
        b_a, b_b, idx_a, idx_b = calc_lds_src2_addresses(0x100, offset0=2, offset1=5)
        self.assertEqual(b_a, 0x100 + 8)
        self.assertEqual(b_b, 0x100 + 20)
        self.assertEqual(idx_a, (0x100 + 8) >> 2)
        self.assertEqual(idx_b, (0x100 + 20) >> 2)

    def test_lds_memory_atomic_max_src2_execution(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_DWORDS)
        # Location A (dword 10): 1.0f, Location B (dword 20): 2.0f
        lds.write_dword(10, ONE_F32)
        lds.write_dword(20, TWO_F32)

        # base = dword 10 * 4 = 40, offset0 = 0 -> A = dword 10, offset1 = 10 -> B = dword 20
        new_a = lds.atomic_max_src2_f32(40, offset0=0, offset1=10)
        self.assertEqual(new_a, TWO_F32)
        # Location A updated to 2.0f
        self.assertEqual(lds.read_dword(10), TWO_F32)
        # Location B remains 2.0f
        self.assertEqual(lds.read_dword(20), TWO_F32)


# ==============================================================================
# Symbolic AST Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Supported DS src2 family opcodes."""
    DS_MAX_SRC2_F32 = "ds_max_src2_f32"


class RegType(Enum):
    """Register file class."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index < DEFAULT_NUM_VGPRS):
                raise ValueError(f"VGPR index out of bounds: v{self.index}")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index < DEFAULT_NUM_SGPRS):
                raise ValueError(f"SGPR index out of bounds: s{self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"

    def __str__(self) -> str:
        return self.name


def vgpr(index: int) -> RegisterOperand:
    return RegisterOperand(index=index, reg_type=RegType.VGPR)


def sgpr(index: int) -> RegisterOperand:
    return RegisterOperand(index=index, reg_type=RegType.SGPR)


@dataclass(frozen=True)
class LDSAddressExpression:
    """Symbolic representation of LDS address with dual offset descriptors."""
    base_reg: RegisterOperand
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset0 <= 0xFF):
            raise ValueError(f"offset0 out of 8-bit range: {self.offset0}")
        if not (0 <= self.offset1 <= 0xFF):
            raise ValueError(f"offset1 out of 8-bit range: {self.offset1}")
        if self.base_reg.reg_type != RegType.VGPR:
            raise TypeError("LDS base address register must be a VGPR")


@dataclass(frozen=True)
class DSSrc2Instruction:
    """Symbolic AST node for DS src2 atomic instruction."""
    opcode: DSOpcode
    addr: LDSAddressExpression

    def disasm(self) -> str:
        """Format canonical assembly mnemonic and operand string."""
        parts = [self.opcode.value, self.addr.base_reg.name]
        if self.addr.offset0 != 0:
            parts.append(f"offset0:{self.addr.offset0}")
        if self.addr.offset1 != 0:
            parts.append(f"offset1:{self.addr.offset1}")
        if self.addr.gds:
            parts.append("gds")
        return " ".join(parts)


@dataclass
class Program:
    """Sequence of symbolic instructions."""
    instructions: List[DSSrc2Instruction] = field(default_factory=list)

    def add(self, instr: DSSrc2Instruction) -> None:
        self.instructions.append(instr)

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        addr = LDSAddressExpression(base_reg=vgpr(2), offset0=4, offset1=8, gds=False)
        instr = DSSrc2Instruction(opcode=DSOpcode.DS_MAX_SRC2_F32, addr=addr)
        self.assertEqual(instr.opcode, DSOpcode.DS_MAX_SRC2_F32)
        self.assertEqual(instr.addr.base_reg, vgpr(2))
        self.assertEqual(instr.addr.offset0, 4)
        self.assertEqual(instr.addr.offset1, 8)
        self.assertFalse(instr.addr.gds)

    def test_ast_disassembly_formatting(self) -> None:
        addr1 = LDSAddressExpression(base_reg=vgpr(0), offset0=2, offset1=4)
        instr1 = DSSrc2Instruction(opcode=DSOpcode.DS_MAX_SRC2_F32, addr=addr1)
        self.assertEqual(instr1.disasm(), "ds_max_src2_f32 v0 offset0:2 offset1:4")

        addr0 = LDSAddressExpression(base_reg=vgpr(5), offset0=0, offset1=0)
        instr0 = DSSrc2Instruction(opcode=DSOpcode.DS_MAX_SRC2_F32, addr=addr0)
        self.assertEqual(instr0.disasm(), "ds_max_src2_f32 v5")

        addr_gds = LDSAddressExpression(base_reg=vgpr(1), offset0=1, offset1=0, gds=True)
        instr_gds = DSSrc2Instruction(opcode=DSOpcode.DS_MAX_SRC2_F32, addr=addr_gds)
        self.assertEqual(instr_gds.disasm(), "ds_max_src2_f32 v1 offset0:1 gds")

    def test_ast_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            vgpr(-1)
        with self.assertRaises(ValueError):
            vgpr(256)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=vgpr(0), offset0=-1)
        with self.assertRaises(ValueError):
            LDSAddressExpression(base_reg=vgpr(0), offset0=256)
        with self.assertRaises(TypeError):
            LDSAddressExpression(base_reg=sgpr(0))

    def test_ast_program_container(self) -> None:
        prog = Program()
        i1 = DSSrc2Instruction(DSOpcode.DS_MAX_SRC2_F32, LDSAddressExpression(vgpr(0)))
        i2 = DSSrc2Instruction(DSOpcode.DS_MAX_SRC2_F32, LDSAddressExpression(vgpr(1), offset0=4))
        prog.add(i1)
        prog.add(i2)
        self.assertEqual(len(prog), 2)
        self.assertEqual(list(prog), [i1, i2])


# ==============================================================================
# Deterministic Scalar Interpreter & Machine State
# ==============================================================================

class ScalarMachineState:
    """Architectural register and memory state for a single scalar GPU thread."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.instruction_count: int = 0
        self.cycle_count: int = 0

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for DS src2 opcodes."""

    def execute(self, instr: DSSrc2Instruction, state: ScalarMachineState) -> int:
        base_addr = state.get_vgpr(instr.addr.base_reg.index)
        _, _, idx_a, idx_b = calc_lds_src2_addresses(
            vaddr=base_addr,
            offset0=instr.addr.offset0,
            offset1=instr.addr.offset1,
            lds_dwords=state.lds.size_dwords,
        )

        val_a = state.lds.read_dword(idx_a)
        val_b = state.lds.read_dword(idx_b)

        if instr.opcode == DSOpcode.DS_MAX_SRC2_F32:
            new_a = oracle_max_f32(val_a, val_b)
        else:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        # In-place atomic update to LDS location A
        state.lds.write_dword(idx_a, new_a)
        state.instruction_count += 1
        state.cycle_count += 4
        return new_a

    def run_program(self, prog: Program, state: ScalarMachineState) -> int:
        count = 0
        for instr in prog:
            self.execute(instr, state)
            count += 1
        return count


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_DWORDS)
        self.state = ScalarMachineState(lds=self.lds)
        self.interp = ScalarInterpreter()

    def test_scalar_execution_basic(self) -> None:
        # Location A = dword 10 (1.0f), Location B = dword 20 (2.0f)
        self.lds.write_dword(10, ONE_F32)
        self.lds.write_dword(20, TWO_F32)
        self.state.set_vgpr(1, 40)  # base = 40 (dword 10)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_MAX_SRC2_F32,
            addr=LDSAddressExpression(base_reg=vgpr(1), offset0=0, offset1=10),
        )
        new_a = self.interp.execute(instr, self.state)
        self.assertEqual(new_a, TWO_F32)
        self.assertEqual(self.lds.read_dword(10), TWO_F32)
        self.assertEqual(self.lds.read_dword(20), TWO_F32)
        # Base register must not be modified
        self.assertEqual(self.state.get_vgpr(1), 40)

    def test_scalar_execution_same_location(self) -> None:
        # offset0 == offset1: A and B are the same location
        self.lds.write_dword(15, TEN_F32)
        self.state.set_vgpr(2, 15 * 4)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_MAX_SRC2_F32,
            addr=LDSAddressExpression(base_reg=vgpr(2), offset0=0, offset1=0),
        )
        new_a = self.interp.execute(instr, self.state)
        self.assertEqual(new_a, TEN_F32)
        self.assertEqual(self.lds.read_dword(15), TEN_F32)

    def test_scalar_program_execution(self) -> None:
        prog = Program()
        self.state.set_vgpr(0, 0)
        # Initialize 3 pairs: (0, 1), (2, 3), (4, 5)
        pairs = [(ONE_F32, TWO_F32), (NEG_TEN_F32, NEG_ONE_F32), (HALF_F32, ONE_F32)]
        for i, (va, vb) in enumerate(pairs):
            idx_a = i * 2
            idx_b = i * 2 + 1
            self.lds.write_dword(idx_a, va)
            self.lds.write_dword(idx_b, vb)
            prog.add(DSSrc2Instruction(
                opcode=DSOpcode.DS_MAX_SRC2_F32,
                addr=LDSAddressExpression(base_reg=vgpr(0), offset0=idx_a, offset1=idx_b),
            ))

        executed = self.interp.run_program(prog, self.state)
        self.assertEqual(executed, 3)
        self.assertEqual(self.lds.read_dword(0), TWO_F32)
        self.assertEqual(self.lds.read_dword(2), NEG_ONE_F32)
        self.assertEqual(self.lds.read_dword(4), ONE_F32)


# ==============================================================================
# SIMD Multi-Lane State and Divergence Interpreter
# ==============================================================================

class SIMDState:
    """SIMD vector machine state with multi-lane VGPR files and shared LDS."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK64,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = (exec_mask & MASK32) if num_lanes == 32 else (exec_mask & MASK64)
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.instruction_count: int = 0
        self.cycle_count: int = 0

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def get_lane_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

    def set_lane_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32


class SIMDInterpreter:
    """SIMD multi-lane execution interpreter handling divergence and EXEC masking."""

    def execute(self, instr: DSSrc2Instruction, state: SIMDState) -> List[Tuple[int, int]]:
        results: List[Tuple[int, int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            base_addr = state.get_lane_vgpr(lane, instr.addr.base_reg.index)
            _, _, idx_a, idx_b = calc_lds_src2_addresses(
                vaddr=base_addr,
                offset0=instr.addr.offset0,
                offset1=instr.addr.offset1,
                lds_dwords=state.lds.size_dwords,
            )

            val_a = state.lds.read_dword(idx_a)
            val_b = state.lds.read_dword(idx_b)

            if instr.opcode == DSOpcode.DS_MAX_SRC2_F32:
                new_a = oracle_max_f32(val_a, val_b)
            else:
                raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

            state.lds.write_dword(idx_a, new_a)
            results.append((lane, new_a))

        state.instruction_count += 1
        state.cycle_count += 4
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_DWORDS)

    def test_simd_wave32_independent_lanes(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        interp = SIMDInterpreter()

        # Each lane i operates on pair (i * 2, i * 2 + 1)
        for lane in range(32):
            idx_a = lane * 2
            idx_b = lane * 2 + 1
            self.lds.write_dword(idx_a, f32_to_bits(float(lane)))
            self.lds.write_dword(idx_b, f32_to_bits(float(lane + 50)))
            state.set_lane_vgpr(lane, 1, lane * 2 * 4)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_MAX_SRC2_F32,
            addr=LDSAddressExpression(base_reg=vgpr(1), offset0=0, offset1=1),
        )
        res = interp.execute(instr, state)
        self.assertEqual(len(res), 32)
        for lane in range(32):
            idx_a = lane * 2
            expected_new_a = f32_to_bits(float(lane + 50))
            self.assertEqual(self.lds.read_dword(idx_a), expected_new_a)

    def test_simd_partial_exec_masking(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000000F, lds=self.lds)  # only lanes 0..3 active
        interp = SIMDInterpreter()

        for lane in range(32):
            idx_a = lane * 2
            idx_b = lane * 2 + 1
            self.lds.write_dword(idx_a, ONE_F32)
            self.lds.write_dword(idx_b, TWO_F32)
            state.set_lane_vgpr(lane, 1, lane * 2 * 4)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_MAX_SRC2_F32,
            addr=LDSAddressExpression(base_reg=vgpr(1), offset0=0, offset1=1),
        )
        res = interp.execute(instr, state)
        self.assertEqual(len(res), 4)

        for lane in range(32):
            idx_a = lane * 2
            if lane < 4:
                self.assertEqual(self.lds.read_dword(idx_a), TWO_F32)
            else:
                self.assertEqual(self.lds.read_dword(idx_a), ONE_F32)  # untouched!

    def test_simd_wave64_execution(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=MASK64, lds=self.lds)
        interp = SIMDInterpreter()

        for lane in range(64):
            idx_a = lane * 2
            idx_b = lane * 2 + 1
            self.lds.write_dword(idx_a, NEG_TEN_F32)
            self.lds.write_dword(idx_b, ONE_F32)
            state.set_lane_vgpr(lane, 0, lane * 2 * 4)

        instr = DSSrc2Instruction(
            opcode=DSOpcode.DS_MAX_SRC2_F32,
            addr=LDSAddressExpression(base_reg=vgpr(0), offset0=0, offset1=1),
        )
        res = interp.execute(instr, state)
        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(self.lds.read_dword(lane * 2), ONE_F32)

    def test_simd_collision_same_address_a(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x000000FF, lds=self.lds)  # 8 lanes active
        interp = SIMDInterpreter()

        # All 8 lanes target destination location A = dword 100
        # Each lane has a different source B
        self.lds.write_dword(100, POS_ZERO_F32)
        candidates = [1.0, 0.5, 3.0, 2.0, 5.0, 4.0, 10.0, 8.0]
        for lane in range(8):
            self.lds.write_dword(200 + lane, f32_to_bits(candidates[lane]))
            state.set_lane_vgpr(lane, 1, 100 * 4)

        for lane in range(8):
            instr = DSSrc2Instruction(
                opcode=DSOpcode.DS_MAX_SRC2_F32,
                addr=LDSAddressExpression(base_reg=vgpr(1), offset0=0, offset1=100 + lane),
            )
            interp.execute(instr, state)

        # After all operations, location A holds global maximum 10.0f
        self.assertEqual(self.lds.read_dword(100), f32_to_bits(10.0))


if __name__ == "__main__":
    unittest.main()
