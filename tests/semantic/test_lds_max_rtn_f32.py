from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_rtn_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Maximum with Return:
Performs a 32-bit atomic single-precision IEEE 754 floating-point maximum operation
on LDS storage and returns the previous value in memory to the destination VGPR.

Syntax:
    ds_max_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = oracle_max_f32(old_val, vdata0)
    LDS[dword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. IEEE 754-2008 / IEEE 754-2019 maxNum Semantics:
     - NaN Suppression: If one operand is NaN and the other is a valid float, the valid float
       is returned. If both operands are NaN, a quiet NaN is returned.
     - Signed Zero Ordering: -0.0 is strictly less than +0.0:
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0, max(-0.0, -0.0) = -0.0.
     - Infinities & Subnormals: -INF < all finite numbers < +INF.
       Subnormal single-precision values are preserved without flushing to zero.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-operation value from LDS[dword_index] and writes it
     into the destination VGPR (vdst).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but dword-aligned for 32-bit operations.
     The bottom 2 bits of the effective address are dropped (eff_addr >> 2).
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (16,384 dwords) organized into 32 memory banks.
     Bank index = dword_index % 32.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# IEEE 754 binary32 architectural bitmasks
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000
CANONICAL_QNAN: int = 0x7FC00000

# Canonical IEEE 754 binary32 bit patterns
POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
ONE_F32: int = 0x3F800000           # 1.0
NEG_ONE_F32: int = 0xBF800000       # -1.0
HALF_F32: int = 0x3F000000          # 0.5
NEG_HALF_F32: int = 0xBF000000      # -0.5
TWO_F32: int = 0x40000000           # 2.0
NEG_TWO_F32: int = 0xC0000000       # -2.0
TEN_F32: int = 0x41200000           # 10.0
NEG_TEN_F32: int = 0xC1200000       # -10.0

FLT_MIN: int = 0x00800000           # Smallest positive normal: 2^-126
NEG_FLT_MIN: int = 0x80800000       # -2^-126
FLT_MAX: int = 0x7F7FFFFF           # Largest finite float
NEG_FLT_MAX: int = 0xFF7FFFFF       # -FLT_MAX
POS_INF: int = 0x7F800000           # +infinity
NEG_INF: int = 0xFF800000           # -infinity
QNAN_PAYLOAD_A: int = 0x7FC10001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFC20002    # Quiet NaN with payload B
SNAN_PAYLOAD_A: int = 0x7F810001    # Signaling NaN
SNAN_PAYLOAD_B: int = 0xFF820002    # Signaling NaN
SUBNORM_MIN: int = 0x00000001       # Smallest positive subnormal: 2^-149
SUBNORM_MAX: int = 0x007FFFFF       # Largest positive subnormal
NEG_SUBNORM_MIN: int = 0x80000001   # Smallest negative subnormal
NEG_SUBNORM_MAX: int = 0x807FFFFF   # Largest negative subnormal

# Architectural LDS constants
DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32              # 32 memory banks


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def split_f32_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack 32-bit integer into (sign, exponent, fraction) components."""
    bits &= MASK32
    sign = (bits & SIGN_MASK) >> 31
    exponent = (bits & EXP_MASK) >> 23
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def is_nan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if 32-bit integer represents an IEEE 754 float Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if 32-bit integer represents IEEE 754 positive or negative infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if 32-bit integer represents a subnormal single-precision float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if 32-bit integer represents positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS with 16-bit offset wrap."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def oracle_max_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_f32 atomic operation.

    Follows IEEE 754-2008 / IEEE 754-2019 maxNum:
      - NaN Suppression: If one operand is NaN, returns the valid numerical operand.
      - If both operands are NaN, returns a quiet NaN (first NaN with quiet bit set).
      - Signed zero ordering: -0.0 < +0.0:
        max(+0.0, -0.0) -> +0.0, max(-0.0, +0.0) -> +0.0, max(-0.0, -0.0) -> -0.0.
      - Subnormals and infinities preserved without flushing.
    """
    a = a_bits & MASK32
    b = b_bits & MASK32

    a_is_nan = is_nan(a)
    b_is_nan = is_nan(b)

    if a_is_nan and b_is_nan:
        return a | QNAN_BIT
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros: -0.0 < +0.0
    if is_zero(a) and is_zero(b):
        return POS_ZERO if ((a == POS_ZERO) or (b == POS_ZERO)) else NEG_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Numerically equal: positive sign bit (0) is greater than negative sign bit (1)
        if (a & SIGN_MASK) == 0:
            return a
        if (b & SIGN_MASK) == 0:
            return b
        return a


def ref_ds_max_rtn_f32(current_mem: int, candidate: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_f32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = oracle_max_f32(old_val, candidate)

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    cand = u32(candidate)
    new_mem = oracle_max_f32(old_val, cand)
    return old_val, new_mem


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        eff_addr = calc_lds_effective_address(byte_addr, offset)
        return calc_lds_dword_index(eff_addr, self.num_dwords)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def read_by_byte_addr(self, byte_addr: int, offset: int = 0) -> int:
        """Read a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        return self.read_dword(idx)

    def write_by_byte_addr(self, byte_addr: int, val: int, offset: int = 0) -> None:
        """Write a 32-bit dword using effective byte address."""
        idx = self.get_dword_index(byte_addr, offset)
        self.write_dword(idx, val)

    def atomic_max_rtn_f32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic float maximum at (byte_addr + offset), returning previous value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_max_f32(old_val, val)
        self.dwords[idx] = new_val
        return old_val


class TestDSMaxRtnF32OracleAndModel(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS model."""

    def test_oracle_standard_floats(self) -> None:
        # Standard positive floats
        self.assertEqual(oracle_max_f32(ONE_F32, TWO_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(TWO_F32, ONE_F32), TWO_F32)
        self.assertEqual(oracle_max_f32(HALF_F32, ONE_F32), ONE_F32)
        # Negative vs Positive
        self.assertEqual(oracle_max_f32(NEG_TEN_F32, ONE_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(NEG_ONE_F32, NEG_TWO_F32), NEG_ONE_F32)

    def test_oracle_signed_zeros(self) -> None:
        # max(+0.0, -0.0) = +0.0
        self.assertEqual(oracle_max_f32(POS_ZERO, NEG_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(NEG_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_max_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_oracle_infinities(self) -> None:
        # +INF is strictly greater than all finite floats
        self.assertEqual(oracle_max_f32(FLT_MAX, POS_INF), POS_INF)
        self.assertEqual(oracle_max_f32(POS_INF, FLT_MAX), POS_INF)
        self.assertEqual(oracle_max_f32(NEG_INF, NEG_FLT_MAX), NEG_FLT_MAX)
        self.assertEqual(oracle_max_f32(POS_INF, NEG_INF), POS_INF)

    def test_oracle_nan_suppression(self) -> None:
        # Valid float vs NaN: valid float is chosen
        self.assertEqual(oracle_max_f32(ONE_F32, QNAN_PAYLOAD_A), ONE_F32)
        self.assertEqual(oracle_max_f32(QNAN_PAYLOAD_A, ONE_F32), ONE_F32)
        self.assertEqual(oracle_max_f32(NEG_TEN_F32, SNAN_PAYLOAD_A), NEG_TEN_F32)
        self.assertEqual(oracle_max_f32(SNAN_PAYLOAD_A, NEG_TEN_F32), NEG_TEN_F32)

        # Both NaN: returns a quieted NaN
        both_nan = oracle_max_f32(SNAN_PAYLOAD_A, QNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(both_nan))

    def test_ref_ds_max_rtn_f32(self) -> None:
        old_val, new_mem = ref_ds_max_rtn_f32(ONE_F32, TWO_F32)
        self.assertEqual(old_val, ONE_F32)
        self.assertEqual(new_mem, TWO_F32)

        # When candidate is smaller, memory remains unchanged
        old_val2, new_mem2 = ref_ds_max_rtn_f32(TWO_F32, ONE_F32)
        self.assertEqual(old_val2, TWO_F32)
        self.assertEqual(new_mem2, TWO_F32)

    def test_lds_memory_init_and_addressing(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.size_bytes, 65536)
        self.assertEqual(lds.num_dwords, 16384)

        # Dword alignment: low 2 bits dropped
        self.assertEqual(lds.get_dword_index(0x0000), 0)
        self.assertEqual(lds.get_dword_index(0x0001), 0)
        self.assertEqual(lds.get_dword_index(0x0002), 0)
        self.assertEqual(lds.get_dword_index(0x0003), 0)
        self.assertEqual(lds.get_dword_index(0x0004), 1)

        # Offset addition
        self.assertEqual(lds.get_dword_index(0x0000, offset=4), 1)
        self.assertEqual(lds.get_dword_index(0x0004, offset=8), 3)

        # Bank calculation
        self.assertEqual(lds.get_bank(0), 0)
        self.assertEqual(lds.get_bank(31), 31)
        self.assertEqual(lds.get_bank(32), 0)

    def test_lds_atomic_max_rtn_f32_execution(self) -> None:
        lds = LDSMemory(65536)
        lds.write_by_byte_addr(0x40, ONE_F32)

        # Atomic max with larger candidate
        old = lds.atomic_max_rtn_f32(0x40, TWO_F32)
        self.assertEqual(old, ONE_F32)
        self.assertEqual(lds.read_by_byte_addr(0x40), TWO_F32)

        # Atomic max with smaller candidate
        old2 = lds.atomic_max_rtn_f32(0x40, HALF_F32)
        self.assertEqual(old2, TWO_F32)
        self.assertEqual(lds.read_by_byte_addr(0x40), TWO_F32)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for LDS floating-point atomic maximum operations."""
    DS_MAX_F32 = "ds_max_f32"
    DS_MAX_RTN_F32 = "ds_max_rtn_f32"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return self == DSOpcode.DS_MAX_RTN_F32

    @property
    def is_max(self) -> bool:
        """True if operation performs floating-point maximum."""
        return True


class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclasses.dataclass(frozen=True)
class Register:
    """Architectural register representation with kind and hardware index."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        elif self.kind == RegKind.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


def vgpr(index: int) -> Register:
    """Convenience helper to create a VGPR."""
    return Register(RegKind.VGPR, index)


def sgpr(index: int) -> Register:
    """Convenience helper to create an SGPR."""
    return Register(RegKind.SGPR, index)


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """Unsigned 16-bit byte offset immediate [0, 65535]."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"Offset immediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSMaxRtnF32Instruction:
    """Symbolic AST node representing an RDNA2 ds_max_rtn_f32 instruction."""
    opcode: DSOpcode
    vdst: Optional[Register]
    vaddr: Register
    vdata0: Register
    offset: OffsetImmediate = OffsetImmediate(0)
    gds: bool = False

    def __post_init__(self) -> None:
        if self.opcode.is_return and self.vdst is None:
            raise ValueError(f"Returning opcode {self.opcode.value} requires destination vdst")
        if self.vdst is not None and self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination register must be VGPR, got {self.vdst.kind}")
        if self.vaddr.kind != RegKind.VGPR:
            raise ValueError(f"Address register must be VGPR, got {self.vaddr.kind}")
        if self.vdata0.kind != RegKind.VGPR:
            raise ValueError(f"Data operand register must be VGPR, got {self.vdata0.kind}")

    def __str__(self) -> str:
        parts = [self.opcode.value]
        if self.vdst is not None:
            parts.append(str(self.vdst) + ",")
        parts.append(str(self.vaddr) + ",")
        parts.append(str(self.vdata0))
        if self.offset.value != 0:
            parts.append(str(self.offset))
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


@dataclasses.dataclass
class Program:
    """Sequence of symbolic instructions."""
    instructions: List[DSMaxRtnF32Instruction] = dataclasses.field(default_factory=list)

    def add(self, inst: DSMaxRtnF32Instruction) -> None:
        self.instructions.append(inst)

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


class TestDSMaxRtnF32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        inst = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(8),
            gds=False,
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_MAX_RTN_F32)
        self.assertEqual(inst.vdst, vgpr(0))
        self.assertEqual(inst.vaddr, vgpr(1))
        self.assertEqual(inst.vdata0, vgpr(2))
        self.assertEqual(inst.offset.value, 8)
        self.assertFalse(inst.gds)

    def test_ast_string_representation(self) -> None:
        inst_offset = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(5),
            vaddr=vgpr(6),
            vdata0=vgpr(7),
            offset=OffsetImmediate(16),
            gds=False,
        )
        self.assertEqual(str(inst_offset), "ds_max_rtn_f32 v5, v6, v7 offset:16")

        inst_no_offset = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(0),
            gds=False,
        )
        self.assertEqual(str(inst_no_offset), "ds_max_rtn_f32 v0, v1, v2")

        inst_gds = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(4),
            gds=True,
        )
        self.assertEqual(str(inst_gds), "ds_max_rtn_f32 v0, v1, v2 offset:4 gds")

    def test_ast_immutability(self) -> None:
        inst = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.opcode = DSOpcode.DS_MAX_F32  # type: ignore

    def test_ast_validation_errors(self) -> None:
        # Invalid VGPR indices
        with self.assertRaises(ValueError):
            vgpr(-1)
        with self.assertRaises(ValueError):
            vgpr(256)

        # Invalid Offset
        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

        # Returning opcode without vdst
        with self.assertRaises(ValueError):
            DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=None,
                vaddr=vgpr(0),
                vdata0=vgpr(1),
            )

        # Non-VGPR registers in operands
        with self.assertRaises(ValueError):
            DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=sgpr(0),
                vaddr=vgpr(1),
                vdata0=vgpr(2),
            )
        with self.assertRaises(ValueError):
            DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=vgpr(0),
                vaddr=sgpr(1),
                vdata0=vgpr(2),
            )
        with self.assertRaises(ValueError):
            DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=vgpr(0),
                vaddr=vgpr(1),
                vdata0=sgpr(2),
            )

    def test_ast_program_container(self) -> None:
        prog = Program()
        inst1 = DSMaxRtnF32Instruction(opcode=DSOpcode.DS_MAX_RTN_F32, vdst=vgpr(0), vaddr=vgpr(1), vdata0=vgpr(2))
        inst2 = DSMaxRtnF32Instruction(opcode=DSOpcode.DS_MAX_RTN_F32, vdst=vgpr(3), vaddr=vgpr(4), vdata0=vgpr(5))
        prog.add(inst1)
        prog.add(inst2)
        self.assertEqual(len(prog), 2)
        instructions = list(prog)
        self.assertEqual(instructions[0], inst1)
        self.assertEqual(instructions[1], inst2)


class ScalarThreadState:
    """Architectural register and execution state for a single scalar GPU thread."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.pc: int = 0
        self.instruction_count: int = 0
        self.cycle_count: int = 0

    def get_vgpr(self, idx: int) -> int:
        """Read 32-bit dword from VGPR file."""
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        """Write 32-bit dword to VGPR file."""
        self.vgprs[idx] = val & MASK32


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for LDS instructions."""

    def __init__(self, state: Optional[ScalarThreadState] = None, lds: Optional[LDSMemory] = None) -> None:
        if state is not None:
            self.state = state
        else:
            self.state = ScalarThreadState(lds=lds)

    def execute_instruction(self, inst: DSMaxRtnF32Instruction) -> int:
        """Execute a single ds_max_rtn_f32 instruction deterministically."""
        if inst.opcode not in (DSOpcode.DS_MAX_RTN_F32, DSOpcode.DS_MAX_F32):
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")

        # In RDNA2, register reads occur before any register write
        vaddr_val = self.state.get_vgpr(inst.vaddr.index)
        vdata0_val = self.state.get_vgpr(inst.vdata0.index)

        eff_addr = calc_lds_effective_address(vaddr_val, inst.offset.value)
        old_val = self.state.lds.atomic_max_rtn_f32(eff_addr, vdata0_val)

        if inst.opcode.is_return and inst.vdst is not None:
            self.state.set_vgpr(inst.vdst.index, old_val)

        self.state.instruction_count += 1
        self.state.cycle_count += 4
        return old_val

    def run_program(self, prog: Program) -> int:
        """Execute all instructions in the given program sequentially."""
        count = 0
        for inst in prog:
            self.execute_instruction(inst)
            count += 1
        return count


class TestDSMaxRtnF32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter(state=self.state)

    def test_scalar_single_execution(self) -> None:
        # v1 (addr) = 0x40, v2 (data) = 2.0f, LDS[0x40] = 1.0f
        self.state.set_vgpr(1, 0x40)
        self.state.set_vgpr(2, TWO_F32)
        self.lds.write_by_byte_addr(0x40, ONE_F32)

        inst = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
        )
        returned = self.interp.execute_instruction(inst)

        # Destination VGPR v0 and returned value should be old LDS value (1.0f)
        self.assertEqual(returned, ONE_F32)
        self.assertEqual(self.state.get_vgpr(0), ONE_F32)
        # LDS memory should now be the maximum (2.0f)
        self.assertEqual(self.lds.read_by_byte_addr(0x40), TWO_F32)
        self.assertEqual(self.state.instruction_count, 1)

    def test_scalar_sequential_reduction(self) -> None:
        # Address at v1 = 0x80; initial memory = -10.0f
        self.state.set_vgpr(1, 0x80)
        self.lds.write_by_byte_addr(0x80, NEG_TEN_F32)

        steps = [
            (NEG_ONE_F32, NEG_TEN_F32, NEG_ONE_F32),  # cand -1.0 > -10.0 -> mem -1.0
            (NEG_TWO_F32, NEG_ONE_F32, NEG_ONE_F32),  # cand -2.0 < -1.0 -> mem unchanged
            (HALF_F32, NEG_ONE_F32, HALF_F32),        # cand 0.5 > -1.0 -> mem 0.5
            (ONE_F32, HALF_F32, ONE_F32),             # cand 1.0 > 0.5 -> mem 1.0
            (TEN_F32, ONE_F32, TEN_F32),              # cand 10.0 > 1.0 -> mem 10.0
            (TWO_F32, TEN_F32, TEN_F32),              # cand 2.0 < 10.0 -> mem unchanged
        ]

        for cand, exp_ret, exp_mem in steps:
            self.state.set_vgpr(2, cand)
            inst = DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=vgpr(0),
                vaddr=vgpr(1),
                vdata0=vgpr(2),
            )
            ret = self.interp.execute_instruction(inst)
            self.assertEqual(ret, exp_ret)
            self.assertEqual(self.state.get_vgpr(0), exp_ret)
            self.assertEqual(self.lds.read_by_byte_addr(0x80), exp_mem)

    def test_scalar_with_offset(self) -> None:
        # v1 (addr) = 0x20, offset = 12 -> byte 0x2C
        self.state.set_vgpr(1, 0x20)
        self.state.set_vgpr(2, TWO_F32)
        self.lds.write_by_byte_addr(0x2C, ONE_F32)

        inst = DSMaxRtnF32Instruction(
            opcode=DSOpcode.DS_MAX_RTN_F32,
            vdst=vgpr(0),
            vaddr=vgpr(1),
            vdata0=vgpr(2),
            offset=OffsetImmediate(12),
        )
        ret = self.interp.execute_instruction(inst)
        self.assertEqual(ret, ONE_F32)
        self.assertEqual(self.state.get_vgpr(0), ONE_F32)
        self.assertEqual(self.lds.read_by_byte_addr(0x2C), TWO_F32)

    def test_scalar_program_execution(self) -> None:
        prog = Program()
        # Initialize 3 memory locations
        inputs = [
            (ONE_F32, TWO_F32),
            (NEG_TEN_F32, NEG_ONE_F32),
            (HALF_F32, ONE_F32),
        ]
        for i, (init_val, cand) in enumerate(inputs):
            addr = i * 4
            self.lds.write_by_byte_addr(addr, init_val)
            self.state.set_vgpr(10 + i, addr)
            self.state.set_vgpr(20 + i, cand)
            prog.add(DSMaxRtnF32Instruction(
                opcode=DSOpcode.DS_MAX_RTN_F32,
                vdst=vgpr(i),
                vaddr=vgpr(10 + i),
                vdata0=vgpr(20 + i),
            ))

        executed = self.interp.run_program(prog)
        self.assertEqual(executed, 3)
        self.assertEqual(self.state.get_vgpr(0), ONE_F32)
        self.assertEqual(self.lds.read_by_byte_addr(0), TWO_F32)
        self.assertEqual(self.state.get_vgpr(1), NEG_TEN_F32)
        self.assertEqual(self.lds.read_by_byte_addr(4), NEG_ONE_F32)
        self.assertEqual(self.state.get_vgpr(2), HALF_F32)
        self.assertEqual(self.lds.read_by_byte_addr(8), ONE_F32)


if __name__ == "__main__":
    unittest.main()
