from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_min_rtn_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision minNum and AMD RDNA2 / GFX10
Local Data Share (LDS) architectural semantics:
  - ds_min_rtn_f32: Atomic floating-point 32-bit minimum on Local Data Share (LDS),
    returning the previous memory content into the destination VGPR.

Syntax:
    ds_min_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = oracle_min_f32(old_val, vdata0)
    LDS[dword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. IEEE 754-2008 / IEEE 754-2019 minNum Semantics:
     - NaN Suppression: If one operand is NaN and the other is a valid float, the valid float
       is returned. If both operands are NaN, a quiet NaN is returned.
     - Signed Zero Ordering: -0.0 is strictly less than +0.0:
       min(+0.0, -0.0) = -0.0, min(-0.0, +0.0) = -0.0.
     - Infinities & Subnormals: -INF < all finite numbers < +INF.
       Subnormal single-precision values are preserved without flushing.
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

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import math
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

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
THREE_F32: int = 0x40400000         # 3.0
TEN_F32: int = 0x41200000           # 10.0
NEG_TEN_F32: int = 0xC1200000       # -10.0

FLT_MIN: int = 0x00800000           # Smallest positive normal: 2^-126
NEG_FLT_MIN: int = 0x80800000       # -2^-126
FLT_MAX: int = 0x7F7FFFFF           # (2 - 2^-23) * 2^127
NEG_FLT_MAX: int = 0xFF7FFFFF
POS_INF: int = 0x7F800000           # +infinity
NEG_INF: int = 0xFF800000           # -infinity
QNAN_PAYLOAD_A: int = 0x7FC10001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFC20002    # Quiet NaN with payload B
SNAN_PAYLOAD_A: int = 0x7F810001    # Signaling NaN
SNAN_PAYLOAD_B: int = 0xFF820002    # Signaling NaN
SUBNORM_MIN: int = 0x00000001       # Smallest positive subnormal: 2^-149
SUBNORM_MAX: int = 0x007FFFFF       # Largest positive subnormal
NEG_SUBNORM_MIN: int = 0x80000001
NEG_SUBNORM_MAX: int = 0x807FFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256         # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106         # Standard GFX10 scalar register file size


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


def oracle_min_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_min_f32 atomic operation.

    Follows IEEE 754-2008/2019 minNum:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (min(+0.0, -0.0) -> -0.0).
      - Subnormals and infinities preserved.
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

    # Signed zeros
    if is_zero(a) and is_zero(b):
        return NEG_ZERO if ((a == NEG_ZERO) or (b == NEG_ZERO)) else POS_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa < fb:
        return a
    elif fb < fa:
        return b
    else:
        # Numerically equal: negative sign bit is strictly smaller
        if (a & SIGN_MASK) != 0:
            return a
        if (b & SIGN_MASK) != 0:
            return b
        return a


def ref_ds_min_rtn_f32(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_min_rtn_f32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = oracle_min_f32(old_val, vdata0)

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    new_mem = oracle_min_f32(old_val, vdata0)
    return old_val, new_mem


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the lowest 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


class LDSMemory:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.dwords = [0] * self.num_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        return self.dwords[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        self.dwords[idx] = u32(val)

    def atomic_min_rtn_f32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit float minimum with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        curr = self.dwords[idx]
        old_val, new_mem = ref_ds_min_rtn_f32(curr, vdata0)
        self.dwords[idx] = new_mem
        return old_val, new_mem


class TestLDSMemoryAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS memory model."""

    def test_oracle_standard_floats(self) -> None:
        # min(1.0, 2.0) = 1.0
        self.assertEqual(oracle_min_f32(ONE_F32, TWO_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(TWO_F32, ONE_F32), ONE_F32)

        # min(10.0, -2.0) = -2.0
        self.assertEqual(oracle_min_f32(TEN_F32, NEG_TWO_F32), NEG_TWO_F32)
        self.assertEqual(oracle_min_f32(NEG_TWO_F32, TEN_F32), NEG_TWO_F32)

        # min(-10.0, -2.0) = -10.0
        self.assertEqual(oracle_min_f32(NEG_TEN_F32, NEG_TWO_F32), NEG_TEN_F32)

    def test_oracle_signed_zeros(self) -> None:
        # min(+0.0, -0.0) = -0.0
        self.assertEqual(oracle_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_oracle_infinities(self) -> None:
        self.assertEqual(oracle_min_f32(POS_INF, ONE_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(NEG_INF, ONE_F32), NEG_INF)
        self.assertEqual(oracle_min_f32(POS_INF, NEG_INF), NEG_INF)
        self.assertEqual(oracle_min_f32(NEG_INF, NEG_INF), NEG_INF)
        self.assertEqual(oracle_min_f32(POS_INF, POS_INF), POS_INF)

    def test_oracle_nan_suppression(self) -> None:
        # If one operand is NaN, return the valid float
        self.assertEqual(oracle_min_f32(QNAN_PAYLOAD_A, ONE_F32), ONE_F32)
        self.assertEqual(oracle_min_f32(ONE_F32, QNAN_PAYLOAD_A), ONE_F32)
        self.assertEqual(oracle_min_f32(SNAN_PAYLOAD_A, NEG_TEN_F32), NEG_TEN_F32)
        self.assertEqual(oracle_min_f32(NEG_TEN_F32, SNAN_PAYLOAD_A), NEG_TEN_F32)

        # If both are NaN, return quiet NaN
        res_nan = oracle_min_f32(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(res_nan))

    def test_oracle_subnormals(self) -> None:
        # Subnormals are smaller than normals but greater than 0
        self.assertEqual(oracle_min_f32(SUBNORM_MIN, FLT_MIN), SUBNORM_MIN)
        self.assertEqual(oracle_min_f32(SUBNORM_MAX, SUBNORM_MIN), SUBNORM_MIN)
        self.assertEqual(oracle_min_f32(NEG_SUBNORM_MAX, POS_ZERO), NEG_SUBNORM_MAX)

    def test_ref_ds_min_rtn_f32(self) -> None:
        # old_val=2.0, vdata0=1.0 -> returns (2.0, 1.0)
        old_val, new_mem = ref_ds_min_rtn_f32(TWO_F32, ONE_F32)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_mem, ONE_F32)

        # old_val=1.0, vdata0=5.0 -> returns (1.0, 1.0)
        old_val, new_mem = ref_ds_min_rtn_f32(ONE_F32, TEN_F32)
        self.assertEqual(old_val, ONE_F32)
        self.assertEqual(new_mem, ONE_F32)

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_dwords, 16384)
        self.assertEqual(len(mem.dwords), 16384)

        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x00, ONE_F32)
        self.assertEqual(mem.read_dword(0x00), ONE_F32)
        mem.write_dword(0x40, NEG_TEN_F32)
        self.assertEqual(mem.read_dword(0x40), NEG_TEN_F32)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x100, TWO_F32)
        self.assertEqual(mem.read_dword(0x101), TWO_F32)
        self.assertEqual(mem.read_dword(0x102), TWO_F32)
        self.assertEqual(mem.read_dword(0x103), TWO_F32)

    def test_lds_memory_bank_calculation(self) -> None:
        self.assertEqual(calc_lds_bank(0), 0)
        self.assertEqual(calc_lds_bank(4), 1)
        self.assertEqual(calc_lds_bank(124), 31)
        self.assertEqual(calc_lds_bank(128), 0)

    def test_lds_atomic_min_rtn_f32(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x20, TEN_F32)

        # min(10.0, 2.0) = 2.0; old=10.0
        old_val, new_mem = mem.atomic_min_rtn_f32(vaddr=0x20, vdata0=TWO_F32)
        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, TWO_F32)
        self.assertEqual(mem.read_dword(0x20), TWO_F32)

        # min(2.0, 3.0) = 2.0; old=2.0
        old_val2, new_mem2 = mem.atomic_min_rtn_f32(vaddr=0x20, vdata0=THREE_F32)
        self.assertEqual(old_val2, TWO_F32)
        self.assertEqual(new_mem2, TWO_F32)
        self.assertEqual(mem.read_dword(0x20), TWO_F32)

    def test_lds_atomic_min_rtn_with_offset(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x50, TEN_F32)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_min_rtn_f32(vaddr=0x40, vdata0=NEG_TWO_F32, offset=0x10)
        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, NEG_TWO_F32)
        self.assertEqual(mem.read_dword(0x50), NEG_TWO_F32)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(enum.Enum):
    """Data Share (DS) atomic opcodes."""
    DS_MIN_RTN_F32 = "ds_min_rtn_f32"


class RegType(enum.Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < limit):
            raise ValueError(f"Register index out of range: {self.reg_type.value}{self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """32-bit immediate value representing float bit pattern."""
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    @property
    def name(self) -> str:
        return hex(self.value)


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """16-bit immediate byte offset descriptor."""
    offset: int

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"Offset out of 16-bit range: {self.offset}")


@dataclasses.dataclass(frozen=True)
class DSMinRtnF32Instruction:
    """Symbolic AST node representing a ds_min_rtn_f32 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Union[RegisterOperand, Immediate]
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_MIN_RTN_F32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be VGPR, got {self.vdst.reg_type.value}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be VGPR, got {self.vaddr.reg_type.value}")
        if isinstance(self.vdata0, RegisterOperand) and self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 register must be VGPR, got {self.vdata0.reg_type.value}")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of 16-bit range: {self.offset}")

    def disasm(self) -> str:
        vdata_str = self.vdata0.name
        parts = [f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {vdata_str}"]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

    @classmethod
    def parse(cls, line: str) -> DSMinRtnF32Instruction:
        import re
        raw = line.split("//")[0].split(";")[0].strip()
        if not raw:
            raise ValueError("Empty instruction line")
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("No tokens found")
        mnemonic = tokens[0].lower()
        if mnemonic != DSOpcode.DS_MIN_RTN_F32.value:
            raise ValueError(f"Unknown opcode: {mnemonic}")

        if len(tokens) < 4:
            raise ValueError(f"Insufficient operands for {mnemonic}")

        def parse_reg(tok: str) -> RegisterOperand:
            m = re.match(r"^([vs])\[?(\d+)\]?$", tok.lower())
            if not m:
                raise ValueError(f"Invalid register operand: {tok}")
            rtype = RegType.VGPR if m.group(1) == "v" else RegType.SGPR
            idx = int(m.group(2))
            return RegisterOperand(idx, rtype)

        vdst = parse_reg(tokens[1])
        vaddr = parse_reg(tokens[2])
        vdata_tok = tokens[3]
        if vdata_tok.lower().startswith("v") or vdata_tok.lower().startswith("s"):
            vdata0: Union[RegisterOperand, Immediate] = parse_reg(vdata_tok)
        else:
            vdata0 = Immediate(int(vdata_tok, 0))

        offset = 0
        gds = False
        for opt in tokens[4:]:
            opt_l = opt.lower()
            if opt_l == "gds":
                gds = True
            elif opt_l.startswith("offset:"):
                offset = int(opt_l.split(":", 1)[1], 0)
            else:
                raise ValueError(f"Unexpected token in instruction: {opt}")

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset, gds=gds)


class TestDSMinRtnF32AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_and_disasm(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v1 = RegisterOperand(1, RegType.VGPR)
        v2 = RegisterOperand(2, RegType.VGPR)

        inst_simple = DSMinRtnF32Instruction(vdst=v0, vaddr=v1, vdata0=v2)
        self.assertEqual(inst_simple.disasm(), "ds_min_rtn_f32 v0, v1, v2")
        self.assertEqual(inst_simple.offset, 0)
        self.assertFalse(inst_simple.gds)

        inst_opts = DSMinRtnF32Instruction(vdst=v0, vaddr=v1, vdata0=v2, offset=32, gds=True)
        self.assertEqual(inst_opts.disasm(), "ds_min_rtn_f32 v0, v1, v2 offset:32 gds")

        inst_imm = DSMinRtnF32Instruction(vdst=v0, vaddr=v1, vdata0=Immediate(ONE_F32), offset=16)
        self.assertEqual(inst_imm.disasm(), f"ds_min_rtn_f32 v0, v1, {hex(ONE_F32)} offset:16")

    def test_ast_parse_canonical(self) -> None:
        inst1 = DSMinRtnF32Instruction.parse("ds_min_rtn_f32 v0, v1, v2")
        self.assertEqual(inst1.vdst, RegisterOperand(0, RegType.VGPR))
        self.assertEqual(inst1.vaddr, RegisterOperand(1, RegType.VGPR))
        self.assertEqual(inst1.vdata0, RegisterOperand(2, RegType.VGPR))
        self.assertEqual(inst1.offset, 0)
        self.assertFalse(inst1.gds)

        inst2 = DSMinRtnF32Instruction.parse("ds_min_rtn_f32 v[5], v[6], v[7] offset:64 gds")
        self.assertEqual(inst2.vdst, RegisterOperand(5, RegType.VGPR))
        self.assertEqual(inst2.vaddr, RegisterOperand(6, RegType.VGPR))
        self.assertEqual(inst2.vdata0, RegisterOperand(7, RegType.VGPR))
        self.assertEqual(inst2.offset, 64)
        self.assertTrue(inst2.gds)

        inst3 = DSMinRtnF32Instruction.parse("ds_min_rtn_f32 v10, v11, 0x3f800000 offset:0x10")
        self.assertEqual(inst3.vdata0, Immediate(ONE_F32))
        self.assertEqual(inst3.offset, 16)

    def test_ast_validation_errors(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)

        with self.assertRaises(ValueError):
            DSMinRtnF32Instruction(vdst=s0, vaddr=v0, vdata0=v0)

        with self.assertRaises(ValueError):
            DSMinRtnF32Instruction(vdst=v0, vaddr=s0, vdata0=v0)

        with self.assertRaises(ValueError):
            DSMinRtnF32Instruction(vdst=v0, vaddr=v0, vdata0=s0)

        with self.assertRaises(ValueError):
            DSMinRtnF32Instruction(vdst=v0, vaddr=v0, vdata0=v0, offset=-1)

        with self.assertRaises(ValueError):
            DSMinRtnF32Instruction(vdst=v0, vaddr=v0, vdata0=v0, offset=65536)

        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    """Architectural state for a single execution thread with scalar register file and LDS."""

    def __init__(
        self,
        lds: Optional[LDSMemory] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.vgprs: List[int] = [0] * num_vgprs
        self.sgprs: List[int] = [0] * num_sgprs

    def get_vgpr(self, reg: Union[RegisterOperand, int]) -> int:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        return self.vgprs[idx]

    def set_vgpr(self, reg: Union[RegisterOperand, int], val: int) -> None:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        self.vgprs[idx] = u32(val)

    def get_sgpr(self, reg: Union[RegisterOperand, int]) -> int:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        return self.sgprs[idx]

    def set_sgpr(self, reg: Union[RegisterOperand, int], val: int) -> None:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        self.sgprs[idx] = u32(val)


class ScalarDSInterpreter:
    """Deterministic scalar interpreter executing DS instructions."""

    def execute(self, state: ScalarThreadState, inst: DSMinRtnF32Instruction) -> Tuple[int, int]:
        # 1. Latch input operands before any register update
        base_addr = state.get_vgpr(inst.vaddr)
        if isinstance(inst.vdata0, RegisterOperand):
            vdata_val = state.get_vgpr(inst.vdata0)
        else:
            vdata_val = inst.vdata0.value

        # 2. Atomic memory transaction on LDS
        old_val, new_mem = state.lds.atomic_min_rtn_f32(
            vaddr=base_addr,
            vdata0=vdata_val,
            offset=inst.offset,
        )

        # 3. Write return value to destination VGPR
        state.set_vgpr(inst.vdst, old_val)

        return old_val, new_mem


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_basic(self) -> None:
        addr = 0x40
        self.lds.write_dword(addr, TEN_F32)

        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, TWO_F32)
        self.state.set_vgpr(0, 0xCAFEBABE)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        # min(10.0, 2.0) = 2.0
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, TWO_F32)
        self.assertEqual(self.state.get_vgpr(0), TEN_F32)
        self.assertEqual(self.lds.read_dword(addr), TWO_F32)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x14
        eff_addr = base_addr + offset

        self.lds.write_dword(eff_addr, ONE_F32)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, NEG_TWO_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=offset,
        )

        # min(1.0, -2.0) = -2.0
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, ONE_F32)
        self.assertEqual(new_mem, NEG_TWO_F32)
        self.assertEqual(self.state.get_vgpr(0), ONE_F32)
        self.assertEqual(self.lds.read_dword(eff_addr), NEG_TWO_F32)

    def test_scalar_execution_with_immediate(self) -> None:
        addr = 0x200
        self.lds.write_dword(addr, TWO_F32)
        self.state.set_vgpr(1, addr)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=Immediate(HALF_F32),
        )

        # min(2.0, 0.5) = 0.5
        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_mem, HALF_F32)
        self.assertEqual(self.state.get_vgpr(0), TWO_F32)
        self.assertEqual(self.lds.read_dword(addr), HALF_F32)

    def test_scalar_unaligned_byte_address(self) -> None:
        self.lds.write_dword(0x300, THREE_F32)
        self.state.set_vgpr(1, 0x303)
        self.state.set_vgpr(2, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, THREE_F32)
        self.assertEqual(new_mem, ONE_F32)
        self.assertEqual(self.lds.read_dword(0x300), ONE_F32)

    def test_scalar_signed_zero_ordering(self) -> None:
        addr = 0x80
        self.lds.write_dword(addr, POS_ZERO)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, NEG_ZERO)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        # min(+0.0, -0.0) = -0.0
        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, POS_ZERO)
        self.assertEqual(new_mem, NEG_ZERO)
        self.assertEqual(self.state.get_vgpr(0), POS_ZERO)
        self.assertEqual(self.lds.read_dword(addr), NEG_ZERO)

    def test_scalar_nan_suppression(self) -> None:
        addr = 0x90
        self.lds.write_dword(addr, QNAN_PAYLOAD_A)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        # min(NaN, 1.0) = 1.0
        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, QNAN_PAYLOAD_A)
        self.assertEqual(new_mem, ONE_F32)
        self.assertEqual(self.state.get_vgpr(0), QNAN_PAYLOAD_A)
        self.assertEqual(self.lds.read_dword(addr), ONE_F32)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(
        self,
        wave_size: int = 64,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.mask_limit: int = (1 << wave_size) - 1
        self.exec_mask: int = (self.mask_limit if exec_mask is None else exec_mask) & self.mask_limit
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()
        self.lane_vgprs: List[List[int]] = [[0] * num_vgprs for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, reg: Union[RegisterOperand, int]) -> int:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        return self.lane_vgprs[lane][idx]

    def set_vgpr(self, lane: int, reg: Union[RegisterOperand, int], val: int) -> None:
        idx = reg.index if isinstance(reg, RegisterOperand) else reg
        self.lane_vgprs[lane][idx] = u32(val)


class SIMDDSInterpreter:
    """SIMD wavefront interpreter handling divergence and EXEC masking."""

    def execute(
        self,
        wave: SIMDWaveState,
        inst: DSMinRtnF32Instruction,
    ) -> List[Optional[Tuple[int, int]]]:
        results: List[Optional[Tuple[int, int]]] = [None] * wave.wave_size

        # In RDNA2, lanes within a wavefront issue memory requests in lane order (0..wave_size-1)
        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue

            base_addr = wave.get_vgpr(lane, inst.vaddr)
            if isinstance(inst.vdata0, RegisterOperand):
                vdata_val = wave.get_vgpr(lane, inst.vdata0)
            else:
                vdata_val = inst.vdata0.value

            old_val, new_mem = wave.lds.atomic_min_rtn_f32(
                vaddr=base_addr,
                vdata0=vdata_val,
                offset=inst.offset,
            )

            wave.set_vgpr(lane, inst.vdst, old_val)
            results[lane] = (old_val, new_mem)

        return results

    def detect_bank_conflicts(
        self,
        wave: SIMDWaveState,
        inst: DSMinRtnF32Instruction,
    ) -> Dict[int, List[int]]:
        bank_map: Dict[int, Dict[int, List[int]]] = {}
        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue
            base_addr = wave.get_vgpr(lane, inst.vaddr)
            eff_addr = calc_lds_effective_address(base_addr, inst.offset)
            dword_idx = calc_lds_dword_index(eff_addr, wave.lds.num_dwords)
            bank = calc_lds_bank(eff_addr)

            if bank not in bank_map:
                bank_map[bank] = {}
            if dword_idx not in bank_map[bank]:
                bank_map[bank][dword_idx] = []
            bank_map[bank][dword_idx].append(lane)

        conflicts: Dict[int, List[int]] = {}
        for bank, dwords in bank_map.items():
            if len(dwords) > 1:
                all_lanes: List[int] = []
                for lane_list in dwords.values():
                    all_lanes.extend(lane_list)
                conflicts[bank] = sorted(all_lanes)

        return conflicts


class TestSIMDInterpreter(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_exec_independent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for lane in range(32):
            addr = lane * 4
            lds.write_dword(addr, TEN_F32)
            wave.set_vgpr(lane, 1, addr)
            wave.set_vgpr(lane, 2, f32_to_bits(float(lane)))
            wave.set_vgpr(lane, 0, 0xCAFEBABE)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        results = interp.execute(wave, inst)

        for lane in range(32):
            self.assertIsNotNone(results[lane])
            old_val, new_mem = results[lane]  # type: ignore[misc]
            self.assertEqual(old_val, TEN_F32)
            expected_new = oracle_min_f32(TEN_F32, f32_to_bits(float(lane)))
            self.assertEqual(new_mem, expected_new)
            self.assertEqual(wave.get_vgpr(lane, 0), TEN_F32)
            self.assertEqual(lds.read_dword(lane * 4), expected_new)

    def test_wave64_full_exec(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=64, exec_mask=MASK64, lds=lds)
        interp = SIMDDSInterpreter()

        for lane in range(64):
            addr = lane * 4
            lds.write_dword(addr, TWO_F32)
            wave.set_vgpr(lane, 1, addr)
            wave.set_vgpr(lane, 2, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        results = interp.execute(wave, inst)

        for lane in range(64):
            self.assertIsNotNone(results[lane])
            old_val, new_mem = results[lane]  # type: ignore[misc]
            self.assertEqual(old_val, TWO_F32)
            self.assertEqual(new_mem, ONE_F32)
            self.assertEqual(wave.get_vgpr(lane, 0), TWO_F32)
            self.assertEqual(lds.read_dword(lane * 4), ONE_F32)

    def test_simd_exec_mask_inactive_lanes(self) -> None:
        lds = LDSMemory(65536)
        # Even lanes active (0, 2, 4...), odd lanes inactive (1, 3, 5...)
        even_mask = 0x5555555555555555
        wave = SIMDWaveState(wave_size=64, exec_mask=even_mask, lds=lds)
        interp = SIMDDSInterpreter()

        for lane in range(64):
            addr = lane * 4
            lds.write_dword(addr, TEN_F32)
            wave.set_vgpr(lane, 1, addr)
            wave.set_vgpr(lane, 2, ONE_F32)
            wave.set_vgpr(lane, 0, 0x12345678)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        results = interp.execute(wave, inst)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(wave.get_vgpr(lane, 0), TEN_F32)
                self.assertEqual(lds.read_dword(lane * 4), ONE_F32)
            else:
                self.assertIsNone(results[lane])
                # Inactive lane VGPR must remain completely untouched
                self.assertEqual(wave.get_vgpr(lane, 0), 0x12345678)
                # LDS cell for inactive lane remains untouched
                self.assertEqual(lds.read_dword(lane * 4), TEN_F32)

    def test_simd_zero_exec_mask(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_dword(0, TEN_F32)
        wave.set_vgpr(0, 1, 0)
        wave.set_vgpr(0, 2, ONE_F32)
        wave.set_vgpr(0, 0, 0xDEAD)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        results = interp.execute(wave, inst)
        self.assertTrue(all(r is None for r in results))
        self.assertEqual(wave.get_vgpr(0, 0), 0xDEAD)
        self.assertEqual(lds.read_dword(0), TEN_F32)

    def test_simd_multi_lane_serialization_same_address(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x7, lds=lds)  # Lanes 0, 1, 2
        interp = SIMDDSInterpreter()

        addr = 0x20
        lds.write_dword(addr, TEN_F32)  # 10.0

        # Lane 0 passes 3.0, Lane 1 passes 2.0, Lane 2 passes 1.0
        wave.set_vgpr(0, 1, addr)
        wave.set_vgpr(0, 2, THREE_F32)

        wave.set_vgpr(1, 1, addr)
        wave.set_vgpr(1, 2, TWO_F32)

        wave.set_vgpr(2, 1, addr)
        wave.set_vgpr(2, 2, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        results = interp.execute(wave, inst)

        # Lane 0 gets 10.0, writes min(10.0, 3.0) = 3.0
        self.assertEqual(results[0], (TEN_F32, THREE_F32))
        self.assertEqual(wave.get_vgpr(0, 0), TEN_F32)

        # Lane 1 gets 3.0, writes min(3.0, 2.0) = 2.0
        self.assertEqual(results[1], (THREE_F32, TWO_F32))
        self.assertEqual(wave.get_vgpr(1, 0), THREE_F32)

        # Lane 2 gets 2.0, writes min(2.0, 1.0) = 1.0
        self.assertEqual(results[2], (TWO_F32, ONE_F32))
        self.assertEqual(wave.get_vgpr(2, 0), TWO_F32)

        self.assertEqual(lds.read_dword(addr), ONE_F32)

    def test_bank_conflict_detection(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x3, lds=lds)  # Lanes 0, 1
        interp = SIMDDSInterpreter()

        # Both lane 0 and lane 1 target bank 0, but different dwords:
        # eff_addr 0 -> dword 0 -> bank 0
        # eff_addr 128 -> dword 32 -> bank 0 (32 % 32 == 0)
        wave.set_vgpr(0, 1, 0)
        wave.set_vgpr(1, 1, 128)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=Immediate(ONE_F32),
        )

        conflicts = interp.detect_bank_conflicts(wave, inst)
        self.assertIn(0, conflicts)
        self.assertEqual(conflicts[0], [0, 1])


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.interp = ScalarDSInterpreter()

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        # vdst = v0, vaddr = v0, vdata0 = v1
        # v0 starts with address 0x40. LDS[0x40] contains TEN_F32.
        # v1 contains TWO_F32.
        state = ScalarThreadState(lds=self.lds)
        state.set_vgpr(0, 0x40)
        state.set_vgpr(1, TWO_F32)
        self.lds.write_dword(0x40, TEN_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(0),
            vdata0=RegisterOperand(1),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        # Returned old_val is TEN_F32, which overwrites v0
        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, TWO_F32)
        self.assertEqual(state.get_vgpr(0), TEN_F32)
        # v1 (vdata0) must be preserved
        self.assertEqual(state.get_vgpr(1), TWO_F32)
        # LDS memory updated correctly
        self.assertEqual(self.lds.read_dword(0x40), TWO_F32)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        # vdst = v1, vaddr = v0, vdata0 = v1
        # Address is in v0 (0x60). LDS[0x60] has THREE_F32.
        # v1 has ONE_F32.
        state = ScalarThreadState(lds=self.lds)
        state.set_vgpr(0, 0x60)
        state.set_vgpr(1, ONE_F32)
        self.lds.write_dword(0x60, THREE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(1),
            vaddr=RegisterOperand(0),
            vdata0=RegisterOperand(1),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        # min(3.0, 1.0) = 1.0; vdst (v1) receives old_val (THREE_F32)
        self.assertEqual(old_val, THREE_F32)
        self.assertEqual(new_mem, ONE_F32)
        self.assertEqual(state.get_vgpr(1), THREE_F32)
        # v0 (vaddr) remains preserved
        self.assertEqual(state.get_vgpr(0), 0x60)
        self.assertEqual(self.lds.read_dword(0x60), ONE_F32)

    def test_aliasing_vdst_equals_vaddr_equals_vdata0(self) -> None:
        # Full 3-way alias: vdst = vaddr = vdata0 = v0
        # Let v0 be 0x00 (address 0x00, which also represents POS_ZERO float).
        # LDS[0x00] has TEN_F32.
        state = ScalarThreadState(lds=self.lds)
        state.set_vgpr(0, POS_ZERO)
        self.lds.write_dword(0x00, TEN_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(0),
            vdata0=RegisterOperand(0),
        )

        # min(10.0, 0.0) = 0.0
        old_val, new_mem = self.interp.execute(state, inst)

        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(new_mem, POS_ZERO)
        # v0 gets old_val
        self.assertEqual(state.get_vgpr(0), TEN_F32)
        self.assertEqual(self.lds.read_dword(0x00), POS_ZERO)

    def test_aliasing_vaddr_equals_vdata0_distinct_vdst(self) -> None:
        # vaddr = v1, vdata0 = v1, vdst = v0
        state = ScalarThreadState(lds=self.lds)
        state.set_vgpr(1, 0x40)  # address 0x40
        state.set_vgpr(0, 0xCAFE)
        self.lds.write_dword(0x40, TEN_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(1),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        self.assertEqual(old_val, TEN_F32)
        self.assertEqual(state.get_vgpr(0), TEN_F32)
        # v1 must remain strictly unaltered
        self.assertEqual(state.get_vgpr(1), 0x40)

    def test_operand_preservation_distinct_registers(self) -> None:
        # All distinct: vdst=v0, vaddr=v1, vdata0=v2
        state = ScalarThreadState(lds=self.lds)
        state.set_vgpr(0, 0xAAAAAAAA)
        state.set_vgpr(1, 0x80)
        state.set_vgpr(2, NEG_TWO_F32)
        self.lds.write_dword(0x80, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(0),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )

        self.interp.execute(state, inst)

        self.assertEqual(state.get_vgpr(0), ONE_F32)
        # Input registers must remain unchanged
        self.assertEqual(state.get_vgpr(1), 0x80)
        self.assertEqual(state.get_vgpr(2), NEG_TWO_F32)

    def test_vgpr_isolation_no_spurious_clobber(self) -> None:
        state = ScalarThreadState(lds=self.lds)
        for i in range(256):
            state.set_vgpr(i, 0x1000 + i)

        addr = 0x20
        self.lds.write_dword(addr, TWO_F32)
        state.set_vgpr(10, addr)
        state.set_vgpr(11, ONE_F32)

        inst = DSMinRtnF32Instruction(
            vdst=RegisterOperand(12),
            vaddr=RegisterOperand(10),
            vdata0=RegisterOperand(11),
        )

        self.interp.execute(state, inst)

        self.assertEqual(state.get_vgpr(12), TWO_F32)
        for i in range(256):
            if i == 12:
                continue
            elif i == 10:
                self.assertEqual(state.get_vgpr(i), addr)
            elif i == 11:
                self.assertEqual(state.get_vgpr(i), ONE_F32)
            else:
                self.assertEqual(state.get_vgpr(i), 0x1000 + i)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class FloatConditionCodes:
    """Architectural condition codes evaluating floating-point minimum results."""
    zero: bool
    negative: bool
    nan: bool
    infinity: bool
    subnormal: bool
    updated: bool


@dataclasses.dataclass(frozen=True)
class LDSStatusFlags:
    """Status flags for LDS addressing and memory bank mapping."""
    dword_aligned: bool
    address_wrapped: bool
    bank_index: int


def compute_float_condition_codes(old_val: int, vdata0: int, result: int) -> FloatConditionCodes:
    """Compute condition codes from float atomic min result."""
    res_u = u32(result)
    old_u = u32(old_val)
    return FloatConditionCodes(
        zero=is_zero(res_u),
        negative=bool(res_u & SIGN_MASK),
        nan=is_nan(res_u),
        infinity=is_inf(res_u),
        subnormal=is_subnormal(res_u),
        updated=(res_u != old_u),
    )


def compute_lds_status_flags(
    vaddr: int,
    offset: int = 0,
    lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES,
) -> LDSStatusFlags:
    """Evaluate alignment, bank assignment, and boundary wrap status for LDS access."""
    raw_eff = vaddr + (offset & OFFSET16_MASK)
    eff_addr = raw_eff & MASK32
    return LDSStatusFlags(
        dword_aligned=((eff_addr & 3) == 0),
        address_wrapped=(raw_eff >= lds_size_bytes),
        bank_index=calc_lds_bank(eff_addr),
    )


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and update detection."""

    def test_condition_code_zero(self) -> None:
        cc_pos = compute_float_condition_codes(ONE_F32, POS_ZERO, POS_ZERO)
        self.assertTrue(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.updated)

        cc_neg = compute_float_condition_codes(POS_ZERO, NEG_ZERO, NEG_ZERO)
        self.assertTrue(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertTrue(cc_neg.updated)

    def test_condition_code_negative(self) -> None:
        cc = compute_float_condition_codes(ONE_F32, NEG_TWO_F32, NEG_TWO_F32)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.nan)
        self.assertFalse(cc.infinity)
        self.assertTrue(cc.updated)

    def test_condition_code_nan(self) -> None:
        # Both NaN results in NaN
        res_nan = oracle_min_f32(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B)
        cc = compute_float_condition_codes(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B, res_nan)
        self.assertTrue(cc.nan)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.infinity)

    def test_condition_code_infinity(self) -> None:
        cc_neg_inf = compute_float_condition_codes(TEN_F32, NEG_INF, NEG_INF)
        self.assertTrue(cc_neg_inf.infinity)
        self.assertTrue(cc_neg_inf.negative)
        self.assertTrue(cc_neg_inf.updated)

        # min(+inf, 10.0) -> 10.0
        cc_pos_inf = compute_float_condition_codes(POS_INF, TEN_F32, TEN_F32)
        self.assertFalse(cc_pos_inf.infinity)
        self.assertTrue(cc_pos_inf.updated)

    def test_condition_code_subnormal(self) -> None:
        cc = compute_float_condition_codes(ONE_F32, SUBNORM_MIN, SUBNORM_MIN)
        self.assertTrue(cc.subnormal)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.updated)

    def test_condition_code_updated_flag(self) -> None:
        # Memory had 10.0, operand was 2.0 -> updated to 2.0 (updated=True)
        cc1 = compute_float_condition_codes(TEN_F32, TWO_F32, TWO_F32)
        self.assertTrue(cc1.updated)

        # Memory had 1.0, operand was 5.0 -> min is 1.0 -> updated=False
        cc2 = compute_float_condition_codes(ONE_F32, TEN_F32, ONE_F32)
        self.assertFalse(cc2.updated)

    def test_lds_status_alignment_and_wrap(self) -> None:
        st_aligned = compute_lds_status_flags(0x100, 0x0)
        self.assertTrue(st_aligned.dword_aligned)
        self.assertFalse(st_aligned.address_wrapped)
        self.assertEqual(st_aligned.bank_index, (0x100 >> 2) % 32)

        st_unaligned = compute_lds_status_flags(0x101, 0x0)
        self.assertFalse(st_unaligned.dword_aligned)

        st_wrapped = compute_lds_status_flags(65536, 0x10)
        self.assertTrue(st_wrapped.address_wrapped)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros_comprehensive(self) -> None:
        self.assertEqual(oracle_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(oracle_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_infinities_comprehensive(self) -> None:
        self.assertEqual(oracle_min_f32(POS_INF, POS_INF), POS_INF)
        self.assertEqual(oracle_min_f32(NEG_INF, NEG_INF), NEG_INF)
        self.assertEqual(oracle_min_f32(POS_INF, NEG_INF), NEG_INF)
        self.assertEqual(oracle_min_f32(NEG_INF, POS_INF), NEG_INF)
        self.assertEqual(oracle_min_f32(POS_INF, FLT_MAX), FLT_MAX)
        self.assertEqual(oracle_min_f32(NEG_INF, NEG_FLT_MAX), NEG_INF)

    def test_subnormals_preservation_and_comparison(self) -> None:
        # Smallest subnormal (2^-149) vs smallest normal (2^-126)
        self.assertEqual(oracle_min_f32(SUBNORM_MIN, FLT_MIN), SUBNORM_MIN)
        self.assertNotEqual(oracle_min_f32(SUBNORM_MIN, FLT_MIN), POS_ZERO)

        # Largest subnormal vs smallest normal
        self.assertEqual(oracle_min_f32(SUBNORM_MAX, FLT_MIN), SUBNORM_MAX)

        # Subnormal vs zeros
        self.assertEqual(oracle_min_f32(SUBNORM_MIN, POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_min_f32(SUBNORM_MIN, NEG_ZERO), NEG_ZERO)

        # Negative subnormals
        self.assertEqual(oracle_min_f32(NEG_SUBNORM_MIN, POS_ZERO), NEG_SUBNORM_MIN)
        self.assertEqual(oracle_min_f32(NEG_SUBNORM_MAX, NEG_SUBNORM_MIN), NEG_SUBNORM_MAX)

    def test_nan_suppression_comprehensive(self) -> None:
        # Single quiet NaN with normal, infinity, zero, subnormal
        for normal_val in (ONE_F32, NEG_ONE_F32, POS_INF, NEG_INF, POS_ZERO, NEG_ZERO, SUBNORM_MIN):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_min_f32(QNAN_PAYLOAD_A, normal_val), normal_val)
                self.assertEqual(oracle_min_f32(normal_val, QNAN_PAYLOAD_A), normal_val)

        # Single signaling NaN
        for normal_val in (TWO_F32, NEG_TWO_F32, FLT_MAX, NEG_FLT_MAX):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_min_f32(SNAN_PAYLOAD_A, normal_val), normal_val)
                self.assertEqual(oracle_min_f32(normal_val, SNAN_PAYLOAD_A), normal_val)

    def test_dual_nans(self) -> None:
        # Quiet + Quiet
        q_q = oracle_min_f32(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(q_q))

        # Signaling + Signaling
        s_s = oracle_min_f32(SNAN_PAYLOAD_A, SNAN_PAYLOAD_B)
        self.assertTrue(is_qnan(s_s))

        # Quiet + Signaling
        q_s = oracle_min_f32(QNAN_PAYLOAD_A, SNAN_PAYLOAD_A)
        self.assertTrue(is_qnan(q_s))

    def test_floating_point_normal_extremes(self) -> None:
        self.assertEqual(oracle_min_f32(FLT_MAX, NEG_FLT_MAX), NEG_FLT_MAX)
        self.assertEqual(oracle_min_f32(NEG_FLT_MAX, FLT_MAX), NEG_FLT_MAX)
        self.assertEqual(oracle_min_f32(FLT_MIN, NEG_FLT_MIN), NEG_FLT_MIN)
        self.assertEqual(oracle_min_f32(NEG_FLT_MIN, FLT_MIN), NEG_FLT_MIN)


if __name__ == "__main__":
    unittest.main()
