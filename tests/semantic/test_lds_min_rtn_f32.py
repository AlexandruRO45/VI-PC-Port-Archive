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


if __name__ == "__main__":
    unittest.main()
