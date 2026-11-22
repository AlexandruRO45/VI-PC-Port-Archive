from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_rtn_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision maxNum and AMD RDNA2 / GFX10
Local Data Share (LDS) architectural semantics:
  - ds_max_rtn_f64: Atomic floating-point 64-bit maximum on Local Data Share (LDS),
    returning the previous 64-bit memory content into destination VGPR pair.

Syntax:
    ds_max_rtn_f64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS64[qword_index]
    new_mem = oracle_max_f64(old_val, vdata0)
    LDS64[qword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. IEEE 754-2008 / IEEE 754-2019 maxNum Semantics (binary64):
     - NaN Suppression: If one operand is NaN and the other is a valid float, the valid float
       is returned. If both operands are NaN, a quiet NaN is returned.
     - Signed Zero Ordering: -0.0 is strictly less than +0.0:
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0.
     - Infinities & Subnormals: -INF < all finite numbers < +INF.
       Subnormal double-precision values are preserved without flushing.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-operation 64-bit value from LDS64[qword_index] and writes it
     into the destination VGPR pair (vdst:vdst+1).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but qword-aligned for 64-bit operations (eff_addr >> 3).
     The bottom 3 bits of the effective address are dropped by hardware.
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (8,192 64-bit qwords) across 32 memory banks.
     Bank index = (effective_byte_address >> 2) % 32.

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

# IEEE 754 binary64 architectural bitmasks
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
SIGN_MASK64: int = 0x8000000000000000
EXP_MASK64: int = 0x7FF0000000000000
FRAC_MASK64: int = 0x000FFFFFFFFFFFFF
QNAN_BIT64: int = 0x0008000000000000
CANONICAL_QNAN64: int = 0x7FF8000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO64: int = 0x0000000000000000
NEG_ZERO64: int = 0x8000000000000000
ONE_F64: int = 0x3FF0000000000000           # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
TWO_F64: int = 0x4000000000000000           # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
THREE_F64: int = 0x4008000000000000         # 3.0
TEN_F64: int = 0x4024000000000000           # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # Largest finite normal double
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF64: int = 0x7FF0000000000000         # +infinity
NEG_INF64: int = 0xFFF0000000000000         # -infinity
QNAN_PAYLOAD_A64: int = 0x7FF8000000000001  # Quiet NaN with payload A
QNAN_PAYLOAD_B64: int = 0xFFF8000000000002  # Quiet NaN with payload B
SNAN_PAYLOAD_A64: int = 0x7FF0000000000001  # Signaling NaN
SNAN_PAYLOAD_B64: int = 0xFFF0000000000002  # Signaling NaN
SUBNORM_MIN64: int = 0x0000000000000001     # Smallest positive subnormal: 2^-1074
SUBNORM_MAX64: int = 0x000FFFFFFFFFFFFF     # Largest positive subnormal
NEG_SUBNORM_MIN64: int = 0x8000000000000001
NEG_SUBNORM_MAX64: int = 0x800FFFFFFFFFFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB default capacity
DEFAULT_LDS_NUM_QWORDS: int = 8192   # 8,192 64-bit qwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256         # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106         # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK64) else val


def f64_to_bits(val: float) -> int:
    """Encode float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF64 if val > 0 else NEG_INF64


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack 64-bit integer into (sign, exponent, fraction) components."""
    bits &= MASK64
    sign = (bits & SIGN_MASK64) >> 63
    exponent = (bits & EXP_MASK64) >> 52
    fraction = bits & FRAC_MASK64
    return sign, exponent, fraction


def is_nan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) != 0)


def is_qnan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Quiet NaN."""
    return is_nan64(bits) and bool(bits & QNAN_BIT64)


def is_snan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Signaling NaN."""
    return is_nan64(bits) and not bool(bits & QNAN_BIT64)


def is_inf64(bits: int) -> bool:
    """Return True if 64-bit integer represents IEEE 754 double positive or negative infinity."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) == 0)


def is_subnormal64(bits: int) -> bool:
    """Return True if 64-bit integer represents a subnormal double-precision float."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == 0) and ((bits & FRAC_MASK64) != 0)


def is_zero64(bits: int) -> bool:
    """Return True if 64-bit integer represents positive or negative zero."""
    return (bits & ~SIGN_MASK64) == 0


def oracle_max_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_f64 atomic operation.

    Follows IEEE 754-2008/2019 maxNum for binary64:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (max(+0.0, -0.0) -> +0.0).
      - Subnormals and infinities preserved.
    """
    a = a_bits & MASK64
    b = b_bits & MASK64

    a_is_nan = is_nan64(a)
    b_is_nan = is_nan64(b)

    if a_is_nan and b_is_nan:
        return a | QNAN_BIT64
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros
    if is_zero64(a) and is_zero64(b):
        return POS_ZERO64 if ((a == POS_ZERO64) or (b == POS_ZERO64)) else NEG_ZERO64

    fa = bits_to_f64(a)
    fb = bits_to_f64(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Numerically equal: positive sign bit is strictly larger (+0.0 > -0.0)
        if (a & SIGN_MASK64) == 0:
            return a
        if (b & SIGN_MASK64) == 0:
            return b
        return a


def ref_ds_max_rtn_f64(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_f64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        new_mem = oracle_max_f64(old_val, vdata0)

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    new_mem = oracle_max_f64(old_val, vdata0)
    return old_val, new_mem


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = DEFAULT_LDS_NUM_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


class LDSMemory64:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS) for 64-bit operations."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_qwords = size_bytes // 8
        self.qwords = [0] * self.num_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        return self.qwords[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        self.qwords[idx] = u64(val)

    def atomic_max_rtn_f64(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit double maximum with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        curr = self.qwords[idx]
        old_val, new_mem = ref_ds_max_rtn_f64(curr, vdata0)
        self.qwords[idx] = new_mem
        return old_val, new_mem


class TestLDSMemoryAndOracle64(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS 64-bit memory model."""

    def test_oracle_standard_doubles(self) -> None:
        # max(1.0, 2.0) = 2.0
        self.assertEqual(oracle_max_f64(ONE_F64, TWO_F64), TWO_F64)
        self.assertEqual(oracle_max_f64(TWO_F64, ONE_F64), TWO_F64)

        # max(10.0, -2.0) = 10.0
        self.assertEqual(oracle_max_f64(TEN_F64, NEG_TWO_F64), TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TWO_F64, TEN_F64), TEN_F64)

        # max(-10.0, -2.0) = -2.0
        self.assertEqual(oracle_max_f64(NEG_TEN_F64, NEG_TWO_F64), NEG_TWO_F64)

    def test_oracle_signed_zeros(self) -> None:
        # max(+0.0, -0.0) = +0.0
        self.assertEqual(oracle_max_f64(POS_ZERO64, NEG_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(POS_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, NEG_ZERO64), NEG_ZERO64)

    def test_oracle_infinities(self) -> None:
        self.assertEqual(oracle_max_f64(POS_INF64, ONE_F64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(POS_INF64, NEG_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_INF64), NEG_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, POS_INF64), POS_INF64)

    def test_oracle_nan_suppression(self) -> None:
        self.assertEqual(oracle_max_f64(QNAN_PAYLOAD_A64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(ONE_F64, QNAN_PAYLOAD_A64), ONE_F64)
        self.assertEqual(oracle_max_f64(SNAN_PAYLOAD_A64, NEG_TEN_F64), NEG_TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TEN_F64, SNAN_PAYLOAD_A64), NEG_TEN_F64)

        res_nan = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(res_nan))

    def test_oracle_subnormals(self) -> None:
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(SUBNORM_MAX64, SUBNORM_MIN64), SUBNORM_MAX64)
        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MAX64, POS_ZERO64), POS_ZERO64)

    def test_ref_ds_max_rtn_f64(self) -> None:
        old_val, new_mem = ref_ds_max_rtn_f64(ONE_F64, TWO_F64)
        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TWO_F64)

        old_val, new_mem = ref_ds_max_rtn_f64(TEN_F64, ONE_F64)
        self.assertEqual(old_val, TEN_F64)
        self.assertEqual(new_mem, TEN_F64)

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory64(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_qwords, 8192)
        self.assertEqual(len(mem.qwords), 8192)

        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory64(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x00, ONE_F64)
        self.assertEqual(mem.read_qword(0x00), ONE_F64)
        mem.write_qword(0x40, NEG_TEN_F64)
        self.assertEqual(mem.read_qword(0x40), NEG_TEN_F64)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x100, TWO_F64)
        for off in range(8):
            self.assertEqual(mem.read_qword(0x100 + off), TWO_F64)

    def test_lds_atomic_max_rtn_f64(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x20, ONE_F64)

        # max(1.0, 10.0) = 10.0
        old_val, new_mem = mem.atomic_max_rtn_f64(vaddr=0x20, vdata0=TEN_F64)
        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TEN_F64)
        self.assertEqual(mem.read_qword(0x20), TEN_F64)

        # max(10.0, 3.0) = 10.0
        old_val2, new_mem2 = mem.atomic_max_rtn_f64(vaddr=0x20, vdata0=THREE_F64)
        self.assertEqual(old_val2, TEN_F64)
        self.assertEqual(new_mem2, TEN_F64)
        self.assertEqual(mem.read_qword(0x20), TEN_F64)

    def test_lds_atomic_max_rtn_with_offset(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x50, NEG_TEN_F64)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_max_rtn_f64(vaddr=0x40, vdata0=TWO_F64, offset=0x10)
        self.assertEqual(old_val, NEG_TEN_F64)
        self.assertEqual(new_mem, TWO_F64)
        self.assertEqual(mem.read_qword(0x50), TWO_F64)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(enum.Enum):
    """Data Share (DS) atomic opcodes."""
    DS_MAX_RTN_F64 = "ds_max_rtn_f64"


class RegType(enum.Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation (scalar or paired registers)."""
    index: int
    count: int = 1
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index and self.index + self.count <= limit):
            raise ValueError(f"Register span out of range: {self.reg_type.value}{self.index} count={self.count}")
        if self.count <= 0:
            raise ValueError(f"Count must be positive, got {self.count}")

    @property
    def name(self) -> str:
        if self.count == 1:
            return f"{self.reg_type.value}{self.index}"
        return f"{self.reg_type.value}[{self.index}:{self.index + self.count - 1}]"


@dataclasses.dataclass(frozen=True)
class Immediate64:
    """64-bit immediate value representing double bit pattern."""
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u64(self.value))

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
class DSMaxRtnF64Instruction:
    """Symbolic AST node representing a ds_max_rtn_f64 instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: Union[RegisterOperand, Immediate64]
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_MAX_RTN_F64

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be VGPR, got {self.vdst.reg_type.value}")
        if self.vdst.count != 2:
            raise ValueError(f"vdst must be a 64-bit paired register (count=2), got {self.vdst.count}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be VGPR, got {self.vaddr.reg_type.value}")
        if self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a 32-bit register (count=1), got {self.vaddr.count}")
        if isinstance(self.vdata0, RegisterOperand):
            if self.vdata0.reg_type != RegType.VGPR:
                raise ValueError(f"vdata0 register must be VGPR, got {self.vdata0.reg_type.value}")
            if self.vdata0.count != 2:
                raise ValueError(f"vdata0 register must be a 64-bit paired register (count=2), got {self.vdata0.count}")
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
    def parse(cls, line: str) -> DSMaxRtnF64Instruction:
        import re
        raw = line.split("//")[0].split(";")[0].strip()
        if not raw:
            raise ValueError("Empty instruction line")
        tokens = raw.replace(",", " ").split()
        if not tokens:
            raise ValueError("No tokens found")
        mnemonic = tokens[0].lower()
        if mnemonic != DSOpcode.DS_MAX_RTN_F64.value:
            raise ValueError(f"Unknown opcode: {mnemonic}")

        if len(tokens) < 4:
            raise ValueError(f"Insufficient operands for {mnemonic}")

        def parse_reg(tok: str, expected_count: int) -> RegisterOperand:
            m_range = re.match(r"^([vs])\[(\d+):(\d+)\]$", tok.lower())
            if m_range:
                rtype = RegType.VGPR if m_range.group(1) == "v" else RegType.SGPR
                start = int(m_range.group(2))
                end = int(m_range.group(3))
                cnt = end - start + 1
                return RegisterOperand(start, cnt, rtype)

            m_single = re.match(r"^([vs])\[?(\d+)\]?$", tok.lower())
            if m_single:
                rtype = RegType.VGPR if m_single.group(1) == "v" else RegType.SGPR
                start = int(m_single.group(2))
                return RegisterOperand(start, expected_count, rtype)

            raise ValueError(f"Invalid register operand: {tok}")

        vdst = parse_reg(tokens[1], expected_count=2)
        vaddr = parse_reg(tokens[2], expected_count=1)
        vdata_tok = tokens[3]
        if vdata_tok.lower().startswith("v") or vdata_tok.lower().startswith("s"):
            vdata0: Union[RegisterOperand, Immediate64] = parse_reg(vdata_tok, expected_count=2)
        else:
            vdata0 = Immediate64(int(vdata_tok, 0))

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


class TestDSMaxRtnF64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_and_disasm(self) -> None:
        v0_1 = RegisterOperand(0, count=2, reg_type=RegType.VGPR)
        v2 = RegisterOperand(2, count=1, reg_type=RegType.VGPR)
        v4_5 = RegisterOperand(4, count=2, reg_type=RegType.VGPR)

        inst_simple = DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=v4_5)
        self.assertEqual(inst_simple.disasm(), "ds_max_rtn_f64 v[0:1], v2, v[4:5]")
        self.assertEqual(inst_simple.offset, 0)
        self.assertFalse(inst_simple.gds)

        inst_opts = DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=v4_5, offset=64, gds=True)
        self.assertEqual(inst_opts.disasm(), "ds_max_rtn_f64 v[0:1], v2, v[4:5] offset:64 gds")

        inst_imm = DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=Immediate64(ONE_F64), offset=16)
        self.assertEqual(inst_imm.disasm(), f"ds_max_rtn_f64 v[0:1], v2, {hex(ONE_F64)} offset:16")

    def test_ast_parse_canonical(self) -> None:
        inst1 = DSMaxRtnF64Instruction.parse("ds_max_rtn_f64 v[0:1], v2, v[4:5]")
        self.assertEqual(inst1.vdst, RegisterOperand(0, count=2, reg_type=RegType.VGPR))
        self.assertEqual(inst1.vaddr, RegisterOperand(2, count=1, reg_type=RegType.VGPR))
        self.assertEqual(inst1.vdata0, RegisterOperand(4, count=2, reg_type=RegType.VGPR))
        self.assertEqual(inst1.offset, 0)
        self.assertFalse(inst1.gds)

        inst2 = DSMaxRtnF64Instruction.parse("ds_max_rtn_f64 v[6:7], v8, v[10:11] offset:128 gds")
        self.assertEqual(inst2.vdst, RegisterOperand(6, count=2, reg_type=RegType.VGPR))
        self.assertEqual(inst2.vaddr, RegisterOperand(8, count=1, reg_type=RegType.VGPR))
        self.assertEqual(inst2.vdata0, RegisterOperand(10, count=2, reg_type=RegType.VGPR))
        self.assertEqual(inst2.offset, 128)
        self.assertTrue(inst2.gds)

        inst3 = DSMaxRtnF64Instruction.parse("ds_max_rtn_f64 v[0:1], v2, 0x3ff0000000000000 offset:0x20")
        self.assertEqual(inst3.vdata0, Immediate64(ONE_F64))
        self.assertEqual(inst3.offset, 32)

    def test_ast_validation_errors(self) -> None:
        v0_1 = RegisterOperand(0, count=2, reg_type=RegType.VGPR)
        v2 = RegisterOperand(2, count=1, reg_type=RegType.VGPR)
        v4_5 = RegisterOperand(4, count=2, reg_type=RegType.VGPR)
        s0_1 = RegisterOperand(0, count=2, reg_type=RegType.SGPR)
        s2 = RegisterOperand(2, count=1, reg_type=RegType.SGPR)
        v_single = RegisterOperand(0, count=1, reg_type=RegType.VGPR)

        # vdst cannot be SGPR
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=s0_1, vaddr=v2, vdata0=v4_5)

        # vdst count must be 2
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v_single, vaddr=v2, vdata0=v4_5)

        # vaddr cannot be SGPR
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v0_1, vaddr=s2, vdata0=v4_5)

        # vaddr count must be 1
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v0_1, vdata0=v4_5)

        # vdata0 cannot be SGPR
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=s0_1)

        # Negative offset
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=v4_5, offset=-1)

        # Offset exceeding 16 bits
        with self.assertRaises(ValueError):
            DSMaxRtnF64Instruction(vdst=v0_1, vaddr=v2, vdata0=v4_5, offset=65536)

        # Register span out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(255, count=2, reg_type=RegType.VGPR)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState64:
    """Architectural state for a single execution thread with scalar register file and 64-bit LDS."""

    def __init__(
        self,
        lds: Optional[LDSMemory64] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
        num_sgprs: int = DEFAULT_NUM_SGPRS,
    ) -> None:
        self.lds: LDSMemory64 = lds if lds is not None else LDSMemory64()
        self.vgprs: List[int] = [0] * num_vgprs
        self.sgprs: List[int] = [0] * num_sgprs

    def get_vgpr32(self, idx: int) -> int:
        return self.vgprs[idx]

    def set_vgpr32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)

    def get_vgpr64(self, idx: int) -> int:
        low = self.vgprs[idx]
        high = self.vgprs[idx + 1]
        return u64(low | (high << 32))

    def set_vgpr64(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)
        self.vgprs[idx + 1] = u32(val >> 32)


class ScalarDSInterpreter64:
    """Deterministic scalar interpreter executing 64-bit DS instructions."""

    def execute(self, state: ScalarThreadState64, inst: DSMaxRtnF64Instruction) -> Tuple[int, int]:
        # 1. Latch input operands before any register update
        base_addr = state.get_vgpr32(inst.vaddr.index)
        if isinstance(inst.vdata0, RegisterOperand):
            vdata_val = state.get_vgpr64(inst.vdata0.index)
        else:
            vdata_val = inst.vdata0.value

        # 2. Atomic memory transaction on LDS
        old_val, new_mem = state.lds.atomic_max_rtn_f64(
            vaddr=base_addr,
            vdata0=vdata_val,
            offset=inst.offset,
        )

        # 3. Write return value to destination VGPR pair
        state.set_vgpr64(inst.vdst.index, old_val)

        return old_val, new_mem


class TestScalarInterpreter64(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory64(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState64(lds=self.lds)
        self.interp = ScalarDSInterpreter64()

    def test_scalar_execution_basic(self) -> None:
        addr = 0x40
        self.lds.write_qword(addr, ONE_F64)

        self.state.set_vgpr32(2, addr)
        self.state.set_vgpr64(4, TEN_F64)
        self.state.set_vgpr64(0, 0xCAFEBABEDEADBEEF)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TEN_F64)
        self.assertEqual(self.state.get_vgpr64(0), ONE_F64)
        self.assertEqual(self.lds.read_qword(addr), TEN_F64)

    def test_scalar_execution_with_offset(self) -> None:
        base_addr = 0x100
        offset = 0x18
        eff_addr = base_addr + offset

        self.lds.write_qword(eff_addr, NEG_TEN_F64)
        self.state.set_vgpr32(2, base_addr)
        self.state.set_vgpr64(4, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
            offset=offset,
        )

        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, NEG_TEN_F64)
        self.assertEqual(new_mem, ONE_F64)
        self.assertEqual(self.state.get_vgpr64(0), NEG_TEN_F64)
        self.assertEqual(self.lds.read_qword(eff_addr), ONE_F64)

    def test_scalar_execution_with_immediate(self) -> None:
        addr = 0x200
        self.lds.write_qword(addr, HALF_F64)
        self.state.set_vgpr32(2, addr)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=Immediate64(TWO_F64),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)

        self.assertEqual(old_val, HALF_F64)
        self.assertEqual(new_mem, TWO_F64)
        self.assertEqual(self.state.get_vgpr64(0), HALF_F64)
        self.assertEqual(self.lds.read_qword(addr), TWO_F64)

    def test_scalar_unaligned_byte_address(self) -> None:
        self.lds.write_qword(0x300, ONE_F64)
        self.state.set_vgpr32(2, 0x307)
        self.state.set_vgpr64(4, THREE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, THREE_F64)
        self.assertEqual(self.lds.read_qword(0x300), THREE_F64)

    def test_scalar_signed_zero_ordering(self) -> None:
        addr = 0x80
        self.lds.write_qword(addr, NEG_ZERO64)
        self.state.set_vgpr32(2, addr)
        self.state.set_vgpr64(4, POS_ZERO64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, NEG_ZERO64)
        self.assertEqual(new_mem, POS_ZERO64)
        self.assertEqual(self.state.get_vgpr64(0), NEG_ZERO64)
        self.assertEqual(self.lds.read_qword(addr), POS_ZERO64)

    def test_scalar_nan_suppression(self) -> None:
        addr = 0x90
        self.lds.write_qword(addr, QNAN_PAYLOAD_A64)
        self.state.set_vgpr32(2, addr)
        self.state.set_vgpr64(4, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        old_val, new_mem = self.interp.execute(self.state, inst)
        self.assertEqual(old_val, QNAN_PAYLOAD_A64)
        self.assertEqual(new_mem, ONE_F64)
        self.assertEqual(self.state.get_vgpr64(0), QNAN_PAYLOAD_A64)
        self.assertEqual(self.lds.read_qword(addr), ONE_F64)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState64:
    """Architectural state for a SIMD wavefront executing 64-bit LDS instructions."""

    def __init__(
        self,
        wave_size: int = 64,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory64] = None,
        num_vgprs: int = DEFAULT_NUM_VGPRS,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.mask_limit: int = (1 << wave_size) - 1
        self.exec_mask: int = (self.mask_limit if exec_mask is None else exec_mask) & self.mask_limit
        self.lds: LDSMemory64 = lds if lds is not None else LDSMemory64()
        self.lane_vgprs: List[List[int]] = [[0] * num_vgprs for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr32(self, lane: int, idx: int) -> int:
        return self.lane_vgprs[lane][idx]

    def set_vgpr32(self, lane: int, idx: int, val: int) -> None:
        self.lane_vgprs[lane][idx] = u32(val)

    def get_vgpr64(self, lane: int, idx: int) -> int:
        low = self.lane_vgprs[lane][idx]
        high = self.lane_vgprs[lane][idx + 1]
        return u64(low | (high << 32))

    def set_vgpr64(self, lane: int, idx: int, val: int) -> None:
        self.lane_vgprs[lane][idx] = u32(val)
        self.lane_vgprs[lane][idx + 1] = u32(val >> 32)


class SIMDDSInterpreter64:
    """SIMD wavefront interpreter handling divergence and EXEC masking for 64-bit DS instructions."""

    def execute(
        self,
        wave: SIMDWaveState64,
        inst: DSMaxRtnF64Instruction,
    ) -> List[Optional[Tuple[int, int]]]:
        results: List[Optional[Tuple[int, int]]] = [None] * wave.wave_size

        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue

            base_addr = wave.get_vgpr32(lane, inst.vaddr.index)
            if isinstance(inst.vdata0, RegisterOperand):
                vdata_val = wave.get_vgpr64(lane, inst.vdata0.index)
            else:
                vdata_val = inst.vdata0.value

            old_val, new_mem = wave.lds.atomic_max_rtn_f64(
                vaddr=base_addr,
                vdata0=vdata_val,
                offset=inst.offset,
            )

            wave.set_vgpr64(lane, inst.vdst.index, old_val)
            results[lane] = (old_val, new_mem)

        return results

    def detect_bank_conflicts(
        self,
        wave: SIMDWaveState64,
        inst: DSMaxRtnF64Instruction,
    ) -> Dict[int, List[int]]:
        bank_map: Dict[int, Dict[int, List[int]]] = {}
        for lane in range(wave.wave_size):
            if not wave.is_lane_active(lane):
                continue
            base_addr = wave.get_vgpr32(lane, inst.vaddr.index)
            eff_addr = calc_lds_effective_address(base_addr, inst.offset)
            qword_idx = calc_lds_qword_index(eff_addr, wave.lds.num_qwords)

            bank0 = calc_lds_bank(eff_addr)
            bank1 = calc_lds_bank(eff_addr + 4)

            for b in (bank0, bank1):
                if b not in bank_map:
                    bank_map[b] = {}
                if qword_idx not in bank_map[b]:
                    bank_map[b][qword_idx] = []
                bank_map[b][qword_idx].append(lane)

        conflicts: Dict[int, List[int]] = {}
        for b, qwords in bank_map.items():
            if len(qwords) > 1:
                all_lanes: List[int] = []
                for lane_list in qwords.values():
                    all_lanes.extend(lane_list)
                conflicts[b] = sorted(list(set(all_lanes)))

        return conflicts


class TestSIMDInterpreter64(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_exec_independent_addresses(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter64()

        for lane in range(32):
            addr = lane * 8
            lds.write_qword(addr, ONE_F64)
            wave.set_vgpr32(lane, 2, addr)
            wave.set_vgpr64(lane, 4, f64_to_bits(float(lane + 2)))
            wave.set_vgpr64(lane, 0, 0xCAFEBABE00000000)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        results = interp.execute(wave, inst)

        for lane in range(32):
            self.assertIsNotNone(results[lane])
            old_val, new_mem = results[lane]  # type: ignore[misc]
            self.assertEqual(old_val, ONE_F64)
            expected_new = oracle_max_f64(ONE_F64, f64_to_bits(float(lane + 2)))
            self.assertEqual(new_mem, expected_new)
            self.assertEqual(wave.get_vgpr64(lane, 0), ONE_F64)
            self.assertEqual(lds.read_qword(lane * 8), expected_new)

    def test_wave64_full_exec(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=64, exec_mask=MASK64, lds=lds)
        interp = SIMDDSInterpreter64()

        for lane in range(64):
            addr = lane * 8
            lds.write_qword(addr, ONE_F64)
            wave.set_vgpr32(lane, 2, addr)
            wave.set_vgpr64(lane, 4, TWO_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        results = interp.execute(wave, inst)

        for lane in range(64):
            self.assertIsNotNone(results[lane])
            old_val, new_mem = results[lane]  # type: ignore[misc]
            self.assertEqual(old_val, ONE_F64)
            self.assertEqual(new_mem, TWO_F64)
            self.assertEqual(wave.get_vgpr64(lane, 0), ONE_F64)
            self.assertEqual(lds.read_qword(lane * 8), TWO_F64)

    def test_simd_exec_mask_inactive_lanes(self) -> None:
        lds = LDSMemory64(65536)
        even_mask = 0x5555555555555555
        wave = SIMDWaveState64(wave_size=64, exec_mask=even_mask, lds=lds)
        interp = SIMDDSInterpreter64()

        for lane in range(64):
            addr = lane * 8
            lds.write_qword(addr, ONE_F64)
            wave.set_vgpr32(lane, 2, addr)
            wave.set_vgpr64(lane, 4, TEN_F64)
            wave.set_vgpr64(lane, 0, 0x1234567890ABCDEF)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        results = interp.execute(wave, inst)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(wave.get_vgpr64(lane, 0), ONE_F64)
                self.assertEqual(lds.read_qword(lane * 8), TEN_F64)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.get_vgpr64(lane, 0), 0x1234567890ABCDEF)
                self.assertEqual(lds.read_qword(lane * 8), ONE_F64)

    def test_simd_zero_exec_mask(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0, lds=lds)
        interp = SIMDDSInterpreter64()

        lds.write_qword(0, ONE_F64)
        wave.set_vgpr32(0, 2, 0)
        wave.set_vgpr64(0, 4, TEN_F64)
        wave.set_vgpr64(0, 0, 0xDEADBEEFCAFEBABE)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        results = interp.execute(wave, inst)
        self.assertTrue(all(r is None for r in results))
        self.assertEqual(wave.get_vgpr64(0, 0), 0xDEADBEEFCAFEBABE)
        self.assertEqual(lds.read_qword(0), ONE_F64)

    def test_simd_multi_lane_serialization_same_address(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0x7, lds=lds)
        interp = SIMDDSInterpreter64()

        addr = 0x20
        lds.write_qword(addr, ONE_F64)

        # Lane 0: 2.0, Lane 1: 3.0, Lane 2: 10.0
        wave.set_vgpr32(0, 2, addr)
        wave.set_vgpr64(0, 4, TWO_F64)

        wave.set_vgpr32(1, 2, addr)
        wave.set_vgpr64(1, 4, THREE_F64)

        wave.set_vgpr32(2, 2, addr)
        wave.set_vgpr64(2, 4, TEN_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        results = interp.execute(wave, inst)

        # Lane 0: sees 1.0, writes max(1.0, 2.0) = 2.0
        self.assertEqual(results[0], (ONE_F64, TWO_F64))
        self.assertEqual(wave.get_vgpr64(0, 0), ONE_F64)

        # Lane 1: sees 2.0, writes max(2.0, 3.0) = 3.0
        self.assertEqual(results[1], (TWO_F64, THREE_F64))
        self.assertEqual(wave.get_vgpr64(1, 0), TWO_F64)

        # Lane 2: sees 3.0, writes max(3.0, 10.0) = 10.0
        self.assertEqual(results[2], (THREE_F64, TEN_F64))
        self.assertEqual(wave.get_vgpr64(2, 0), THREE_F64)

        self.assertEqual(lds.read_qword(addr), TEN_F64)

    def test_bank_conflict_detection(self) -> None:
        lds = LDSMemory64(65536)
        wave = SIMDWaveState64(wave_size=32, exec_mask=0x3, lds=lds)
        interp = SIMDDSInterpreter64()

        wave.set_vgpr32(0, 2, 0)
        wave.set_vgpr32(1, 2, 128)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=Immediate64(ONE_F64),
        )

        conflicts = interp.detect_bank_conflicts(wave, inst)
        self.assertIn(0, conflicts)
        self.assertEqual(conflicts[0], [0, 1])


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing64(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory64(65536)
        self.interp = ScalarDSInterpreter64()

    def test_aliasing_vdst_overlaps_vaddr(self) -> None:
        # vdst = v[0:1], vaddr = v0, vdata0 = v[2:3]
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr32(0, 0x40)
        state.set_vgpr64(2, TEN_F64)
        self.lds.write_qword(0x40, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(0, count=1),
            vdata0=RegisterOperand(2, count=2),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TEN_F64)
        self.assertEqual(state.get_vgpr64(0), ONE_F64)
        self.assertEqual(state.get_vgpr64(2), TEN_F64)
        self.assertEqual(self.lds.read_qword(0x40), TEN_F64)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        # vdst = v[2:3], vaddr = v0, vdata0 = v[2:3]
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr32(0, 0x60)
        state.set_vgpr64(2, THREE_F64)
        self.lds.write_qword(0x60, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(2, count=2),
            vaddr=RegisterOperand(0, count=1),
            vdata0=RegisterOperand(2, count=2),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, THREE_F64)
        self.assertEqual(state.get_vgpr64(2), ONE_F64)
        self.assertEqual(state.get_vgpr32(0), 0x60)
        self.assertEqual(self.lds.read_qword(0x60), THREE_F64)

    def test_aliasing_vdst_partially_overlaps_vdata0(self) -> None:
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr32(0, 0x80)
        state.set_vgpr64(3, TEN_F64)
        self.lds.write_qword(0x80, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(2, count=2),
            vaddr=RegisterOperand(0, count=1),
            vdata0=RegisterOperand(3, count=2),
        )

        old_val, new_mem = self.interp.execute(state, inst)

        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TEN_F64)
        self.assertEqual(state.get_vgpr64(2), ONE_F64)
        self.assertEqual(self.lds.read_qword(0x80), TEN_F64)

    def test_operand_preservation_distinct_registers(self) -> None:
        state = ScalarThreadState64(lds=self.lds)
        state.set_vgpr64(0, 0xAAAAAAAAAAAAAAAA)
        state.set_vgpr32(2, 0xA0)
        state.set_vgpr64(4, TEN_F64)
        self.lds.write_qword(0xA0, ONE_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(0, count=2),
            vaddr=RegisterOperand(2, count=1),
            vdata0=RegisterOperand(4, count=2),
        )

        self.interp.execute(state, inst)

        self.assertEqual(state.get_vgpr64(0), ONE_F64)
        self.assertEqual(state.get_vgpr32(2), 0xA0)
        self.assertEqual(state.get_vgpr64(4), TEN_F64)

    def test_vgpr_isolation_no_spurious_clobber(self) -> None:
        state = ScalarThreadState64(lds=self.lds)
        for i in range(256):
            state.set_vgpr32(i, 0x1000 + i)

        addr = 0x20
        self.lds.write_qword(addr, ONE_F64)
        state.set_vgpr32(10, addr)
        state.set_vgpr64(12, TEN_F64)

        inst = DSMaxRtnF64Instruction(
            vdst=RegisterOperand(14, count=2),
            vaddr=RegisterOperand(10, count=1),
            vdata0=RegisterOperand(12, count=2),
        )

        self.interp.execute(state, inst)

        self.assertEqual(state.get_vgpr64(14), ONE_F64)
        for i in range(256):
            if i in (14, 15):
                continue
            elif i == 10:
                self.assertEqual(state.get_vgpr32(i), addr)
            elif i == 12:
                self.assertEqual(state.get_vgpr32(i), TEN_F64 & MASK32)
            elif i == 13:
                self.assertEqual(state.get_vgpr32(i), (TEN_F64 >> 32) & MASK32)
            else:
                self.assertEqual(state.get_vgpr32(i), 0x1000 + i)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class Float64ConditionCodes:
    """Architectural condition codes evaluating double-precision float maximum results."""
    zero: bool
    negative: bool
    nan: bool
    infinity: bool
    subnormal: bool
    updated: bool


@dataclasses.dataclass(frozen=True)
class LDSStatusFlags64:
    """Status flags for 64-bit LDS addressing and dual-bank mapping."""
    qword_aligned: bool
    address_wrapped: bool
    bank_span: Tuple[int, int]


def compute_float64_condition_codes(old_val: int, vdata0: int, result: int) -> Float64ConditionCodes:
    """Compute condition codes from double-precision float atomic max result."""
    res_u = u64(result)
    old_u = u64(old_val)
    return Float64ConditionCodes(
        zero=is_zero64(res_u),
        negative=bool(res_u & SIGN_MASK64),
        nan=is_nan64(res_u),
        infinity=is_inf64(res_u),
        subnormal=is_subnormal64(res_u),
        updated=(res_u != old_u),
    )


def compute_lds64_status_flags(
    vaddr: int,
    offset: int = 0,
    lds_size_bytes: int = DEFAULT_LDS_SIZE_BYTES,
) -> LDSStatusFlags64:
    """Evaluate 8-byte alignment, dual-bank span, and boundary wrap status for 64-bit LDS access."""
    raw_eff = vaddr + (offset & OFFSET16_MASK)
    eff_addr = raw_eff & MASK32
    return LDSStatusFlags64(
        qword_aligned=((eff_addr & 7) == 0),
        address_wrapped=(raw_eff >= lds_size_bytes),
        bank_span=(calc_lds_bank(eff_addr), calc_lds_bank(eff_addr + 4)),
    )


class TestConditionCodesAndStatusFlags64(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and update detection."""

    def test_condition_code_zero(self) -> None:
        cc_pos = compute_float64_condition_codes(NEG_ONE_F64, POS_ZERO64, POS_ZERO64)
        self.assertTrue(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.updated)

        cc_neg = compute_float64_condition_codes(NEG_TWO_F64, NEG_ZERO64, NEG_ZERO64)
        self.assertTrue(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertTrue(cc_neg.updated)

    def test_condition_code_negative(self) -> None:
        cc = compute_float64_condition_codes(NEG_TEN_F64, NEG_TWO_F64, NEG_TWO_F64)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.nan)
        self.assertFalse(cc.infinity)
        self.assertTrue(cc.updated)

    def test_condition_code_nan(self) -> None:
        res_nan = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        cc = compute_float64_condition_codes(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64, res_nan)
        self.assertTrue(cc.nan)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.infinity)

    def test_condition_code_infinity(self) -> None:
        cc_pos_inf = compute_float64_condition_codes(TEN_F64, POS_INF64, POS_INF64)
        self.assertTrue(cc_pos_inf.infinity)
        self.assertFalse(cc_pos_inf.negative)
        self.assertTrue(cc_pos_inf.updated)

        # max(-inf, -10.0) -> -10.0
        cc_neg_inf = compute_float64_condition_codes(NEG_INF64, NEG_TEN_F64, NEG_TEN_F64)
        self.assertFalse(cc_neg_inf.infinity)
        self.assertTrue(cc_neg_inf.updated)

    def test_condition_code_subnormal(self) -> None:
        cc = compute_float64_condition_codes(NEG_ONE_F64, SUBNORM_MIN64, SUBNORM_MIN64)
        self.assertTrue(cc.subnormal)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.updated)

    def test_condition_code_updated_flag(self) -> None:
        # Memory had 1.0, operand was 10.0 -> updated to 10.0
        cc1 = compute_float64_condition_codes(ONE_F64, TEN_F64, TEN_F64)
        self.assertTrue(cc1.updated)

        # Memory had 10.0, operand was 1.0 -> max is 10.0 (unchanged)
        cc2 = compute_float64_condition_codes(TEN_F64, ONE_F64, TEN_F64)
        self.assertFalse(cc2.updated)

    def test_lds_status_alignment_and_wrap(self) -> None:
        st_aligned = compute_lds64_status_flags(0x100, 0x0)
        self.assertTrue(st_aligned.qword_aligned)
        self.assertFalse(st_aligned.address_wrapped)
        self.assertEqual(st_aligned.bank_span, ((0x100 >> 2) % 32, (0x104 >> 2) % 32))

        st_unaligned = compute_lds64_status_flags(0x104, 0x0)
        self.assertFalse(st_unaligned.qword_aligned)

        st_wrapped = compute_lds64_status_flags(65536, 0x10)
        self.assertTrue(st_wrapped.address_wrapped)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BoundaryExtremes64(unittest.TestCase):
    """Phase 7: Unit tests for double-precision IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros_comprehensive(self) -> None:
        self.assertEqual(oracle_max_f64(POS_ZERO64, NEG_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(POS_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, NEG_ZERO64), NEG_ZERO64)

    def test_infinities_comprehensive(self) -> None:
        self.assertEqual(oracle_max_f64(POS_INF64, POS_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_INF64), NEG_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, NEG_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, POS_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, DBL_MAX), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_DBL_MAX), NEG_DBL_MAX)

    def test_subnormals_preservation_and_comparison(self) -> None:
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), DBL_MIN)
        self.assertNotEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), POS_ZERO64)

        self.assertEqual(oracle_max_f64(SUBNORM_MAX64, DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, POS_ZERO64), SUBNORM_MIN64)
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, NEG_ZERO64), SUBNORM_MIN64)

        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MIN64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MAX64, NEG_SUBNORM_MIN64), NEG_SUBNORM_MIN64)

    def test_nan_suppression_comprehensive(self) -> None:
        for normal_val in (ONE_F64, NEG_ONE_F64, POS_INF64, NEG_INF64, POS_ZERO64, NEG_ZERO64, SUBNORM_MIN64):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_max_f64(QNAN_PAYLOAD_A64, normal_val), normal_val)
                self.assertEqual(oracle_max_f64(normal_val, QNAN_PAYLOAD_A64), normal_val)

        for normal_val in (TWO_F64, NEG_TWO_F64, DBL_MAX, NEG_DBL_MAX):
            with self.subTest(val=hex(normal_val)):
                self.assertEqual(oracle_max_f64(SNAN_PAYLOAD_A64, normal_val), normal_val)
                self.assertEqual(oracle_max_f64(normal_val, SNAN_PAYLOAD_A64), normal_val)

    def test_dual_nans(self) -> None:
        q_q = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(q_q))

        s_s = oracle_max_f64(SNAN_PAYLOAD_A64, SNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(s_s))

        q_s = oracle_max_f64(QNAN_PAYLOAD_A64, SNAN_PAYLOAD_A64)
        self.assertTrue(is_qnan64(q_s))

    def test_floating_point_normal_extremes(self) -> None:
        self.assertEqual(oracle_max_f64(DBL_MAX, NEG_DBL_MAX), DBL_MAX)
        self.assertEqual(oracle_max_f64(NEG_DBL_MAX, DBL_MAX), DBL_MAX)
        self.assertEqual(oracle_max_f64(DBL_MIN, NEG_DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(NEG_DBL_MIN, DBL_MIN), DBL_MIN)


if __name__ == "__main__":
    unittest.main()
